"""Knob selection for the trained-head consultant, and per-seed head builds.

Two consultants now select their own `top_k_percent` / `injection_scale`. The
things that must hold:

  1. their sweeps never share a directory — a stale prototype candidate score
     read as a head score would silently pick the wrong knob;
  2. `finalize --use-llm-head` promotes the head block but keeps the prototype
     knobs under `prototype_superseded`, because those are what
     `results/*_current.json` schema 4/6 was measured under and the ladder there
     cannot be re-derived without them;
  3. a head built at seed 42 still reproduces the committed canonical file, and
     any other seed writes somewhere else so it cannot clobber it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.pipeline.common.datasets import get_dataset_config
from src.pipeline.step4 import build_llm_heads as bh
from src.pipeline.step4 import select_feedback_knobs as sel

DATASETS = ("unsw_nb15", "ton_iot")


# --- 1. the two consultants' sweeps are kept apart ----------------------------

def test_head_and_prototype_sweeps_use_separate_trees():
    for dataset in DATASETS:
        proto = sel.run_root(dataset, use_llm_head=False)
        head = sel.run_root(dataset, use_llm_head=True)
        assert proto != head
        assert proto.parts[1] == "knob_selection_v2"
        assert head.parts[1] == "knob_selection_head"


def test_collect_reads_the_matching_tree(tmp_path, monkeypatch):
    """`_collect` must not fall back to the other consultant's curves."""
    monkeypatch.setattr(sel, "RESULT_ROOT", tmp_path / "proto")
    monkeypatch.setattr(sel, "HEAD_RESULT_ROOT", tmp_path / "head")
    for use_head, value in ((False, 0.1), (True, 0.9)):
        root = sel.run_root("unsw_nb15", use_head)
        root.mkdir(parents=True, exist_ok=True)
        for seed in sel.SELECTION_SEEDS:
            (root / f"top_k_seed{seed}.json").write_text(json.dumps(
                {"validation_by_candidate": {"20.0": value, "25.0": value / 2}}
            ))
    assert sel.selected_top_k("unsw_nb15", use_llm_head=False) == 20.0
    proto = sel._collect("unsw_nb15", "top_k", use_llm_head=False)
    head = sel._collect("unsw_nb15", "top_k", use_llm_head=True)
    assert proto[42][20.0] == 0.1
    assert head[42][20.0] == 0.9


def test_missing_head_sweep_raises_rather_than_using_prototype_curves(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(sel, "RESULT_ROOT", tmp_path / "proto")
    monkeypatch.setattr(sel, "HEAD_RESULT_ROOT", tmp_path / "head")
    root = sel.run_root("unsw_nb15", False)
    root.mkdir(parents=True, exist_ok=True)
    for seed in sel.SELECTION_SEEDS:
        (root / f"top_k_seed{seed}.json").write_text(
            json.dumps({"validation_by_candidate": {"20.0": 0.5}})
        )
    with pytest.raises(FileNotFoundError):
        sel._collect("unsw_nb15", "top_k", use_llm_head=True)


# --- 2. finalize keeps the prototype knobs ------------------------------------

def _fake_curves():
    return {
        "top_k": {str(s): {"20.0": 0.5} for s in sel.SELECTION_SEEDS},
        "injection_scale": {str(s): {"2.0": 0.5} for s in sel.SELECTION_SEEDS},
    }


def test_finalize_head_promotes_the_head_and_preserves_the_prototype(
    tmp_path, monkeypatch
):
    previous = {
        "dataset": "unsw_nb15",
        "source": "multiseed_validation_sweep",
        "top_k_percent": 31.0,
        "injection_scale": 2.0,
        "bias_confidence_fraction": 0.5,
        "effective_feedback_percent": 15.5,
        "semantic_consultant": "whitened_prototype_scorer",
        "trained_llm_head": False,
        "active_condition": "A",
        "condition_c": {"top_k_percent": 35.0},
        "selection_uses_test_labels": False,
    }
    written = {}
    monkeypatch.setattr(
        sel, "selected_config_path",
        lambda ds, root=None: tmp_path / "selected_feedback_config.json",
    )
    path = sel._finalize_head(
        "unsw_nb15", previous, 22.0, 5.0, sel.SELECTION_SEEDS, _fake_curves()
    )
    payload = json.loads(Path(path).read_text())

    # Active block is the head.
    assert payload["consultant"] == "trained_llm_head"
    assert payload["semantic_consultant"] == "trained_llm_head"
    assert payload["trained_llm_head"] is True
    assert payload["top_k_percent"] == 22.0
    assert payload["injection_scale"] == 5.0
    assert payload["effective_feedback_percent"] == 11.0
    assert payload["selection_uses_test_labels"] is False

    # The prototype knobs the schema-4/6 contracts were measured under survive.
    proto = payload["prototype_superseded"]
    assert proto["top_k_percent"] == 31.0
    assert proto["injection_scale"] == 2.0
    assert proto["semantic_consultant"] == "whitened_prototype_scorer"
    assert "condition_c" in proto and proto["condition_c"]["top_k_percent"] == 35.0
    assert "condition_c" not in payload, "condition_c must move under the prototype block"

    head = payload["consultant_trained_llm_head"]
    assert head["top_k_percent"] == 22.0 and head["injection_scale"] == 5.0
    assert head["selection_curves"] == _fake_curves()


def test_finalize_head_flags_a_boundary_selection(tmp_path, monkeypatch):
    """A knob chosen at the edge of its range means the range was too narrow;
    the config has to say so rather than leaving it to be noticed."""
    previous = {
        "dataset": "ton_iot", "top_k_percent": 25.0, "injection_scale": 20.0,
        "bias_confidence_fraction": 0.5, "selection_uses_test_labels": False,
    }
    monkeypatch.setattr(
        sel, "selected_config_path",
        lambda ds, root=None: tmp_path / "cfg.json",
    )
    edge = sel._finalize_head(
        "ton_iot", previous, float(max(sel.TOP_K_RANGE)),
        float(max(sel.SCALE_CANDIDATES)), sel.SELECTION_SEEDS, _fake_curves()
    )
    block = json.loads(Path(edge).read_text())["consultant_trained_llm_head"]
    assert block["top_k_on_range_boundary"] is True
    assert block["scale_on_range_boundary"] is True

    inside = sel._finalize_head(
        "ton_iot", previous, 22.0, 5.0, sel.SELECTION_SEEDS, _fake_curves()
    )
    block = json.loads(Path(inside).read_text())["consultant_trained_llm_head"]
    assert block["top_k_on_range_boundary"] is False
    assert block["scale_on_range_boundary"] is False


# --- 3. per-seed head builds ---------------------------------------------------

def test_seed_42_head_still_reproduces_the_canonical_file():
    for dataset in DATASETS:
        cfg = get_dataset_config(dataset)
        canonical = Path(
            f"data/{dataset}/processed/step4_feedback/llm_head_logits.pt"
        )
        if not (Path(cfg.graph_path).exists() and canonical.exists()):
            pytest.skip(f"missing artifacts for {dataset}")
        labels, emb, folds = bh._load_inputs(cfg)
        stored = torch.load(canonical, weights_only=False).float()
        got = bh._train_fold(emb, labels, folds[0], 0, cfg.eval_classes, seed=bh.SEED)
        torch.testing.assert_close(got, stored[0], rtol=0, atol=0)


def test_a_non_default_seed_writes_a_different_head():
    """Seeds 1 and 2 exist so the head-alone baseline carries its own +/-. If
    they matched seed 42 the baseline's std would be a fiction."""
    for dataset in DATASETS:
        base = Path(f"data/{dataset}/processed/step4_feedback")
        paths = [base / "llm_head_logits.pt"] + [
            base / f"llm_head_logits_seed{s}.pt" for s in (1, 2)
        ]
        if not all(p.exists() for p in paths):
            pytest.skip(f"missing per-seed head files for {dataset}")
        tensors = [torch.load(p, weights_only=False).float() for p in paths]
        assert tensors[1].shape == tensors[0].shape
        assert not torch.allclose(tensors[0], tensors[1])
        assert not torch.allclose(tensors[1], tensors[2])


def test_non_default_seed_never_targets_the_canonical_path():
    """The CLI must route S != 42 to its own file. Losing llm_head_logits.pt to a
    seed sweep would silently change every rung that consults it."""
    import inspect

    src = inspect.getsource(bh.main)
    assert "llm_head_logits_seed" in src
    assert "args.seed != SEED" in src
