"""Task 3 — close the injection_scale trap.

`selected_feedback_config.json` records a per-dataset `injection_scale` (2.0 UNSW /
20.0 ToN), but `train_feedback` resolved only `top_k_percent` and
`bias_confidence_fraction` from that file; `--injection-scale` was a plain argparse
default of 10.0. The canonical run passed the flag explicitly, the 3-seed sweep behind
`results/multiseed_ladder.json` did not, and nothing in the artifacts made the
difference visible: `benchmark_summary.json` recorded `injection_scale: 10.0` beside a
`selected_config` block saying 2.0.

These tests make the config the single source of truth, make the artifacts report the
value that RAN, and make the multi-seed aggregator refuse to pool runs that disagree.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch

from src.pipeline.common.datasets import get_dataset_config

DATASET = "unsw_nb15"
CANON = Path(f"data/{DATASET}/processed/step4_feedback")


def _require(*paths):
    for p in paths:
        if not Path(p).exists():
            pytest.skip(f"missing pipeline artifact {p}")


def _write_config(root: Path, *, injection_scale=2.0, top_k=31.0, dataset=DATASET):
    payload = {
        "schema_version": 1,
        "dataset": dataset,
        "source": "test_fixture",
        "selection_uses_test_labels": False,
        "top_k_percent": top_k,
        "bias_confidence_fraction": 0.5,
        "gate_mode": "confidence",
        "max_feedback_iterations": 3,
        "churn_tolerance": 0.01,
        "semantic_consultant": "whitened_prototype_scorer",
        "trained_llm_head": False,
        "injection_mode": "edge",
    }
    if injection_scale is not None:
        payload["injection_scale"] = injection_scale
    root.mkdir(parents=True, exist_ok=True)
    (root / "selected_feedback_config.json").write_text(json.dumps(payload, indent=2))
    return payload


def _run_feedback(root: Path, **kwargs):
    """Cheapest possible real train_feedback run into an isolated output dir."""
    import src.pipeline.step4.train_feedback as tf

    cfg = get_dataset_config(DATASET)
    _require(cfg.graph_path, cfg.llm_embedding_path, cfg.splits_path,
             CANON / "prototypes.pt")
    saved = (tf.MAX_EPOCHS, tf.EARLY_STOPPING_PATIENCE, tf.SEED)
    tf.MAX_EPOCHS, tf.EARLY_STOPPING_PATIENCE = 2, 1
    try:
        return tf.train_feedback(
            DATASET, modes=["real"], output_dir=root,
            prototypes_path=CANON / "prototypes.pt", **kwargs,
        )
    finally:
        tf.MAX_EPOCHS, tf.EARLY_STOPPING_PATIENCE, tf.SEED = saved


# --- 3.0 / 3.1 ---------------------------------------------------------------

def test_injection_scale_comes_from_the_config_when_no_flag_is_given(tmp_path):
    """The trap itself: no flag must mean the config's value, not 10.0."""
    root = tmp_path / "fb"
    _write_config(root, injection_scale=2.0)
    _run_feedback(root)
    benchmark = json.loads((root / "benchmark_summary.json").read_text())
    assert benchmark["injection_scale"] == 2.0, (
        f"train_feedback ran at injection_scale={benchmark['injection_scale']} while "
        f"the selected config said 2.0"
    )


def test_explicit_injection_scale_overrides_the_config(tmp_path):
    root = tmp_path / "fb"
    _write_config(root, injection_scale=2.0)
    _run_feedback(root, injection_scale=7.0)
    benchmark = json.loads((root / "benchmark_summary.json").read_text())
    assert benchmark["injection_scale"] == 7.0


def test_missing_injection_scale_raises_rather_than_defaulting(tmp_path):
    """No silent fallback to a number — that is how the trap was sprung."""
    root = tmp_path / "fb"
    _write_config(root, injection_scale=None)
    with pytest.raises(Exception) as exc:
        _run_feedback(root)
    assert "injection_scale" in str(exc.value)


def test_sweep_top_k_also_refuses_to_default_the_scale(tmp_path, monkeypatch):
    """A sweep must not rank candidates under a scale the feedback stage will not use."""
    from src.pipeline.step4 import sweep_top_k

    # A config that has everything except the scale — the shape `load_feedback_config`
    # returns when no selected config file exists.
    monkeypatch.setattr(
        sweep_top_k, "load_feedback_config",
        lambda dataset, *a, **k: {"top_k_percent": 31.0, "bias_confidence_fraction": 0.5},
    )
    root = tmp_path / "sweep"
    root.mkdir(parents=True)
    with pytest.raises(ValueError) as exc:
        sweep_top_k.run_top_k_sweep(
            DATASET, candidates=(20,), output_dir=root, injection_scale=None,
        )
    assert "injection_scale" in str(exc.value)


# --- 3.2 ---------------------------------------------------------------------

def test_assemble_ladder_reports_the_scale_that_ran(tmp_path):
    """The ladder must echo benchmark_summary.json, not the config it drifted from."""
    from src.pipeline.step4.assemble_ladder import assemble_ladder

    _require(CANON / "benchmark_summary.json", CANON / "feedback_oof_real.pt",
             CANON / "oof_logits.pt", CANON / "prototypes.pt")
    root = tmp_path / "fb"
    root.mkdir(parents=True)
    _write_config(root, injection_scale=2.0)
    shutil.copy(CANON / "feedback_oof_real.pt", root / "feedback_oof_real.pt")
    benchmark = json.loads((CANON / "benchmark_summary.json").read_text())
    benchmark["injection_scale"] = 7.0  # what RAN, deliberately != the config's 2.0
    (root / "benchmark_summary.json").write_text(json.dumps(benchmark, indent=2))

    assemble_ladder(DATASET, feedback_dir=root)
    ladder = json.loads((root / "ladder_summary.json").read_text())
    assert ladder["configuration"]["injection_scale"] == 7.0, (
        "ladder reported the config's scale, not the one the run actually used"
    )
    assert ladder["configuration"]["selected_feedback_config"]["injection_scale"] == 2.0


# --- 3.3 ---------------------------------------------------------------------

def _fake_capture(root: Path, dataset: str, seed: int, *, scale: float, top_k=31.0):
    src = Path(f"results/multiseed/{dataset}_seed{seed}")
    _require(src)
    dst = root / f"{dataset}_seed{seed}"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("oof_logits.pt", "metrics.json", "feedback_oof_real.pt",
                 "ladder_summary.json"):
        shutil.copy(src / name, dst / name)
    (dst / "benchmark_summary.json").write_text(json.dumps(
        {"injection_scale": scale, "top_k_percent": top_k, "injection_mode": "edge"}
    ))
    return dst


def test_aggregate_multiseed_accepts_a_capture_root(tmp_path):
    from src.pipeline.step4 import aggregate_multiseed as agg

    for seed in (42, 1, 2):
        _fake_capture(tmp_path, DATASET, seed, scale=2.0)
    out = agg.aggregate(DATASET, seeds=(42, 1, 2), capture_root=tmp_path)
    assert out["feedback_configuration"]["injection_scale"] == 2.0
    assert out["feedback_configuration"]["top_k_percent"] == 31.0


def test_aggregate_multiseed_refuses_mismatched_knobs(tmp_path):
    from src.pipeline.step4 import aggregate_multiseed as agg

    _fake_capture(tmp_path, DATASET, 42, scale=2.0)
    _fake_capture(tmp_path, DATASET, 1, scale=10.0)
    _fake_capture(tmp_path, DATASET, 2, scale=2.0)
    with pytest.raises(RuntimeError) as exc:
        agg.aggregate(DATASET, seeds=(42, 1, 2), capture_root=tmp_path)
    assert "injection_scale" in str(exc.value)


# --- 3.4 ---------------------------------------------------------------------

def test_v1_multiseed_ladder_is_annotated_invalid():
    path = Path("results/multiseed_ladder.json")
    _require(path)
    payload = json.loads(path.read_text())
    validity = payload.get("validity")
    assert validity is not None, "results/multiseed_ladder.json carries no validity block"
    assert validity["invalid_rungs"] == ["loop"]
    assert set(validity["valid_rungs"]) == {"gnn", "llm", "agaf"}
    assert validity["injection_scale_that_ran"] == {"unsw_nb15": 10.0, "ton_iot": 10.0}
    # The datasets' own payloads must still be present and untouched.
    for ds in ("unsw_nb15", "ton_iot"):
        assert payload[ds]["rungs"]["agaf"]["mean"] > 0


# --- 3.5 ---------------------------------------------------------------------

def test_defaults_are_condition_a_the_contract_configuration(tmp_path):
    """A plain `train_feedback --dataset X` must reproduce what
    `results/*_current.json` was measured under: selector head unsupervised and
    the consultant temperature pinned at the T=10 init. The two "signal fixes"
    are real fixes, but they measured WORSE at 3 seeds, so they are not the
    default -- the default has to be the configuration the contracts describe."""
    root = tmp_path / "fb"
    _write_config(root, injection_scale=2.0)
    _run_feedback(root)
    benchmark = json.loads((root / "benchmark_summary.json").read_text())
    assert benchmark["selector_head_loss_weight"] == 0.0
    assert benchmark["legacy_temperature"] is True
    trace = json.loads((root / "feedback_trace_real.json").read_text())
    diag = trace["folds"][0]["bias_diagnostics"]
    # T=10 on whitened cosines is the near-uniform softmax of condition A.
    assert diag["consultant_temperature"] == pytest.approx(10.0, rel=1e-3)
    assert diag["mean_disagreement"] > 0.85


def test_fixed_signals_stay_reachable_by_flag(tmp_path):
    """Condition B/C must be reproducible on purpose, not only by checking out an
    old commit."""
    root = tmp_path / "fb"
    _write_config(root, injection_scale=2.0)
    _run_feedback(root, selector_head_loss_weight=1.0, legacy_temperature=False)
    benchmark = json.loads((root / "benchmark_summary.json").read_text())
    assert benchmark["selector_head_loss_weight"] == 1.0
    assert benchmark["legacy_temperature"] is False
    trace = json.loads((root / "feedback_trace_real.json").read_text())
    diag = trace["folds"][0]["bias_diagnostics"]
    assert diag["consultant_temperature"] < 1.0


def test_cli_calibrate_temperature_flag_maps_to_legacy_temperature():
    """--calibrate-temperature is the ONLY way to turn calibration on; absent it
    the CLI must pass legacy_temperature=True."""
    import src.pipeline.step4.train_feedback as tf

    captured = {}
    parser_argv = ["prog", "--dataset", DATASET]
    import sys as _sys
    saved_argv, saved_fn = _sys.argv, tf.train_feedback
    tf.train_feedback = lambda *a, **k: captured.update(k)
    try:
        _sys.argv = parser_argv
        tf.main()
        assert captured["legacy_temperature"] is True
        assert captured["selector_head_loss_weight"] == 0.0

        captured.clear()
        _sys.argv = parser_argv + ["--calibrate-temperature",
                                   "--selector-head-loss-weight", "1.0"]
        tf.main()
        assert captured["legacy_temperature"] is False
        assert captured["selector_head_loss_weight"] == 1.0
    finally:
        _sys.argv, tf.train_feedback = saved_argv, saved_fn


def test_a_sweep_written_config_still_resolves(tmp_path):
    """The config a sweep writes must be usable by the next `train_feedback`.

    `resolve_injection_scale` deliberately has no numeric fallback, so a config
    without `injection_scale` raises. Until 2026-09-16
    `write_selected_feedback_config` never wrote one, which meant selecting a
    knob and then training on it failed unless the caller repeated
    `--injection-scale` by hand. That is the shape of mistake the no-default
    rule was added to prevent, reintroduced one step upstream.
    """
    from src.pipeline.step4.feedback_config import (
        load_feedback_config,
        resolve_injection_scale,
        write_selected_feedback_config,
    )

    selected = {
        "top_k_percent": 29.0,
        "mean_best_val_macro_f1": 0.8277,
        "std_best_val_macro_f1": 0.0121,
        "pooled_oof_test_macro_f1": 0.8341,
        "sweep_summary_path": "unused",
    }

    written = write_selected_feedback_config(
        DATASET, selected, root=tmp_path, injection_scale=2.0
    )
    payload = json.loads(written.read_text())
    assert payload["injection_scale"] == 2.0
    assert resolve_injection_scale(None, payload, DATASET) == 2.0
    assert resolve_injection_scale(None, load_feedback_config(DATASET, tmp_path),
                                   DATASET) == 2.0

    # Omitting it stays an error rather than becoming a silent default.
    bare = write_selected_feedback_config(DATASET, selected, root=tmp_path)
    assert "injection_scale" not in json.loads(bare.read_text())
    with pytest.raises(ValueError, match="No injection_scale"):
        resolve_injection_scale(None, json.loads(bare.read_text()), DATASET)


def test_the_sweep_records_the_scale_it_ranked_candidates_under():
    """`sweep_top_k` must hand its resolved scale to the config it writes.

    The sweep resolves a scale before ranking anything. If it does not record
    that value, the selected top-k describes one mechanism strength and the
    training run that uses the config describes another.
    """
    import inspect

    from src.pipeline.step4 import sweep_top_k

    source = inspect.getsource(sweep_top_k.run_top_k_sweep)
    assert "injection_scale=injection_scale" in source, (
        "run_top_k_sweep no longer passes its resolved injection_scale to "
        "write_selected_feedback_config"
    )
