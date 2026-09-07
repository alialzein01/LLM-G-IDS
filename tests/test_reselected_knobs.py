"""Task 3.2 / Task 4 condition C — the re-selection must be recorded, not asserted.

History, not scratch space. The active knobs are the trained-head consultant's
(2026-09-07); the prototype's condition-A knobs live under `prototype_superseded`
with condition C nested inside them. Every layer stays reproducible: A and C by
running `train_feedback` WITHOUT `--use-llm-head` at their knobs, the head
condition by running it with. These tests pin that nothing was dropped.
"""

from __future__ import annotations

import json

import pytest

from src.pipeline.step4.feedback_config import selected_config_path
from src.pipeline.step4.select_feedback_knobs import (
    SELECTION_SEEDS,
    SELECTION_SOURCE,
    select_by_mean_validation,
)

# Active knobs (condition A) and the recorded condition-C re-selection.
CONDITION_A = {"unsw_nb15": (31.0, 2.0), "ton_iot": (25.0, 20.0)}
CONDITION_C = {"unsw_nb15": (35.0, 5.0), "ton_iot": (21.0, 10.0)}


def test_selection_takes_the_across_seed_mean_and_breaks_ties_downward():
    per_seed = {42: {15.0: 0.50, 20.0: 0.60}, 1: {15.0: 0.70, 20.0: 0.55}}
    # means: 15 -> 0.60, 20 -> 0.575. Neither seed alone would pick 15.
    assert select_by_mean_validation(per_seed) == 15.0
    tie = {42: {15.0: 0.5, 20.0: 0.5}, 1: {15.0: 0.5, 20.0: 0.5}}
    assert select_by_mean_validation(tie) == 15.0


def test_selection_refuses_seeds_that_scored_different_candidates():
    with pytest.raises(ValueError, match="different candidate sets"):
        select_by_mean_validation({42: {15.0: 0.5}, 1: {20.0: 0.5}})


# The trained-head consultant's selected knobs (2026-09-07), now active.
CONDITION_HEAD = {"unsw_nb15": (29.0, 2.0), "ton_iot": (16.0, 20.0)}


@pytest.mark.parametrize("dataset", sorted(CONDITION_A))
def test_the_head_consultant_is_active_and_the_prototype_is_preserved(dataset):
    path = selected_config_path(dataset)
    if not path.exists():
        pytest.skip(f"missing {path}")
    cfg = json.loads(path.read_text())
    assert cfg["selection_uses_test_labels"] is False
    assert cfg["selection_seeds"] == list(SELECTION_SEEDS)

    # Active: the trained-head consultant.
    h_k, h_scale = CONDITION_HEAD[dataset]
    assert cfg["consultant"] == "trained_llm_head"
    assert cfg["trained_llm_head"] is True
    assert cfg["top_k_percent"] == h_k
    assert cfg["injection_scale"] == h_scale
    head = cfg["consultant_trained_llm_head"]
    assert head["top_k_percent"] == h_k and head["injection_scale"] == h_scale
    for knob in ("top_k", "injection_scale"):
        curves = head["selection_curves"][knob]
        assert sorted(int(s) for s in curves) == sorted(SELECTION_SEEDS)

    # Preserved: the prototype's condition A, with condition C nested inside it.
    a_k, a_scale = CONDITION_A[dataset]
    proto = cfg["prototype_superseded"]
    assert proto["top_k_percent"] == a_k
    assert proto["injection_scale"] == a_scale
    assert proto["semantic_consultant"] == "whitened_prototype_scorer"
    assert proto["active_condition"] == "A"

    c_k, c_scale = CONDITION_C[dataset]
    condition_c = proto["condition_c"]
    assert condition_c["source"] == SELECTION_SOURCE
    assert condition_c["top_k_percent"] == c_k
    assert condition_c["injection_scale"] == c_scale
    for knob in ("top_k", "injection_scale"):
        curves = condition_c["selection_curves"][knob]
        assert sorted(int(s) for s in curves) == sorted(SELECTION_SEEDS)
    assert "condition_c" not in cfg, "condition_c belongs under the prototype block"


def test_the_head_selection_records_whether_a_knob_hit_a_range_boundary():
    """A knob chosen at the edge of its swept range means the range was too
    narrow. ToN's injection_scale did exactly that and the config must say so."""
    boundaries = {}
    for dataset in sorted(CONDITION_A):
        path = selected_config_path(dataset)
        if not path.exists():
            pytest.skip(f"missing {path}")
        head = json.loads(path.read_text())["consultant_trained_llm_head"]
        boundaries[dataset] = (
            head["top_k_on_range_boundary"], head["scale_on_range_boundary"]
        )
    assert boundaries["unsw_nb15"] == (False, False)
    assert boundaries["ton_iot"] == (False, True)


def test_knobs_are_not_carried_across_datasets():
    """The two datasets must be selected independently, not copied."""
    cfgs = {}
    for dataset in CONDITION_A:
        path = selected_config_path(dataset)
        if not path.exists():
            pytest.skip(f"missing {path}")
        cfgs[dataset] = json.loads(path.read_text())
    for dataset, cfg in cfgs.items():
        assert cfg["dataset"] == dataset
        assert cfg["consultant_trained_llm_head"]["selection_curves"]["top_k"], (
            f"{dataset} has no own top_k curve"
        )
        assert cfg["prototype_superseded"]["condition_c"]["selection_curves"]["top_k"]
    assert (
        cfgs["unsw_nb15"]["top_k_percent"] != cfgs["ton_iot"]["top_k_percent"]
        or cfgs["unsw_nb15"]["injection_scale"] != cfgs["ton_iot"]["injection_scale"]
    )
