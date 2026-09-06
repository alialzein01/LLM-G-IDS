"""Task 3.2 / Task 4 condition C — the re-selection must be recorded, not asserted.

Condition C is recorded, not active. The active knobs are condition A, the ones
`results/*_current.json` was measured under; C's values and its full selection
curves live under the config's `condition_c` block so the choice stays auditable
and reachable by flag. These tests pin exactly that split.
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


@pytest.mark.parametrize("dataset", sorted(CONDITION_A))
def test_config_records_the_reselection_but_condition_a_is_active(dataset):
    path = selected_config_path(dataset)
    if not path.exists():
        pytest.skip(f"missing {path}")
    cfg = json.loads(path.read_text())
    assert cfg["selection_uses_test_labels"] is False
    assert cfg["selection_seeds"] == list(SELECTION_SEEDS)

    a_k, a_scale = CONDITION_A[dataset]
    assert cfg["top_k_percent"] == a_k
    assert cfg["injection_scale"] == a_scale
    assert cfg["active_condition"] == "A"

    c_k, c_scale = CONDITION_C[dataset]
    condition_c = cfg["condition_c"]
    assert condition_c["source"] == SELECTION_SOURCE
    assert condition_c["top_k_percent"] == c_k
    assert condition_c["injection_scale"] == c_scale
    # Curves must be present for all three seeds, both knobs, so the recorded
    # choice stays auditable even though it is not the one in force.
    for knob in ("top_k", "injection_scale"):
        curves = condition_c["selection_curves"][knob]
        assert sorted(int(s) for s in curves) == sorted(SELECTION_SEEDS)


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
        assert cfg["condition_c"]["selection_curves"]["top_k"], (
            f"{dataset} has no own top_k curve"
        )
    assert (
        cfgs["unsw_nb15"]["top_k_percent"] != cfgs["ton_iot"]["top_k_percent"]
        or cfgs["unsw_nb15"]["injection_scale"] != cfgs["ton_iot"]["injection_scale"]
    )
