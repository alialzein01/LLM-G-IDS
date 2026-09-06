"""Task 3.2 / Task 4 condition C — the re-selection must be recorded, not asserted."""

from __future__ import annotations

import json

import pytest

from src.pipeline.step4.feedback_config import selected_config_path
from src.pipeline.step4.select_feedback_knobs import (
    SELECTION_SEEDS,
    SELECTION_SOURCE,
    select_by_mean_validation,
)

SUPERSEDED = {"unsw_nb15": (31.0, 2.0), "ton_iot": (25.0, 20.0)}


def test_selection_takes_the_across_seed_mean_and_breaks_ties_downward():
    per_seed = {42: {15.0: 0.50, 20.0: 0.60}, 1: {15.0: 0.70, 20.0: 0.55}}
    # means: 15 -> 0.60, 20 -> 0.575. Neither seed alone would pick 15.
    assert select_by_mean_validation(per_seed) == 15.0
    tie = {42: {15.0: 0.5, 20.0: 0.5}, 1: {15.0: 0.5, 20.0: 0.5}}
    assert select_by_mean_validation(tie) == 15.0


def test_selection_refuses_seeds_that_scored_different_candidates():
    with pytest.raises(ValueError, match="different candidate sets"):
        select_by_mean_validation({42: {15.0: 0.5}, 1: {20.0: 0.5}})


@pytest.mark.parametrize("dataset", sorted(SUPERSEDED))
def test_config_records_the_reselection_and_the_superseded_values(dataset):
    path = selected_config_path(dataset)
    if not path.exists():
        pytest.skip(f"missing {path}")
    cfg = json.loads(path.read_text())
    assert cfg["source"] == SELECTION_SOURCE
    assert cfg["selection_uses_test_labels"] is False
    assert cfg["selection_seeds"] == list(SELECTION_SEEDS)
    assert "injection_scale" in cfg
    old_k, old_scale = SUPERSEDED[dataset]
    superseded = cfg["superseded"]
    assert superseded["top_k_percent"] == old_k
    assert superseded["injection_scale"] == old_scale
    # Curves must be present for all three seeds, both knobs, so the choice is auditable.
    for knob in ("top_k", "injection_scale"):
        curves = cfg["selection_curves"][knob]
        assert sorted(int(s) for s in curves) == sorted(SELECTION_SEEDS)


def test_knobs_are_not_carried_across_datasets():
    """The two datasets must be selected independently, not copied."""
    cfgs = {}
    for dataset in SUPERSEDED:
        path = selected_config_path(dataset)
        if not path.exists():
            pytest.skip(f"missing {path}")
        cfgs[dataset] = json.loads(path.read_text())
    for dataset, cfg in cfgs.items():
        assert cfg["dataset"] == dataset
        assert cfg["selection_curves"]["top_k"], f"{dataset} has no own top_k curve"
