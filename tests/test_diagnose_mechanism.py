"""The diagnostics must describe the run, not a run-like thing of its own.

The one place that is easy to get wrong is the selector: the model ranks entropy
on the RAW softmax over all NUM_CLASSES, and only masks dropped classes when it
turns logits into predictions. Masking before the entropy flags a different edge
set on ToN (classes 3 and 7 are dropped), which quietly changes every
"on flagged edges" number the diagnostics report.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.models.feedback_classifier import UncertaintySelector
from src.pipeline.common.datasets import get_dataset_config
from src.pipeline.common.splits import mask_dropped_logits
from src.pipeline.step4 import diagnose_mechanism as dm


TON_RUN = Path("results/dev/task0/ton_iot")
UNSW_RUN = Path("results/dev/task0/unsw_nb15")


def _require(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"missing artifacts: {', '.join(missing)}")


def test_selector_uses_the_unmasked_softmax_like_the_model():
    """A synthetic case where masking changes the flagged set, pinned against the
    model's own convention rather than against a number."""
    torch.manual_seed(0)
    logits = torch.randn(400, 10)
    # Make the dropped classes carry most of the mass on half the edges, so
    # masking them changes those edges' entropy a lot.
    logits[:200, [3, 7]] += 4.0
    selector = UncertaintySelector(top_k_percent=25.0)
    raw = selector(logits.softmax(dim=-1))
    masked = selector(mask_dropped_logits(logits, (3, 7)).softmax(dim=-1))
    assert not torch.equal(raw, masked), (
        "test is vacuous: masking did not change the flagged set"
    )


def test_diagnostics_reproduce_the_runs_pooled_macro_f1():
    """The diagnostics reload the run's own tensors, so their pooled numbers must
    equal the ones the run reported. A mismatch means it is reading something
    else."""
    for run, dataset in ((UNSW_RUN, "unsw_nb15"), (TON_RUN, "ton_iot")):
        _require(run / "feedback_oof_real.pt", run / "benchmark_summary.json",
                 run / "mechanism_diagnostics.json")
        diag = json.loads((run / "mechanism_diagnostics.json").read_text())
        bench = json.loads((run / "benchmark_summary.json").read_text())
        pooled = bench["per_mode_pooled_macro_f1"]
        assert diag["pooled_macro_f1"] == pooled
        # `all_test_edges` macro-F1 is computed over classes present, which for a
        # pooled OOF stitch is every eval class, so it must match the run.
        for rung, mode in (("gnn_head_only", "head_only"), ("loop_real", "real")):
            got = diag["subset_performance"][rung]["all_test_edges"][
                "macro_f1_classes_present"
            ]
            assert got == pytest.approx(pooled[mode], abs=5e-3), (
                f"{dataset} {rung}: diagnostics {got} vs run {pooled[mode]}"
            )


def test_flagged_count_is_top_k_percent_of_all_edges():
    for run, dataset in ((UNSW_RUN, "unsw_nb15"), (TON_RUN, "ton_iot")):
        _require(run / "mechanism_diagnostics.json")
        diag = json.loads((run / "mechanism_diagnostics.json").read_text())
        c = diag["consultation"]
        expected = round(c["n_edges"] * c["top_k_percent"] / 100.0)
        assert abs(c["n_flagged"] - expected) <= 1, (
            f"{dataset}: flagged {c['n_flagged']}, expected ~{expected}"
        )
        assert c["n_confidence_gated"] == round(
            c["bias_confidence_fraction"] * c["n_flagged"]
        )
