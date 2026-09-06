"""Task 1 — the loop's selector head must be a classifier.

Uncertainty selection ranks edges by the entropy of `fusion_out(edge_emb)`.
In the canonical prototype path `_output_fusion` returns `logits = correction`,
so that head never enters the loss except as two scalar gate features
(`h_gnn`, `conf_gnn`). It therefore collapses onto one or two classes, and the
top-k% "most uncertain" set it produces — the set the semantic advice is aimed
at — is arbitrary.

These tests pin the head to a real classifier *when it is supervised*, and pin
`head_only` to the plain GNN it is supposed to be so the ablation cannot drift
under the fix.

Supervising it is NOT the default. `SELECTOR_HEAD_LOSS_WEIGHT` defaults to 0.0,
condition A, the configuration `results/*_current.json` was measured under —
turning the signal on measured worse at 3 seeds. So the two tests about the
fixed head ask for the supervision explicitly, and a companion test pins that
the default still collapses. Both facts have to stay true.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.pipeline.common.datasets import get_dataset_config
from src.pipeline.common.splits import eval_macro_f1, mask_dropped_logits
from src.pipeline.step4.feedback_config import load_feedback_config

DATASET = "unsw_nb15"
FOLD = 0
# Short schedule: enough for the head to separate classes if it is in the loss
# at all, far short of the 300-epoch production run.
SHORT_EPOCHS = 150
FIXTURE_DIR = Path("tests/fixtures")
HEAD_ONLY_FIXTURE = FIXTURE_DIR / f"head_only_fold{FOLD}_{DATASET}_e{SHORT_EPOCHS}.pt"

_CACHE: dict[str, object] = {}


def _artifacts():
    cfg = get_dataset_config(DATASET)
    root = Path(f"data/{DATASET}/processed/step4_feedback")
    for p in (cfg.graph_path, cfg.llm_embedding_path, cfg.splits_path, root / "prototypes.pt"):
        if not Path(p).exists():
            pytest.skip(f"missing pipeline artifact {p}")
    return cfg, root


def train_fold0(mode: str, selector_head_loss_weight: float | None = None):
    """Train one UNSW fold on a short schedule; cached per (mode, weight).

    `selector_head_loss_weight=None` means "whatever the code default is", which
    is the point of the default-behaviour test below.
    """
    key = (mode, selector_head_loss_weight)
    if key in _CACHE:
        return _CACHE[key]
    import src.pipeline.step4.train_feedback as tf

    cfg, root = _artifacts()
    data = torch.load(cfg.graph_path, weights_only=False)
    tf.IN_DIM = data.x.shape[1]
    emb = torch.load(cfg.llm_embedding_path, weights_only=False).float()
    folds = torch.load(cfg.splits_path, weights_only=False)
    protos = torch.load(root / "prototypes.pt", weights_only=False)
    sel = load_feedback_config(DATASET)
    extra = (
        {} if selector_head_loss_weight is None
        else {"selector_head_loss_weight": selector_head_loss_weight}
    )

    saved = tf.MAX_EPOCHS
    tf.MAX_EPOCHS = SHORT_EPOCHS
    try:
        result = tf._train_one_fold(
            data, emb, folds[FOLD], protos["folds"][FOLD], FOLD, mode,
            top_k_percent=float(sel["top_k_percent"]),
            bias_confidence_fraction=float(sel["bias_confidence_fraction"]),
            gate_mode=sel.get("gate_mode", "confidence"),
            eval_classes=cfg.eval_classes,
            dropped_classes=cfg.dropped_classes,
            injection_mode=sel.get("injection_mode", "edge"),
            injection_scale=float(sel.get("injection_scale", 10.0)),
            **extra,
        )
    finally:
        tf.MAX_EPOCHS = saved
    _CACHE[key] = result
    return result


def _iter1_row(result):
    assert result.iterations, "no trace rows collected"
    return result.iterations[0]


def test_selector_head_predicts_more_than_one_class_when_supervised():
    """Iteration-1 head must have nonzero F1 on >= 6 of the 10 eval classes."""
    cfg, _ = _artifacts()
    row = _iter1_row(train_fold0("real", selector_head_loss_weight=1.0))
    per_class = row["per_class_f1"]
    nonzero = [c for c in cfg.eval_classes if per_class[c] > 0.0]
    assert len(nonzero) >= 6, (
        f"selector head has nonzero F1 on only {len(nonzero)}/{len(cfg.eval_classes)} "
        f"eval classes (classes {nonzero}); per_class_f1={per_class}"
    )


def test_selector_head_tracks_head_only_quality_when_supervised():
    """Iteration-1 head macro-F1 must be within 0.15 of the plain-GNN rung."""
    row = _iter1_row(train_fold0("real", selector_head_loss_weight=1.0))
    head_only_f1 = train_fold0("head_only").test_macro_f1
    gap = head_only_f1 - row["test_macro_f1"]
    assert gap < 0.15, (
        f"selector head macro-F1 {row['test_macro_f1']:.4f} trails head_only "
        f"{head_only_f1:.4f} by {gap:.4f} (> 0.15)"
    )


def test_selector_head_macro_f1_is_reported_in_bias_diagnostics():
    """Task 1.3 — the head's quality must be visible without reading the trace."""
    diag = train_fold0("real").bias_diagnostics
    assert diag is not None and "selector_head_macro_f1" in diag, (
        f"bias_diagnostics missing selector_head_macro_f1: {sorted(diag or {})}"
    )
    row = _iter1_row(train_fold0("real"))
    assert diag["selector_head_macro_f1"] == pytest.approx(
        row["test_macro_f1"], abs=1e-6
    ), (
        f"selector_head_macro_f1={diag['selector_head_macro_f1']} disagrees with the "
        f"iteration-1 trace row {row['test_macro_f1']}"
    )


def test_head_only_logits_unchanged_by_the_selector_fix():
    """`head_only` must stay a plain GNN — the ablation's whole point.

    The fixture was captured on the pre-fix code at commit 5bc37e4. Any change
    to the selector head that also moves `head_only` breaks the ablation.
    """
    if not HEAD_ONLY_FIXTURE.exists():
        pytest.skip(f"fixture not captured: {HEAD_ONLY_FIXTURE}")
    cfg, _ = _artifacts()
    expected = torch.load(HEAD_ONLY_FIXTURE, weights_only=False)
    got = train_fold0("head_only").logits
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    # Guard the metric too, so a same-logits/different-scoring change is caught.
    folds = torch.load(cfg.splits_path, weights_only=False)
    labels = torch.load(cfg.graph_path, weights_only=False).edge_label
    tm = folds[FOLD]["test_mask"]
    preds = mask_dropped_logits(got[tm], cfg.dropped_classes).argmax(1)
    assert eval_macro_f1(labels[tm], preds, cfg.eval_classes) == pytest.approx(
        train_fold0("head_only").test_macro_f1, abs=1e-9
    )


def test_default_selector_head_collapses_as_condition_a_says():
    """The mirror of the two tests above. At the default weight of 0.0 the head
    reaches the loss only as two scalar gate features and collapses onto one or
    two classes. That is the documented condition-A behaviour, and the ladder in
    `results/*_current.json` was measured with it. If this starts passing on
    many classes, the default has silently moved off A."""
    import src.pipeline.step4.train_feedback as tf

    assert tf.SELECTOR_HEAD_LOSS_WEIGHT == 0.0
    cfg, _ = _artifacts()
    per_class = _iter1_row(train_fold0("real"))["per_class_f1"]
    nonzero = [c for c in cfg.eval_classes if per_class[c] > 0.0]
    assert len(nonzero) <= 2, (
        f"default selector head has nonzero F1 on {len(nonzero)} classes "
        f"({nonzero}) — expected the collapse condition A describes"
    )
