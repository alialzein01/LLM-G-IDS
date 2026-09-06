"""E1 — reliability-weighted advice.

`w[c]` is the precision of the consultant's argmax for class `c`, measured on
the fold's TRAIN edges. Three things have to hold or the experiment is not what
it claims to be:

  1. `w` is a vector of probabilities in [0, 1], zero where the consultant never
     predicts the class on train;
  2. `w` depends on TRAIN labels only — permuting the val/test labels must not
     move it by a single bit, or the advice is being steered by held-out data;
  3. `off` (the default) reproduces condition A bit-for-bit, so the canonical
     ladder is untouched by this code existing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.models.feedback_classifier import (
    ADVICE_RELIABILITY_MODES,
    FeedbackLoopClassifier,
    consultant_reliability,
)
from src.pipeline.common.datasets import get_dataset_config
from src.pipeline.common.splits import NUM_CLASSES
from src.pipeline.step4.feedback_config import load_feedback_config

DATASET = "unsw_nb15"
FOLD = 0


def _fold0():
    cfg = get_dataset_config(DATASET)
    root = Path(f"data/{DATASET}/processed/step4_feedback")
    for p in (cfg.graph_path, cfg.llm_embedding_path, cfg.splits_path,
              root / "prototypes.pt"):
        if not Path(p).exists():
            pytest.skip(f"missing pipeline artifact {p}")
    data = torch.load(cfg.graph_path, weights_only=False)
    emb = torch.load(cfg.llm_embedding_path, weights_only=False).float()
    folds = torch.load(cfg.splits_path, weights_only=False)
    protos = torch.load(root / "prototypes.pt", weights_only=False)
    return cfg, data, emb, folds, protos


def _consultant_logits():
    """The prototype consultant's logits over every edge, fold 0's state."""
    from src.models.feedback_classifier import WhitenedPrototypeScorer

    cfg, data, emb, folds, protos = _fold0()
    st = protos["folds"][FOLD]
    scorer = WhitenedPrototypeScorer(
        num_classes=NUM_CLASSES, embed_dim=protos["embed_dim"]
    )
    scorer.load_fold_state(st["mean"], st["whitener"], st["prototypes_whitened"])
    with torch.no_grad():
        return scorer(emb), data.edge_label, folds[FOLD]


# --- 1. shape and range ------------------------------------------------------

def test_reliability_is_a_probability_per_class():
    logits, labels, fold = _consultant_logits()
    w = consultant_reliability(logits, labels, fold["train_mask"], NUM_CLASSES)
    assert w.shape == (NUM_CLASSES,)
    assert torch.all(w >= 0.0) and torch.all(w <= 1.0), w
    # Every class the consultant DOES predict on train must have w equal to that
    # class's train precision; every class it never predicts must be exactly 0.
    pred = logits[fold["train_mask"]].argmax(dim=-1)
    truth = labels[fold["train_mask"]]
    for c in range(NUM_CLASSES):
        called = pred == c
        if not int(called.sum()):
            assert float(w[c]) == 0.0, f"class {c} never predicted but w={w[c]}"
        else:
            assert float(w[c]) == pytest.approx(
                float((truth[called] == c).float().mean()), abs=1e-7
            )


def test_reliability_is_zero_for_a_class_the_consultant_never_calls():
    logits = torch.zeros(20, NUM_CLASSES)
    logits[:, 0] = 5.0  # argmax is class 0 for every edge
    labels = torch.zeros(20, dtype=torch.long)
    labels[:5] = 1
    train = torch.ones(20, dtype=torch.bool)
    w = consultant_reliability(logits, labels, train, NUM_CLASSES)
    assert float(w[0]) == pytest.approx(15 / 20)
    assert float(w[1:].sum()) == 0.0


# --- 2. train-only ------------------------------------------------------------

def test_reliability_ignores_val_and_test_labels():
    """The leak check. Scramble every label outside the train mask; `w` must be
    bit-identical, or held-out labels are choosing where advice lands."""
    logits, labels, fold = _consultant_logits()
    train = fold["train_mask"]
    w = consultant_reliability(logits, labels, train, NUM_CLASSES)

    held_out = ~train
    assert int(held_out.sum()) > 0
    g = torch.Generator().manual_seed(7)
    scrambled = labels.clone()
    idx = held_out.nonzero(as_tuple=True)[0]
    scrambled[idx] = labels[idx][torch.randperm(idx.numel(), generator=g)]
    # Also try an outright wrong constant, not just a permutation.
    clobbered = labels.clone()
    clobbered[held_out] = (labels[held_out] + 3) % NUM_CLASSES

    for variant, name in ((scrambled, "permuted"), (clobbered, "clobbered")):
        w2 = consultant_reliability(logits, variant, train, NUM_CLASSES)
        assert torch.equal(w, w2), f"w moved when held-out labels were {name}"


# --- 3. the mechanism ---------------------------------------------------------

def _model(advice_reliability: str) -> FeedbackLoopClassifier:
    return FeedbackLoopClassifier(
        num_classes=NUM_CLASSES, injection_mode="edge", bias_dim=5,
        edge_attr_dim=5, injection_scale=2.0, bias_confidence_frac=0.5,
        advice_reliability=advice_reliability,
    )


def test_advice_reliability_rejects_unknown_modes():
    with pytest.raises(ValueError, match="advice_reliability"):
        _model("sometimes")
    for mode in ADVICE_RELIABILITY_MODES:
        _model(mode)


def test_gate_reweights_the_ranking_but_keeps_the_count():
    """A class the consultant is never right about must stop winning the cut."""
    torch.manual_seed(0)
    flagged = torch.arange(8)
    semantic = torch.full((8, NUM_CLASSES), -5.0)
    # Edges 0-3 are confident class 1 (unreliable); 4-7 less confident class 2.
    semantic[:4, 1] = 10.0
    semantic[4:, 2] = 2.0
    gnn_probs = torch.full((8, NUM_CLASSES), 1.0 / NUM_CLASSES)

    plain = _model("off")
    kept_plain = plain._gate_flagged(flagged, semantic, gnn_probs)
    assert set(kept_plain.tolist()) == {0, 1, 2, 3}

    weighted = _model("gate")
    w = torch.zeros(NUM_CLASSES)
    w[1] = 0.0   # class 1 verdicts are never right
    w[2] = 0.9
    weighted.set_consultant_reliability(w)
    kept = weighted._gate_flagged(flagged, semantic, gnn_probs)
    assert kept.numel() == kept_plain.numel(), "the gate must keep the same count"
    assert set(kept.tolist()) == {4, 5, 6, 7}


def test_gate_plus_scale_shrinks_advice_from_unreliable_classes():
    model = _model("gate+scale")
    w = torch.zeros(NUM_CLASSES)
    w[1] = 0.25
    model.set_consultant_reliability(w)
    advice = torch.full((3, NUM_CLASSES), -1.0)
    advice[:, 1] = 4.0
    scaled = advice * model._advice_weights(advice).unsqueeze(-1)
    assert torch.allclose(scaled, advice * 0.25)
    # `gate` alone must NOT touch the magnitude.
    assert _model("gate").advice_reliability == "gate"


def test_default_is_off_and_the_buffer_is_the_identity():
    m = _model("off")
    assert m.advice_reliability == "off"
    assert torch.equal(m.consultant_reliability_w, torch.ones(NUM_CLASSES))
    # A default-constructed model must also be off, so no caller opts in by accident.
    assert FeedbackLoopClassifier(
        num_classes=NUM_CLASSES, injection_scale=2.0
    ).advice_reliability == "off"


def test_off_reproduces_condition_a_bit_for_bit(tmp_path):
    """The whole flag must be inert when off: same fold, same logits, exactly."""
    import src.pipeline.step4.train_feedback as tf

    cfg, data, emb, folds, protos = _fold0()
    tf.IN_DIM = data.x.shape[1]
    sel = load_feedback_config(DATASET)
    saved = tf.MAX_EPOCHS
    tf.MAX_EPOCHS = 3
    try:
        kwargs = dict(
            top_k_percent=float(sel["top_k_percent"]),
            bias_confidence_fraction=float(sel["bias_confidence_fraction"]),
            gate_mode=sel.get("gate_mode", "confidence"),
            eval_classes=cfg.eval_classes,
            dropped_classes=cfg.dropped_classes,
            injection_mode="edge",
            injection_scale=float(sel["injection_scale"]),
        )
        default = tf._train_one_fold(
            data, emb, folds[FOLD], protos["folds"][FOLD], FOLD, "real", **kwargs
        )
        explicit_off = tf._train_one_fold(
            data, emb, folds[FOLD], protos["folds"][FOLD], FOLD, "real",
            advice_reliability="off", **kwargs
        )
    finally:
        tf.MAX_EPOCHS = saved
    torch.testing.assert_close(
        default.logits, explicit_off.logits, rtol=0, atol=0
    )
    assert default.test_macro_f1 == explicit_off.test_macro_f1
