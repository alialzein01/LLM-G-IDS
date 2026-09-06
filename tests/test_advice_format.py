"""Advice format — the shape of the verdict handed to the bias projection.

The hypothesis this supports: the projection may be unable to use raw,
unnormalised consultant logits while being able to use saturated one-hot advice,
which is the only format the oracle ever gave it. Testing that requires the
realistic consultant's advice to be reshaped into *exactly* the oracle's format,
so the two experiments differ in content and nothing else. Hence:

  1. `onehot` applied to logits whose argmax is the true label must equal the
     oracle vector, element for element, from the oracle's own constructor;
  2. the magnitude is one shared constant, not two that can drift;
  3. `logits` (the default) reproduces condition A bit-for-bit;
  4. selection and the confidence gate keep ranking the ORIGINAL logits.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.models.feedback_classifier import (
    ADVICE_FORMATS,
    ADVICE_SATURATION_MAGNITUDE,
    FeedbackLoopClassifier,
    saturated_advice,
)
from src.pipeline.common.datasets import get_dataset_config
from src.pipeline.common.splits import NUM_CLASSES
from src.pipeline.step4.feedback_config import load_feedback_config

DATASET = "unsw_nb15"
FOLD = 0


# --- 1 & 2. it is the oracle's format, from the oracle's constant -------------

def test_onehot_of_the_true_label_is_exactly_the_oracle_vector():
    from src.pipeline.step4.mechanism_only_edge_injection import (
        ORACLE_LOGIT_MAGNITUDE,
        _oracle_logits,
    )

    torch.manual_seed(0)
    labels = torch.randint(0, NUM_CLASSES, (32,))
    oracle = _oracle_logits(labels, NUM_CLASSES, ORACLE_LOGIT_MAGNITUDE)

    # Any logits whose argmax is the true label must format to the oracle vector.
    consultant = torch.randn(32, NUM_CLASSES) * 0.1
    consultant.scatter_(1, labels.unsqueeze(1), 10.0)
    assert torch.equal(consultant.argmax(dim=-1), labels), "test setup is wrong"

    got = saturated_advice(consultant, "onehot")
    torch.testing.assert_close(got, oracle, rtol=0, atol=0)


def test_the_saturation_magnitude_is_one_shared_constant():
    from src.pipeline.step4 import mechanism_only_edge_injection as gate0

    assert gate0.ORACLE_LOGIT_MAGNITUDE is ADVICE_SATURATION_MAGNITUDE


def test_softmax_format_peaks_at_the_oracle_magnitude():
    torch.manual_seed(1)
    logits = torch.randn(16, NUM_CLASSES) * 3.0
    out = saturated_advice(logits, "softmax")
    peaks = out.max(dim=-1).values
    torch.testing.assert_close(
        peaks, torch.full((16,), ADVICE_SATURATION_MAGNITUDE), rtol=1e-6, atol=1e-6
    )
    # The soft version must preserve the consultant's ranking of classes.
    assert torch.equal(out.argmax(dim=-1), logits.argmax(dim=-1))


def test_logits_format_is_the_identity():
    logits = torch.randn(8, NUM_CLASSES)
    assert saturated_advice(logits, "logits") is logits


def test_unknown_format_is_rejected():
    with pytest.raises(ValueError, match="advice_format"):
        saturated_advice(torch.randn(4, NUM_CLASSES), "argmax")
    with pytest.raises(ValueError, match="advice_format"):
        FeedbackLoopClassifier(
            num_classes=NUM_CLASSES, injection_scale=2.0, advice_format="argmax"
        )
    for fmt in ADVICE_FORMATS:
        saturated_advice(torch.randn(4, NUM_CLASSES), fmt)


def test_empty_selection_is_handled():
    for fmt in ADVICE_FORMATS:
        out = saturated_advice(torch.empty(0, NUM_CLASSES), fmt)
        assert out.shape == (0, NUM_CLASSES)


# --- 4. selection is untouched -----------------------------------------------

def test_the_confidence_gate_still_ranks_the_raw_logits():
    """The format changes what is injected, not who is consulted. Rewriting the
    advice to one-hot makes every row's max-prob identical, so a gate that
    ranked the FORMATTED advice would pick an arbitrary half."""
    torch.manual_seed(2)
    flagged = torch.arange(10)
    semantic = torch.randn(10, NUM_CLASSES) * 2.0
    gnn_probs = torch.full((10, NUM_CLASSES), 1.0 / NUM_CLASSES)

    kept = {}
    for fmt in ADVICE_FORMATS:
        model = FeedbackLoopClassifier(
            num_classes=NUM_CLASSES, injection_mode="edge", bias_dim=5,
            edge_attr_dim=5, injection_scale=2.0, bias_confidence_frac=0.5,
            advice_format=fmt,
        )
        kept[fmt] = model._gate_flagged(flagged, semantic, gnn_probs).tolist()
    assert kept["logits"] == kept["onehot"] == kept["softmax"], kept


def test_default_is_logits():
    assert FeedbackLoopClassifier(
        num_classes=NUM_CLASSES, injection_scale=2.0
    ).advice_format == "logits"


# --- 3. the default is inert --------------------------------------------------

def _fold0():
    cfg = get_dataset_config(DATASET)
    root = Path(f"data/{DATASET}/processed/step4_feedback")
    for p in (cfg.graph_path, cfg.llm_embedding_path, cfg.splits_path,
              root / "prototypes.pt"):
        if not Path(p).exists():
            pytest.skip(f"missing pipeline artifact {p}")
    return (
        cfg,
        torch.load(cfg.graph_path, weights_only=False),
        torch.load(cfg.llm_embedding_path, weights_only=False).float(),
        torch.load(cfg.splits_path, weights_only=False),
        torch.load(root / "prototypes.pt", weights_only=False),
    )


def _train(mode: str, **extra):
    import src.pipeline.step4.train_feedback as tf

    cfg, data, emb, folds, protos = _fold0()
    tf.IN_DIM = data.x.shape[1]
    sel = load_feedback_config(DATASET)
    saved = tf.MAX_EPOCHS
    tf.MAX_EPOCHS = 3
    try:
        return tf._train_one_fold(
            data, emb, folds[FOLD], protos["folds"][FOLD], FOLD, mode,
            top_k_percent=float(sel["top_k_percent"]),
            bias_confidence_fraction=float(sel["bias_confidence_fraction"]),
            gate_mode=sel.get("gate_mode", "confidence"),
            eval_classes=cfg.eval_classes,
            dropped_classes=cfg.dropped_classes,
            injection_mode="edge",
            injection_scale=float(sel["injection_scale"]),
            **extra,
        )
    finally:
        tf.MAX_EPOCHS = saved


def test_logits_format_reproduces_condition_a_bit_for_bit():
    default = _train("real")
    explicit = _train("real", advice_format="logits")
    torch.testing.assert_close(default.logits, explicit.logits, rtol=0, atol=0)
    assert default.test_macro_f1 == explicit.test_macro_f1


def test_head_only_is_untouched_by_the_format():
    """`head_only` has no semantic logits, so it never reaches the format at
    all. Pinned because the E3 work showed that allocation-time changes can move
    an untreated arm through the shared RNG stream."""
    base = _train("head_only")
    for fmt in ("onehot", "softmax"):
        swapped = _train("head_only", advice_format=fmt)
        torch.testing.assert_close(base.logits, swapped.logits, rtol=0, atol=0)


def test_saturated_formats_actually_change_the_real_mode():
    base = _train("real")
    for fmt in ("onehot", "softmax"):
        swapped = _train("real", advice_format=fmt)
        assert not torch.allclose(base.logits, swapped.logits), fmt
