"""E3 — give the loop's output fusion AGAF's gate.

The swap must be exactly that: a different block combining the same two
projected branches. Three things have to hold:

  1. the `agaf` block runs AGAF's arithmetic, not a lookalike — it is pinned
     against `AGAFFusionEdgeClassifier`'s own layers on the same inputs;
  2. `head_only` is bit-for-bit unchanged, because it never reaches
     `_output_fusion` at all — if it moves, the ablation's baseline moved with
     the treatment and nothing can be read off the comparison;
  3. `loop` (the default) reproduces condition A bit-for-bit.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.models.feedback_classifier import FUSION_BLOCKS, FeedbackLoopClassifier
from src.models.fusion_classifier import (
    AGAFFusionEdgeClassifier,
    agaf_feature_attention,
    agaf_feature_gate_fuse,
)
from src.pipeline.common.datasets import get_dataset_config
from src.pipeline.common.splits import NUM_CLASSES
from src.pipeline.step4.feedback_config import load_feedback_config

DATASET = "unsw_nb15"
FOLD = 0


def _model(fusion_block: str) -> FeedbackLoopClassifier:
    return FeedbackLoopClassifier(
        num_classes=NUM_CLASSES, injection_mode="edge", bias_dim=5,
        edge_attr_dim=5, injection_scale=2.0, bias_confidence_frac=0.5,
        fusion_block=fusion_block,
    )


# --- 1. it really is AGAF's block --------------------------------------------

def test_agaf_block_matches_agafs_own_layers_exactly():
    """Same weights, same inputs, same output — through AGAF's module and
    through the loop's copy of the shapes."""
    torch.manual_seed(0)
    proj_dim = 16
    agaf = AGAFFusionEdgeClassifier(
        gnn_dim=8, llm_dim=32, proj_dim=proj_dim, hidden_dim=8,
        num_classes=NUM_CLASSES, fusion_mode="feature_gate",
    )
    agaf.eval()
    h = torch.randn(7, proj_dim)
    s = torch.randn(7, proj_dim)

    fused_ref, gate_ref = agaf.fuse(h, s)
    attended_ref, attn_ref = agaf_feature_attention(agaf.feature_attention, fused_ref)

    fused, gate = agaf_feature_gate_fuse(agaf.gate, h, s)
    attended, attn = agaf_feature_attention(agaf.feature_attention, fused)

    torch.testing.assert_close(fused, fused_ref, rtol=0, atol=0)
    torch.testing.assert_close(gate, gate_ref, rtol=0, atol=0)
    torch.testing.assert_close(attended, attended_ref, rtol=0, atol=0)
    # The attention really is a distribution over the fused vector's dimensions.
    torch.testing.assert_close(attn.sum(dim=1), torch.ones(7), rtol=0, atol=1e-6)


def test_agaf_block_layer_shapes_match_agafs():
    """4*proj -> proj gate, proj -> proj attention. A different shape would be a
    different mechanism wearing the same flag."""
    m = _model("agaf")
    proj = m.fusion_llm_proj.out_features
    assert m.fusion_feature_gate.in_features == 4 * proj
    assert m.fusion_feature_gate.out_features == proj
    assert m.fusion_feature_attention.in_features == proj
    assert m.fusion_feature_attention.out_features == proj
    # The classifier reads a proj-wide vector, not the loop block's 2*proj concat.
    assert m.fusion_classifier[0].in_features == proj
    assert _model("loop").fusion_classifier[0].in_features == 2 * proj


def test_only_the_block_in_use_is_allocated():
    """Dead parameters would make the reported capacity a fiction."""
    loop, agaf = _model("loop"), _model("agaf")
    assert hasattr(loop, "fusion_gate_conf") and not hasattr(loop, "fusion_feature_gate")
    assert hasattr(agaf, "fusion_feature_gate") and not hasattr(agaf, "fusion_gate_conf")
    assert agaf.fusion_block_parameter_count() > loop.fusion_block_parameter_count()
    # `fusion_hidden` / `fusion_out` are the GNN's own variant fusion, not the
    # output-fusion block, so they must NOT be in the count.
    block_prefixes = ("fusion_gnn_proj", "fusion_llm_proj", "fusion_head_proj",
                      "fusion_gate", "fusion_feature_", "fusion_classifier")
    for m in (loop, agaf):
        assert m.fusion_block_parameter_count() == sum(
            p.numel() for n, p in m.named_parameters()
            if n.startswith(block_prefixes)
        )
    assert loop.fusion_block_parameter_count() == 125266
    assert agaf.fusion_block_parameter_count() == 199242


def test_the_fusion_block_does_not_shift_the_global_rng_stream():
    """Construction must advance the RNG identically for both blocks. It did
    not, and that alone moved `head_only` (which never reaches this block) by
    shifting every dropout mask drawn afterwards."""
    def build(block):
        torch.manual_seed(42)
        _model(block)
        return torch.rand(4)

    torch.testing.assert_close(build("loop"), build("agaf"), rtol=0, atol=0)


def test_fusion_block_rejects_unknown_values():
    with pytest.raises(ValueError, match="fusion_block"):
        _model("transformer")
    for b in FUSION_BLOCKS:
        _model(b)


def test_default_is_the_loop_block():
    assert FeedbackLoopClassifier(
        num_classes=NUM_CLASSES, injection_scale=2.0
    ).fusion_block == "loop"


# --- 2 & 3. the run-level guarantees -----------------------------------------

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


def test_head_only_is_unchanged_by_the_fusion_block():
    """`head_only` never enters `_output_fusion`, so swapping that block must
    not move it by a single bit. If this fails, the E3 comparison has no
    baseline."""
    base = _train("head_only")
    swapped = _train("head_only", fusion_block="agaf")
    torch.testing.assert_close(base.logits, swapped.logits, rtol=0, atol=0)
    assert base.test_macro_f1 == swapped.test_macro_f1


def test_loop_block_reproduces_the_default_bit_for_bit():
    default = _train("real")
    explicit = _train("real", fusion_block="loop")
    torch.testing.assert_close(default.logits, explicit.logits, rtol=0, atol=0)
    assert default.test_macro_f1 == explicit.test_macro_f1


def test_agaf_block_actually_changes_the_real_mode():
    """The treatment must do something, or a null result is uninformative."""
    base = _train("real")
    swapped = _train("real", fusion_block="agaf")
    assert not torch.allclose(base.logits, swapped.logits)
