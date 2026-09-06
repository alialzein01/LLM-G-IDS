"""Task 2 — the semantic consultant must have a usable confidence.

`WhitenedPrototypeScorer` returns `cos / T` with `T` initialised to 10.0.
Whitened CySecBERT cosines span roughly [-0.25, 0.76], so dividing by 10 leaves
logits spanning ~0.1 and a softmax that is uniform to three decimals. Two things
downstream rank on that softmax: the second-stage confidence gate
(`bias_confidence_frac`, keep the most confident half) and the injected advice
vector itself. Both were ranking noise, and `mean_disagreement` sat pinned at
1 - 1/C = 0.900 in every fold of both datasets.

These tests pin the consultant's softmax to something a gate can rank.

Calibration is NOT the default: the canonical runs behind `results/*_current.json`
pin T=10 on purpose (condition A), because turning the signal on measured worse
at 3 seeds. So the training test below asks for calibration explicitly, and a
companion test pins that the default is still the legacy pin. Both facts matter:
the fix must keep working, and it must stay off by default.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.models.feedback_classifier import WhitenedPrototypeScorer
from src.pipeline.common.datasets import get_dataset_config
from src.pipeline.common.splits import NUM_CLASSES

DATASETS = ("unsw_nb15", "ton_iot")
FOLD = 0
# Plan thresholds. Uniform over 10 classes is max-prob 0.100 / disagreement 0.900.
MIN_MAX_PROB = 0.25
MAX_DISAGREEMENT = 0.80

_TRAIN_CACHE: dict[str, object] = {}


def _fold0(dataset: str):
    cfg = get_dataset_config(dataset)
    root = Path(f"data/{dataset}/processed/step4_feedback")
    for p in (cfg.llm_embedding_path, cfg.splits_path, root / "prototypes.pt"):
        if not Path(p).exists():
            pytest.skip(f"missing pipeline artifact {p}")
    emb = torch.load(cfg.llm_embedding_path, weights_only=False).float()
    folds = torch.load(cfg.splits_path, weights_only=False)
    protos = torch.load(root / "prototypes.pt", weights_only=False)
    return cfg, emb, folds, protos


def _fold0_scorer(dataset: str):
    """The scorer exactly as the canonical path builds it, at its init."""
    cfg, emb, folds, protos = _fold0(dataset)
    st = protos["folds"][FOLD]
    scorer = WhitenedPrototypeScorer(
        num_classes=NUM_CLASSES, embed_dim=protos["embed_dim"]
    )
    scorer.load_fold_state(st["mean"], st["whitener"], st["prototypes_whitened"])
    return scorer, emb, folds


@pytest.mark.parametrize("dataset", DATASETS)
def test_consultant_softmax_is_not_uniform_at_init(dataset):
    """Plan 2.1 — 30% of edges, mean max-prob > 0.25, disagreement < 0.80."""
    scorer, emb, folds = _fold0_scorer(dataset)
    g = torch.Generator().manual_seed(0)
    n = emb.shape[0]
    idx = torch.randperm(n, generator=g)[: max(1, int(round(0.30 * n)))]
    with torch.no_grad():
        probs = scorer(emb[idx]).softmax(dim=-1)
    max_prob = float(probs.max(dim=-1).values.mean())
    # Disagreement against the consultant's own runner-up class is the cleanest
    # label-free stand-in for "1 - p[the GNN's class]": if the top class does not
    # carry mass, nothing the gate ranks does.
    disagreement = 1.0 - max_prob
    assert max_prob > MIN_MAX_PROB, (
        f"[{dataset}] consultant mean max-prob {max_prob:.4f} <= {MIN_MAX_PROB} "
        f"(uniform over {NUM_CLASSES} classes is {1 / NUM_CLASSES:.3f})"
    )
    assert disagreement < MAX_DISAGREEMENT, (
        f"[{dataset}] mean disagreement {disagreement:.4f} >= {MAX_DISAGREEMENT}"
    )


def _train_fold0_real(dataset: str):
    if dataset in _TRAIN_CACHE:
        return _TRAIN_CACHE[dataset]
    import src.pipeline.step4.train_feedback as tf
    from src.pipeline.step4.feedback_config import load_feedback_config

    cfg, emb, folds, protos = _fold0(dataset)
    data = torch.load(cfg.graph_path, weights_only=False)
    tf.IN_DIM = data.x.shape[1]
    sel = load_feedback_config(dataset)
    saved = tf.MAX_EPOCHS
    tf.MAX_EPOCHS = 150
    try:
        result = tf._train_one_fold(
            data, emb, folds[FOLD], protos["folds"][FOLD], FOLD, "real",
            top_k_percent=float(sel["top_k_percent"]),
            bias_confidence_fraction=float(sel["bias_confidence_fraction"]),
            gate_mode=sel.get("gate_mode", "confidence"),
            eval_classes=cfg.eval_classes,
            dropped_classes=cfg.dropped_classes,
            injection_mode=sel.get("injection_mode", "edge"),
            injection_scale=float(sel.get("injection_scale", 10.0)),
            # Explicit: this test is about the calibrated consultant, which is
            # reachable but not the default.
            legacy_temperature=False,
        )
    finally:
        tf.MAX_EPOCHS = saved
    _TRAIN_CACHE[dataset] = result
    return result


@pytest.mark.parametrize("dataset", DATASETS)
def test_trained_consultant_confidence_is_reported_and_moved(dataset):
    """Plan 2.3 — the two numbers must be in `bias_diagnostics`, and
    post-training `mean_disagreement` must be off the uniform pin."""
    diag = _train_fold0_real(dataset).bias_diagnostics or {}
    for key in ("consultant_mean_max_prob", "mean_disagreement"):
        assert key in diag, f"[{dataset}] bias_diagnostics missing {key}: {sorted(diag)}"
    assert diag["mean_disagreement"] < MAX_DISAGREEMENT, (
        f"[{dataset}] post-training mean_disagreement {diag['mean_disagreement']:.4f} "
        f">= {MAX_DISAGREEMENT}"
    )
    assert diag["consultant_mean_max_prob"] > MIN_MAX_PROB, (
        f"[{dataset}] post-training consultant_mean_max_prob "
        f"{diag['consultant_mean_max_prob']:.4f} <= {MIN_MAX_PROB}"
    )


@pytest.mark.parametrize("dataset", DATASETS)
def test_legacy_pin_is_what_a_default_fold_gets(dataset):
    """The mirror of the test above: with no flag, the consultant is the T=10
    near-uniform softmax the contracts were measured under. If this starts
    failing, the default silently moved off condition A."""
    import src.pipeline.step4.train_feedback as tf
    from src.pipeline.step4.feedback_config import load_feedback_config

    cfg, emb, folds, protos = _fold0(dataset)
    data = torch.load(cfg.graph_path, weights_only=False)
    tf.IN_DIM = data.x.shape[1]
    sel = load_feedback_config(dataset)
    saved = tf.MAX_EPOCHS
    tf.MAX_EPOCHS = 3
    try:
        result = tf._train_one_fold(
            data, emb, folds[FOLD], protos["folds"][FOLD], FOLD, "real",
            top_k_percent=float(sel["top_k_percent"]),
            bias_confidence_fraction=float(sel["bias_confidence_fraction"]),
            gate_mode=sel.get("gate_mode", "confidence"),
            eval_classes=cfg.eval_classes,
            dropped_classes=cfg.dropped_classes,
            injection_mode=sel.get("injection_mode", "edge"),
            injection_scale=float(sel.get("injection_scale", 10.0)),
        )
    finally:
        tf.MAX_EPOCHS = saved
    diag = result.bias_diagnostics or {}
    assert diag["consultant_temperature"] == pytest.approx(10.0, rel=1e-3)
    assert diag["mean_disagreement"] > MAX_DISAGREEMENT
