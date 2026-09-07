"""Cross-fitted consultant advice.

`build_llm_heads` writes, for outer fold f, the fold-f head applied to EVERY
edge. On f's own train edges those logits are in-fold — the head saw those
labels — so the consultant looks far more reliable there than on the test edges
the loop is scored on. `--crossfit K` replaces the train rows with inner
out-of-fold logits so the advice the loop trains on is as reliable as the advice
it is scored with.

The load-bearing property is that this changes NOTHING about out-of-fold
discipline: no outer test label reaches any head that scores it, before or
after. That is asserted per fold — permute fold f's test labels and row f must
come back bit-identical. Stated per fold on purpose: the five test masks
partition the edge set, so permuting them all at once also permutes every other
fold's TRAIN labels and proves nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.pipeline.common.datasets import get_dataset_config
from src.pipeline.common.splits import NUM_CLASSES, mask_dropped_logits
from src.pipeline.step4 import build_llm_heads as bh

DATASET = "unsw_nb15"
K = 3  # smaller than the production 5 so the test builds 4 heads per fold, not 6


def _require(dataset: str):
    cfg = get_dataset_config(dataset)
    for p in (cfg.graph_path, cfg.llm_embedding_path, cfg.splits_path):
        if not Path(p).exists():
            pytest.skip(f"missing pipeline artifact {p}")
    return cfg


def _build(tmp_path, monkeypatch, labels=None, epochs=6):
    """Build a cross-fitted file on a short schedule, optionally with substituted
    labels, and return the tensor."""
    cfg = _require(DATASET)
    real_labels, emb, folds = bh._load_inputs(cfg)
    used = real_labels if labels is None else labels
    monkeypatch.setattr(bh, "_load_inputs", lambda c: (used, emb, folds))
    monkeypatch.setattr(bh, "MAX_EPOCHS", epochs)
    monkeypatch.setattr(bh, "EARLY_STOPPING_PATIENCE", epochs)
    out = tmp_path / "crossfit.pt"
    bh.build_llm_heads_crossfit(DATASET, k=K, output_path=out)
    return torch.load(out, weights_only=False), real_labels, folds


@pytest.mark.parametrize("fold_idx", (0, 2))
def test_no_outer_test_label_reaches_the_head_that_scores_it(
    tmp_path, monkeypatch, fold_idx
):
    """The leakage check, stated per fold.

    Scrambling EVERY fold's test labels at once proves nothing: the five test
    masks partition the edge set, so fold 0's test edges are fold 1's TRAIN
    edges, and every row is entitled to move. The property that matters is
    per-fold — permute only fold f's test labels and row f must be identical,
    bit for bit. Rows other than f may legitimately move.
    """
    baseline, labels, folds = _build(tmp_path, monkeypatch)

    scrambled = labels.clone()
    idx = folds[fold_idx]["test_mask"].nonzero(as_tuple=True)[0]
    g = torch.Generator().manual_seed(11)
    scrambled[idx] = labels[idx][torch.randperm(idx.numel(), generator=g)]
    assert not torch.equal(scrambled, labels), "test setup did not change anything"

    with_scrambled, _, _ = _build(tmp_path, monkeypatch, labels=scrambled)
    torch.testing.assert_close(
        with_scrambled[fold_idx], baseline[fold_idx], rtol=0, atol=0
    )


def test_shape_and_determinism(tmp_path, monkeypatch):
    first, labels, folds = _build(tmp_path, monkeypatch)
    assert first.shape == (len(folds), labels.shape[0], NUM_CLASSES)
    second, _, _ = _build(tmp_path, monkeypatch)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_only_the_train_rows_change(tmp_path, monkeypatch):
    """Val and test rows must still come from the full-outer-train head, so the
    file's out-of-fold predictions — and its pooled OOF macro-F1 — are unchanged.
    Only the train rows are cross-fitted."""
    cfg = _require(DATASET)
    labels, emb, folds = bh._load_inputs(cfg)
    monkeypatch.setattr(bh, "MAX_EPOCHS", 6)
    monkeypatch.setattr(bh, "EARLY_STOPPING_PATIENCE", 6)
    crossfit, _, _ = _build(tmp_path, monkeypatch)

    for fi, fold in enumerate(folds):
        full, _ = bh._train_head(
            emb, labels, fold["train_mask"], fold["val_mask"],
            bh.SEED + fi, cfg.eval_classes,
        )
        for name in ("val_mask", "test_mask"):
            m = fold[name]
            torch.testing.assert_close(
                crossfit[fi][m], full[m], rtol=0, atol=0,
            ), f"fold {fi} {name} rows moved"
        tm = fold["train_mask"]
        assert not torch.allclose(crossfit[fi][tm], full[tm]), (
            f"fold {fi} train rows are unchanged — nothing was cross-fitted"
        )


def test_crossfit_lowers_train_accuracy_toward_test_accuracy(tmp_path, monkeypatch):
    """The point of the exercise: the consultant must stop looking better on the
    edges the loop trains against than on the edges it is scored on."""
    cfg = _require(DATASET)
    labels, emb, folds = bh._load_inputs(cfg)
    monkeypatch.setattr(bh, "MAX_EPOCHS", 6)
    monkeypatch.setattr(bh, "EARLY_STOPPING_PATIENCE", 6)
    crossfit, _, _ = _build(tmp_path, monkeypatch)

    for fi, fold in enumerate(folds):
        full, _ = bh._train_head(
            emb, labels, fold["train_mask"], fold["val_mask"],
            bh.SEED + fi, cfg.eval_classes,
        )
        tm = fold["train_mask"]
        infold = float(
            (mask_dropped_logits(full, cfg.dropped_classes)[tm].argmax(1)
             == labels[tm]).float().mean()
        )
        crossed = float(
            (mask_dropped_logits(crossfit[fi], cfg.dropped_classes)[tm].argmax(1)
             == labels[tm]).float().mean()
        )
        assert crossed <= infold, (
            f"fold {fi}: cross-fit train accuracy {crossed} exceeds in-fold {infold}"
        )


def test_inner_splits_cover_the_train_edges_exactly_once():
    cfg = _require(DATASET)
    labels, _, folds = bh._load_inputs(cfg)
    for fi, fold in enumerate(folds):
        held = bh._inner_splits(labels, fold["train_mask"], 5, bh.SEED + fi)
        assert len(held) == 5
        allv = torch.cat([torch.as_tensor(h) for h in held])
        expected = fold["train_mask"].nonzero(as_tuple=True)[0]
        assert torch.equal(allv.sort().values, expected.sort().values)


def test_crossfit_k_is_validated():
    with pytest.raises(ValueError, match="crossfit K"):
        bh.build_llm_heads_crossfit(DATASET, k=1)


def test_the_canonical_builder_is_unchanged_by_default():
    """`--crossfit` is opt-in; the default path must still write the in-fold file."""
    import inspect

    src = inspect.getsource(bh.build_llm_heads)
    assert "llm_head_logits.pt" in src
    assert "crossfit" not in src
