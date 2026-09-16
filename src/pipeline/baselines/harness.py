from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from src.pipeline.common.datasets import DatasetConfig
from src.pipeline.common.metrics import classification_metrics
from src.pipeline.common.splits import (
    eval_macro_f1,
    get_class_weights,
    mask_dropped_logits,
)

GRAD_CLIP = 1.0


def run_out_of_fold(
    model_factory: Callable[[], nn.Module],
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    labels: torch.Tensor,
    folds: list[dict[str, torch.Tensor]],
    *,
    seed: int,
    class_weighting: str,
    eval_classes: tuple[int, ...],
    dropped_classes: tuple[int, ...] = (),
    max_epochs: int = 300,
    patience: int = 25,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    optimizer: str = "adam",
) -> np.ndarray:
    """Train one model per fold, predict that fold's test edges, pool the result.

    Early stopping selects on validation macro-F1 restricted to `eval_classes`.
    Test masks are never read for selection.

    `dropped_classes` are driven to -inf before every argmax, so a baseline is
    held to the same protocol as the rungs it is compared against. Before
    2026-09-16 these models could predict a class the metric does not score,
    which cost them recall on edges the rungs could not lose.
    """
    pooled = np.full(labels.shape[0], -1, dtype=np.int64)

    for fold_idx, fold in enumerate(folds):
        torch.manual_seed(seed + fold_idx)
        np.random.seed(seed + fold_idx)

        model = model_factory()
        # Both source implementations use plain Adam, not AdamW. Defaulting to AdamW
        # here would silently apply decoupled weight decay neither paper specifies.
        opt_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[optimizer]
        opt = opt_cls(model.parameters(), lr=lr, weight_decay=weight_decay)

        train_mask, val_mask = fold["train_mask"], fold["val_mask"]
        if class_weighting == "inverse_frequency":
            criterion = nn.CrossEntropyLoss(weight=get_class_weights(labels, train_mask))
        elif class_weighting == "none":
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"unknown class_weighting {class_weighting!r}")

        best_f1, best_state, stale = -1.0, None, 0
        for _ in range(max_epochs):
            model.train()
            opt.zero_grad()
            loss = criterion(
                model(x, edge_index, edge_attr)[train_mask], labels[train_mask]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            model.eval()
            with torch.no_grad():
                val_logits = model(x, edge_index, edge_attr)[val_mask]
                val_preds = mask_dropped_logits(
                    val_logits, dropped_classes
                ).argmax(dim=1)
            val_f1 = eval_macro_f1(labels[val_mask], val_preds, eval_classes)

            if val_f1 > best_f1:
                best_f1, stale = val_f1, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            fold_preds = mask_dropped_logits(
                model(x, edge_index, edge_attr), dropped_classes
            ).argmax(dim=1).cpu().numpy()
        test_mask = fold["test_mask"].cpu().numpy()
        pooled[test_mask] = fold_preds[test_mask]

    if (pooled < 0).any():
        raise ValueError("some edges never appeared in a test fold")
    return pooled


def pooled_scores(
    labels: torch.Tensor, preds: np.ndarray, config: DatasetConfig
) -> dict[str, object]:
    y_true = labels.cpu().numpy()
    row_mask = np.isin(y_true, list(config.eval_classes))
    detail = classification_metrics(
        preds[row_mask], y_true[row_mask], config.label_names
    )
    return {
        "macro_f1": eval_macro_f1(labels, preds, config.eval_classes),
        "accuracy": float((preds[row_mask] == y_true[row_mask]).mean()),
        "weighted_f1": detail["overall_weighted_f1"],
        "per_class_f1": [
            {"class_id": c["class_id"], "class_name": c["class_name"], "f1": c["f1"]}
            for c in detail["per_class"]
            if c["class_id"] in config.eval_classes
        ],
    }
