"""Phase 4.0 setup — build out-of-fold GNN predictions for Step 4.

Retrains a fresh `GATEdgeClassifier` on each of the 5 CV folds (same
hyperparameters as `src.pipeline.step3.train_gnn`), records the trained
model's logits on that fold's test edges, and stitches them into a single
`[E, num_classes]` OOF logit tensor written to
`data/{dataset}/processed/step4_feedback/oof_logits.pt`.

Downstream phases (uncertainty selector, prototype builder, feedback
loop training) consume these OOF logits as honest evidence of GNN
uncertainty. In-sample logits from the final Step 3 model would
underestimate entropy because the model has seen every edge.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.models.gnn_classifier import GATEdgeClassifier, VARIANT_NAMES
from src.pipeline.common.datasets import get_dataset_config
from src.pipeline.common.splits import (
    NUM_CLASSES,
    FocalLoss,
    get_class_weights,
)


IN_DIM = 10
HIDDEN_DIM = 64
EDGE_ATTR_DIM = 5
HEADS = 8
DROPOUT = 0.2
AUXILIARY_DETECTOR_LOSS_WEIGHT = 0.30

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
MAX_EPOCHS = 300
EARLY_STOPPING_PATIENCE = 40
GRAD_CLIP = 1.0
LOG_EVERY = 40
SEED = 42
FOCAL_GAMMA = 2.0


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _build_model() -> GATEdgeClassifier:
    return GATEdgeClassifier(
        in_dim=IN_DIM,
        hidden_dim=HIDDEN_DIM,
        edge_attr_dim=EDGE_ATTR_DIM,
        num_classes=NUM_CLASSES,
        heads=HEADS,
        dropout=DROPOUT,
    )


def _train_step(
    model: GATEdgeClassifier,
    data: Data,
    train_mask: torch.Tensor,
    criterion: FocalLoss,
) -> torch.Tensor:
    model.train()
    logits, _, aux_logits = model.forward_with_aux(
        data.x, data.edge_index, data.edge_attr
    )
    labels = data.edge_label[train_mask]
    loss = criterion(logits[train_mask], labels)
    aux_loss = torch.stack(
        [criterion(aux_logits[name][train_mask], labels) for name in VARIANT_NAMES]
    ).mean()
    return loss + AUXILIARY_DETECTOR_LOSS_WEIGHT * aux_loss


def _macro_f1(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> float:
    from sklearn.metrics import f1_score

    preds = logits[mask].argmax(dim=1).cpu().numpy()
    targets = labels[mask].cpu().numpy()
    return float(
        f1_score(
            targets,
            preds,
            average="macro",
            labels=list(range(NUM_CLASSES)),
            zero_division=0,
        )
    )


def _train_fold_capture_logits(
    data: Data,
    fold: dict[str, torch.Tensor],
    fold_idx: int,
) -> torch.Tensor:
    """Retrain one fold and return `[E, C]` logits from the best-val checkpoint.

    Only the test-fold rows will be used downstream, but returning the full
    tensor keeps the caller simple.
    """
    _set_seed(SEED + fold_idx)
    model = _build_model()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    train_mask = fold["train_mask"]
    val_mask = fold["val_mask"]

    class_weights = get_class_weights(data.edge_label, train_mask)
    criterion = FocalLoss(alpha=class_weights, gamma=FOCAL_GAMMA)

    best_val_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        optimizer.zero_grad()
        loss = _train_step(model, data, train_mask, criterion)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_logits, _ = model(data.x, data.edge_index, data.edge_attr)
            val_f1 = _macro_f1(eval_logits, data.edge_label, val_mask)

        if epoch % LOG_EVERY == 0 or epoch == 1:
            print(
                f"  fold {fold_idx} | epoch {epoch:3d} | "
                f"train_loss={loss.item():.4f} | val_macro_f1={val_f1:.4f}"
            )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        eval_logits, _ = model(data.x, data.edge_index, data.edge_attr)

    test_mask = fold["test_mask"]
    test_f1 = _macro_f1(eval_logits, data.edge_label, test_mask)
    print(
        f"  fold {fold_idx} | done | best val_f1={best_val_f1:.4f} | test_f1={test_f1:.4f}"
    )

    return eval_logits.detach().cpu()


def build_oof_logits(dataset: str) -> Path:
    config = get_dataset_config(dataset)
    data: Data = torch.load(config.graph_path, weights_only=False)
    global IN_DIM
    IN_DIM = data.x.shape[1]  # derive from graph (supports pruned node features)
    folds = torch.load(config.splits_path, weights_only=False)

    num_edges = data.edge_label.shape[0]
    oof_logits = torch.full(
        (num_edges, NUM_CLASSES), float("nan"), dtype=torch.float32
    )

    for fold_idx, fold in enumerate(folds):
        print(f"\n=== Fold {fold_idx} ===")
        fold_logits = _train_fold_capture_logits(data, fold, fold_idx)
        test_mask = fold["test_mask"]
        oof_logits[test_mask] = fold_logits[test_mask]

    if torch.isnan(oof_logits).any():
        raise RuntimeError(
            "Some edges were not covered by any fold's test set — check folds.pt."
        )

    output_dir = Path(f"data/{dataset}/processed/step4_feedback")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "oof_logits.pt"
    torch.save(oof_logits, output_path)

    oof_probs = oof_logits.softmax(dim=-1)
    oof_preds = oof_probs.argmax(dim=-1)
    acc = float((oof_preds == data.edge_label).float().mean())
    manifest = {
        "dataset": dataset,
        "num_edges": num_edges,
        "num_classes": NUM_CLASSES,
        "pooled_oof_accuracy": acc,
        "output_path": str(output_path),
    }
    with open(output_dir / "oof_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved OOF logits to {output_path}")
    print(f"Pooled OOF accuracy: {acc:.4f}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ton_iot", choices=["ton_iot", "unsw_nb15"])
    args = parser.parse_args()
    build_oof_logits(args.dataset)


if __name__ == "__main__":
    main()
