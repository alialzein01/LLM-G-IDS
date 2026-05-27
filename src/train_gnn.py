from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch_geometric.data import Data

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.gnn_classifier import GATEdgeClassifier
import torch.nn as nn

from src.splits import (
    NUM_CLASSES,
    create_edge_splits,
    get_class_weights,
)


DATA_PATH = "data/processed/step1/pyg_data.pt"
SPLITS_PATH = "data/processed/splits/folds.pt"
OUTPUT_DIR = "data/processed/step3_gnn"

IN_DIM = 10
HIDDEN_DIM = 64
EDGE_ATTR_DIM = 5
HEADS = 4
DROPOUT = 0.2

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
MAX_EPOCHS = 300
EARLY_STOPPING_PATIENCE = 40
GRAD_CLIP = 1.0
LOG_EVERY = 20
SEED = 42

LABEL_NAMES = [
    "Benign",
    "backdoor",
    "ddos",
    "dos",
    "injection",
    "mitm",
    "password",
    "ransomware",
    "scanning",
    "xss",
]


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


def _macro_f1(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> float:
    preds = logits[mask].argmax(dim=1).cpu().numpy()
    targets = labels[mask].cpu().numpy()
    return float(
        f1_score(targets, preds, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0)
    )


def _train_step(
    model: GATEdgeClassifier,
    data: Data,
    train_mask: torch.Tensor,
    criterion: nn.CrossEntropyLoss,
) -> torch.Tensor:
    """
    One training step.

    HOW THIS WORKS (step by step):
    1. Run the full GNN forward pass to get one logit vector per edge [E, 10].
    2. Slice to the training edges only.
    3. Compute weighted CrossEntropyLoss — rare classes have higher weight so
       the gradient for a ransomware mistake is much larger than for a Benign
       mistake, pushing the model to pay attention to rare attacks.

    WHY NOT FOCAL LOSS:
    Focal loss with class weights caused NaN gradients on this small dataset
    (2127 edges). The inverse-frequency class weights already provide the
    imbalance correction; focal loss on top creates gradient magnitudes that
    blow up in the GATv2 BatchNorm layers.

    WHY NOT SMOTE/OVERSAMPLING:
    With only 2–3 real samples for the rarest classes, synthetic interpolations
    are essentially noise. They overwhelmed the training signal for medium-size
    classes (mitm: 157 samples, injection: 116) and reduced their F1 from 0.69
    to 0.0. Removed.
    """
    model.train()
    logits, _ = model(data.x, data.edge_index, data.edge_attr)
    return criterion(logits[train_mask], data.edge_label[train_mask])


def _train_one_fold(
    data: Data,
    fold: dict[str, torch.Tensor],
    fold_idx: int,
) -> dict[str, object]:
    _set_seed(SEED + fold_idx)
    model = _build_model()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    train_mask = fold["train_mask"]
    val_mask = fold["val_mask"]

    class_weights = get_class_weights(data.edge_label, train_mask)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    history: list[dict[str, float]] = []
    best_val_f1 = -1.0
    best_epoch = -1
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

        history.append(
            {"epoch": epoch, "train_loss": float(loss.item()), "val_macro_f1": val_f1}
        )

        if epoch % LOG_EVERY == 0 or epoch == 1:
            print(
                f"  fold {fold_idx} | epoch {epoch:3d} | "
                f"train_loss={loss.item():.4f} | val_macro_f1={val_f1:.4f}"
            )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(
                    f"  fold {fold_idx} | early stopping at epoch {epoch} "
                    f"(best epoch={best_epoch}, best val_macro_f1={best_val_f1:.4f})"
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        eval_logits, _ = model(data.x, data.edge_index, data.edge_attr)
        test_mask = fold["test_mask"]
        test_f1 = _macro_f1(eval_logits, data.edge_label, test_mask)

    print(
        f"  fold {fold_idx} | best epoch={best_epoch} | "
        f"best val_macro_f1={best_val_f1:.4f} | test_macro_f1={test_f1:.4f}"
    )

    return {
        "fold": fold_idx,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "test_macro_f1": test_f1,
    }


def _train_final_model(data: Data) -> GATEdgeClassifier:
    _set_seed(SEED)
    model = _build_model()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    full_mask = torch.ones(data.edge_label.shape[0], dtype=torch.bool)
    class_weights = get_class_weights(data.edge_label, full_mask)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print("Training final model on all edges:")
    for epoch in range(1, MAX_EPOCHS + 1):
        optimizer.zero_grad()
        loss = _train_step(model, data, full_mask, criterion)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if epoch % LOG_EVERY == 0 or epoch == 1:
            print(f"  final | epoch {epoch:3d} | train_loss={loss.item():.4f}")

    return model


def _per_class_metrics(
    preds: np.ndarray, targets: np.ndarray
) -> list[dict[str, object]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        preds,
        labels=list(range(NUM_CLASSES)),
        zero_division=0,
    )
    return [
        {
            "class_id": cls,
            "class_name": LABEL_NAMES[cls],
            "precision": float(precision[cls]),
            "recall": float(recall[cls]),
            "f1": float(f1[cls]),
            "support": int(support[cls]),
        }
        for cls in range(NUM_CLASSES)
    ]


def _print_class_report(preds: np.ndarray, targets: np.ndarray) -> None:
    """Print a per-class results table sorted by sample count (most common first)."""
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, preds, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    print(f"\n  {'Class':<12} {'Support':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print(f"  {'-'*52}")
    order = np.argsort(-support)
    for cls in order:
        print(
            f"  {LABEL_NAMES[cls]:<12} {int(support[cls]):>8d} "
            f"{precision[cls]:>10.4f} {recall[cls]:>8.4f} {f1[cls]:>8.4f}"
        )


def main() -> None:
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading graph data from {DATA_PATH}")
    data: Data = torch.load(DATA_PATH, weights_only=False)

    splits_file = Path(SPLITS_PATH)
    if splits_file.exists():
        print(f"Loading splits from {SPLITS_PATH}")
        folds = torch.load(SPLITS_PATH, weights_only=False)
    else:
        print(f"Splits not found — creating new splits at {SPLITS_PATH}")
        folds = create_edge_splits(data)

    fold_results: list[dict[str, object]] = []
    for fold_idx, fold in enumerate(folds):
        print(f"\n=== Fold {fold_idx} ===")
        result = _train_one_fold(data, fold, fold_idx)
        fold_results.append(result)

    val_f1s = np.array([r["best_val_macro_f1"] for r in fold_results])
    test_f1s = np.array([r["test_macro_f1"] for r in fold_results])
    print("\n=== Cross-validation summary ===")
    print(f"val_macro_f1:  {val_f1s.mean():.4f} ± {val_f1s.std():.4f}")
    print(f"test_macro_f1: {test_f1s.mean():.4f} ± {test_f1s.std():.4f}")

    history_payload = {
        "folds": fold_results,
        "val_macro_f1_mean": float(val_f1s.mean()),
        "val_macro_f1_std": float(val_f1s.std()),
        "test_macro_f1_mean": float(test_f1s.mean()),
        "test_macro_f1_std": float(test_f1s.std()),
    }
    with open(output_path / "training_history.json", "w") as f:
        json.dump(history_payload, f, indent=2)

    print("\n=== Final model ===")
    final_model = _train_final_model(data)

    torch.save(final_model.state_dict(), output_path / "model.pt")

    final_model.eval()
    with torch.no_grad():
        logits, edge_emb = final_model(data.x, data.edge_index, data.edge_attr)
        preds = logits.argmax(dim=1).cpu().numpy()

    torch.save(edge_emb.detach().cpu(), output_path / "edge_embeddings.pt")

    targets = data.edge_label.cpu().numpy()
    overall_macro_f1 = float(
        f1_score(targets, preds, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0)
    )
    overall_weighted_f1 = float(
        f1_score(targets, preds, average="weighted", labels=list(range(NUM_CLASSES)), zero_division=0)
    )
    overall_accuracy = float((preds == targets).mean())
    per_class = _per_class_metrics(preds, targets)

    print(f"\nFinal model results:")
    print(f"  Accuracy:    {overall_accuracy:.4f}")
    print(f"  Macro-F1:    {overall_macro_f1:.4f}   ← primary metric")
    print(f"  Weighted-F1: {overall_weighted_f1:.4f}")
    _print_class_report(preds, targets)

    metrics_payload = {
        "overall_accuracy": overall_accuracy,
        "overall_macro_f1": overall_macro_f1,
        "overall_weighted_f1": overall_weighted_f1,
        "per_class": per_class,
        "predictions": preds.tolist(),
    }
    with open(output_path / "metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"\nSaved model, embeddings, and metrics to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
