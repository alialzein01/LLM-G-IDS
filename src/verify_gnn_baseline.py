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
    get_class_weights,
)


DATA_PATH = "data/processed/step1/pyg_data.pt"
SPLITS_PATH = "data/processed/splits/folds.pt"
MODEL_PATH = "data/processed/step3_gnn/model.pt"
EMBEDDINGS_PATH = "data/processed/step3_gnn/edge_embeddings.pt"
HISTORY_PATH = "data/processed/step3_gnn/training_history.json"

IN_DIM = 10
HIDDEN_DIM = 64
EDGE_ATTR_DIM = 5
HEADS = 4
DROPOUT = 0.2
MAX_EPOCHS = 300
GRAD_CLIP = 1.0
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


def _build_model() -> GATEdgeClassifier:
    return GATEdgeClassifier(
        in_dim=IN_DIM,
        hidden_dim=HIDDEN_DIM,
        edge_attr_dim=EDGE_ATTR_DIM,
        num_classes=NUM_CLASSES,
        heads=HEADS,
        dropout=DROPOUT,
    )


def _train_fold_for_eval(
    data: Data,
    fold: dict[str, torch.Tensor],
    fold_idx: int,
    max_epochs: int,
) -> np.ndarray:
    torch.manual_seed(SEED + fold_idx)
    np.random.seed(SEED + fold_idx)

    model = _build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)

    train_mask = fold["train_mask"]
    val_mask = fold["val_mask"]

    class_weights = get_class_weights(data.edge_label, train_mask)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    patience = 25
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(logits[train_mask], data.edge_label[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_logits, _ = model(data.x, data.edge_index, data.edge_attr)
            preds = eval_logits[val_mask].argmax(dim=1).cpu().numpy()
            targets = data.edge_label[val_mask].cpu().numpy()
            val_f1 = float(
                f1_score(
                    targets,
                    preds,
                    average="macro",
                    labels=list(range(NUM_CLASSES)),
                    zero_division=0,
                )
            )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits, _ = model(data.x, data.edge_index, data.edge_attr)
        test_preds = logits[fold["test_mask"]].argmax(dim=1).cpu().numpy()
    return test_preds


def main() -> None:
    print(f"Loading graph data from {DATA_PATH}")
    data: Data = torch.load(DATA_PATH, weights_only=False)
    num_edges = data.edge_label.shape[0]

    print(f"Loading splits from {SPLITS_PATH}")
    folds = torch.load(SPLITS_PATH, weights_only=False)

    print(f"Loading final model from {MODEL_PATH}")
    model = _build_model()
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    print(f"Loading training history from {HISTORY_PATH}")
    with open(HISTORY_PATH) as f:
        training_history = json.load(f)

    pooled_preds = np.full(num_edges, fill_value=-1, dtype=np.int64)
    pooled_targets = data.edge_label.cpu().numpy()

    for fold_idx, fold in enumerate(folds):
        print(f"Re-training fold {fold_idx} to evaluate test edges...")
        test_preds = _train_fold_for_eval(data, fold, fold_idx, MAX_EPOCHS)
        test_indices = fold["test_mask"].nonzero(as_tuple=True)[0].cpu().numpy()
        pooled_preds[test_indices] = test_preds

    assert (pooled_preds >= 0).all(), "Some edges were never assigned a fold prediction"

    accuracy = float((pooled_preds == pooled_targets).mean())
    macro_f1 = float(
        f1_score(
            pooled_targets,
            pooled_preds,
            average="macro",
            labels=list(range(NUM_CLASSES)),
            zero_division=0,
        )
    )
    weighted_f1 = float(
        f1_score(
            pooled_targets,
            pooled_preds,
            average="weighted",
            labels=list(range(NUM_CLASSES)),
            zero_division=0,
        )
    )

    print("\n=== Out-of-fold pooled metrics ===")
    print(f"accuracy:    {accuracy:.4f}")
    print(f"macro-F1:    {macro_f1:.4f}   ← primary metric for IDS comparisons")
    print(f"weighted-F1: {weighted_f1:.4f}")

    precision, recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        pooled_targets,
        pooled_preds,
        labels=list(range(NUM_CLASSES)),
        zero_division=0,
    )
    print(f"\nPer-class results (sorted by sample count):")
    print(f"  {'Class':<12} {'Support':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print(f"  {'-'*52}")
    order = np.argsort(-per_class_support)
    for cls in order:
        print(
            f"  {LABEL_NAMES[cls]:<12} {int(per_class_support[cls]):>8d} "
            f"{precision[cls]:>10.4f} {recall[cls]:>8.4f} {per_class_f1[cls]:>8.4f}"
        )

    print("\nAssertions:")
    for fold_result in training_history["folds"]:
        history_len = len(fold_result["history"])
        assert 1 <= history_len <= MAX_EPOCHS, (
            f"Fold {fold_result['fold']} history length {history_len} out of range"
        )
    print(f"  history length 1..{MAX_EPOCHS} per fold: OK")

    edge_embeddings = torch.load(EMBEDDINGS_PATH, weights_only=True)
    assert tuple(edge_embeddings.shape) == (num_edges, HIDDEN_DIM), (
        f"edge_embeddings.pt shape {tuple(edge_embeddings.shape)} != ({num_edges}, {HIDDEN_DIM})"
    )
    print(f"  edge_embeddings.pt shape == ({num_edges}, {HIDDEN_DIM}): OK")

    unique_preds = set(int(p) for p in pooled_preds)
    missing = set(range(NUM_CLASSES)) - unique_preds
    if missing:
        print(f"  Warning: pooled predictions missing classes: {sorted(missing)}")
        print(f"  This is expected for ultra-rare classes (dos=4, ransomware=3 samples).")
    else:
        print(f"  all {NUM_CLASSES} classes present in pooled predictions: OK")


if __name__ == "__main__":
    main()
