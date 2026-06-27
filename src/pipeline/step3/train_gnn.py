from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch_geometric.data import Data

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.models.gnn_classifier import GATEdgeClassifier, VARIANT_NAMES
import torch.nn as nn

from src.pipeline.common.datasets import DATASETS, get_dataset_config
from src.pipeline.common.metrics import (
    classification_metrics,
    print_class_report,
    write_benchmark_summary,
)
from src.pipeline.common.splits import (
    NUM_CLASSES,
    FocalLoss,
    create_edge_splits,
    get_class_weights,
)


DATA_PATH = "data/ton_iot/processed/step1/pyg_data.pt"
SPLITS_PATH = "data/ton_iot/processed/splits/folds.pt"
OUTPUT_DIR = "data/ton_iot/processed/step3_gnn"

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
LOG_EVERY = 20
SEED = 42
FOCAL_GAMMA = 2.0

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
    criterion: nn.Module,
) -> torch.Tensor:
    model.train()
    logits, _, aux_logits = model.forward_with_aux(data.x, data.edge_index, data.edge_attr)
    labels = data.edge_label[train_mask]
    loss = criterion(logits[train_mask], labels)
    aux_loss = torch.stack(
        [criterion(aux_logits[name][train_mask], labels) for name in VARIANT_NAMES]
    ).mean()
    return loss + AUXILIARY_DETECTOR_LOSS_WEIGHT * aux_loss


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
    criterion = FocalLoss(alpha=class_weights, gamma=FOCAL_GAMMA)

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
        test_indices = test_mask.nonzero(as_tuple=True)[0].cpu().numpy()
        test_predictions = eval_logits[test_mask].argmax(dim=1).cpu().numpy()

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
        "test_indices": test_indices.tolist(),
        "test_predictions": test_predictions.tolist(),
    }


def _train_final_model(data: Data) -> GATEdgeClassifier:
    _set_seed(SEED)
    model = _build_model()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    full_mask = torch.ones(data.edge_label.shape[0], dtype=torch.bool)
    class_weights = get_class_weights(data.edge_label, full_mask)
    criterion = FocalLoss(alpha=class_weights, gamma=FOCAL_GAMMA)

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


def main(
    data_path: str = DATA_PATH,
    splits_path: str = SPLITS_PATH,
    output_dir: str = OUTPUT_DIR,
    dataset: str = "ton_iot",
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading graph data from {data_path}")
    data: Data = torch.load(data_path, weights_only=False)
    config = get_dataset_config(dataset)

    # Derive label names from the mapping stored in the Data object
    inv_mapping = {v: k for k, v in data.label_mapping.items()}
    label_names = [inv_mapping.get(i, config.label_names[i]) for i in range(NUM_CLASSES)]

    splits_file = Path(splits_path)
    if splits_file.exists():
        print(f"Loading splits from {splits_path}")
        folds = torch.load(splits_path, weights_only=False)
        n_edges = data.edge_label.shape[0]
        split_size = folds[0]["train_mask"].shape[0]
        if split_size != n_edges:
            raise ValueError(
                f"Stale splits detected: splits have {split_size} edges but "
                f"graph has {n_edges}. Delete {splits_path} and re-run to regenerate."
            )
    else:
        print(f"Splits not found — creating new splits at {splits_path}")
        splits_file.parent.mkdir(parents=True, exist_ok=True)
        folds = create_edge_splits(data, output_path=splits_path)

    fold_results: list[dict[str, object]] = []
    targets = data.edge_label.cpu().numpy()
    pooled_preds = np.full(data.edge_label.shape[0], fill_value=-1, dtype=np.int64)
    for fold_idx, fold in enumerate(folds):
        print(f"\n=== Fold {fold_idx} ===")
        result = _train_one_fold(data, fold, fold_idx)
        fold_results.append(result)
        pooled_preds[np.array(result["test_indices"], dtype=np.int64)] = np.array(
            result["test_predictions"], dtype=np.int64
        )

    if (pooled_preds < 0).any():
        raise RuntimeError("Some edges were not covered by out-of-fold predictions.")

    val_f1s = np.array([r["best_val_macro_f1"] for r in fold_results])
    test_f1s = np.array([r["test_macro_f1"] for r in fold_results])
    pooled_metrics = classification_metrics(pooled_preds, targets, label_names)
    print("\n=== Cross-validation summary ===")
    print(f"val_macro_f1:  {val_f1s.mean():.4f} ± {val_f1s.std():.4f}")
    print(f"test_macro_f1: {test_f1s.mean():.4f} ± {test_f1s.std():.4f}")
    print(f"pooled_test_macro_f1: {pooled_metrics['overall_macro_f1']:.4f}")
    print_class_report(pooled_preds, targets, label_names)

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

    final_model_train_metrics = classification_metrics(preds, targets, label_names)

    print(f"\nFinal model fit-on-all-edges results (not benchmark performance):")
    print(f"  Accuracy:    {final_model_train_metrics['overall_accuracy']:.4f}")
    print(f"  Macro-F1:    {final_model_train_metrics['overall_macro_f1']:.4f}")
    print(f"  Weighted-F1: {final_model_train_metrics['overall_weighted_f1']:.4f}")

    metrics_payload = {
        **pooled_metrics,
        "metric_source": "pooled_out_of_fold_test_predictions",
        "predictions": pooled_preds.tolist(),
        "final_model_train_metrics": final_model_train_metrics,
    }
    with open(output_path / "metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    write_benchmark_summary(
        output_path,
        dataset=dataset,
        model_name="end_to_end_gnn",
        history_payload=history_payload,
        pooled_metrics=pooled_metrics,
        artifact_paths={
            "model": str(output_path / "model.pt"),
            "edge_embeddings": str(output_path / "edge_embeddings.pt"),
            "metrics": str(output_path / "metrics.json"),
        },
    )

    print(f"\nSaved model, embeddings, and metrics to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the GNN edge classifier.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="ton_iot")
    parser.add_argument("--data-path")
    parser.add_argument("--splits-path")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    config = get_dataset_config(args.dataset)
    main(
        data_path=args.data_path or config.graph_path,
        splits_path=args.splits_path or config.splits_path,
        output_dir=args.output_dir or config.gnn_output_dir,
        dataset=args.dataset,
    )
