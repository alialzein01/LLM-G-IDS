from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.models.fusion_classifier import UnimodalEdgeClassifier
from src.pipeline.common.datasets import DATASETS, DatasetConfig, get_dataset_config
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


PROJ_DIM = 128
HIDDEN_DIM = 128
DROPOUT = 0.2

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
MAX_EPOCHS = 300
EARLY_STOPPING_PATIENCE = 40
GRAD_CLIP = 1.0
LOG_EVERY = 20
SEED = 42
FOCAL_GAMMA = 2.0

MODALITIES = ("gnn", "llm", "both")


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _build_model(in_dim: int) -> UnimodalEdgeClassifier:
    return UnimodalEdgeClassifier(
        in_dim=in_dim,
        proj_dim=PROJ_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    )


def _fit_scaler(emb: torch.Tensor, train_mask: torch.Tensor) -> StandardScaler:
    return StandardScaler().fit(emb.numpy()[train_mask.numpy()])


def _apply_scaler(emb: torch.Tensor, scaler: StandardScaler) -> torch.Tensor:
    return torch.from_numpy(scaler.transform(emb.numpy())).float()


def _macro_f1(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    preds = logits[mask].argmax(dim=1).cpu().numpy()
    targets = labels[mask].cpu().numpy()
    return float(
        f1_score(targets, preds, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0)
    )


def _train_step(
    model: UnimodalEdgeClassifier,
    emb_t: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    model.train()
    logits, _ = model(emb_t)
    return criterion(logits[train_mask], labels[train_mask])


def _train_one_fold(
    emb: torch.Tensor,
    labels: torch.Tensor,
    fold: dict[str, torch.Tensor],
    fold_idx: int,
    in_dim: int,
) -> dict[str, object]:
    _set_seed(SEED + fold_idx)

    train_mask = fold["train_mask"]
    val_mask = fold["val_mask"]
    test_mask = fold["test_mask"]

    scaler = _fit_scaler(emb, train_mask)
    emb_t = _apply_scaler(emb, scaler)

    model = _build_model(in_dim)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    class_weights = get_class_weights(labels, train_mask)
    criterion = FocalLoss(alpha=class_weights, gamma=FOCAL_GAMMA)

    history: list[dict[str, float]] = []
    best_val_f1 = -1.0
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        optimizer.zero_grad()
        loss = _train_step(model, emb_t, labels, train_mask, criterion)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_logits, _ = model(emb_t)
            val_f1 = _macro_f1(eval_logits, labels, val_mask)

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
        eval_logits, _ = model(emb_t)
        test_f1 = _macro_f1(eval_logits, labels, test_mask)
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


def _train_final_model(
    emb: torch.Tensor, labels: torch.Tensor, in_dim: int
) -> tuple[UnimodalEdgeClassifier, StandardScaler]:
    _set_seed(SEED)
    full_mask = torch.ones(labels.shape[0], dtype=torch.bool)
    scaler = _fit_scaler(emb, full_mask)
    emb_t = _apply_scaler(emb, scaler)

    model = _build_model(in_dim)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    class_weights = get_class_weights(labels, full_mask)
    criterion = FocalLoss(alpha=class_weights, gamma=FOCAL_GAMMA)

    print("Training final model on all edges:")
    for epoch in range(1, MAX_EPOCHS + 1):
        optimizer.zero_grad()
        loss = _train_step(model, emb_t, labels, full_mask, criterion)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if epoch % LOG_EVERY == 0 or epoch == 1:
            print(f"  final | epoch {epoch:3d} | train_loss={loss.item():.4f}")

    return model, scaler


def _load_embeddings(config: DatasetConfig, modality: str) -> tuple[torch.Tensor, int, str]:
    if modality == "gnn":
        path = config.gnn_embedding_path
        in_dim = config.gnn_dim
    elif modality == "llm":
        path = config.llm_embedding_path
        in_dim = config.llm_dim
    else:
        raise ValueError(f"Unsupported single modality: {modality}")

    print(f"Loading {modality.upper()} embeddings from {path}")
    emb = torch.load(path, weights_only=True)
    return emb, in_dim, path


def _train_modality(dataset: str, modality: str) -> None:
    config = get_dataset_config(dataset)
    output_path = Path(config.baseline_output_dir) / f"{modality}_embedding"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading graph data from {config.graph_path}")
    data = torch.load(config.graph_path, weights_only=False)
    labels = data.edge_label
    n_edges = labels.shape[0]
    inv_mapping = {v: k for k, v in data.label_mapping.items()}
    label_names = [inv_mapping.get(i, config.label_names[i]) for i in range(NUM_CLASSES)]

    emb, in_dim, emb_path = _load_embeddings(config, modality)
    assert emb.shape == (n_edges, in_dim), (
        f"{modality} embeddings shape mismatch: {tuple(emb.shape)} vs ({n_edges}, {in_dim})"
    )

    splits_file = Path(config.splits_path)
    if splits_file.exists():
        print(f"Loading splits from {config.splits_path}")
        folds = torch.load(config.splits_path, weights_only=False)
    else:
        print(f"Splits not found — creating new splits at {config.splits_path}")
        folds = create_edge_splits(data, output_path=config.splits_path)

    split_size = folds[0]["train_mask"].shape[0]
    if split_size != n_edges:
        raise ValueError(
            f"Stale splits detected: splits have {split_size} edges but graph has {n_edges}."
        )

    fold_results: list[dict[str, object]] = []
    targets = labels.cpu().numpy()
    pooled_preds = np.full(n_edges, fill_value=-1, dtype=np.int64)
    for fold_idx, fold in enumerate(folds):
        print(f"\n=== {modality.upper()} fold {fold_idx} ===")
        result = _train_one_fold(emb, labels, fold, fold_idx, in_dim)
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
    print(f"val_macro_f1:  {val_f1s.mean():.4f} +/- {val_f1s.std():.4f}")
    print(f"test_macro_f1: {test_f1s.mean():.4f} +/- {test_f1s.std():.4f}")
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
    final_model, scaler = _train_final_model(emb, labels, in_dim)
    torch.save(final_model.state_dict(), output_path / "model.pt")
    torch.save(scaler, output_path / "scaler.pt")

    final_model.eval()
    emb_t = _apply_scaler(emb, scaler)
    with torch.no_grad():
        logits, edge_emb = final_model(emb_t)
        final_preds = logits.argmax(dim=1).cpu().numpy()
    torch.save(edge_emb.detach().cpu(), output_path / "edge_embeddings.pt")

    final_model_train_metrics = classification_metrics(final_preds, targets, label_names)

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
        model_name=f"frozen_{modality}_embedding_mlp",
        history_payload=history_payload,
        pooled_metrics=pooled_metrics,
        artifact_paths={
            "source_embeddings": emb_path,
            "model": str(output_path / "model.pt"),
            "scaler": str(output_path / "scaler.pt"),
            "edge_embeddings": str(output_path / "edge_embeddings.pt"),
            "metrics": str(output_path / "metrics.json"),
        },
    )

    print(f"\nSaved {modality.upper()} baseline artifacts to {output_path}")


def main(dataset: str = "ton_iot", modality: str = "both") -> None:
    modalities = ("gnn", "llm") if modality == "both" else (modality,)
    for selected in modalities:
        _train_modality(dataset, selected)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train frozen-embedding baselines.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="ton_iot")
    parser.add_argument("--modality", choices=MODALITIES, default="both")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(dataset=args.dataset, modality=args.modality)
