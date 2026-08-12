from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.models.fusion_classifier import AGAFFusionEdgeClassifier
from src.pipeline.common.datasets import DATASETS, get_dataset_config
from src.pipeline.common.metrics import (
    classification_metrics,
    fusion_diagnostics_summary,
    print_attention_summary,
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
GNN_EMB_PATH = "data/ton_iot/processed/step3_gnn/edge_embeddings.pt"
LLM_EMB_PATH = "data/ton_iot/processed/step3_llm/edge_embeddings.pt"
SPLITS_PATH = "data/ton_iot/processed/splits/folds.pt"
OUTPUT_DIR = "data/ton_iot/processed/step3_fusion"

GNN_DIM = 64
LLM_DIM = 768
PROJ_DIM = 128
HIDDEN_DIM = 128
DROPOUT = 0.2
GATE_ENTROPY_LAMBDA = 0.01
# "gate"   = convex blend g*h + (1-g)*s (original)
# "concat" = todo.md Step 3: concatenate both modalities, per-sample attention
FUSION_MODE = "gate"
FEATURE_ATTENTION_ENTROPY_LAMBDA = 0.0

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
MAX_EPOCHS = 300
EARLY_STOPPING_PATIENCE = 40
GRAD_CLIP = 1.0
LOG_EVERY = 20
SEED = 42
FOCAL_GAMMA = 2.0


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


# M3: when True, AGAF's semantic modality is the trained LLM head's per-fold OOF
# logits (dim == NUM_CLASSES) and those logits are re-injected at the output via a
# learned gate (see AGAFFusionEdgeClassifier.head_fusion). Set in main().
USE_HEAD_LOGITS = False


def _build_model() -> AGAFFusionEdgeClassifier:
    return AGAFFusionEdgeClassifier(
        gnn_dim=GNN_DIM,
        llm_dim=NUM_CLASSES if USE_HEAD_LOGITS else LLM_DIM,
        proj_dim=PROJ_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
        head_fusion=USE_HEAD_LOGITS,
        fusion_mode=FUSION_MODE,
    )


def _fit_scalers(
    gnn_emb: torch.Tensor, llm_emb: torch.Tensor, train_mask: torch.Tensor
) -> tuple[StandardScaler, StandardScaler]:
    mask_np = train_mask.numpy()
    gnn_scaler = StandardScaler().fit(gnn_emb.numpy()[mask_np])
    llm_scaler = StandardScaler().fit(llm_emb.numpy()[mask_np])
    return gnn_scaler, llm_scaler


def _apply_scalers(
    gnn_emb: torch.Tensor,
    llm_emb: torch.Tensor,
    gnn_scaler: StandardScaler,
    llm_scaler: StandardScaler,
) -> tuple[torch.Tensor, torch.Tensor]:
    gnn_t = torch.from_numpy(gnn_scaler.transform(gnn_emb.numpy())).float()
    llm_t = torch.from_numpy(llm_scaler.transform(llm_emb.numpy())).float()
    return gnn_t, llm_t


# Classes used for in-fold model selection (early stopping). Set to the dataset's
# eval_classes in main() so we select epochs on the metric we actually report.
EVAL_CLASSES: tuple[int, ...] = tuple(range(NUM_CLASSES))


def _macro_f1(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> float:
    preds = logits[mask].argmax(dim=1).cpu().numpy()
    targets = labels[mask].cpu().numpy()
    return float(
        f1_score(targets, preds, average="macro", labels=list(EVAL_CLASSES), zero_division=0)
    )


def _train_step(
    model: AGAFFusionEdgeClassifier,
    gnn_t: torch.Tensor,
    llm_t: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    criterion: nn.Module,
    gate_entropy_lambda: float = GATE_ENTROPY_LAMBDA,
    feature_attention_entropy_lambda: float = FEATURE_ATTENTION_ENTROPY_LAMBDA,
    head_raw: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.train()
    logits, _, diagnostics = model(gnn_t, llm_t, head_logits=head_raw)
    cls_loss = criterion(logits[train_mask], labels[train_mask])
    train_gate = diagnostics["gate"][train_mask]
    gate_entropy = -(
        train_gate * (train_gate + 1e-12).log()
        + (1.0 - train_gate) * (1.0 - train_gate + 1e-12).log()
    ).mean()
    feature_entropy = diagnostics["feature_attention_entropy"][train_mask].mean()
    loss = (
        cls_loss
        - gate_entropy_lambda * gate_entropy
        - feature_attention_entropy_lambda * feature_entropy
    )
    return loss, cls_loss.detach(), gate_entropy.detach()


def _gate_means(diagnostics: dict[str, torch.Tensor], mask: torch.Tensor) -> tuple[float, float]:
    gate_gnn = diagnostics["gate_gnn_mean"][mask]
    gate_llm = diagnostics["gate_llm_mean"][mask]
    return float(gate_gnn.mean().item()), float(gate_llm.mean().item())


def _train_one_fold(
    gnn_emb: torch.Tensor,
    llm_emb: torch.Tensor,
    labels: torch.Tensor,
    fold: dict[str, torch.Tensor],
    fold_idx: int,
    gate_entropy_lambda: float,
    feature_attention_entropy_lambda: float,
    head_all: torch.Tensor | None = None,
) -> dict[str, object]:
    _set_seed(SEED + fold_idx)

    train_mask = fold["train_mask"]
    val_mask = fold["val_mask"]
    test_mask = fold["test_mask"]

    # M3: in head-fusion mode the semantic modality is this fold's leak-safe OOF
    # head logits (used both as the scaled branch input and the raw residual).
    head_raw: torch.Tensor | None = None
    if head_all is not None:
        head_raw = head_all[fold_idx]
        llm_emb = head_raw

    gnn_scaler, llm_scaler = _fit_scalers(gnn_emb, llm_emb, train_mask)
    gnn_t, llm_t = _apply_scalers(gnn_emb, llm_emb, gnn_scaler, llm_scaler)

    model = _build_model()
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
        loss, cls_loss, gate_entropy = _train_step(
            model,
            gnn_t,
            llm_t,
            labels,
            train_mask,
            criterion,
            gate_entropy_lambda=gate_entropy_lambda,
            feature_attention_entropy_lambda=feature_attention_entropy_lambda,
            head_raw=head_raw,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_logits, _, eval_diagnostics = model(gnn_t, llm_t, head_logits=head_raw)
            val_f1 = _macro_f1(eval_logits, labels, val_mask)
            train_gate_gnn, train_gate_llm = _gate_means(eval_diagnostics, train_mask)
            val_gate_gnn, val_gate_llm = _gate_means(eval_diagnostics, val_mask)

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(loss.item()),
                "classification_loss": float(cls_loss.item()),
                "gate_entropy": float(gate_entropy.item()),
                "val_macro_f1": val_f1,
                "train_gate_mean_gnn": train_gate_gnn,
                "train_gate_mean_llm": train_gate_llm,
                "val_gate_mean_gnn": val_gate_gnn,
                "val_gate_mean_llm": val_gate_llm,
            }
        )

        if epoch % LOG_EVERY == 0 or epoch == 1:
            print(
                f"  fold {fold_idx} | epoch {epoch:3d} | "
                f"train_loss={loss.item():.4f} | cls_loss={cls_loss.item():.4f} | "
                f"gate_entropy={gate_entropy.item():.4f} | val_macro_f1={val_f1:.4f} | "
                f"gate_train=({train_gate_gnn:.3f},{train_gate_llm:.3f}) | "
                f"gate_val=({val_gate_gnn:.3f},{val_gate_llm:.3f})"
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
        eval_logits, _, diagnostics = model(gnn_t, llm_t, head_logits=head_raw)
        test_f1 = _macro_f1(eval_logits, labels, test_mask)
        test_indices = test_mask.nonzero(as_tuple=True)[0].cpu().numpy()
        test_predictions = eval_logits[test_mask].argmax(dim=1).cpu().numpy()
        test_gate = diagnostics["gate"][test_mask]
        test_feature_attention = diagnostics["feature_attention"][test_mask]
        gate_test_mean_gnn = float(diagnostics["gate_gnn_mean"][test_mask].mean().item())
        gate_test_mean_llm = float(diagnostics["gate_llm_mean"][test_mask].mean().item())

    print(
        f"  fold {fold_idx} | best epoch={best_epoch} | "
        f"best val_macro_f1={best_val_f1:.4f} | test_macro_f1={test_f1:.4f} | "
        f"gate(gnn,llm)=({gate_test_mean_gnn:.3f}, {gate_test_mean_llm:.3f})"
    )

    return {
        "fold": fold_idx,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "test_macro_f1": test_f1,
        "gate_test_mean_gnn": gate_test_mean_gnn,
        "gate_test_mean_llm": gate_test_mean_llm,
        "test_indices": test_indices.tolist(),
        "test_predictions": test_predictions.tolist(),
        "test_gate": test_gate.detach().cpu().tolist(),
        "test_feature_attention": test_feature_attention.detach().cpu().tolist(),
    }


def _train_final_model(
    gnn_emb: torch.Tensor,
    llm_emb: torch.Tensor,
    labels: torch.Tensor,
    gate_entropy_lambda: float,
    feature_attention_entropy_lambda: float,
    head_all: torch.Tensor | None = None,
) -> tuple[AGAFFusionEdgeClassifier, StandardScaler, StandardScaler]:
    _set_seed(SEED)
    full_mask = torch.ones(labels.shape[0], dtype=torch.bool)

    # M3: the fit-on-all-edges artifact has no single OOF head, so use the mean
    # of the per-fold head logits (this artifact is not the benchmark metric).
    head_raw: torch.Tensor | None = None
    if head_all is not None:
        head_raw = head_all.mean(dim=0)
        llm_emb = head_raw

    gnn_scaler, llm_scaler = _fit_scalers(gnn_emb, llm_emb, full_mask)
    gnn_t, llm_t = _apply_scalers(gnn_emb, llm_emb, gnn_scaler, llm_scaler)

    model = _build_model()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    class_weights = get_class_weights(labels, full_mask)
    criterion = FocalLoss(alpha=class_weights, gamma=FOCAL_GAMMA)

    print("Training final model on all edges:")
    for epoch in range(1, MAX_EPOCHS + 1):
        optimizer.zero_grad()
        loss, cls_loss, gate_entropy = _train_step(
            model,
            gnn_t,
            llm_t,
            labels,
            full_mask,
            criterion,
            gate_entropy_lambda=gate_entropy_lambda,
            feature_attention_entropy_lambda=feature_attention_entropy_lambda,
            head_raw=head_raw,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if epoch % LOG_EVERY == 0 or epoch == 1:
            print(
                f"  final | epoch {epoch:3d} | train_loss={loss.item():.4f} | "
                f"cls_loss={cls_loss.item():.4f} | gate_entropy={gate_entropy.item():.4f}"
            )

    return model, gnn_scaler, llm_scaler


def main(
    data_path: str = DATA_PATH,
    gnn_emb_path: str = GNN_EMB_PATH,
    llm_emb_path: str = LLM_EMB_PATH,
    splits_path: str = SPLITS_PATH,
    output_dir: str = OUTPUT_DIR,
    dataset: str = "ton_iot",
    gate_entropy_lambda: float = GATE_ENTROPY_LAMBDA,
    feature_attention_entropy_lambda: float = FEATURE_ATTENTION_ENTROPY_LAMBDA,
    use_head_logits: bool = False,
    head_logits_path: str | None = None,
    fusion_mode: str = FUSION_MODE,
) -> None:
    global USE_HEAD_LOGITS, EVAL_CLASSES, FUSION_MODE
    USE_HEAD_LOGITS = use_head_logits
    FUSION_MODE = fusion_mode
    EVAL_CLASSES = get_dataset_config(dataset).eval_classes
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading graph data from {data_path}")
    data = torch.load(data_path, weights_only=False)
    labels = data.edge_label
    config = get_dataset_config(dataset)
    inv_mapping = {v: k for k, v in data.label_mapping.items()}
    label_names = [inv_mapping.get(i, config.label_names[i]) for i in range(NUM_CLASSES)]
    n_edges = labels.shape[0]

    print(f"Loading GNN embeddings from {gnn_emb_path}")
    gnn_emb = torch.load(gnn_emb_path, weights_only=True)
    print(f"Loading LLM embeddings from {llm_emb_path}")
    llm_emb = torch.load(llm_emb_path, weights_only=True)

    assert gnn_emb.shape == (n_edges, config.gnn_dim), (
        f"GNN embeddings shape mismatch: {tuple(gnn_emb.shape)} vs ({n_edges}, {config.gnn_dim})"
    )
    assert llm_emb.shape == (n_edges, config.llm_dim), (
        f"LLM embeddings shape mismatch: {tuple(llm_emb.shape)} vs ({n_edges}, {config.llm_dim})"
    )

    splits_file = Path(splits_path)
    if splits_file.exists():
        print(f"Loading splits from {splits_path}")
        folds = torch.load(splits_path, weights_only=False)
    else:
        print(f"Splits not found — creating new splits at {splits_path}")
        folds = create_edge_splits(data, output_path=splits_path)
    split_size = folds[0]["train_mask"].shape[0]
    if split_size != n_edges:
        raise ValueError(
            f"Stale splits detected: splits have {split_size} edges but "
            f"graph has {n_edges}. Delete {splits_path} and re-run to regenerate."
        )

    head_all: torch.Tensor | None = None
    if use_head_logits:
        hp = head_logits_path or f"data/{dataset}/processed/step4_feedback/llm_head_logits.pt"
        print(f"[M3] Loading trained LLM head logits from {hp}")
        head_all = torch.load(hp, weights_only=True).float()
        assert head_all.shape == (len(folds), n_edges, NUM_CLASSES), (
            f"head logits shape {tuple(head_all.shape)} != "
            f"({len(folds)}, {n_edges}, {NUM_CLASSES})"
        )
        print("[M3] AGAF semantic modality = LLM head logits + gated output fusion")

    fold_results: list[dict[str, object]] = []
    targets = labels.cpu().numpy()
    pooled_preds = np.full(n_edges, fill_value=-1, dtype=np.int64)
    pooled_gate = np.full((n_edges, PROJ_DIM), fill_value=np.nan, dtype=np.float32)
    # Attention width is mode-dependent: PROJ_DIM per-feature weights in "gate"
    # mode, but a single interpretable [structural, semantic] pair in "concat".
    attention_dim = PROJ_DIM if FUSION_MODE == "gate" else 2
    pooled_feature_attention = np.full(
        (n_edges, attention_dim), fill_value=np.nan, dtype=np.float32
    )
    for fold_idx, fold in enumerate(folds):
        print(f"\n=== Fold {fold_idx} ===")
        result = _train_one_fold(
            gnn_emb,
            llm_emb,
            labels,
            fold,
            fold_idx,
            gate_entropy_lambda,
            feature_attention_entropy_lambda,
            head_all=head_all,
        )
        fold_results.append(result)
        test_indices = np.array(result["test_indices"], dtype=np.int64)
        pooled_preds[test_indices] = np.array(result["test_predictions"], dtype=np.int64)
        pooled_gate[test_indices] = np.array(result["test_gate"], dtype=np.float32)
        pooled_feature_attention[test_indices] = np.array(
            result["test_feature_attention"], dtype=np.float32
        )

    if (
        (pooled_preds < 0).any()
        or np.isnan(pooled_gate).any()
        or np.isnan(pooled_feature_attention).any()
    ):
        raise RuntimeError("Some edges were not covered by out-of-fold predictions.")

    val_f1s = np.array([r["best_val_macro_f1"] for r in fold_results])
    test_f1s = np.array([r["test_macro_f1"] for r in fold_results])
    pooled_metrics = classification_metrics(pooled_preds, targets, label_names)
    pooled_diagnostics_summary = fusion_diagnostics_summary(
        pooled_gate, pooled_feature_attention, targets, label_names
    )
    print("\n=== Cross-validation summary ===")
    print(f"val_macro_f1:  {val_f1s.mean():.4f} ± {val_f1s.std():.4f}")
    print(f"test_macro_f1: {test_f1s.mean():.4f} ± {test_f1s.std():.4f}")
    print(f"pooled_test_macro_f1: {pooled_metrics['overall_macro_f1']:.4f}")
    print_class_report(pooled_preds, targets, label_names)
    print("\n=== Pooled out-of-fold AGAF diagnostics ===")
    print_attention_summary(pooled_diagnostics_summary)

    history_payload = {
        "folds": fold_results,
        "val_macro_f1_mean": float(val_f1s.mean()),
        "val_macro_f1_std": float(val_f1s.std()),
        "test_macro_f1_mean": float(test_f1s.mean()),
        "test_macro_f1_std": float(test_f1s.std()),
        "gate_entropy_lambda": gate_entropy_lambda,
        "feature_attention_entropy_lambda": feature_attention_entropy_lambda,
    }
    with open(output_path / "training_history.json", "w") as f:
        json.dump(history_payload, f, indent=2)

    print("\n=== Final model ===")
    final_model, gnn_scaler, llm_scaler = _train_final_model(
        gnn_emb,
        llm_emb,
        labels,
        gate_entropy_lambda,
        feature_attention_entropy_lambda,
        head_all=head_all,
    )

    torch.save(final_model.state_dict(), output_path / "model.pt")
    torch.save({"gnn": gnn_scaler, "llm": llm_scaler}, output_path / "scalers.pt")

    final_model.eval()
    final_head_raw = head_all.mean(dim=0) if head_all is not None else None
    final_llm_emb = final_head_raw if head_all is not None else llm_emb
    gnn_t, llm_t = _apply_scalers(gnn_emb, final_llm_emb, gnn_scaler, llm_scaler)
    with torch.no_grad():
        logits, edge_emb, diagnostics = final_model(gnn_t, llm_t, head_logits=final_head_raw)
        preds = logits.argmax(dim=1).cpu().numpy()

    torch.save(edge_emb.detach().cpu(), output_path / "edge_embeddings.pt")
    torch.save(diagnostics["gate"].detach().cpu(), output_path / "gate_weights.pt")
    torch.save(
        diagnostics["feature_attention"].detach().cpu(),
        output_path / "feature_attention_weights.pt",
    )

    gate_np = diagnostics["gate"].detach().cpu().numpy()
    feature_attention_np = diagnostics["feature_attention"].detach().cpu().numpy()

    final_model_train_metrics = classification_metrics(preds, targets, label_names)
    final_model_train_diagnostics_summary = fusion_diagnostics_summary(
        gate_np, feature_attention_np, targets, label_names
    )

    print("\nFinal model fit-on-all-edges results (not benchmark performance):")
    print(f"  Accuracy:    {final_model_train_metrics['overall_accuracy']:.4f}")
    print(f"  Macro-F1:    {final_model_train_metrics['overall_macro_f1']:.4f}")
    print(f"  Weighted-F1: {final_model_train_metrics['overall_weighted_f1']:.4f}")

    print("\n=== Final model AGAF diagnostics (fit-on-all-edges artifact) ===")
    print_attention_summary(final_model_train_diagnostics_summary)

    metrics_payload = {
        **pooled_metrics,
        "metric_source": "pooled_out_of_fold_test_predictions",
        "fusion_diagnostics_summary": pooled_diagnostics_summary,
        "predictions": pooled_preds.tolist(),
        "gate_weights": pooled_gate.tolist(),
        "feature_attention_weights": pooled_feature_attention.tolist(),
        "final_model_train_metrics": final_model_train_metrics,
        "final_model_train_diagnostics_summary": final_model_train_diagnostics_summary,
    }
    with open(output_path / "metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    write_benchmark_summary(
        output_path,
        dataset=dataset,
        model_name="agaf_fusion",
        history_payload=history_payload,
        pooled_metrics=pooled_metrics,
        artifact_paths={
            "model": str(output_path / "model.pt"),
            "scalers": str(output_path / "scalers.pt"),
            "edge_embeddings": str(output_path / "edge_embeddings.pt"),
            "gate_weights": str(output_path / "gate_weights.pt"),
            "feature_attention_weights": str(output_path / "feature_attention_weights.pt"),
            "metrics": str(output_path / "metrics.json"),
        },
    )

    print(f"\nSaved model, scalers, embeddings, AGAF diagnostics, and metrics to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the GNN+LLM fusion classifier.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="ton_iot")
    parser.add_argument("--data-path")
    parser.add_argument("--gnn-emb-path")
    parser.add_argument("--llm-emb-path")
    parser.add_argument("--splits-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--gate-entropy-lambda", type=float, default=GATE_ENTROPY_LAMBDA)
    parser.add_argument(
        "--fusion-mode", choices=("gate", "concat"), default=FUSION_MODE,
        help="concat follows todo.md Step 3: concatenate both modalities and let "
             "per-sample attention weight them, instead of averaging via a gate.",
    )
    parser.add_argument(
        "--feature-attention-entropy-lambda",
        type=float,
        default=FEATURE_ATTENTION_ENTROPY_LAMBDA,
    )
    parser.add_argument(
        "--use-head-logits",
        action="store_true",
        help="M3: feed AGAF the trained LLM head's OOF logits as the semantic "
        "modality + gated output fusion (instead of raw 768-d embeddings).",
    )
    parser.add_argument("--head-logits-path")
    args = parser.parse_args()

    config = get_dataset_config(args.dataset)
    main(
        data_path=args.data_path or config.graph_path,
        gnn_emb_path=args.gnn_emb_path or config.gnn_embedding_path,
        llm_emb_path=args.llm_emb_path or config.llm_embedding_path,
        splits_path=args.splits_path or config.splits_path,
        output_dir=args.output_dir or config.fusion_output_dir,
        dataset=args.dataset,
        gate_entropy_lambda=args.gate_entropy_lambda,
        feature_attention_entropy_lambda=args.feature_attention_entropy_lambda,
        use_head_logits=args.use_head_logits,
        head_logits_path=args.head_logits_path,
        fusion_mode=args.fusion_mode,
    )
