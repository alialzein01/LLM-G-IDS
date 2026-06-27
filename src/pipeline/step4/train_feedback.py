from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.models.feedback_classifier import FeedbackFusionClassifier
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
STRONG_HIDDEN_DIM = 128
FEEDBACK_HIDDEN_DIM = 64
FEEDBACK_DIM = 8
DROPOUT = 0.2

LR_STAGE_A = 1e-3
LR_STAGE_B = 1e-4
STRONG_LR = 1e-3
WEIGHT_DECAY = 5e-4
MAX_EPOCHS = 300
STRONG_MAX_EPOCHS = 300
FINAL_MAX_EPOCHS = 300
EARLY_STOPPING_PATIENCE = 40
GRAD_CLIP = 1.0
LOG_EVERY = 20
SEED = 42
FOCAL_GAMMA = 2.0
MAX_ITERATIONS = 3
DAMPING = 0.5
LAMBDA_SPARSE = 1e-3
UNCERTAIN_FRACTION = 0.30
ALPHA_INIT = 0.10
ALPHA_WARMUP_EPOCHS = 10
ALPHA_WARMUP_VALUE = 1.0
REENTRY_ADVANTAGE_LAMBDA = 0.10
REENTRY_ADVANTAGE_MARGIN = 0.05
ERROR_FOCUS_WRONG_BONUS = 2.0
ERROR_FOCUS_LOW_CONF_BONUS = 1.0
ERROR_FOCUS_UNCERTAIN_BONUS = 1.0
ERROR_FOCUS_ENTROPY_BONUS = 0.5
ALLOW_INITIAL_STAGE_BEST = False


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _macro_f1(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
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


def _fit_scaler(emb: torch.Tensor, train_mask: torch.Tensor) -> StandardScaler:
    return StandardScaler().fit(emb.numpy()[train_mask.numpy()])


def _apply_scaler(emb: torch.Tensor, scaler: StandardScaler) -> torch.Tensor:
    return torch.from_numpy(scaler.transform(emb.numpy())).float()


def _strong_embedding(config: DatasetConfig) -> tuple[torch.Tensor, int, str]:
    if config.strong_modality == "gnn":
        path = config.gnn_embedding_path
        in_dim = config.gnn_dim
    elif config.strong_modality == "llm":
        path = config.llm_embedding_path
        in_dim = config.llm_dim
    else:
        raise ValueError(
            f"Unsupported strong modality for {config.key}: {config.strong_modality}"
        )
    print(f"Loading strong {config.strong_modality.upper()} embeddings from {path}")
    return torch.load(path, weights_only=True).float(), in_dim, path


def _build_strong_head(in_dim: int) -> UnimodalEdgeClassifier:
    return UnimodalEdgeClassifier(
        in_dim=in_dim,
        proj_dim=PROJ_DIM,
        hidden_dim=STRONG_HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    )


def _train_strong_head(
    emb: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor | None,
    in_dim: int,
    *,
    seed: int,
    max_epochs: int,
    patience: int,
    log_prefix: str,
) -> tuple[UnimodalEdgeClassifier, StandardScaler, torch.Tensor, dict[str, Any]]:
    _set_seed(seed)
    scaler = _fit_scaler(emb, train_mask)
    emb_t = _apply_scaler(emb, scaler)

    model = _build_strong_head(in_dim)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=STRONG_LR, weight_decay=WEIGHT_DECAY
    )
    criterion = FocalLoss(alpha=get_class_weights(labels, train_mask), gamma=FOCAL_GAMMA)

    history: list[dict[str, float]] = []
    best_score = -1.0
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(emb_t)
        loss = criterion(logits[train_mask], labels[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_logits, _ = model(emb_t)
            score_mask = val_mask if val_mask is not None else train_mask
            score = _macro_f1(eval_logits, labels, score_mask)

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(loss.item()),
                "score_macro_f1": score,
            }
        )

        if epoch % LOG_EVERY == 0 or epoch == 1:
            print(
                f"  {log_prefix} strong | epoch {epoch:3d} | "
                f"loss={loss.item():.4f} | score_macro_f1={score:.4f}"
            )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if val_mask is not None and epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    with torch.no_grad():
        strong_logits, _ = model(emb_t)

    return (
        model,
        scaler,
        strong_logits.detach(),
        {
            "history": history,
            "best_epoch": best_epoch,
            "best_score_macro_f1": best_score,
        },
    )


def _build_feedback_model(
    config: DatasetConfig,
    uncertain_fraction: float,
    prototype_temperature: float,
    initial_alpha: float,
) -> FeedbackFusionClassifier:
    return FeedbackFusionClassifier(
        hidden_dim=FEEDBACK_HIDDEN_DIM,
        feedback_dim=FEEDBACK_DIM,
        llm_dim=config.llm_dim,
        num_classes=NUM_CLASSES,
        uncertain_fraction=uncertain_fraction,
        prototype_temperature=prototype_temperature,
        initial_alpha=initial_alpha,
    )


def _weighted_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    focal_weight = class_weights.to(logits.device)[targets] * (1.0 - pt) ** FOCAL_GAMMA
    loss = focal_weight * ce
    if sample_weights is not None:
        weights = sample_weights.to(dtype=loss.dtype, device=loss.device)
        weights = weights / weights.mean().clamp_min(1e-6)
        loss = loss * weights
    return loss.mean()


def _hard_case_weights(
    strong_logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    entropy: torch.Tensor | None,
    uncertain_mask: torch.Tensor | None,
) -> torch.Tensor:
    weights = torch.ones(int(mask.sum().item()), dtype=strong_logits.dtype)
    with torch.no_grad():
        masked_strong = strong_logits[mask]
        masked_labels = labels[mask]
        probs = masked_strong.softmax(dim=1)
        confidence, pred = probs.max(dim=1)
        weights = weights.to(masked_strong.device)
        weights = weights + ERROR_FOCUS_WRONG_BONUS * (pred != masked_labels).to(weights.dtype)
        weights = weights + ERROR_FOCUS_LOW_CONF_BONUS * (1.0 - confidence)
        if uncertain_mask is not None:
            uncertain_weight = uncertain_mask[mask].to(weights.dtype)
            weights = weights + ERROR_FOCUS_UNCERTAIN_BONUS * uncertain_weight
        if entropy is not None:
            weights = weights + ERROR_FOCUS_ENTROPY_BONUS * entropy[mask].to(weights.dtype)
    return weights


def _target_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    target_logits = logits.gather(1, labels.view(-1, 1)).flatten()
    other_logits = logits.masked_fill(
        F.one_hot(labels, num_classes=logits.size(1)).bool(),
        -torch.inf,
    )
    return target_logits - other_logits.max(dim=1).values


def _reentry_advantage_loss(
    logits: torch.Tensor,
    no_reentry_logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    on_margin = _target_margin(logits[mask], labels[mask])
    off_margin = _target_margin(no_reentry_logits[mask].detach(), labels[mask])
    return F.relu(off_margin + margin - on_margin).mean()


def _feedback_loss(
    model: FeedbackFusionClassifier,
    logits: torch.Tensor,
    correction: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    strong_logits: torch.Tensor,
    entropy: torch.Tensor | None,
    uncertain_mask: torch.Tensor | None,
    class_weights: torch.Tensor,
    lambda_sparse: float,
    no_reentry_logits: torch.Tensor | None,
    reentry_advantage_lambda: float,
    reentry_advantage_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sample_weights = _hard_case_weights(
        strong_logits=strong_logits,
        labels=labels,
        mask=mask,
        entropy=entropy,
        uncertain_mask=uncertain_mask,
    )
    cls_loss = _weighted_focal_loss(
        logits[mask],
        labels[mask],
        class_weights=class_weights,
        sample_weights=sample_weights,
    )
    sparse_loss = model.sparse_correction_loss(correction[mask])
    if no_reentry_logits is None or reentry_advantage_lambda <= 0:
        reentry_loss = logits.new_tensor(0.0)
    else:
        reentry_loss = _reentry_advantage_loss(
            logits,
            no_reentry_logits,
            labels,
            mask,
            margin=reentry_advantage_margin,
        )
    total = (
        cls_loss
        + lambda_sparse * sparse_loss
        + reentry_advantage_lambda * reentry_loss
    )
    return total, cls_loss.detach(), sparse_loss.detach(), reentry_loss.detach()


def _set_alpha_warmup(
    model: FeedbackFusionClassifier,
    epoch: int,
    warmup_epochs: int,
    warmup_value: float,
) -> bool:
    in_warmup = warmup_epochs > 0 and epoch <= warmup_epochs
    model.alpha.requires_grad_(not in_warmup)
    if in_warmup:
        with torch.no_grad():
            model.alpha.fill_(warmup_value)
    return in_warmup


def _train_feedback_stage(
    model: FeedbackFusionClassifier,
    stage_name: str,
    inject_feedback: bool,
    lr: float,
    data: Any,
    llm_emb: torch.Tensor,
    prototypes: dict[str, Any],
    strong_logits: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    *,
    max_epochs: int,
    patience: int,
    lambda_sparse: float,
    reentry_advantage_lambda: float,
    reentry_advantage_margin: float,
    max_iterations: int,
    damping: float,
    selection_mode: str,
    feedback_mode: str,
    alpha_warmup_epochs: int,
    alpha_warmup_value: float,
    allow_initial_best: bool,
    log_prefix: str,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    criterion = FocalLoss(alpha=get_class_weights(labels, train_mask), gamma=FOCAL_GAMMA)
    class_weights = criterion.alpha.detach().clone()

    with torch.no_grad():
        eval_logits, _, eval_trace = model.run_feedback_loop(
            data.x,
            data.edge_index,
            data.edge_attr,
            llm_emb,
            prototypes,
            strong_logits,
            inject_feedback=inject_feedback,
            max_iterations=max_iterations,
            damping=damping,
            selection_mode=selection_mode,
            feedback_mode=feedback_mode,
            return_trace=True,
        )
        initial_val_f1 = _macro_f1(eval_logits, labels, val_mask)
    best_val_f1 = initial_val_f1 if allow_initial_best else -1.0
    best_epoch = 0 if allow_initial_best else -1
    best_state = deepcopy(model.state_dict()) if allow_initial_best else None
    best_trace = eval_trace["iterations"]
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "train_loss": None,
            "classification_loss": None,
            "sparse_loss": None,
            "reentry_advantage_loss": None,
            "val_macro_f1": initial_val_f1,
            "alpha": float(model.alpha.detach().item()),
            "alpha_warmup": False,
            "selection_trace": best_trace,
        }
    ]

    for epoch in range(1, max_epochs + 1):
        model.train()
        alpha_warmup = _set_alpha_warmup(
            model,
            epoch,
            warmup_epochs=alpha_warmup_epochs,
            warmup_value=alpha_warmup_value,
        )
        optimizer.zero_grad()
        logits, correction, train_trace = model.run_feedback_loop(
            data.x,
            data.edge_index,
            data.edge_attr,
            llm_emb,
            prototypes,
            strong_logits,
            inject_feedback=inject_feedback,
            max_iterations=max_iterations,
            damping=damping,
            selection_mode=selection_mode,
            feedback_mode=feedback_mode,
            return_trace=False,
        )
        no_reentry_logits = None
        if inject_feedback and reentry_advantage_lambda > 0:
            with torch.no_grad():
                no_reentry_logits, _, _ = model.run_feedback_loop(
                    data.x,
                    data.edge_index,
                    data.edge_attr,
                    llm_emb,
                    prototypes,
                    strong_logits,
                    inject_feedback=False,
                    max_iterations=max_iterations,
                    damping=damping,
                    selection_mode=selection_mode,
                    feedback_mode=feedback_mode,
                    return_trace=False,
                )
        loss, cls_loss, sparse_loss, reentry_loss = _feedback_loss(
            model,
            logits,
            correction,
            labels,
            train_mask,
            strong_logits,
            train_trace.get("entropy"),
            train_trace.get("uncertain_mask"),
            class_weights,
            lambda_sparse,
            no_reentry_logits,
            reentry_advantage_lambda if inject_feedback else 0.0,
            reentry_advantage_margin,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        if alpha_warmup:
            with torch.no_grad():
                model.alpha.fill_(alpha_warmup_value)

        model.eval()
        with torch.no_grad():
            eval_logits, _, eval_trace = model.run_feedback_loop(
                data.x,
                data.edge_index,
                data.edge_attr,
                llm_emb,
                prototypes,
                strong_logits,
                inject_feedback=inject_feedback,
                max_iterations=max_iterations,
                damping=damping,
                selection_mode=selection_mode,
                feedback_mode=feedback_mode,
                return_trace=True,
            )
            val_f1 = _macro_f1(eval_logits, labels, val_mask)

        row = {
            "epoch": epoch,
            "train_loss": float(loss.item()),
            "classification_loss": float(cls_loss.item()),
            "sparse_loss": float(sparse_loss.item()),
            "reentry_advantage_loss": float(reentry_loss.item()),
            "val_macro_f1": val_f1,
            "alpha": float(model.alpha.detach().item()),
            "alpha_warmup": alpha_warmup,
            "selection_trace": eval_trace["iterations"],
        }
        history.append(row)

        if epoch % LOG_EVERY == 0 or epoch == 1:
            print(
                f"  {log_prefix} {stage_name} | epoch {epoch:3d} | "
                f"loss={loss.item():.4f} | cls={cls_loss.item():.4f} | "
                f"sparse={sparse_loss.item():.4f} | "
                f"reentry={reentry_loss.item():.4f} | "
                f"val_macro_f1={val_f1:.4f} | "
                f"alpha={model.alpha.detach().item():.4f}"
            )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            best_trace = eval_trace["iterations"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"  {log_prefix} {stage_name} | early stopping at epoch {epoch} "
                    f"(best epoch={best_epoch}, best val_macro_f1={best_val_f1:.4f})"
                )
                break

    model.alpha.requires_grad_(True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "stage": stage_name,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "best_selection_trace": best_trace,
    }


def _evaluate_feedback(
    model: FeedbackFusionClassifier,
    data: Any,
    llm_emb: torch.Tensor,
    prototypes: dict[str, Any],
    strong_logits: torch.Tensor,
    *,
    inject_feedback: bool,
    max_iterations: int,
    damping: float,
    selection_mode: str,
    feedback_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    model.eval()
    with torch.no_grad():
        return model.run_feedback_loop(
            data.x,
            data.edge_index,
            data.edge_attr,
            llm_emb,
            prototypes,
            strong_logits,
            inject_feedback=inject_feedback,
            max_iterations=max_iterations,
            damping=damping,
            selection_mode=selection_mode,
            feedback_mode=feedback_mode,
            return_trace=True,
        )


def _train_one_fold(
    config: DatasetConfig,
    data: Any,
    llm_emb: torch.Tensor,
    strong_emb: torch.Tensor,
    strong_dim: int,
    prototype_artifact: dict[str, Any],
    fold: dict[str, torch.Tensor],
    fold_idx: int,
    *,
    stage: str,
    uncertain_fraction: float,
    max_iterations: int,
    lambda_sparse: float,
    reentry_advantage_lambda: float,
    reentry_advantage_margin: float,
    damping: float,
    max_epochs: int,
    strong_max_epochs: int,
    patience: int,
    selection_mode: str,
    feedback_mode: str,
    initial_alpha: float,
    alpha_warmup_epochs: int,
    alpha_warmup_value: float,
    allow_initial_best: bool,
) -> dict[str, Any]:
    _set_seed(SEED + fold_idx)
    labels = data.edge_label.long()
    train_mask = fold["train_mask"]
    val_mask = fold["val_mask"]
    test_mask = fold["test_mask"]
    log_prefix = f"fold {fold_idx}"

    strong_model, strong_scaler, strong_logits, strong_history = _train_strong_head(
        strong_emb,
        labels,
        train_mask,
        val_mask,
        strong_dim,
        seed=SEED + fold_idx,
        max_epochs=strong_max_epochs,
        patience=patience,
        log_prefix=log_prefix,
    )

    prototype_temperature = float(prototype_artifact.get("prototype_temperature", 10.0))
    prototypes = prototype_artifact["folds"][fold_idx]
    model = _build_feedback_model(
        config,
        uncertain_fraction,
        prototype_temperature,
        initial_alpha=initial_alpha,
    )

    stage_results = []
    if stage in {"a", "both"}:
        stage_results.append(
            _train_feedback_stage(
                model,
                "stage_a_no_reentry",
                False,
                LR_STAGE_A,
                data,
                llm_emb,
                prototypes,
                strong_logits,
                labels,
                train_mask,
                val_mask,
                max_epochs=max_epochs,
                patience=patience,
                lambda_sparse=lambda_sparse,
                reentry_advantage_lambda=0.0,
                reentry_advantage_margin=reentry_advantage_margin,
                max_iterations=max_iterations,
                damping=damping,
                selection_mode=selection_mode,
                feedback_mode=feedback_mode,
                alpha_warmup_epochs=alpha_warmup_epochs,
                alpha_warmup_value=alpha_warmup_value,
                allow_initial_best=allow_initial_best,
                log_prefix=log_prefix,
            )
        )
    if stage in {"b", "both"}:
        stage_results.append(
            _train_feedback_stage(
                model,
                "stage_b_feedback_loop",
                True,
                LR_STAGE_B,
                data,
                llm_emb,
                prototypes,
                strong_logits,
                labels,
                train_mask,
                val_mask,
                max_epochs=max_epochs,
                patience=patience,
                lambda_sparse=lambda_sparse,
                reentry_advantage_lambda=reentry_advantage_lambda,
                reentry_advantage_margin=reentry_advantage_margin,
                max_iterations=max_iterations,
                damping=damping,
                selection_mode=selection_mode,
                feedback_mode=feedback_mode,
                alpha_warmup_epochs=alpha_warmup_epochs,
                alpha_warmup_value=alpha_warmup_value,
                allow_initial_best=allow_initial_best,
                log_prefix=log_prefix,
            )
        )

    inject_feedback = stage != "a"
    logits, _, trace = _evaluate_feedback(
        model,
        data,
        llm_emb,
        prototypes,
        strong_logits,
        inject_feedback=inject_feedback,
        max_iterations=max_iterations,
        damping=damping,
        selection_mode=selection_mode,
        feedback_mode=feedback_mode,
    )
    test_f1 = _macro_f1(logits, labels, test_mask)
    test_indices = test_mask.nonzero(as_tuple=True)[0].cpu().numpy()
    test_predictions = logits[test_mask].argmax(dim=1).cpu().numpy()
    feedback = trace["next_feedback"][test_mask].detach().cpu().numpy()

    print(
        f"  fold {fold_idx} | test_macro_f1={test_f1:.4f} | "
        f"stage={stage} | alpha={model.alpha.detach().item():.4f}"
    )

    return {
        "fold": fold_idx,
        "stage_results": stage_results,
        "strong_history": strong_history,
        "test_macro_f1": test_f1,
        "test_indices": test_indices.tolist(),
        "test_predictions": test_predictions.tolist(),
        "test_feedback": feedback.tolist(),
        "test_selection_trace": trace["iterations"],
        "alpha": float(model.alpha.detach().item()),
        "strong_model_state": strong_model.state_dict(),
        "strong_scaler": strong_scaler,
    }


def _load_or_create_folds(data: Any, splits_path: str) -> list[dict[str, torch.Tensor]]:
    splits_file = Path(splits_path)
    if splits_file.exists():
        print(f"Loading splits from {splits_path}")
        folds = torch.load(splits_file, weights_only=False)
    else:
        print(f"Splits not found - creating new splits at {splits_path}")
        folds = create_edge_splits(data, output_path=splits_path)
    return folds


def _label_names(data: Any, config: DatasetConfig) -> list[str]:
    inv_mapping = {v: k for k, v in data.label_mapping.items()}
    return [inv_mapping.get(i, config.label_names[i]) for i in range(NUM_CLASSES)]


def _validate_inputs(
    data: Any,
    llm_emb: torch.Tensor,
    strong_emb: torch.Tensor,
    strong_dim: int,
    folds: list[dict[str, torch.Tensor]],
    config: DatasetConfig,
) -> None:
    n_edges = data.edge_label.shape[0]
    if llm_emb.shape != (n_edges, config.llm_dim):
        raise ValueError(
            f"LLM embeddings shape mismatch: {tuple(llm_emb.shape)} "
            f"vs ({n_edges}, {config.llm_dim})"
        )
    if strong_emb.shape != (n_edges, strong_dim):
        raise ValueError(
            f"Strong embeddings shape mismatch: {tuple(strong_emb.shape)} "
            f"vs ({n_edges}, {strong_dim})"
        )
    if folds[0]["train_mask"].shape[0] != n_edges:
        raise ValueError("Stale splits detected: fold masks do not match edge count.")
    for name, tensor in {"llm_emb": llm_emb, "strong_emb": strong_emb}.items():
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} contains NaN or infinite values.")


def _train_final_model(
    config: DatasetConfig,
    data: Any,
    llm_emb: torch.Tensor,
    strong_emb: torch.Tensor,
    strong_dim: int,
    prototype_artifact: dict[str, Any],
    *,
    stage: str,
    uncertain_fraction: float,
    max_iterations: int,
    lambda_sparse: float,
    reentry_advantage_lambda: float,
    reentry_advantage_margin: float,
    damping: float,
    max_epochs: int,
    strong_max_epochs: int,
    selection_mode: str,
    feedback_mode: str,
    initial_alpha: float,
    alpha_warmup_epochs: int,
    alpha_warmup_value: float,
    allow_initial_best: bool,
) -> tuple[
    FeedbackFusionClassifier,
    UnimodalEdgeClassifier,
    StandardScaler,
    torch.Tensor,
    dict[str, Any],
]:
    labels = data.edge_label.long()
    full_mask = torch.ones(labels.shape[0], dtype=torch.bool)
    strong_model, strong_scaler, strong_logits, strong_history = _train_strong_head(
        strong_emb,
        labels,
        full_mask,
        None,
        strong_dim,
        seed=SEED,
        max_epochs=strong_max_epochs,
        patience=EARLY_STOPPING_PATIENCE,
        log_prefix="final",
    )

    prototype_temperature = float(prototype_artifact.get("prototype_temperature", 10.0))
    model = _build_feedback_model(
        config,
        uncertain_fraction,
        prototype_temperature,
        initial_alpha=initial_alpha,
    )
    prototypes = prototype_artifact["all_edges"]
    stage_results = []
    if stage in {"a", "both"}:
        stage_results.append(
            _train_feedback_stage(
                model,
                "stage_a_no_reentry",
                False,
                LR_STAGE_A,
                data,
                llm_emb,
                prototypes,
                strong_logits,
                labels,
                full_mask,
                full_mask,
                max_epochs=max_epochs,
                patience=max_epochs + 1,
                lambda_sparse=lambda_sparse,
                reentry_advantage_lambda=0.0,
                reentry_advantage_margin=reentry_advantage_margin,
                max_iterations=max_iterations,
                damping=damping,
                selection_mode=selection_mode,
                feedback_mode=feedback_mode,
                alpha_warmup_epochs=alpha_warmup_epochs,
                alpha_warmup_value=alpha_warmup_value,
                allow_initial_best=allow_initial_best,
                log_prefix="final",
            )
        )
    if stage in {"b", "both"}:
        stage_results.append(
            _train_feedback_stage(
                model,
                "stage_b_feedback_loop",
                True,
                LR_STAGE_B,
                data,
                llm_emb,
                prototypes,
                strong_logits,
                labels,
                full_mask,
                full_mask,
                max_epochs=max_epochs,
                patience=max_epochs + 1,
                lambda_sparse=lambda_sparse,
                reentry_advantage_lambda=reentry_advantage_lambda,
                reentry_advantage_margin=reentry_advantage_margin,
                max_iterations=max_iterations,
                damping=damping,
                selection_mode=selection_mode,
                feedback_mode=feedback_mode,
                alpha_warmup_epochs=alpha_warmup_epochs,
                alpha_warmup_value=alpha_warmup_value,
                allow_initial_best=allow_initial_best,
                log_prefix="final",
            )
        )

    inject_feedback = stage != "a"
    logits, _, trace = _evaluate_feedback(
        model,
        data,
        llm_emb,
        prototypes,
        strong_logits,
        inject_feedback=inject_feedback,
        max_iterations=max_iterations,
        damping=damping,
        selection_mode=selection_mode,
        feedback_mode=feedback_mode,
    )
    return (
        model,
        strong_model,
        strong_scaler,
        logits,
        {
            "strong_history": strong_history,
            "stage_results": stage_results,
            "trace": trace,
        },
    )


def main(
    dataset: str = "ton_iot",
    data_path: str | None = None,
    splits_path: str | None = None,
    output_dir: str | None = None,
    prototypes_path: str | None = None,
    stage: str = "both",
    uncertain_fraction: float = UNCERTAIN_FRACTION,
    max_iterations: int = MAX_ITERATIONS,
    lambda_sparse: float = LAMBDA_SPARSE,
    reentry_advantage_lambda: float = REENTRY_ADVANTAGE_LAMBDA,
    reentry_advantage_margin: float = REENTRY_ADVANTAGE_MARGIN,
    damping: float = DAMPING,
    max_epochs: int = MAX_EPOCHS,
    strong_max_epochs: int = STRONG_MAX_EPOCHS,
    final_max_epochs: int = FINAL_MAX_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    selection_mode: str = "entropy",
    feedback_mode: str = "prototype",
    initial_alpha: float = ALPHA_INIT,
    alpha_warmup_epochs: int = ALPHA_WARMUP_EPOCHS,
    alpha_warmup_value: float = ALPHA_WARMUP_VALUE,
    allow_initial_best: bool = ALLOW_INITIAL_STAGE_BEST,
) -> None:
    config = get_dataset_config(dataset)
    output_path = Path(output_dir or config.feedback_output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data_path = data_path or config.graph_path
    splits_path = splits_path or config.splits_path
    prototypes_path = prototypes_path or config.prototypes_path

    print(f"Loading graph data from {data_path}")
    data = torch.load(data_path, weights_only=False)
    labels = data.edge_label.long()
    label_names = _label_names(data, config)
    n_edges = labels.shape[0]

    print(f"Loading LLM embeddings from {config.llm_embedding_path}")
    llm_emb = torch.load(config.llm_embedding_path, weights_only=True).float()
    strong_emb, strong_dim, strong_emb_path = _strong_embedding(config)
    print(f"Loading prototypes from {prototypes_path}")
    prototype_artifact = torch.load(prototypes_path, weights_only=False)
    folds = _load_or_create_folds(data, splits_path)
    _validate_inputs(data, llm_emb, strong_emb, strong_dim, folds, config)

    fold_results: list[dict[str, Any]] = []
    targets = labels.cpu().numpy()
    pooled_preds = np.full(n_edges, fill_value=-1, dtype=np.int64)
    pooled_feedback = np.full((n_edges, FEEDBACK_DIM), fill_value=np.nan, dtype=np.float32)

    for fold_idx, fold in enumerate(folds):
        print(f"\n=== Feedback fold {fold_idx} ===")
        result = _train_one_fold(
            config,
            data,
            llm_emb,
            strong_emb,
            strong_dim,
            prototype_artifact,
            fold,
            fold_idx,
            stage=stage,
            uncertain_fraction=uncertain_fraction,
            max_iterations=max_iterations,
            lambda_sparse=lambda_sparse,
            reentry_advantage_lambda=reentry_advantage_lambda,
            reentry_advantage_margin=reentry_advantage_margin,
            damping=damping,
            max_epochs=max_epochs,
            strong_max_epochs=strong_max_epochs,
            patience=patience,
            selection_mode=selection_mode,
            feedback_mode=feedback_mode,
            initial_alpha=initial_alpha,
            alpha_warmup_epochs=alpha_warmup_epochs,
            alpha_warmup_value=alpha_warmup_value,
            allow_initial_best=allow_initial_best,
        )
        fold_results.append(
            {k: v for k, v in result.items() if k not in {"strong_model_state", "strong_scaler"}}
        )
        test_indices = np.array(result["test_indices"], dtype=np.int64)
        pooled_preds[test_indices] = np.array(result["test_predictions"], dtype=np.int64)
        pooled_feedback[test_indices] = np.array(result["test_feedback"], dtype=np.float32)

    if (pooled_preds < 0).any() or np.isnan(pooled_feedback).any():
        raise RuntimeError("Some edges were not covered by out-of-fold predictions.")

    val_f1s = np.array(
        [fold["stage_results"][-1]["best_val_macro_f1"] for fold in fold_results]
    )
    test_f1s = np.array([fold["test_macro_f1"] for fold in fold_results])
    pooled_metrics = classification_metrics(pooled_preds, targets, label_names)

    print("\n=== Feedback cross-validation summary ===")
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
        "stage": stage,
        "strong_modality": config.strong_modality,
        "uncertain_fraction": uncertain_fraction,
        "max_iterations": max_iterations,
        "lambda_sparse": lambda_sparse,
        "reentry_advantage_lambda": reentry_advantage_lambda,
        "reentry_advantage_margin": reentry_advantage_margin,
        "damping": damping,
        "selection_mode": selection_mode,
        "feedback_mode": feedback_mode,
        "initial_alpha": initial_alpha,
        "alpha_warmup_epochs": alpha_warmup_epochs,
        "alpha_warmup_value": alpha_warmup_value,
        "allow_initial_stage_best": allow_initial_best,
    }
    with open(output_path / "training_history.json", "w") as f:
        json.dump(history_payload, f, indent=2)

    print("\n=== Final feedback model ===")
    final_model, strong_model, strong_scaler, final_logits, final_payload = (
        _train_final_model(
            config,
            data,
            llm_emb,
            strong_emb,
            strong_dim,
            prototype_artifact,
            stage=stage,
            uncertain_fraction=uncertain_fraction,
            max_iterations=max_iterations,
            lambda_sparse=lambda_sparse,
            reentry_advantage_lambda=reentry_advantage_lambda,
            reentry_advantage_margin=reentry_advantage_margin,
            damping=damping,
            max_epochs=final_max_epochs,
            strong_max_epochs=strong_max_epochs,
            selection_mode=selection_mode,
            feedback_mode=feedback_mode,
            initial_alpha=initial_alpha,
            alpha_warmup_epochs=alpha_warmup_epochs,
            alpha_warmup_value=alpha_warmup_value,
            allow_initial_best=allow_initial_best,
        )
    )
    final_preds = final_logits.argmax(dim=1).cpu().numpy()
    final_metrics = classification_metrics(final_preds, targets, label_names)
    final_trace = final_payload["trace"]

    torch.save(final_model.state_dict(), output_path / "model.pt")
    torch.save(strong_model.state_dict(), output_path / "strong_head.pt")
    torch.save(
        {
            "strong": strong_scaler,
            "strong_modality": config.strong_modality,
            "strong_embedding_path": strong_emb_path,
        },
        output_path / "scalers.pt",
    )
    torch.save(final_trace["next_feedback"].detach().cpu(), output_path / "feedback_weights.pt")
    torch.save(final_trace["edge_emb"].detach().cpu(), output_path / "edge_embeddings.pt")

    metrics_payload = {
        **pooled_metrics,
        "metric_source": "pooled_out_of_fold_test_predictions",
        "predictions": pooled_preds.tolist(),
        "feedback_weights": pooled_feedback.tolist(),
        "selection_stats": [fold["test_selection_trace"] for fold in fold_results],
        "final_model_train_metrics": final_metrics,
        "final_model_selection_trace": final_trace["iterations"],
        "stage": stage,
        "strong_modality": config.strong_modality,
        "prototype_mode": prototype_artifact.get("prototype_mode", "single"),
        "selection_mode": selection_mode,
        "feedback_mode": feedback_mode,
    }
    with open(output_path / "metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    write_benchmark_summary(
        output_path,
        dataset=dataset,
        model_name="feedback_loop",
        history_payload=history_payload,
        pooled_metrics=pooled_metrics,
        artifact_paths={
            "model": str(output_path / "model.pt"),
            "strong_head": str(output_path / "strong_head.pt"),
            "scalers": str(output_path / "scalers.pt"),
            "feedback_weights": str(output_path / "feedback_weights.pt"),
            "edge_embeddings": str(output_path / "edge_embeddings.pt"),
            "metrics": str(output_path / "metrics.json"),
            "prototypes": prototypes_path,
        },
    )

    print(f"\nSaved feedback-loop artifacts to {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Phase 2 feedback loop.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="ton_iot")
    parser.add_argument("--data-path")
    parser.add_argument("--splits-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--prototypes-path")
    parser.add_argument("--stage", choices=("a", "b", "both"), default="both")
    parser.add_argument("--uncertain-fraction", type=float, default=UNCERTAIN_FRACTION)
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--lambda-sparse", type=float, default=LAMBDA_SPARSE)
    parser.add_argument(
        "--reentry-advantage-lambda",
        type=float,
        default=REENTRY_ADVANTAGE_LAMBDA,
    )
    parser.add_argument(
        "--reentry-advantage-margin",
        type=float,
        default=REENTRY_ADVANTAGE_MARGIN,
    )
    parser.add_argument("--damping", type=float, default=DAMPING)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--strong-max-epochs", type=int, default=STRONG_MAX_EPOCHS)
    parser.add_argument("--final-max-epochs", type=int, default=FINAL_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument(
        "--selection-mode",
        choices=("entropy", "random"),
        default="entropy",
    )
    parser.add_argument(
        "--feedback-mode",
        choices=("prototype", "random", "zero"),
        default="prototype",
    )
    parser.add_argument("--initial-alpha", type=float, default=ALPHA_INIT)
    parser.add_argument(
        "--alpha-warmup-epochs",
        type=int,
        default=ALPHA_WARMUP_EPOCHS,
    )
    parser.add_argument(
        "--alpha-warmup-value",
        type=float,
        default=ALPHA_WARMUP_VALUE,
    )
    parser.add_argument(
        "--allow-initial-stage-best",
        action="store_true",
        default=ALLOW_INITIAL_STAGE_BEST,
        help="Allow an untrained epoch-0 feedback stage to be restored as best.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        dataset=args.dataset,
        data_path=args.data_path,
        splits_path=args.splits_path,
        output_dir=args.output_dir,
        prototypes_path=args.prototypes_path,
        stage=args.stage,
        uncertain_fraction=args.uncertain_fraction,
        max_iterations=args.max_iterations,
        lambda_sparse=args.lambda_sparse,
        reentry_advantage_lambda=args.reentry_advantage_lambda,
        reentry_advantage_margin=args.reentry_advantage_margin,
        damping=args.damping,
        max_epochs=args.max_epochs,
        strong_max_epochs=args.strong_max_epochs,
        final_max_epochs=args.final_max_epochs,
        patience=args.patience,
        selection_mode=args.selection_mode,
        feedback_mode=args.feedback_mode,
        initial_alpha=args.initial_alpha,
        alpha_warmup_epochs=args.alpha_warmup_epochs,
        alpha_warmup_value=args.alpha_warmup_value,
        allow_initial_best=args.allow_initial_stage_best,
    )
