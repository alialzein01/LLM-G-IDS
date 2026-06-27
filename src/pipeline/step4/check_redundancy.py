from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.models.feedback_classifier import SemanticFeedbackScorer
from src.pipeline.common.datasets import DATASETS, DatasetConfig, get_dataset_config
from src.pipeline.common.splits import NUM_CLASSES


REDUNDANCY_WARNING_THRESHOLD = 0.60


def _load_graph_labels(config: DatasetConfig) -> torch.Tensor:
    data = torch.load(config.graph_path, weights_only=False)
    if not hasattr(data, "edge_label"):
        raise ValueError(f"Graph at {config.graph_path} does not contain edge_label.")
    return data.edge_label.long()


def _load_gate_weights(config: DatasetConfig, n_edges: int) -> torch.Tensor:
    gate_path = Path(config.fusion_output_dir) / "gate_weights.pt"
    if not gate_path.exists():
        raise FileNotFoundError(f"Missing AGAF gate weights: {gate_path}")
    gate = torch.load(gate_path, weights_only=True).float()
    if gate.size(0) != n_edges:
        raise ValueError(
            f"Gate edge count mismatch: {gate.size(0)} vs expected {n_edges}"
        )
    if gate.ndim == 1:
        return gate
    return gate.mean(dim=1)


def _load_prototypes(config: DatasetConfig) -> dict[str, Any]:
    prototype_path = Path(config.prototypes_path)
    if not prototype_path.exists():
        raise FileNotFoundError(f"Missing prototype artifact: {prototype_path}")
    return torch.load(prototype_path, weights_only=False)


def _load_llm_embeddings(config: DatasetConfig, n_edges: int) -> torch.Tensor:
    llm_emb = torch.load(config.llm_embedding_path, weights_only=True).float()
    expected_shape = (n_edges, config.llm_dim)
    if tuple(llm_emb.shape) != expected_shape:
        raise ValueError(
            f"LLM embedding shape mismatch: {tuple(llm_emb.shape)} vs {expected_shape}"
        )
    return llm_emb


def _load_folds(config: DatasetConfig) -> list[dict[str, torch.Tensor]]:
    splits_path = Path(config.splits_path)
    if not splits_path.exists():
        raise FileNotFoundError(f"Missing split file: {splits_path}")
    return torch.load(splits_path, weights_only=False)


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float | None:
    if x.numel() < 2:
        return None
    x_centered = x.float() - x.float().mean()
    y_centered = y.float() - y.float().mean()
    denom = torch.linalg.vector_norm(x_centered) * torch.linalg.vector_norm(y_centered)
    if float(denom.item()) == 0.0:
        return None
    return float((x_centered @ y_centered / denom).item())


def _prototype_confidence(
    llm_emb_subset: torch.Tensor,
    prototype_payload: dict[str, Any],
    temperature: float,
    pooling: str,
) -> torch.Tensor:
    scorer = SemanticFeedbackScorer(
        temperature=temperature,
        num_classes=NUM_CLASSES,
        pooling=pooling,
    )
    with torch.no_grad():
        soft_votes = scorer(llm_emb_subset, prototype_payload)
    return soft_votes.max(dim=1).values


def _classwise_correlations(
    labels: torch.Tensor,
    signal: torch.Tensor,
    gate: torch.Tensor,
) -> list[dict[str, float | int | None]]:
    results = []
    for class_idx in range(NUM_CLASSES):
        class_mask = labels == class_idx
        support = int(class_mask.sum().item())
        corr = _pearson(signal[class_mask], gate[class_mask]) if support >= 2 else None
        results.append(
            {
                "class": class_idx,
                "support": support,
                "pearson": corr,
            }
        )
    return results


def check_redundancy(config: DatasetConfig) -> dict[str, Any]:
    labels = _load_graph_labels(config)
    n_edges = labels.numel()
    llm_emb = _load_llm_embeddings(config, n_edges)
    gate_scalar = _load_gate_weights(config, n_edges)
    folds = _load_folds(config)
    prototype_artifact = _load_prototypes(config)

    temperature = float(
        prototype_artifact.get("prototype_temperature", 10.0)
    )
    pooling = str(prototype_artifact.get("pooling", "max"))

    pooled_signal = torch.full((n_edges,), float("nan"))
    pooled_gate = torch.full((n_edges,), float("nan"))
    pooled_labels = torch.full((n_edges,), -1, dtype=torch.long)
    fold_results = []

    for fold_idx, fold in enumerate(folds):
        test_mask = fold["test_mask"]
        fold_payload = prototype_artifact["folds"][fold_idx]
        signal = _prototype_confidence(
            llm_emb[test_mask],
            fold_payload,
            temperature=temperature,
            pooling=pooling,
        )
        gate = gate_scalar[test_mask]
        test_labels = labels[test_mask]
        corr = _pearson(signal, gate)

        pooled_signal[test_mask] = signal
        pooled_gate[test_mask] = gate
        pooled_labels[test_mask] = test_labels

        fold_results.append(
            {
                "fold": fold_idx,
                "test_edges": int(test_mask.sum().item()),
                "pearson": corr,
                "prototype_confidence_mean": float(signal.mean().item()),
                "prototype_confidence_std": float(signal.std(unbiased=False).item()),
                "gate_mean": float(gate.mean().item()),
                "gate_std": float(gate.std(unbiased=False).item()),
            }
        )

    covered = torch.isfinite(pooled_signal) & torch.isfinite(pooled_gate)
    if not bool(covered.all()):
        missing = int((~covered).sum().item())
        raise RuntimeError(f"Some edges were not covered by test folds: {missing}")

    pooled_corr = _pearson(pooled_signal, pooled_gate)
    redundant = (
        pooled_corr is not None
        and abs(pooled_corr) > REDUNDANCY_WARNING_THRESHOLD
    )

    return {
        "dataset": config.key,
        "prototype_path": config.prototypes_path,
        "gate_path": str(Path(config.fusion_output_dir) / "gate_weights.pt"),
        "prototype_mode": prototype_artifact.get("prototype_mode", "single"),
        "k_per_class": int(
            prototype_artifact.get("k_per_class", 1)
            if not isinstance(prototype_artifact.get("k_per_class", 1), torch.Tensor)
            else prototype_artifact.get("k_per_class").item()
        ),
        "prototype_temperature": temperature,
        "pooling": pooling,
        "redundancy_warning_threshold": REDUNDANCY_WARNING_THRESHOLD,
        "pooled_pearson": pooled_corr,
        "is_redundant": redundant,
        "folds": fold_results,
        "classwise": _classwise_correlations(
            pooled_labels, pooled_signal, pooled_gate
        ),
    }


def main(dataset: str = "ton_iot") -> dict[str, Any]:
    config = get_dataset_config(dataset)
    result = check_redundancy(config)

    output_dir = Path(config.feedback_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "redundancy_check.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Dataset: {result['dataset']}")
    print(f"Prototype mode: {result['prototype_mode']}")
    print(f"Pooled Pearson correlation: {result['pooled_pearson']:.4f}")
    print(f"Redundant: {result['is_redundant']}")
    print(f"Saved redundancy check to {output_path}")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check prototype feedback redundancy against AGAF gates."
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="ton_iot")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(dataset=args.dataset)
