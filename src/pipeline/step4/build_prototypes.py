from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.models.feedback_classifier import SemanticFeedbackScorer
from src.pipeline.common.datasets import DATASETS, DatasetConfig, get_dataset_config
from src.pipeline.common.splits import NUM_CLASSES, create_edge_splits


PROTOTYPE_TEMPERATURE = 10.0
DEFAULT_PROTOTYPE_MODE = "single"
DEFAULT_K_PER_CLASS = 1
DEFAULT_POOLING = "max"


def _load_graph_labels(config: DatasetConfig) -> torch.Tensor:
    print(f"Loading graph data from {config.graph_path}")
    data = torch.load(config.graph_path, weights_only=False)
    if not hasattr(data, "edge_label"):
        raise ValueError(f"Graph at {config.graph_path} does not contain edge_label.")
    return data.edge_label.long()


def _load_llm_embeddings(config: DatasetConfig, n_edges: int) -> torch.Tensor:
    print(f"Loading LLM embeddings from {config.llm_embedding_path}")
    llm_emb = torch.load(config.llm_embedding_path, weights_only=True).float()
    expected_shape = (n_edges, config.llm_dim)
    if tuple(llm_emb.shape) != expected_shape:
        raise ValueError(
            f"LLM embedding shape mismatch: {tuple(llm_emb.shape)} vs {expected_shape}"
        )
    if not bool(torch.isfinite(llm_emb).all()):
        raise ValueError("LLM embeddings contain NaN or infinite values.")
    return llm_emb


def _load_or_create_folds(config: DatasetConfig, labels: torch.Tensor) -> list[dict[str, torch.Tensor]]:
    splits_path = Path(config.splits_path)
    if splits_path.exists():
        print(f"Loading splits from {config.splits_path}")
        folds = torch.load(splits_path, weights_only=False)
    else:
        print(f"Splits not found. Creating splits at {config.splits_path}")
        data = torch.load(config.graph_path, weights_only=False)
        folds = create_edge_splits(data, output_path=config.splits_path)

    n_edges = labels.numel()
    for fold_idx, fold in enumerate(folds):
        for key in ("train_mask", "val_mask", "test_mask"):
            if key not in fold:
                raise ValueError(f"Fold {fold_idx} is missing {key}.")
            if fold[key].shape != (n_edges,):
                raise ValueError(
                    f"Fold {fold_idx} {key} shape mismatch: "
                    f"{tuple(fold[key].shape)} vs ({n_edges},)"
                )
    return folds


def _class_counts(labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.bincount(labels[mask], minlength=NUM_CLASSES)


def _prototype_dict(
    prototypes: torch.Tensor,
    prototype_classes: torch.Tensor,
    prototype_counts: torch.Tensor,
    prototype_mode: str,
    k_per_class: int,
    pooling: str,
) -> dict[str, torch.Tensor | str]:
    return {
        "prototypes": prototypes.cpu(),
        "prototype_classes": prototype_classes.cpu(),
        "prototype_counts": prototype_counts.cpu(),
        "prototype_mode": prototype_mode,
        "pooling": pooling,
        "k_per_class": torch.tensor(k_per_class, dtype=torch.long),
    }


def _single_prototypes(
    llm_emb: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prototypes = SemanticFeedbackScorer.build_prototypes(
        llm_emb, labels, train_mask, num_classes=NUM_CLASSES
    )
    prototype_classes = torch.arange(NUM_CLASSES, dtype=torch.long)
    prototype_counts = _class_counts(labels, train_mask)
    return prototypes, prototype_classes, prototype_counts


def _multi_prototypes(
    llm_emb: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    k_per_class: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if k_per_class < 1:
        raise ValueError("k_per_class must be >= 1")

    normalized = F.normalize(llm_emb, dim=1)
    train_fallback = F.normalize(normalized[train_mask].mean(dim=0), dim=0)
    prototypes: list[torch.Tensor] = []
    prototype_classes: list[torch.Tensor] = []
    prototype_counts: list[torch.Tensor] = []

    for class_idx in range(NUM_CLASSES):
        class_mask = train_mask & (labels == class_idx)
        class_emb = normalized[class_mask]
        class_count = int(class_emb.size(0))

        if class_count == 0:
            prototypes.append(train_fallback.view(1, -1))
            prototype_classes.append(torch.tensor([class_idx], dtype=torch.long))
            prototype_counts.append(torch.tensor([0], dtype=torch.long))
            continue

        n_clusters = min(k_per_class, class_count)
        if n_clusters == 1:
            centers = F.normalize(class_emb.mean(dim=0, keepdim=True), dim=1)
            counts = torch.tensor([class_count], dtype=torch.long)
        else:
            kmeans = KMeans(
                n_clusters=n_clusters,
                n_init=10,
                random_state=seed + class_idx,
            )
            assignments = kmeans.fit_predict(class_emb.cpu().numpy())
            centers = torch.from_numpy(kmeans.cluster_centers_).to(
                dtype=llm_emb.dtype
            )
            centers = F.normalize(centers, dim=1)
            counts = torch.bincount(
                torch.from_numpy(assignments), minlength=n_clusters
            ).long()

        prototypes.append(centers)
        prototype_classes.append(
            torch.full((n_clusters,), class_idx, dtype=torch.long)
        )
        prototype_counts.append(counts)

    return (
        torch.cat(prototypes, dim=0),
        torch.cat(prototype_classes, dim=0),
        torch.cat(prototype_counts, dim=0),
    )


def _build_prototype_payload(
    llm_emb: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    prototype_mode: str,
    k_per_class: int,
    pooling: str,
    seed: int,
) -> dict[str, torch.Tensor | str]:
    if prototype_mode == "single":
        prototypes, prototype_classes, prototype_counts = _single_prototypes(
            llm_emb, labels, train_mask
        )
    elif prototype_mode == "multi":
        prototypes, prototype_classes, prototype_counts = _multi_prototypes(
            llm_emb, labels, train_mask, k_per_class=k_per_class, seed=seed
        )
    else:
        raise ValueError("prototype_mode must be 'single' or 'multi'")

    return _prototype_dict(
        prototypes=prototypes,
        prototype_classes=prototype_classes,
        prototype_counts=prototype_counts,
        prototype_mode=prototype_mode,
        k_per_class=k_per_class,
        pooling=pooling,
    )


def _prototype_metrics(
    llm_emb: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    prototype_payload: dict[str, torch.Tensor | str],
    temperature: float,
    pooling: str,
) -> dict[str, Any]:
    if not bool(mask.any()):
        return {
            "overall_top1_match": None,
            "per_class": [],
            "confusion_counts": [],
        }
    scorer = SemanticFeedbackScorer(
        temperature=temperature, num_classes=NUM_CLASSES, pooling=pooling
    )
    with torch.no_grad():
        soft_votes = scorer(llm_emb[mask], prototype_payload)
        pred = soft_votes.argmax(dim=1)
        target = labels[mask]
        top1 = float((pred == target).float().mean().item())

        confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
        for true_class, pred_class in zip(target.tolist(), pred.tolist(), strict=False):
            confusion[true_class, pred_class] += 1

        per_class = []
        max_confidence = soft_votes.max(dim=1).values
        true_confidence = soft_votes.gather(1, target.view(-1, 1)).flatten()
        for class_idx in range(NUM_CLASSES):
            class_mask = target == class_idx
            support = int(class_mask.sum().item())
            if support == 0:
                per_class.append(
                    {
                        "class": class_idx,
                        "support": 0,
                        "top1_match": None,
                        "avg_predicted_confidence": None,
                        "avg_true_class_confidence": None,
                    }
                )
                continue
            per_class.append(
                {
                    "class": class_idx,
                    "support": support,
                    "top1_match": float(
                        (pred[class_mask] == target[class_mask]).float().mean().item()
                    ),
                    "avg_predicted_confidence": float(
                        max_confidence[class_mask].mean().item()
                    ),
                    "avg_true_class_confidence": float(
                        true_confidence[class_mask].mean().item()
                    ),
                }
            )

        return {
            "overall_top1_match": top1,
            "per_class": per_class,
            "confusion_counts": confusion.tolist(),
        }


def _fold_payload(
    fold_idx: int,
    fold: dict[str, torch.Tensor],
    llm_emb: torch.Tensor,
    labels: torch.Tensor,
    prototype_mode: str,
    k_per_class: int,
    temperature: float,
    pooling: str,
) -> dict[str, Any]:
    train_mask = fold["train_mask"]
    val_mask = fold["val_mask"]
    test_mask = fold["test_mask"]
    prototype_payload = _build_prototype_payload(
        llm_emb=llm_emb,
        labels=labels,
        train_mask=train_mask,
        prototype_mode=prototype_mode,
        k_per_class=k_per_class,
        pooling=pooling,
        seed=42 + fold_idx * 100,
    )
    prototypes = prototype_payload["prototypes"]
    train_counts = _class_counts(labels, train_mask)
    missing_classes = (train_counts == 0).nonzero(as_tuple=False).flatten()

    if not isinstance(prototypes, torch.Tensor):
        raise TypeError("prototype payload contains a non-tensor prototypes value.")
    if not bool(torch.isfinite(prototypes).all()):
        raise ValueError(f"Fold {fold_idx} prototypes contain NaN or infinite values.")

    train_metrics = _prototype_metrics(
        llm_emb, labels, train_mask, prototype_payload, temperature, pooling
    )
    val_metrics = _prototype_metrics(
        llm_emb, labels, val_mask, prototype_payload, temperature, pooling
    )
    test_metrics = _prototype_metrics(
        llm_emb, labels, test_mask, prototype_payload, temperature, pooling
    )
    diagnostics = {
        "fold": fold_idx,
        "prototype_mode": prototype_mode,
        "k_per_class": k_per_class,
        "num_prototypes": int(prototypes.size(0)),
        "pooling": pooling,
        "temperature": temperature,
        "train_edges": int(train_mask.sum().item()),
        "val_edges": int(val_mask.sum().item()),
        "test_edges": int(test_mask.sum().item()),
        "train_class_counts": train_counts.tolist(),
        "missing_train_classes": missing_classes.tolist(),
        "prototype_counts": prototype_payload["prototype_counts"].tolist(),
        "prototype_classes": prototype_payload["prototype_classes"].tolist(),
        "train_top1_match": train_metrics["overall_top1_match"],
        "val_top1_match": val_metrics["overall_top1_match"],
        "test_top1_match": test_metrics["overall_top1_match"],
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    return {
        "fold": fold_idx,
        **prototype_payload,
        "train_class_counts": train_counts.cpu(),
        "missing_train_classes": missing_classes.cpu(),
        "diagnostics": diagnostics,
    }


def build_prototypes(
    config: DatasetConfig,
    prototype_mode: str = DEFAULT_PROTOTYPE_MODE,
    k_per_class: int = DEFAULT_K_PER_CLASS,
    temperature: float = PROTOTYPE_TEMPERATURE,
    pooling: str = DEFAULT_POOLING,
) -> dict[str, Any]:
    labels = _load_graph_labels(config)
    n_edges = labels.numel()
    llm_emb = _load_llm_embeddings(config, n_edges)
    folds = _load_or_create_folds(config, labels)

    fold_payloads = []
    diagnostics = []
    print("\nBuilding per-fold prototypes:")
    for fold_idx, fold in enumerate(folds):
        payload = _fold_payload(
            fold_idx,
            fold,
            llm_emb,
            labels,
            prototype_mode=prototype_mode,
            k_per_class=k_per_class,
            temperature=temperature,
            pooling=pooling,
        )
        fold_payloads.append(
            {
                "fold": payload["fold"],
                "prototypes": payload["prototypes"],
                "prototype_classes": payload["prototype_classes"],
                "prototype_counts": payload["prototype_counts"],
                "prototype_mode": payload["prototype_mode"],
                "k_per_class": payload["k_per_class"],
                "pooling": payload["pooling"],
                "train_class_counts": payload["train_class_counts"],
                "missing_train_classes": payload["missing_train_classes"],
            }
        )
        diagnostics.append(payload["diagnostics"])
        print(
            f"  fold {fold_idx}: prototypes={tuple(payload['prototypes'].shape)} | "
            f"train_top1_match={payload['diagnostics']['train_top1_match']:.4f} | "
            f"test_top1_match={payload['diagnostics']['test_top1_match']:.4f}"
        )

    full_mask = torch.ones(n_edges, dtype=torch.bool)
    all_edge_payload = _build_prototype_payload(
        llm_emb=llm_emb,
        labels=labels,
        train_mask=full_mask,
        prototype_mode=prototype_mode,
        k_per_class=k_per_class,
        pooling=pooling,
        seed=42,
    )
    all_edge_prototypes = all_edge_payload["prototypes"]
    all_edge_counts = _class_counts(labels, full_mask)
    if not isinstance(all_edge_prototypes, torch.Tensor):
        raise TypeError("all-edge payload contains a non-tensor prototypes value.")
    if not bool(torch.isfinite(all_edge_prototypes).all()):
        raise ValueError("All-edge prototypes contain NaN or infinite values.")

    return {
        "dataset": config.key,
        "display_name": config.display_name,
        "source_embedding_path": config.llm_embedding_path,
        "graph_path": config.graph_path,
        "splits_path": config.splits_path,
        "num_classes": NUM_CLASSES,
        "llm_dim": config.llm_dim,
        "prototype_temperature": temperature,
        "prototype_mode": prototype_mode,
        "k_per_class": k_per_class,
        "pooling": pooling,
        "folds": fold_payloads,
        "all_edges": {
            **all_edge_payload,
            "class_counts": all_edge_counts.cpu(),
        },
        "diagnostics": diagnostics,
    }


def _write_diagnostics(payload: dict[str, Any], output_dir: Path) -> None:
    diagnostics_path = output_dir / "prototype_diagnostics.json"
    serializable = {
        "dataset": payload["dataset"],
        "display_name": payload["display_name"],
        "source_embedding_path": payload["source_embedding_path"],
        "graph_path": payload["graph_path"],
        "splits_path": payload["splits_path"],
        "num_classes": payload["num_classes"],
        "llm_dim": payload["llm_dim"],
        "prototype_temperature": payload["prototype_temperature"],
        "prototype_mode": payload["prototype_mode"],
        "k_per_class": payload["k_per_class"],
        "pooling": payload["pooling"],
        "all_edge_class_counts": payload["all_edges"]["class_counts"].tolist(),
        "all_edge_num_prototypes": int(payload["all_edges"]["prototypes"].size(0)),
        "folds": payload["diagnostics"],
    }
    with open(diagnostics_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved diagnostics to {diagnostics_path}")


def _mean_metric(fold_diagnostics: list[dict[str, Any]], key: str) -> float:
    values = [fold[key] for fold in fold_diagnostics if fold[key] is not None]
    return float(np.mean(values)) if values else float("nan")


def _evaluate_prototype_config(
    llm_emb: torch.Tensor,
    labels: torch.Tensor,
    folds: list[dict[str, torch.Tensor]],
    prototype_mode: str,
    k_per_class: int,
    temperature: float,
    pooling: str,
) -> dict[str, Any]:
    diagnostics = []
    num_prototypes = []
    for fold_idx, fold in enumerate(folds):
        payload = _fold_payload(
            fold_idx,
            fold,
            llm_emb,
            labels,
            prototype_mode=prototype_mode,
            k_per_class=k_per_class,
            temperature=temperature,
            pooling=pooling,
        )
        diagnostics.append(payload["diagnostics"])
        num_prototypes.append(payload["diagnostics"]["num_prototypes"])

    return {
        "prototype_mode": prototype_mode,
        "k_per_class": k_per_class,
        "temperature": temperature,
        "pooling": pooling,
        "num_prototypes_mean": float(np.mean(num_prototypes)),
        "train_top1_mean": _mean_metric(diagnostics, "train_top1_match"),
        "val_top1_mean": _mean_metric(diagnostics, "val_top1_match"),
        "test_top1_mean": _mean_metric(diagnostics, "test_top1_match"),
        "folds": diagnostics,
    }


def run_validation_sweep(
    config: DatasetConfig,
    k_values: list[int],
    temperature_values: list[float],
    pooling: str,
) -> dict[str, Any]:
    labels = _load_graph_labels(config)
    llm_emb = _load_llm_embeddings(config, labels.numel())
    folds = _load_or_create_folds(config, labels)

    results = []
    print("\nValidation sweep:")
    for k_per_class in k_values:
        prototype_mode = "single" if k_per_class == 1 else "multi"
        for temperature in temperature_values:
            result = _evaluate_prototype_config(
                llm_emb=llm_emb,
                labels=labels,
                folds=folds,
                prototype_mode=prototype_mode,
                k_per_class=k_per_class,
                temperature=temperature,
                pooling=pooling,
            )
            results.append(result)
            print(
                f"  k={k_per_class} temp={temperature:g} mode={prototype_mode} | "
                f"val={result['val_top1_mean']:.4f} | "
                f"test={result['test_top1_mean']:.4f}"
            )

    best = max(results, key=lambda item: item["val_top1_mean"])
    return {
        "dataset": config.key,
        "selection_metric": "val_top1_mean",
        "pooling": pooling,
        "k_values": k_values,
        "temperature_values": temperature_values,
        "best": {
            "prototype_mode": best["prototype_mode"],
            "k_per_class": best["k_per_class"],
            "temperature": best["temperature"],
            "pooling": best["pooling"],
            "val_top1_mean": best["val_top1_mean"],
            "test_top1_mean_after_selection": best["test_top1_mean"],
        },
        "results": results,
    }


def _write_sweep(payload: dict[str, Any], output_dir: Path) -> None:
    sweep_path = output_dir / "prototype_validation_sweep.json"
    with open(sweep_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved validation sweep to {sweep_path}")


def _parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _parse_float_list(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def main(
    dataset: str = "ton_iot",
    prototype_mode: str = DEFAULT_PROTOTYPE_MODE,
    k_per_class: int = DEFAULT_K_PER_CLASS,
    temperature: float = PROTOTYPE_TEMPERATURE,
    pooling: str = DEFAULT_POOLING,
    validation_sweep: bool = False,
    sweep_k_values: list[int] | None = None,
    sweep_temperatures: list[float] | None = None,
) -> dict[str, Any]:
    config = get_dataset_config(dataset)
    output_path = Path(config.prototypes_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if validation_sweep:
        sweep_payload = run_validation_sweep(
            config=config,
            k_values=sweep_k_values or [1, 2, 3],
            temperature_values=sweep_temperatures or [5.0, 10.0, 20.0],
            pooling=pooling,
        )
        _write_sweep(sweep_payload, output_path.parent)
        best = sweep_payload["best"]
        prototype_mode = best["prototype_mode"]
        k_per_class = best["k_per_class"]
        temperature = best["temperature"]
        pooling = best["pooling"]
        print(
            "\nSelected by validation: "
            f"mode={prototype_mode}, k={k_per_class}, "
            f"temperature={temperature:g}, pooling={pooling}"
        )

    payload = build_prototypes(
        config,
        prototype_mode=prototype_mode,
        k_per_class=k_per_class,
        temperature=temperature,
        pooling=pooling,
    )
    torch.save(payload, output_path)
    _write_diagnostics(payload, output_path.parent)
    print(f"Saved prototypes to {output_path}")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build CySecBERT class prototypes for Phase 2 feedback."
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="ton_iot")
    parser.add_argument(
        "--prototype-mode",
        choices=("single", "multi"),
        default=DEFAULT_PROTOTYPE_MODE,
    )
    parser.add_argument("--k-per-class", type=int, default=DEFAULT_K_PER_CLASS)
    parser.add_argument("--temperature", type=float, default=PROTOTYPE_TEMPERATURE)
    parser.add_argument("--pooling", choices=("max", "logsumexp"), default=DEFAULT_POOLING)
    parser.add_argument(
        "--validation-sweep",
        action="store_true",
        help="Select k/temperature by mean validation top-1 before saving artifacts.",
    )
    parser.add_argument("--sweep-k-values", default="1,2,3")
    parser.add_argument("--sweep-temperatures", default="5,10,20")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        dataset=args.dataset,
        prototype_mode=args.prototype_mode,
        k_per_class=args.k_per_class,
        temperature=args.temperature,
        pooling=args.pooling,
        validation_sweep=args.validation_sweep,
        sweep_k_values=_parse_int_list(args.sweep_k_values),
        sweep_temperatures=_parse_float_list(args.sweep_temperatures),
    )
