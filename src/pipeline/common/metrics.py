from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support


def classification_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    label_names: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    num_classes = len(label_names)
    labels = list(range(num_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, preds, labels=labels, zero_division=0
    )
    return {
        "overall_accuracy": float((preds == targets).mean()),
        "overall_macro_f1": float(
            f1_score(targets, preds, average="macro", labels=labels, zero_division=0)
        ),
        "overall_weighted_f1": float(
            f1_score(targets, preds, average="weighted", labels=labels, zero_division=0)
        ),
        "per_class": [
            {
                "class_id": cls,
                "class_name": label_names[cls],
                "precision": float(precision[cls]),
                "recall": float(recall[cls]),
                "f1": float(f1[cls]),
                "support": int(support[cls]),
            }
            for cls in labels
        ],
    }


def attention_summary(
    attn: np.ndarray,
    targets: np.ndarray,
    label_names: list[str] | tuple[str, ...],
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for cls, class_name in enumerate(label_names):
        cls_mask = targets == cls
        support = int(cls_mask.sum())
        if support == 0:
            summary.append(
                {
                    "class_id": cls,
                    "class_name": class_name,
                    "support": 0,
                    "attn_gnn_mean": float("nan"),
                    "attn_llm_mean": float("nan"),
                }
            )
            continue
        summary.append(
            {
                "class_id": cls,
                "class_name": class_name,
                "support": support,
                "attn_gnn_mean": float(attn[cls_mask, 0].mean()),
                "attn_llm_mean": float(attn[cls_mask, 1].mean()),
            }
        )
    return summary


def print_class_report(
    preds: np.ndarray,
    targets: np.ndarray,
    label_names: list[str] | tuple[str, ...],
) -> None:
    metrics = classification_metrics(preds, targets, label_names)
    print(f"\n  {'Class':<16} {'Support':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print(f"  {'-'*56}")
    rows = sorted(metrics["per_class"], key=lambda row: -int(row["support"]))
    for row in rows:
        print(
            f"  {row['class_name']:<16} {int(row['support']):>8d} "
            f"{float(row['precision']):>10.4f} {float(row['recall']):>8.4f} "
            f"{float(row['f1']):>8.4f}"
        )


def print_attention_summary(rows: list[dict[str, Any]]) -> None:
    print(f"\n  {'Class':<16} {'Support':>8} {'attn(GNN)':>10} {'attn(LLM)':>10}")
    print(f"  {'-'*48}")
    for row in sorted(rows, key=lambda r: -int(r["support"])):
        if int(row["support"]) == 0:
            continue
        print(
            f"  {row['class_name']:<16} {int(row['support']):>8d} "
            f"{float(row['attn_gnn_mean']):>10.4f} "
            f"{float(row['attn_llm_mean']):>10.4f}"
        )


def write_benchmark_summary(
    output_dir: str | Path,
    *,
    dataset: str,
    model_name: str,
    history_payload: dict[str, Any],
    pooled_metrics: dict[str, Any],
    artifact_paths: dict[str, str],
) -> dict[str, Any]:
    summary = {
        "dataset": dataset,
        "model_name": model_name,
        "cv_val_macro_f1_mean": history_payload["val_macro_f1_mean"],
        "cv_val_macro_f1_std": history_payload["val_macro_f1_std"],
        "cv_test_macro_f1_mean": history_payload["test_macro_f1_mean"],
        "cv_test_macro_f1_std": history_payload["test_macro_f1_std"],
        "pooled_cv_accuracy": pooled_metrics["overall_accuracy"],
        "pooled_cv_macro_f1": pooled_metrics["overall_macro_f1"],
        "pooled_cv_weighted_f1": pooled_metrics["overall_weighted_f1"],
        "per_class_support": [
            {
                "class_id": row["class_id"],
                "class_name": row["class_name"],
                "support": row["support"],
            }
            for row in pooled_metrics["per_class"]
        ],
        "artifact_paths": artifact_paths,
    }
    out = Path(output_dir)
    with open(out / "benchmark_summary.json", "w") as f:
        import json

        json.dump(summary, f, indent=2)
    return summary
