from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from src.pipeline.common.datasets import DATASETS, get_dataset_config
from src.pipeline.common.reports import check, finalize_report, report_dir, write_json


def _load_json(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _fold_f1s(history: dict) -> np.ndarray:
    return np.array([float(fold["test_macro_f1"]) for fold in history.get("folds", [])], dtype=float)


def _bootstrap_delta(deltas: np.ndarray, seed: int = 42, samples: int = 10000) -> dict:
    if deltas.size == 0:
        return {"mean": None, "ci95_low": None, "ci95_high": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, deltas.size, size=(samples, deltas.size))
    means = deltas[idx].mean(axis=1)
    return {
        "mean": float(deltas.mean()),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
    }


def main(dataset: str = "ton_iot") -> dict:
    config = get_dataset_config(dataset)
    data = torch.load(config.graph_path, weights_only=False)
    n_edges = int(data.edge_label.shape[0])
    targets = data.edge_label.cpu().numpy()

    gnn_summary_path = Path(config.gnn_output_dir) / "benchmark_summary.json"
    llm_summary_path = Path(config.baseline_output_dir) / "llm_embedding" / "benchmark_summary.json"
    fusion_summary_path = Path(config.fusion_output_dir) / "benchmark_summary.json"
    gnn_history_path = Path(config.gnn_output_dir) / "training_history.json"
    llm_history_path = Path(config.baseline_output_dir) / "llm_embedding" / "training_history.json"
    fusion_history_path = Path(config.fusion_output_dir) / "training_history.json"
    fusion_metrics_path = Path(config.fusion_output_dir) / "metrics.json"

    gnn_summary = _load_json(gnn_summary_path)
    llm_summary = _load_json(llm_summary_path)
    fusion_summary = _load_json(fusion_summary_path)
    gnn_history = _load_json(gnn_history_path)
    llm_history = _load_json(llm_history_path)
    fusion_history = _load_json(fusion_history_path)
    fusion_metrics = _load_json(fusion_metrics_path)

    gnn_emb = torch.load(config.gnn_embedding_path, weights_only=True)
    llm_emb = torch.load(config.llm_embedding_path, weights_only=True)
    fused_emb = torch.load(Path(config.fusion_output_dir) / "edge_embeddings.pt", weights_only=True)
    gate = torch.load(Path(config.fusion_output_dir) / "gate_weights.pt", weights_only=True)
    feature_attention = torch.load(
        Path(config.fusion_output_dir) / "feature_attention_weights.pt", weights_only=True
    )
    folds = torch.load(config.splits_path, weights_only=False)

    fusion_macro = float(fusion_summary["pooled_cv_macro_f1"])
    gnn_macro = float(gnn_summary["pooled_cv_macro_f1"])
    llm_macro = float(llm_summary["pooled_cv_macro_f1"])
    fusion_fold = _fold_f1s(fusion_history)
    gnn_fold = _fold_f1s(gnn_history)
    llm_fold = _fold_f1s(llm_history)
    delta_gnn = fusion_fold - gnn_fold if fusion_fold.shape == gnn_fold.shape else np.array([])
    delta_llm = fusion_fold - llm_fold if fusion_fold.shape == llm_fold.shape else np.array([])
    stats = {
        "fusion_minus_gnn": _bootstrap_delta(delta_gnn),
        "fusion_minus_cysecbert": _bootstrap_delta(delta_llm),
    }

    predictions = np.array(fusion_metrics.get("predictions", []), dtype=int)
    cm = confusion_matrix(targets, predictions, labels=list(range(config.num_classes))).tolist() if predictions.size == n_edges else []
    out = report_dir(config, "phase4")
    write_json(out / "confusion_matrix.json", {"labels": list(config.label_names), "matrix": cm})
    write_json(out / "significance_report.json", stats)

    checks = [
        check("fusion model artifact exists", (Path(config.fusion_output_dir) / "model.pt").exists()),
        check("fusion scaler artifact exists", (Path(config.fusion_output_dir) / "scalers.pt").exists()),
        check("GNN embeddings align to edge count", tuple(gnn_emb.shape) == (n_edges, config.gnn_dim), str(tuple(gnn_emb.shape))),
        check("LLM embeddings align to edge count", tuple(llm_emb.shape) == (n_edges, config.llm_dim), str(tuple(llm_emb.shape))),
        check("fused embeddings align to edge count", int(fused_emb.shape[0]) == n_edges, str(tuple(fused_emb.shape))),
        check("gate weights align to edge count", int(gate.shape[0]) == n_edges, str(tuple(gate.shape))),
        check("feature attention weights align to edge count", int(feature_attention.shape[0]) == n_edges, str(tuple(feature_attention.shape))),
        check("all fusion tensors are finite", bool(torch.isfinite(fused_emb).all() and torch.isfinite(gate).all() and torch.isfinite(feature_attention).all())),
        check("split masks align to edge count", all(fold["train_mask"].shape[0] == n_edges for fold in folds)),
        check("fusion predictions cover every edge", predictions.size == n_edges, f"predictions={predictions.size}, edges={n_edges}"),
        check("fusion improves over GATv2 macro-F1", fusion_macro > gnn_macro, f"fusion={fusion_macro:.4f}, gnn={gnn_macro:.4f}"),
        check("fusion improves over CySecBERT macro-F1", fusion_macro > llm_macro, f"fusion={fusion_macro:.4f}, cysecbert={llm_macro:.4f}"),
        check(
            "paired fold delta over GATv2 is positive",
            bool(stats["fusion_minus_gnn"]["mean"] is not None and stats["fusion_minus_gnn"]["mean"] > 0),
            str(stats["fusion_minus_gnn"]),
        ),
        check(
            "paired fold delta over CySecBERT is positive",
            bool(stats["fusion_minus_cysecbert"]["mean"] is not None and stats["fusion_minus_cysecbert"]["mean"] > 0),
            str(stats["fusion_minus_cysecbert"]),
        ),
    ]

    metrics = {
        "fusion_pooled_macro_f1": fusion_macro,
        "gnn_pooled_macro_f1": gnn_macro,
        "cysecbert_pooled_macro_f1": llm_macro,
        "fusion_minus_gnn_macro_f1": fusion_macro - gnn_macro,
        "fusion_minus_cysecbert_macro_f1": fusion_macro - llm_macro,
        "paired_fold_bootstrap": stats,
    }
    artifacts = {
        "fusion_model": str(Path(config.fusion_output_dir) / "model.pt"),
        "fusion_metrics": str(fusion_metrics_path),
        "gate_weights": str(Path(config.fusion_output_dir) / "gate_weights.pt"),
        "feature_attention_weights": str(Path(config.fusion_output_dir) / "feature_attention_weights.pt"),
        "significance_report": str(out / "significance_report.json"),
        "confusion_matrix": str(out / "confusion_matrix.json"),
    }
    notes = []
    if fusion_macro <= llm_macro:
        notes.append(
            "Phase 4 is intentionally not approvable until fusion exceeds the CySecBERT baseline on pooled macro-F1."
        )
    return finalize_report(
        config, "phase4", f"Phase 4 Validation - {config.display_name}", checks, artifacts, metrics, notes
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Phase 4 fusion artifacts.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="ton_iot")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    report = main(args.dataset)
    print(f"Phase 4 {args.dataset}: {report['status']}")
