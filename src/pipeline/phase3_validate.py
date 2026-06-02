from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch_geometric.nn import GATv2Conv

from src.models.gnn_classifier import GATEdgeClassifier
from src.pipeline.common.datasets import DATASETS, get_dataset_config
from src.pipeline.common.reports import check, finalize_report


def _load_json(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _history_ok(history: dict) -> bool:
    folds = history.get("folds", [])
    return bool(folds) and all(
        fold.get("best_epoch", -1) > 0 and fold.get("test_macro_f1") is not None
        for fold in folds
    )


def main(dataset: str = "ton_iot") -> dict:
    config = get_dataset_config(dataset)
    data = torch.load(config.graph_path, weights_only=False)
    n_edges = int(data.edge_label.shape[0])

    gnn_model = GATEdgeClassifier(hidden_dim=config.gnn_dim)
    conv_types = {type(module).__name__ for module in gnn_model.modules() if "Conv" in type(module).__name__}
    gnn_emb = torch.load(config.gnn_embedding_path, weights_only=True)
    llm_emb = torch.load(config.llm_embedding_path, weights_only=True)

    gnn_history_path = Path(config.gnn_output_dir) / "training_history.json"
    gnn_metrics_path = Path(config.gnn_output_dir) / "metrics.json"
    gnn_summary_path = Path(config.gnn_output_dir) / "benchmark_summary.json"
    llm_base_dir = Path(config.baseline_output_dir) / "llm_embedding"
    llm_history_path = llm_base_dir / "training_history.json"
    llm_metrics_path = llm_base_dir / "metrics.json"
    llm_summary_path = llm_base_dir / "benchmark_summary.json"

    gnn_history = _load_json(gnn_history_path)
    gnn_metrics = _load_json(gnn_metrics_path)
    gnn_summary = _load_json(gnn_summary_path)
    llm_history = _load_json(llm_history_path)
    llm_metrics = _load_json(llm_metrics_path)
    llm_summary = _load_json(llm_summary_path)
    sentences = Path(config.kg_nl_path).read_text().strip().splitlines()
    leakage_terms = {
        name.lower()
        for name in config.label_names
        if name.lower() not in {"benign", "normal"}
    }
    leaks = [
        (i, term)
        for i, sentence in enumerate(sentences)
        for term in leakage_terms
        if term in sentence.lower()
    ]

    checks = [
        check("GNN model artifact exists", (Path(config.gnn_output_dir) / "model.pt").exists()),
        check("GNN architecture uses only GATv2 graph convolutions", conv_types == {"GATv2Conv"}, str(sorted(conv_types))),
        check("GNN embeddings shape is [E, 64]", tuple(gnn_emb.shape) == (n_edges, config.gnn_dim), str(tuple(gnn_emb.shape))),
        check("GNN embeddings are finite", bool(torch.isfinite(gnn_emb).all())),
        check("GNN training converged to valid fold results", _history_ok(gnn_history)),
        check("GNN benchmark summary exists", gnn_summary_path.exists(), str(gnn_summary_path)),
        check("GNN metrics include pooled predictions", len(gnn_metrics.get("predictions", [])) == n_edges),
        check("CySecBERT embeddings shape is [E, 768]", tuple(llm_emb.shape) == (n_edges, config.llm_dim), str(tuple(llm_emb.shape))),
        check("CySecBERT embeddings are finite", bool(torch.isfinite(llm_emb).all())),
        check("KG sentence count equals edge count", len(sentences) == n_edges, f"sentences={len(sentences)}, edges={n_edges}"),
        check("CySecBERT inputs are label-free", len(leaks) == 0, f"violations={len(leaks)}"),
        check("LLM baseline model artifact exists", (llm_base_dir / "model.pt").exists()),
        check("LLM baseline training converged to valid fold results", _history_ok(llm_history)),
        check("LLM benchmark summary exists", llm_summary_path.exists(), str(llm_summary_path)),
        check("LLM metrics include pooled predictions", len(llm_metrics.get("predictions", [])) == n_edges),
    ]

    metrics = {
        "n_edges": n_edges,
        "gnn": {
            "pooled_cv_accuracy": gnn_summary.get("pooled_cv_accuracy"),
            "pooled_cv_macro_f1": gnn_summary.get("pooled_cv_macro_f1"),
            "pooled_cv_weighted_f1": gnn_summary.get("pooled_cv_weighted_f1"),
            "cv_test_macro_f1_mean": gnn_summary.get("cv_test_macro_f1_mean"),
            "cv_test_macro_f1_std": gnn_summary.get("cv_test_macro_f1_std"),
        },
        "cysecbert": {
            "pooled_cv_accuracy": llm_summary.get("pooled_cv_accuracy"),
            "pooled_cv_macro_f1": llm_summary.get("pooled_cv_macro_f1"),
            "pooled_cv_weighted_f1": llm_summary.get("pooled_cv_weighted_f1"),
            "cv_test_macro_f1_mean": llm_summary.get("cv_test_macro_f1_mean"),
            "cv_test_macro_f1_std": llm_summary.get("cv_test_macro_f1_std"),
        },
    }
    artifacts = {
        "gnn_model": str(Path(config.gnn_output_dir) / "model.pt"),
        "gnn_embeddings": config.gnn_embedding_path,
        "gnn_metrics": str(gnn_metrics_path),
        "llm_embeddings": config.llm_embedding_path,
        "llm_model": str(llm_base_dir / "model.pt"),
        "llm_metrics": str(llm_metrics_path),
    }
    return finalize_report(
        config, "phase3", f"Phase 3 Validation - {config.display_name}", checks, artifacts, metrics
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Phase 3 baseline artifacts.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="ton_iot")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    report = main(args.dataset)
    print(f"Phase 3 {args.dataset}: {report['status']}")
