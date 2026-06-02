from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import pandas as pd
import torch

from src.pipeline.common.datasets import DATASETS, get_dataset_config
from src.pipeline.common.reports import check, finalize_report
from src.pipeline.step1.graph_construction import (
    ATTACK_COL,
    DST_COL,
    EDGE_ATTR_COLUMNS,
    SRC_COL,
)


def _read_input_counts(path: str) -> tuple[int, int, dict[str, int]]:
    df = pd.read_csv(path, usecols=[SRC_COL, DST_COL, ATTACK_COL])
    df[ATTACK_COL] = df[ATTACK_COL].str.strip()
    node_count = int(pd.concat([df[SRC_COL], df[DST_COL]], ignore_index=True).nunique())
    edge_count = int(df.groupby([SRC_COL, DST_COL, ATTACK_COL], sort=False).size().shape[0])
    labels = df[ATTACK_COL].value_counts().sort_index().astype(int).to_dict()
    return node_count, edge_count, labels


def _connectivity(edge_df: pd.DataFrame) -> dict[str, int | float]:
    graph = nx.DiGraph()
    graph.add_edges_from(zip(edge_df[SRC_COL], edge_df[DST_COL]))
    weak_components = list(nx.weakly_connected_components(graph))
    strong_components = list(nx.strongly_connected_components(graph))
    largest_weak = max((len(c) for c in weak_components), default=0)
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "weak_components": len(weak_components),
        "strong_components": len(strong_components),
        "largest_weak_component_nodes": largest_weak,
        "density": float(nx.density(graph)) if graph.number_of_nodes() > 1 else 0.0,
    }


def main(dataset: str = "ton_iot") -> dict:
    config = get_dataset_config(dataset)
    data = torch.load(config.graph_path, weights_only=False)
    edge_df = pd.read_csv(config.aggregated_edges_path)
    expected_nodes, expected_edges, raw_label_distribution = _read_input_counts(
        config.phase1_input_path
    )

    attr = data.edge_attr
    normalized = attr[:, :3]
    label_ids = data.edge_label.cpu().numpy()
    inv_mapping = {v: k for k, v in data.label_mapping.items()}
    edge_label_distribution = {
        inv_mapping.get(int(k), str(int(k))): int(v)
        for k, v in pd.Series(label_ids).value_counts().sort_index().items()
    }
    connectivity = _connectivity(edge_df)

    checks = [
        check("graph file exists", Path(config.graph_path).exists(), config.graph_path),
        check(
            "aggregated edge file exists",
            Path(config.aggregated_edges_path).exists(),
            config.aggregated_edges_path,
        ),
        check(
            "node count matches unique input IPs",
            int(data.x.shape[0]) == expected_nodes,
            f"graph={int(data.x.shape[0])}, input={expected_nodes}",
        ),
        check(
            "edge count matches grouped input flows",
            int(data.edge_index.shape[1]) == expected_edges == len(edge_df),
            f"graph={int(data.edge_index.shape[1])}, grouped={expected_edges}, csv={len(edge_df)}",
        ),
        check("node feature width is 10", int(data.x.shape[1]) == 10, str(tuple(data.x.shape))),
        check(
            "edge attribute width is 5",
            int(data.edge_attr.shape[1]) == len(EDGE_ATTR_COLUMNS),
            str(tuple(data.edge_attr.shape)),
        ),
        check(
            "edge labels align with edges",
            int(data.edge_label.shape[0]) == int(data.edge_index.shape[1]),
            f"labels={int(data.edge_label.shape[0])}, edges={int(data.edge_index.shape[1])}",
        ),
        check(
            "edge indices reference valid nodes",
            bool(data.edge_index.max() < data.x.shape[0]) if data.edge_index.numel() else True,
        ),
        check("edge attributes are finite", bool(torch.isfinite(data.edge_attr).all())),
        check("node features are finite", bool(torch.isfinite(data.x).all())),
        check(
            "normalized numeric edge attrs have near-zero mean",
            bool(torch.all(torch.abs(normalized.mean(dim=0)) < 1e-4)),
            [float(v) for v in normalized.mean(dim=0)],
        ),
        check(
            "attack labels preserved in edge table",
            sorted(edge_df[ATTACK_COL].str.strip().unique()) == sorted(data.label_mapping.keys()),
            f"edge_labels={sorted(edge_df[ATTACK_COL].str.strip().unique())}",
        ),
    ]

    metrics = {
        "expected_nodes": expected_nodes,
        "expected_edges": expected_edges,
        "graph_nodes": int(data.x.shape[0]),
        "graph_edges": int(data.edge_index.shape[1]),
        "input_label_distribution": raw_label_distribution,
        "edge_label_distribution": edge_label_distribution,
        "connectivity": connectivity,
    }
    artifacts = {
        "graph": config.graph_path,
        "aggregated_edges": config.aggregated_edges_path,
        "graph_visualization": str(Path(config.aggregated_edges_path).with_name("graph.html")),
    }
    return finalize_report(
        config, "phase1", f"Phase 1 Validation - {config.display_name}", checks, artifacts, metrics
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Phase 1 graph artifacts.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="ton_iot")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    report = main(args.dataset)
    print(f"Phase 1 {args.dataset}: {report['status']}")
