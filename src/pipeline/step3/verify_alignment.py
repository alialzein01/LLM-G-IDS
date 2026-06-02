"""
Pre-fusion alignment verification.

Run before fusion training to confirm that GNN embeddings, LLM embeddings,
structured KG rows, NL sentences, splits, and graph edge labels are row-aligned.
Row i in every artifact must refer to the same network edge.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.pipeline.common.datasets import DATASETS, get_dataset_config


def _check(condition: bool, msg: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {msg}")
    if not condition:
        raise AssertionError(msg)


def main(dataset: str = "ton_iot") -> None:
    config = get_dataset_config(dataset)
    print(f"=== Pre-Fusion Alignment Verification: {config.display_name} ===\n")
    all_passed = True

    print("Loading artifacts...")
    data = torch.load(config.graph_path, weights_only=False)
    gnn_emb = torch.load(config.gnn_embedding_path, weights_only=True)
    llm_emb = torch.load(config.llm_embedding_path, weights_only=True)
    folds = torch.load(config.splits_path, weights_only=False)
    agg = pd.read_csv(config.aggregated_edges_path)
    kg = pd.read_csv(config.kg_csv_path)
    sentences = Path(config.kg_nl_path).read_text().strip().splitlines()
    n_edges = data.edge_label.shape[0]
    print(f"  Graph edges (N): {n_edges}\n")

    print("Shape checks:")
    try:
        _check(
            gnn_emb.shape == (n_edges, config.gnn_dim),
            f"GNN embeddings shape == ({n_edges}, {config.gnn_dim}), got {tuple(gnn_emb.shape)}",
        )
        _check(
            llm_emb.shape == (n_edges, config.llm_dim),
            f"LLM embeddings shape == ({n_edges}, {config.llm_dim}), got {tuple(llm_emb.shape)}",
        )
        _check(data.edge_label.shape[0] == n_edges, f"Graph edge_label length == {n_edges}")
        _check(
            len(agg) == n_edges,
            f"aggregated_edges.csv rows == {n_edges}, got {len(agg)}",
        )
        _check(len(kg) == n_edges, f"kg_triples.csv rows == {n_edges}, got {len(kg)}")
        _check(
            len(sentences) == n_edges,
            f"kg_triples_nl.txt lines == {n_edges}, got {len(sentences)}",
        )
    except AssertionError:
        all_passed = False

    print("\nFiniteness checks:")
    try:
        _check(bool(torch.isfinite(gnn_emb).all()), "GNN embeddings contain no NaN/Inf")
        _check(bool(torch.isfinite(llm_emb).all()), "LLM embeddings contain no NaN/Inf")
        _check(bool(torch.isfinite(data.edge_label.float()).all()), "Edge labels contain no NaN/Inf")
    except AssertionError:
        all_passed = False

    print("\nRow alignment (CSV <-> CSV):")
    try:
        src_match = (agg["IPV4_SRC_ADDR"].values == kg["IPV4_SRC_ADDR"].values).all()
        dst_match = (agg["IPV4_DST_ADDR"].values == kg["IPV4_DST_ADDR"].values).all()
        _check(bool(src_match), "aggregated_edges SRC IPs match kg_triples SRC IPs")
        _check(bool(dst_match), "aggregated_edges DST IPs match kg_triples DST IPs")
    except AssertionError:
        all_passed = False

    print("\nSplits integrity:")
    try:
        _check(len(folds) == 5, f"folds.pt contains 5 folds, got {len(folds)}")
        for i, fold in enumerate(folds):
            split_size = fold["train_mask"].shape[0]
            _check(split_size == n_edges, f"Fold {i} masks length == {n_edges}")
            total = int(
                fold["train_mask"].sum() + fold["val_mask"].sum() + fold["test_mask"].sum()
            )
            _check(total == n_edges, f"Fold {i} train+val+test = {total} (expected {n_edges})")
    except AssertionError:
        all_passed = False

    print("\nLabel-free integrity:")
    try:
        leakage_terms = {name.lower() for name in config.label_names if name.lower() not in {"benign", "normal"}}
        leaks = [
            (i, term)
            for i, sentence in enumerate(sentences)
            for term in leakage_terms
            if term in sentence.lower()
        ]
        _check(
            len(leaks) == 0,
            f"No attack labels leaked into NL sentences (found {len(leaks)} violations)",
        )
    except AssertionError:
        all_passed = False

    print()
    if all_passed:
        print("All checks passed. Both paths are aligned and ready for fusion.")
        print(f"\n  GNN embeddings : ({n_edges}, {config.gnn_dim})")
        print(f"  LLM embeddings : ({n_edges}, {config.llm_dim})")
        print(f"  Edge labels    : ({n_edges},)")
        print(f"  Splits         : 5 folds x {n_edges} edges each")
    else:
        print("One or more checks FAILED. Do not proceed to fusion training.")
        sys.exit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify pre-fusion artifact alignment.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="ton_iot")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(dataset=args.dataset)
