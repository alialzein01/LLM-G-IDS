"""Invariant checks for a signature-grouped (capped) Step 1 graph.

The first attempt at per-flow edges scored 0.7597 macro-F1 and was worthless:
48% of every test fold was an exact duplicate of a training edge, so the model
was recalling answers rather than generalising. Signature grouping is supposed to
make that impossible. This module proves it did, rather than assuming it.

    python -m src.pipeline.step1.verify_flow_graph --dataset ton_iot_capped
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.pipeline.common.datasets import DATASETS, get_dataset_config


def _signatures(data) -> np.ndarray:
    """Hash of everything the classifier can see for each edge."""
    frame = pd.DataFrame(
        np.column_stack(
            [
                data.edge_index.T.numpy(),
                data.edge_attr.numpy(),
                data.edge_label.numpy(),
            ]
        )
    )
    joined = frame.round(6).astype(str).agg("|".join, axis=1)
    return pd.util.hash_pandas_object(joined, index=False).to_numpy()


def verify(dataset: str, expected_total_flows: int | None = None) -> bool:
    config = get_dataset_config(dataset)
    data = torch.load(config.graph_path, weights_only=False)
    key = _signatures(data)
    n_edges = len(key)
    failures: list[str] = []

    distinct = len(np.unique(key))
    print(f"edges={n_edges}  distinct signatures={distinct}")
    if distinct != n_edges:
        failures.append(
            f"{n_edges - distinct} duplicate signatures remain "
            f"({1 - distinct / n_edges:.1%}); grouping did not collapse them"
        )

    splits_path = Path(config.splits_path)
    if splits_path.exists():
        folds = torch.load(splits_path, weights_only=False)
        print("\nper-fold share of test edges also present in train:")
        for i, fold in enumerate(folds):
            train = np.unique(key[fold["train_mask"].numpy()])
            test = key[fold["test_mask"].numpy()]
            leaked = float(np.isin(test, train).mean())
            print(f"  fold {i}: {leaked:.2%}")
            if leaked > 0:
                failures.append(f"fold {i} leaks {leaked:.2%} of its test edges")
    else:
        print(f"\n(no splits at {splits_path} — skipping leakage check)")

    edges_csv = Path(config.aggregated_edges_path)
    if edges_csv.exists():
        total = int(pd.read_csv(edges_csv, usecols=["flow_count"])["flow_count"].sum())
        print(f"\nraw flows represented by flow_count: {total}")
        if expected_total_flows is not None and total != expected_total_flows:
            failures.append(
                f"flow_count sums to {total}, expected {expected_total_flows}"
            )

    nl_path = Path(config.kg_nl_path)
    if nl_path.exists():
        n_sentences = sum(1 for _ in nl_path.open())
        print(f"NL sentences: {n_sentences}")
        if n_sentences != n_edges:
            failures.append(f"{n_sentences} sentences for {n_edges} edges")

    emb_path = Path(config.llm_embedding_path)
    if emb_path.exists():
        rows = torch.load(emb_path, weights_only=True).shape[0]
        print(f"LLM embedding rows: {rows}")
        if rows != n_edges:
            failures.append(f"{rows} embedding rows for {n_edges} edges")

    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return False
    print("PASS: no duplicate signatures, no fold leakage, all stages aligned.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ton_iot_capped", choices=sorted(DATASETS))
    parser.add_argument("--expected-total-flows", type=int)
    args = parser.parse_args()
    sys.exit(0 if verify(args.dataset, args.expected_total_flows) else 1)


if __name__ == "__main__":
    main()
