from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.pipeline.common.datasets import DATASETS, get_dataset_config
from src.pipeline.common.splits import DEFAULT_K, DEFAULT_SEED, create_edge_splits


def main(dataset: str = "ton_iot", k: int = DEFAULT_K, seed: int = DEFAULT_SEED) -> None:
    config = get_dataset_config(dataset)
    print(f"Loading graph data from {config.graph_path}")
    data: Data = torch.load(config.graph_path, weights_only=False)
    print(f"Creating {k} folds for {config.display_name} at {config.splits_path}")
    create_edge_splits(data, k=k, seed=seed, output_path=config.splits_path)
    print("Saved splits.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stratified edge splits.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="ton_iot")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(dataset=args.dataset, k=args.k, seed=args.seed)
