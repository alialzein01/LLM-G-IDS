"""Prune node features down to the five centralities the ToN-IoT models use.

`run_step1` emits all ten measures in `CENTRALITY_MEASURES`. The live ToN-IoT
graph, however, carries only five structural columns before traffic augmentation
— betweenness, k_truss, global_betweenness, global_pagerank and
modularity_vitality were dropped.

That selection was previously applied outside version control: the repo contained
`pyg_data_5feat.pt` and `pyg_data_pre_traffic.pt` but no code that produced them.
The kept set below was recovered by matching those artifacts column-by-column
against `pyg_data_10feat.pt` (exact match, zero difference), so this module
reproduces the existing graph rather than redefining it.

Run before `augment_traffic_features`, which appends its 7 traffic columns to
whatever structural features remain:

    python -m src.pipeline.step1.prune_node_features --dataset ton_iot_capped
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.pipeline.common.datasets import DATASETS, get_dataset_config
from src.pipeline.step1.graph_construction import CENTRALITY_MEASURES

KEEP_CENTRALITIES = (
    "pagerank",
    "degree",
    "closeness",
    "eigenvector",
    "k_core",
)


def prune(dataset: str, graph_path: str | None = None) -> Path:
    config = get_dataset_config(dataset)
    path = Path(graph_path or config.graph_path)
    data = torch.load(path, weights_only=False)

    if data.x.shape[1] == len(KEEP_CENTRALITIES):
        print(f"Already pruned to {data.x.shape[1]} features — nothing to do.")
        return path
    if data.x.shape[1] != len(CENTRALITY_MEASURES):
        raise ValueError(
            f"Expected {len(CENTRALITY_MEASURES)} node features to prune from, "
            f"found {data.x.shape[1]}. Prune before traffic augmentation."
        )

    keep_idx = [CENTRALITY_MEASURES.index(m) for m in KEEP_CENTRALITIES]
    backup = path.with_name(path.stem + "_10feat.pt")
    if not backup.exists():
        torch.save(data, backup)
        print(f"backed up -> {backup.name}")

    data.x = data.x[:, keep_idx].contiguous()
    data.num_struct_features = len(KEEP_CENTRALITIES)
    data.kept_centralities = list(KEEP_CENTRALITIES)
    torch.save(data, path)

    dropped = [m for m in CENTRALITY_MEASURES if m not in KEEP_CENTRALITIES]
    print(f"kept {len(KEEP_CENTRALITIES)}: {', '.join(KEEP_CENTRALITIES)}")
    print(f"dropped {len(dropped)}: {', '.join(dropped)}")
    print(f"saved pruned graph -> {path}  (x={tuple(data.x.shape)})")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ton_iot", choices=sorted(DATASETS))
    parser.add_argument("--graph-path")
    args = parser.parse_args()
    prune(args.dataset, args.graph_path)


if __name__ == "__main__":
    main()
