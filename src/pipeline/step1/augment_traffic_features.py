"""Augment node features with per-node traffic statistics (FedGATSage-style).

Our nodes originally carried only structural centralities. FedGATSage
(Al Tfaily et al., Sci. Reports 2025), which reports macro-F1 0.62 on
NF-ToN-IoT, also gives each node *behavioural* traffic statistics (how much
each IP sends/receives). This adds that signal: for every node it aggregates
its incident edges' flow statistics into in/out traffic features and appends
them to `data.x`.

Features appended per node (7):
  out_flow_count, in_flow_count   (log1p)   — volume of sessions initiated / received
  out_bytes,      in_bytes        (log1p)   — byte volume sent / received
  out_edges,      in_edges        (log1p)   — distinct out / in edges (fan-out / fan-in)
  avg_duration                    (z-score) — mean flow duration over incident edges
Counts/bytes are log1p then z-scored; all features standardised across nodes.

The graph is read from / written back to `config.graph_path`; the pre-augment
version is backed up alongside it. Downstream models derive `in_dim` from
`data.x.shape[1]`, so no other change is needed.

Run:
    python -m src.pipeline.step1.augment_traffic_features --dataset ton_iot
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

from src.pipeline.common.datasets import get_dataset_config
from src.pipeline.step1.graph_construction import DST_COL, SRC_COL


def _zscore(a: np.ndarray) -> np.ndarray:
    mu, sd = a.mean(0), a.std(0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return (a - mu) / sd


def build_traffic_node_features(edge_df: pd.DataFrame, node_to_idx: dict, num_nodes: int) -> np.ndarray:
    """[num_nodes, 7] per-node in/out traffic statistics."""
    feats = np.zeros((num_nodes, 7), dtype=np.float64)
    dur_sum = np.zeros(num_nodes)
    dur_cnt = np.zeros(num_nodes)

    for _, r in edge_df.iterrows():
        s = node_to_idx.get(r[SRC_COL])
        d = node_to_idx.get(r[DST_COL])
        fc = float(r["flow_count"])
        tb = float(r["total_bytes"])
        dur = float(r["avg_duration"])
        if s is not None:
            feats[s, 0] += fc     # out_flow_count
            feats[s, 2] += tb     # out_bytes
            feats[s, 4] += 1      # out_edges
            dur_sum[s] += dur; dur_cnt[s] += 1
        if d is not None:
            feats[d, 1] += fc     # in_flow_count
            feats[d, 3] += tb     # in_bytes
            feats[d, 5] += 1      # in_edges
            dur_sum[d] += dur; dur_cnt[d] += 1

    feats[:, 6] = np.where(dur_cnt > 0, dur_sum / np.where(dur_cnt > 0, dur_cnt, 1), 0.0)
    # log1p the heavy-tailed count/byte columns (0..5), keep duration raw
    feats[:, :6] = np.log1p(feats[:, :6])
    return _zscore(feats).astype(np.float32)


def augment(dataset: str) -> Path:
    config = get_dataset_config(dataset)
    graph_path = Path(config.graph_path)
    data = torch.load(graph_path, weights_only=False)

    node_to_idx = getattr(data, "node_to_idx", None)
    if node_to_idx is None:
        idx_to_node = getattr(data, "idx_to_node", None)
        if idx_to_node is None:
            raise RuntimeError("Graph has no node_to_idx / idx_to_node mapping.")
        node_to_idx = {v: k for k, v in idx_to_node.items()}

    edge_df = pd.read_csv(config.aggregated_edges_path)
    num_nodes = data.x.shape[0]
    traffic = build_traffic_node_features(edge_df, node_to_idx, num_nodes)
    traffic_t = torch.tensor(traffic, dtype=torch.float)

    base = data.x
    # if already augmented (has traffic cols), replace them rather than stack again
    n_struct = getattr(data, "num_struct_features", base.shape[1])
    base = base[:, :n_struct]
    data.num_struct_features = int(n_struct)
    data.x = torch.cat([base, traffic_t], dim=1)

    backup = graph_path.with_name(graph_path.stem + "_pre_traffic.pt")
    if not backup.exists():
        torch.save(torch.load(graph_path, weights_only=False), backup)
        print(f"backed up -> {backup.name}")
    torch.save(data, graph_path)
    print(f"node features: {n_struct} structural + 7 traffic = {data.x.shape[1]}  "
          f"(nodes={num_nodes}, edges={data.edge_label.shape[0]})")
    print(f"saved augmented graph -> {graph_path}")
    return graph_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ton_iot", choices=["ton_iot", "unsw_nb15"])
    args = parser.parse_args()
    augment(args.dataset)


if __name__ == "__main__":
    main()
