"""Expanded Step-1 graph: keep the (src,dst,attack) aggregation mandated by
todo.md, but split each group's flows into CHUNKS so a heavy triple yields many
parallel edges instead of a single count-aggregated one.

Rationale (2026-07-16): the original Step 1 collapsed 1,379,274 NF-ToN-IoT
flows into 2,127 edges (one per src,dst,attack triple), starving the GNN and
inverting class balance (dos -> 4 edges, ransomware -> 3). The structural
fingerprints (hub-and-spoke = DDoS, star = scanning) live in the src->dst
topology, which chunking preserves — the graph is a directed multigraph, so a
triple simply contributes several parallel edges. Each edge still carries the
same 5 aggregated attributes; per-class chunking rebalances the labels.

Per class: chunk_size = round(class_flows / TARGET_EDGES_PER_CLASS) (>=1), so
big classes are capped near TARGET and tiny classes (ransomware 142, mitm 1295)
keep ~all their flows as edges. No timestamp column exists, so chunks follow
row order within each group.

Node features = 5 kept centralities (pagerank, degree, closeness, eigenvector,
k_core; averaged per IP from the raw CSV's src_/dst_ columns) + 7 traffic
statistics computed on the expanded edge set = 12 dims.

Run:
    python -m src.pipeline.step1.build_expanded_graph --dataset ton_iot \
        --target-per-class 3000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.pipeline.common.datasets import get_dataset_config
from src.pipeline.step1.augment_traffic_features import build_traffic_node_features
from src.pipeline.step1.graph_construction import (
    ATTACK_COL,
    DST_COL,
    LABEL_MAPPING,
    SRC_COL,
    _mode_first,
)

KEEP_CENTRALITIES = ["pagerank", "degree", "closeness", "eigenvector", "k_core"]
EDGE_ATTR_COLUMNS = [
    "flow_count", "total_bytes", "avg_duration",
    "most_common_protocol", "most_common_port",
]


def build_expanded_graph(dataset: str, target_per_class: int = 3000) -> Path:
    config = get_dataset_config(dataset)
    raw_path = config.phase1_input_path
    print(f"Reading {raw_path}")
    df = pd.read_csv(raw_path)
    df[ATTACK_COL] = df[ATTACK_COL].astype(str).str.strip()
    n_raw = len(df)

    # --- per-class chunk size ---
    class_counts = df[ATTACK_COL].value_counts()
    chunk_size = {c: max(1, int(round(n / target_per_class))) for c, n in class_counts.items()}
    df["_cs"] = df[ATTACK_COL].map(chunk_size)
    # sequential index within each (src,dst,attack) group -> chunk id
    df["_within"] = df.groupby([SRC_COL, DST_COL, ATTACK_COL]).cumcount()
    df["_chunk"] = (df["_within"] // df["_cs"]).astype(int)

    df["_bytes"] = df["IN_BYTES"] + df.get("OUT_BYTES", 0)
    edge_df = df.groupby([SRC_COL, DST_COL, ATTACK_COL, "_chunk"], as_index=False).agg(
        flow_count=("_bytes", "count"),
        total_bytes=("_bytes", "sum"),
        avg_duration=("FLOW_DURATION_MILLISECONDS", "mean"),
        most_common_protocol=("PROTOCOL", _mode_first),
        most_common_port=("L4_DST_PORT", _mode_first),
    )
    print(f"Raw flows: {n_raw:,}  ->  expanded edges: {len(edge_df):,}")
    print("Edges per class:")
    print(edge_df[ATTACK_COL].map(str).value_counts().to_string())

    # --- node index ---
    nodes = pd.unique(pd.concat([edge_df[SRC_COL], edge_df[DST_COL]], ignore_index=True))
    nodes = sorted(nodes)
    node_to_idx = {ip: i for i, ip in enumerate(nodes)}
    idx_to_node = {i: ip for ip, i in node_to_idx.items()}
    num_nodes = len(node_to_idx)

    # --- node centralities (avg per IP from raw src_/dst_ columns) ---
    cent = np.zeros((num_nodes, len(KEEP_CENTRALITIES)), dtype=np.float64)
    for j, m in enumerate(KEEP_CENTRALITIES):
        src_part = df[[SRC_COL, f"src_{m}"]].rename(columns={SRC_COL: "node", f"src_{m}": "v"})
        dst_part = df[[DST_COL, f"dst_{m}"]].rename(columns={DST_COL: "node", f"dst_{m}": "v"})
        both = pd.concat([src_part, dst_part], ignore_index=True)
        means = both.groupby("node")["v"].mean()
        for ip, val in means.items():
            if ip in node_to_idx:
                cent[node_to_idx[ip], j] = val
    # z-score centralities
    mu, sd = cent.mean(0), cent.std(0)
    cent = (cent - mu) / np.where(sd > 1e-8, sd, 1.0)

    # --- traffic node features on the expanded edge set ---
    traffic = build_traffic_node_features(edge_df, node_to_idx, num_nodes)
    x = torch.tensor(np.concatenate([cent, traffic], axis=1), dtype=torch.float)

    # --- edge tensors ---
    edge_index = torch.tensor(
        np.stack([edge_df[SRC_COL].map(node_to_idx).to_numpy(),
                  edge_df[DST_COL].map(node_to_idx).to_numpy()]), dtype=torch.long,
    )
    edge_attr = torch.tensor(edge_df[EDGE_ATTR_COLUMNS].to_numpy(), dtype=torch.float)
    edge_label = torch.tensor(edge_df[ATTACK_COL].map(LABEL_MAPPING).to_numpy(), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_label=edge_label)
    data.node_to_idx = node_to_idx
    data.idx_to_node = idx_to_node
    data.label_mapping = LABEL_MAPPING.copy()
    data.num_struct_features = len(KEEP_CENTRALITIES)
    # z-score first 3 edge attrs (counts/bytes/duration)
    sl = data.edge_attr[:, :3]
    data.edge_attr_mean = sl.mean(0)
    data.edge_attr_std = sl.std(0, unbiased=False)
    data.edge_attr[:, :3] = (sl - data.edge_attr_mean) / (data.edge_attr_std + 1e-8)

    # also persist an aggregated_edges.csv aligned to edges (for Step 2 NL)
    out_dir = Path(config.graph_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    edge_df.rename(columns={ATTACK_COL: "Attack"})[
        [SRC_COL, DST_COL, "Attack"] + EDGE_ATTR_COLUMNS
    ].to_csv(config.aggregated_edges_path, index=False)

    graph_path = Path(config.graph_path)
    backup = graph_path.with_name("pyg_data_2127.pt")
    if graph_path.exists() and not backup.exists():
        torch.save(torch.load(graph_path, weights_only=False), backup)
        print(f"backed up old graph -> {backup.name}")
    torch.save(data, graph_path)
    print(f"nodes={num_nodes}  edges={edge_label.shape[0]}  node_feat_dim={x.shape[1]}")
    print(f"saved expanded graph -> {graph_path}")
    return graph_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="ton_iot", choices=["ton_iot", "unsw_nb15"])
    p.add_argument("--target-per-class", type=int, default=3000)
    args = p.parse_args()
    build_expanded_graph(args.dataset, args.target_per_class)


if __name__ == "__main__":
    main()
