from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data


SRC_COL = "IPV4_SRC_ADDR"
DST_COL = "IPV4_DST_ADDR"
ATTACK_COL = "Attack"

CENTRALITY_MEASURES = [
    "betweenness",
    "pagerank",
    "degree",
    "closeness",
    "eigenvector",
    "k_core",
    "k_truss",
    "global_betweenness",
    "global_pagerank",
    "modularity_vitality",
]

SRC_CENTRALITY_COLUMNS = [f"src_{measure}" for measure in CENTRALITY_MEASURES]
DST_CENTRALITY_COLUMNS = [f"dst_{measure}" for measure in CENTRALITY_MEASURES]
ALL_CENTRALITY_COLUMNS = SRC_CENTRALITY_COLUMNS + DST_CENTRALITY_COLUMNS

EDGE_ATTR_COLUMNS = [
    "flow_count",
    "total_bytes",
    "avg_duration",
    "most_common_protocol",
    "most_common_port",
]

LABEL_MAPPING = {
    "Benign": 0,
    "backdoor": 1,
    "ddos": 2,
    "dos": 3,
    "injection": 4,
    "mitm": 5,
    "password": 6,
    "ransomware": 7,
    "scanning": 8,
    "xss": 9,
}


def _mode_first(series: pd.Series) -> int:
    mode_values = series.mode(dropna=False)
    return int(mode_values.iloc[0])


def _build_node_index(edge_df: pd.DataFrame) -> tuple[dict[str, int], dict[int, str]]:
    unique_ips = (
        pd.concat([edge_df[SRC_COL], edge_df[DST_COL]], ignore_index=True)
        .drop_duplicates()
        .sort_values(kind="mergesort")
        .reset_index(drop=True)
    )
    node_indices = pd.RangeIndex(len(unique_ips))
    node_to_idx = pd.Series(node_indices, index=unique_ips).to_dict()
    idx_to_node = pd.Series(unique_ips.to_numpy(), index=node_indices).to_dict()
    return node_to_idx, idx_to_node


def _build_node_features(
    edge_df: pd.DataFrame, node_to_idx: dict[str, int], num_nodes: int
) -> torch.Tensor:
    node_lookup = (
        pd.Series(node_to_idx, name="node_idx").rename_axis("node").reset_index()
    )

    src_long = edge_df[[SRC_COL, *SRC_CENTRALITY_COLUMNS]].melt(
        id_vars=SRC_COL, var_name="measure", value_name="value"
    )
    src_long = src_long.rename(columns={SRC_COL: "node"})
    src_long["measure"] = src_long["measure"].str.removeprefix("src_")

    dst_long = edge_df[[DST_COL, *DST_CENTRALITY_COLUMNS]].melt(
        id_vars=DST_COL, var_name="measure", value_name="value"
    )
    dst_long = dst_long.rename(columns={DST_COL: "node"})
    dst_long["measure"] = dst_long["measure"].str.removeprefix("dst_")

    centrality_long = pd.concat([src_long, dst_long], ignore_index=True)
    centrality_long = centrality_long.merge(node_lookup, on="node", how="left")

    node_features = (
        centrality_long.groupby(["node_idx", "measure"], sort=False)["value"]
        .mean()
        .unstack("measure")
        .reindex(index=pd.RangeIndex(num_nodes), columns=CENTRALITY_MEASURES)
        .fillna(0.0)
    )

    return torch.tensor(node_features.to_numpy(), dtype=torch.float)


def _build_edge_tensors(
    edge_df: pd.DataFrame, node_to_idx: dict[str, int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    edge_index_array = pd.concat(
        [edge_df[SRC_COL].map(node_to_idx), edge_df[DST_COL].map(node_to_idx)], axis=1
    ).T.to_numpy()
    edge_index = torch.tensor(edge_index_array, dtype=torch.long)

    edge_attr = torch.tensor(edge_df[EDGE_ATTR_COLUMNS].to_numpy(), dtype=torch.float)

    mapped_labels = edge_df[ATTACK_COL].map(LABEL_MAPPING)
    if mapped_labels.isna().any():
        unknown_labels = sorted(edge_df.loc[mapped_labels.isna(), ATTACK_COL].unique())
        raise ValueError(f"Unknown attack labels found: {unknown_labels}")
    edge_label = torch.tensor(mapped_labels.to_numpy(), dtype=torch.long)

    return edge_index, edge_attr, edge_label


def _print_edge_attr_stats(edge_attr: torch.Tensor) -> None:
    stats = pd.DataFrame(
        {
            "column": EDGE_ATTR_COLUMNS,
            "min": edge_attr.min(dim=0).values.tolist(),
            "max": edge_attr.max(dim=0).values.tolist(),
            "mean": edge_attr.mean(dim=0).tolist(),
        }
    )
    print("Edge attribute statistics after normalization:")
    print(stats.to_string(index=False))


def run_step1(csv_path: str, output_dir: str) -> tuple[Data, pd.DataFrame]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df[ATTACK_COL] = df[ATTACK_COL].str.strip()

    null_count = int(df.isnull().sum().sum())
    print(f"Loaded CSV shape: {df.shape}")
    print(f"No null values: {null_count == 0} (total nulls: {null_count})")

    aggregations = {
        "flow_count": ("IN_BYTES", "count"),
        "total_bytes": ("IN_BYTES", "sum"),
        "avg_duration": ("FLOW_DURATION_MILLISECONDS", "mean"),
        "most_common_protocol": ("PROTOCOL", _mode_first),
        "most_common_port": ("L4_DST_PORT", _mode_first),
    }
    aggregations.update({column: (column, "mean") for column in ALL_CENTRALITY_COLUMNS})

    edge_df = df.groupby([SRC_COL, DST_COL, ATTACK_COL], as_index=False).agg(
        **aggregations
    )

    node_to_idx, idx_to_node = _build_node_index(edge_df)
    num_nodes = len(node_to_idx)

    print(f"Aggregated edges: {len(edge_df)}")
    print(f"Unique IP count: {num_nodes}")

    x = _build_node_features(edge_df, node_to_idx, num_nodes)
    edge_index, edge_attr, edge_label = _build_edge_tensors(edge_df, node_to_idx)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_label=edge_label)
    data.node_to_idx = node_to_idx
    data.idx_to_node = idx_to_node
    data.label_mapping = LABEL_MAPPING.copy()

    normalization_slice = data.edge_attr[:, :3]
    data.edge_attr_mean = normalization_slice.mean(dim=0)
    data.edge_attr_std = normalization_slice.std(dim=0, unbiased=False)
    data.edge_attr[:, :3] = (
        normalization_slice - data.edge_attr_mean
    ) / (data.edge_attr_std + 1e-8)

    assert data.x.shape[1] == 10
    assert data.edge_index.shape[0] == 2
    assert data.edge_index.max() < data.x.shape[0]
    assert data.edge_attr.shape[1] == 5
    assert data.edge_label.shape[0] == data.edge_index.shape[1]

    print("Validation passed:")
    print(f"  data.x.shape[1] == 10: {data.x.shape[1] == 10}")
    print(f"  data.edge_index.shape[0] == 2: {data.edge_index.shape[0] == 2}")
    print(
        "  data.edge_index.max() < data.x.shape[0]: "
        f"{data.edge_index.max() < data.x.shape[0]}"
    )
    print(f"  data.edge_attr.shape[1] == 5: {data.edge_attr.shape[1] == 5}")
    print(
        "  data.edge_label.shape[0] == data.edge_index.shape[1]: "
        f"{data.edge_label.shape[0] == data.edge_index.shape[1]}"
    )

    label_distribution = (
        pd.Series(edge_label.numpy()).value_counts().sort_index().rename("edge_count")
    )
    print("Edge label distribution:")
    print(label_distribution.to_string())
    _print_edge_attr_stats(data.edge_attr)

    edge_df.to_csv(output_path / "aggregated_edges.csv", index=False)
    torch.save(data, output_path / "pyg_data.pt")

    return data, edge_df
