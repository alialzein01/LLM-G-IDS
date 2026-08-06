from __future__ import annotations

from pathlib import Path

import numpy as np
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


FLOW_SIGNATURE_COLUMNS = [
    SRC_COL,
    DST_COL,
    ATTACK_COL,
    "IN_BYTES",
    "FLOW_DURATION_MILLISECONDS",
    "PROTOCOL",
    "L4_DST_PORT",
]


def _group_by_signature(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse flow records into one row per distinct model-visible behaviour.

    Two flows sharing endpoints and all five edge attributes are indistinguishable
    to the classifier. Left as separate rows, a random split scatters copies of the
    same record across train and test and the reported score measures memorisation
    rather than generalisation.

    Collapsing them is not the same as discarding the repetition: `flow_count`
    carries how often each behaviour occurred, so the frequency signal the
    aggregated path captured survives as a feature instead of as duplicate rows.

    `total_bytes` is stored as a true sum (bytes x occurrences) because Step 2
    recovers per-flow payload size as `total_bytes / flow_count`.
    """
    before = len(df)
    grouped = df.groupby(FLOW_SIGNATURE_COLUMNS, as_index=False, sort=False).agg(
        flow_count=("IN_BYTES", "size"),
        **{column: (column, "mean") for column in ALL_CENTRALITY_COLUMNS},
    )
    print(
        f"Grouped by signature: {before} flows -> {len(grouped)} distinct behaviours "
        f"({1 - len(grouped) / before:.1%} collapsed)"
    )
    return grouped


def _cap_per_class(df: pd.DataFrame, cap: int | None, seed: int) -> pd.DataFrame:
    """Keep at most `cap` rows per attack class.

    Classes with fewer than `cap` rows are kept in full, so capping only ever
    thins the majority classes and never touches the rare ones.
    """
    if cap is None:
        return df

    rng = np.random.default_rng(seed)
    kept: list[np.ndarray] = []
    for label, positions in df.groupby(ATTACK_COL, sort=False).indices.items():
        if len(positions) > cap:
            positions = rng.choice(positions, size=cap, replace=False)
        kept.append(np.asarray(positions))

    keep_positions = np.sort(np.concatenate(kept))
    capped = df.iloc[keep_positions].reset_index(drop=True)

    before = df[ATTACK_COL].value_counts().rename("before")
    after = capped[ATTACK_COL].value_counts().rename("after")
    print(f"Capped at {cap} flows per class:")
    print(pd.concat([before, after], axis=1).fillna(0).astype(int).to_string())
    return capped


def _build_flow_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Map signature-grouped rows onto the EDGE_ATTR_COLUMNS schema.

    Emits exactly the columns the aggregated path emits, so Step 2 and every later
    stage run unmodified.
    """
    edge_df = df[
        [SRC_COL, DST_COL, ATTACK_COL, "flow_count", *ALL_CENTRALITY_COLUMNS]
    ].copy()
    edge_df["flow_count"] = df["flow_count"].astype(float)
    edge_df["total_bytes"] = (df["IN_BYTES"] * df["flow_count"]).astype(float)
    edge_df["avg_duration"] = df["FLOW_DURATION_MILLISECONDS"].astype(float)
    edge_df["most_common_protocol"] = df["PROTOCOL"].astype(int)
    edge_df["most_common_port"] = df["L4_DST_PORT"].astype(int)
    return edge_df


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
    edge_df: pd.DataFrame,
    node_to_idx: dict[str, int],
    label_mapping: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    edge_index_array = pd.concat(
        [edge_df[SRC_COL].map(node_to_idx), edge_df[DST_COL].map(node_to_idx)], axis=1
    ).T.to_numpy()
    edge_index = torch.tensor(edge_index_array, dtype=torch.long)

    edge_attr = torch.tensor(edge_df[EDGE_ATTR_COLUMNS].to_numpy(), dtype=torch.float)

    mapped_labels = edge_df[ATTACK_COL].map(label_mapping)
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


def run_step1(
    csv_path: str,
    output_dir: str,
    label_mapping: dict[str, int] | None = None,
    per_flow: bool = False,
    cap_per_class: int | None = None,
    seed: int = 42,
) -> tuple[Data, pd.DataFrame]:
    """Build the Step 1 communication graph.

    By default flows are aggregated by (src, dst, attack), which is the
    todo.md-mandated form. With `per_flow=True` one edge is emitted per distinct
    model-visible flow signature, carrying its occurrence count in `flow_count`,
    and `cap_per_class` bounds how many distinct behaviours each class contributes.
    """
    if label_mapping is None:
        label_mapping = LABEL_MAPPING

    if cap_per_class is not None and not per_flow:
        raise ValueError("cap_per_class only applies when per_flow=True")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df[ATTACK_COL] = df[ATTACK_COL].str.strip()

    null_count = int(df.isnull().sum().sum())
    print(f"Loaded CSV shape: {df.shape}")
    print(f"No null values: {null_count == 0} (total nulls: {null_count})")

    if per_flow:
        df = _group_by_signature(df)
        df = _cap_per_class(df, cap_per_class, seed)
        edge_df = _build_flow_edges(df)
        print(f"Total flows represented: {int(edge_df['flow_count'].sum())}")
    else:
        aggregations = {
            "flow_count": ("IN_BYTES", "count"),
            "total_bytes": ("IN_BYTES", "sum"),
            "avg_duration": ("FLOW_DURATION_MILLISECONDS", "mean"),
            "most_common_protocol": ("PROTOCOL", _mode_first),
            "most_common_port": ("L4_DST_PORT", _mode_first),
        }
        aggregations.update(
            {column: (column, "mean") for column in ALL_CENTRALITY_COLUMNS}
        )

        edge_df = df.groupby([SRC_COL, DST_COL, ATTACK_COL], as_index=False).agg(
            **aggregations
        )

    node_to_idx, idx_to_node = _build_node_index(edge_df)
    num_nodes = len(node_to_idx)

    print(f"{'Flow' if per_flow else 'Aggregated'} edges: {len(edge_df)}")
    print(f"Unique IP count: {num_nodes}")

    x = _build_node_features(edge_df, node_to_idx, num_nodes)
    edge_index, edge_attr, edge_label = _build_edge_tensors(edge_df, node_to_idx, label_mapping)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_label=edge_label)
    data.node_to_idx = node_to_idx
    data.idx_to_node = idx_to_node
    data.label_mapping = label_mapping.copy()

    if per_flow:
        # Occurrence counts span 1 to many thousands. Z-scoring that raw leaves a
        # feature that is ~0 almost everywhere with a handful of huge outliers, so
        # compress it first. Aggregated mode is left untouched to keep the existing
        # baseline byte-identical.
        data.edge_attr[:, 0] = torch.log1p(data.edge_attr[:, 0])
        data.log1p_flow_count = True

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

    edges_name = "flow_edges.csv" if per_flow else "aggregated_edges.csv"
    graph_name = "pyg_data_perflow.pt" if per_flow else "pyg_data.pt"
    edge_df.to_csv(output_path / edges_name, index=False)
    torch.save(data, output_path / graph_name)
    print(f"Saved {edges_name} and {graph_name} to {output_path}")

    return data, edge_df
