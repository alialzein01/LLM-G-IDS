from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

from src.pipeline.common.datasets import DatasetConfig

CONTINUOUS_COLUMNS = ["flow_count", "total_bytes", "avg_duration"]
CATEGORICAL_COLUMNS = ["most_common_protocol", "most_common_port"]
RARE_TOKEN = "__rare__"


def load_aligned_edges(config: DatasetConfig) -> tuple[pd.DataFrame, Data]:
    """Load the aggregated edge table alongside its graph, order-checked.

    `folds.pt` indexes edges positionally, so a mismatch between CSV row order
    and `edge_index` column order silently corrupts every downstream number.
    """
    df = pd.read_csv(config.aggregated_edges_path)
    df["Attack"] = df["Attack"].astype(str).str.strip()
    data = torch.load(config.graph_path, weights_only=False)

    if len(df) != int(data.edge_index.shape[1]):
        raise ValueError(
            f"{config.key}: {len(df)} CSV rows vs "
            f"{int(data.edge_index.shape[1])} graph edges"
        )

    label_names = list(config.label_names)
    csv_labels = np.array([label_names.index(a) for a in df["Attack"]])
    graph_labels = data.edge_label.cpu().numpy()
    if not np.array_equal(csv_labels, graph_labels):
        raise ValueError(
            f"{config.key}: CSV row order does not match graph edge order"
        )
    return df, data


def egraphsage_edge_features(df: pd.DataFrame) -> np.ndarray:
    """E-GraphSAGE preprocessing: the paper documents no categorical scheme,
    so all five NetFlow columns are used numerically and standard-scaled."""
    raw = df[CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS].to_numpy(dtype=np.float64)
    return StandardScaler().fit_transform(raw).astype(np.float32)


def te_g_sage_edge_features(
    df: pd.DataFrame, rare_min_freq: int = 50
) -> tuple[np.ndarray, list[str]]:
    """TE-G-SAGE preprocessing: log1p + StandardScaler + correlation pruning
    at 0.995 on numerics; one-hot on categoricals with rare-category folding."""
    numeric = np.log1p(
        df[CONTINUOUS_COLUMNS].to_numpy(dtype=np.float64).clip(min=0.0)
    )
    numeric = StandardScaler().fit_transform(numeric)
    numeric_names = list(CONTINUOUS_COLUMNS)

    keep = _prune_correlated(numeric, threshold=0.995)
    numeric = numeric[:, keep]
    numeric_names = [numeric_names[i] for i in keep]

    blocks = [numeric]
    names = list(numeric_names)
    for col in CATEGORICAL_COLUMNS:
        values = df[col].astype(str)
        counts = values.value_counts()
        frequent = set(counts[counts >= rare_min_freq].index)
        folded = values.where(values.isin(frequent), RARE_TOKEN)
        dummies = pd.get_dummies(folded, prefix=col, dtype=np.float64)
        dummies = dummies.reindex(sorted(dummies.columns), axis=1)
        blocks.append(dummies.to_numpy())
        names.extend(dummies.columns.tolist())

    return np.hstack(blocks).astype(np.float32), names


def _prune_correlated(matrix: np.ndarray, threshold: float) -> list[int]:
    """Drop later columns correlating above `threshold` with an earlier kept one."""
    if matrix.shape[1] < 2:
        return list(range(matrix.shape[1]))
    corr = np.corrcoef(matrix, rowvar=False)
    corr = np.nan_to_num(corr)
    keep: list[int] = []
    for col in range(matrix.shape[1]):
        if all(abs(corr[col, kept]) <= threshold for kept in keep):
            keep.append(col)
    return keep
