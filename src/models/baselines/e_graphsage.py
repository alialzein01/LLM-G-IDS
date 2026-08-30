from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

PUBLISHED_HIDDEN_DIM = 128
PUBLISHED_NUM_LAYERS = 2
PUBLISHED_DROPOUT = 0.2
PUBLISHED_LR = 1e-3


class EGraphSAGELayer(MessagePassing):
    """One E-GraphSAGE layer (Lo et al., NOMS 2022).

    Message is built from the source node state concatenated with the edge
    feature, mean-aggregated over incident edges, then concatenated with the
    node's own state -- the form used by the authors' released implementation.
    """

    def __init__(self, in_dim: int, edge_dim: int, out_dim: int) -> None:
        super().__init__(aggr="mean", flow="source_to_target")
        self.w_msg = nn.Linear(in_dim + edge_dim, out_dim)
        self.w_apply = nn.Linear(in_dim + out_dim, out_dim)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        neigh = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return F.relu(self.w_apply(torch.cat([x, neigh], dim=1)))

    # Source: https://raw.githubusercontent.com/waimorris/E-GraphSAGE/master/E-GraphSAGE/netflow/ton-iot/Unsw_ton_iot_multiclass_mean_agg.ipynb
    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.w_msg(torch.cat([x_j, edge_attr], dim=1))


class EGraphSAGE(nn.Module):
    """E-GraphSAGE edge classifier.

    `node_init="ones"` reproduces the paper: nodes carry x_v = {1,...,1} with
    dimensionality equal to the edge-feature count, so node features are unused.
    `node_init="node_features"` is the labelled, NON-FAITHFUL variant that
    substitutes our centrality measures; it is never reported as E-GraphSAGE.
    """

    def __init__(
        self,
        edge_dim: int,
        hidden_dim: int = PUBLISHED_HIDDEN_DIM,
        num_layers: int = PUBLISHED_NUM_LAYERS,
        num_classes: int = 10,
        dropout: float = PUBLISHED_DROPOUT,
        node_init: str = "ones",
        node_feat_dim: int | None = None,
    ) -> None:
        super().__init__()
        if node_init not in ("ones", "node_features"):
            raise ValueError(f"unknown node_init {node_init!r}")
        if node_init == "node_features" and node_feat_dim is None:
            raise ValueError("node_init='node_features' requires node_feat_dim")

        self.node_init = node_init
        self.dropout = dropout
        in_dim = edge_dim if node_init == "ones" else int(node_feat_dim)

        dims = [in_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList(
            EGraphSAGELayer(dims[i], edge_dim, dims[i + 1]) for i in range(num_layers)
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        self.in_dim = in_dim

    def forward(
        self,
        x: torch.Tensor | None,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = int(edge_index.max()) + 1 if x is None else x.shape[0]
        if self.node_init == "ones":
            h = torch.ones(
                num_nodes, self.in_dim, dtype=edge_attr.dtype, device=edge_attr.device
            )
        else:
            h = x

        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index, edge_attr)
            if i < len(self.layers) - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)

        src, dst = edge_index[0], edge_index[1]
        return self.classifier(torch.cat([h[src], h[dst]], dim=1))
