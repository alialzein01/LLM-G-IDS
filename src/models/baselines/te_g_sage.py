from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

PUBLISHED_HIDDEN_DIM = 128
PUBLISHED_NUM_LAYERS = 2
PUBLISHED_DROPOUT = 0.3
PUBLISHED_EDGE_MLP_HIDDEN = 128
PUBLISHED_LR = 3e-4
PUBLISHED_WEIGHT_DECAY = 1e-4
PUBLISHED_EPOCHS = 20
PUBLISHED_FANOUT = (25, 15)


class EdgeHead(nn.Module):
    """Two-layer MLP over [h_src || h_dst || e_feat] -- the released EdgeHead."""

    def __init__(self, in_dim: int, hidden: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class TEGSage(nn.Module):
    """TE-G-SAGE edge classifier, per the authors' released implementation.

    Nodes carry no structural features: `node_init="learned_constant"` uses one
    `nn.Embedding(1, hidden)` broadcast to every node, so `data.x` is unused.
    `node_init="node_features"` is the labelled, NON-FAITHFUL ablation.
    Edge features enter at the head only, never during convolution.
    """

    def __init__(
        self,
        edge_dim: int,
        hidden_dim: int = PUBLISHED_HIDDEN_DIM,
        num_layers: int = PUBLISHED_NUM_LAYERS,
        num_classes: int = 10,
        dropout: float = PUBLISHED_DROPOUT,
        edge_mlp_hidden: int = PUBLISHED_EDGE_MLP_HIDDEN,
        node_init: str = "learned_constant",
        node_feat_dim: int | None = None,
    ) -> None:
        super().__init__()
        if node_init not in ("learned_constant", "node_features"):
            raise ValueError(f"unknown node_init {node_init!r}")
        if node_init == "node_features" and node_feat_dim is None:
            raise ValueError("node_init='node_features' requires node_feat_dim")

        self.node_init = node_init
        self.hidden_dim = hidden_dim
        in_dim = hidden_dim if node_init == "learned_constant" else int(node_feat_dim)
        if node_init == "learned_constant":
            self.node_embed = nn.Embedding(1, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                SAGEConv(in_dim if layer == 0 else hidden_dim, hidden_dim, aggr="mean")
            )
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.head = EdgeHead(
            hidden_dim * 2 + edge_dim, edge_mlp_hidden, num_classes, dropout
        )

    def forward(
        self,
        x: torch.Tensor | None,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        if self.node_init == "learned_constant":
            num_nodes = int(edge_index.max()) + 1 if x is None else x.shape[0]
            h = self.node_embed.weight[0].expand(num_nodes, self.hidden_dim)
        else:
            h = x

        for conv, norm in zip(self.convs, self.norms):
            h = F.relu(norm(conv(h, edge_index)))

        src, dst = edge_index[0], edge_index[1]
        return self.head(torch.cat([h[src], h[dst], edge_attr], dim=1))
