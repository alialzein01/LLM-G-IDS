from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


DEFAULT_IN_DIM = 10
DEFAULT_HIDDEN_DIM = 32
DEFAULT_EDGE_ATTR_DIM = 5
DEFAULT_NUM_CLASSES = 10
DEFAULT_HEADS = 4
DEFAULT_DROPOUT = 0.2


class GATStructuralEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int = DEFAULT_IN_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        edge_attr_dim: int = DEFAULT_EDGE_ATTR_DIM,
        heads: int = DEFAULT_HEADS,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})"
            )

        self.dropout = dropout

        self.gat1 = GATv2Conv(
            in_channels=in_dim,
            out_channels=hidden_dim // heads,
            heads=heads,
            concat=True,
            edge_dim=edge_attr_dim,
            add_self_loops=False,
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.res1 = nn.Linear(in_dim, hidden_dim)

        self.gat2 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=1,
            concat=False,
            edge_dim=edge_attr_dim,
            add_self_loops=False,
        )
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        h = self.gat1(x, edge_index, edge_attr=edge_attr)
        h = self.bn1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + self.res1(x)

        h_in = h
        h = self.gat2(h, edge_index, edge_attr=edge_attr)
        h = self.bn2(h)
        h = F.elu(h)
        h = h + h_in

        return h


class GATEdgeClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int = DEFAULT_IN_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        edge_attr_dim: int = DEFAULT_EDGE_ATTR_DIM,
        num_classes: int = DEFAULT_NUM_CLASSES,
        heads: int = DEFAULT_HEADS,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.encoder = GATStructuralEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            edge_attr_dim=edge_attr_dim,
            heads=heads,
            dropout=dropout,
        )
        self.mlp_hidden = nn.Linear(hidden_dim * 2 + edge_attr_dim, hidden_dim)
        self.mlp_out = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_v = self.encoder(x, edge_index, edge_attr)

        src, dst = edge_index[0], edge_index[1]
        edge_repr = torch.cat([h_v[src], h_v[dst], edge_attr], dim=1)

        edge_emb = self.mlp_hidden(edge_repr)
        edge_emb = F.relu(edge_emb)
        edge_emb = F.dropout(edge_emb, p=self.dropout, training=self.training)

        logits = self.mlp_out(edge_emb)
        return logits, edge_emb
