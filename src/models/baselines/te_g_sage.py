from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

PUBLISHED_HIDDEN_DIM = 128
PUBLISHED_NUM_LAYERS = 2
PUBLISHED_DROPOUT = 0.3
PUBLISHED_FANOUT = (25, 15)


class TEGSage(nn.Module):
    """TE-G-SAGE edge classifier (MDPI 2025), edge-aware GraphSAGE.

    Published fanout (25, 15) exceeds the degree available in our aggregated
    graphs, so full-neighbourhood aggregation is used -- deviation D2.
    """

    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        hidden_dim: int = PUBLISHED_HIDDEN_DIM,
        num_layers: int = PUBLISHED_NUM_LAYERS,
        num_classes: int = 10,
        dropout: float = PUBLISHED_DROPOUT,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        dims = [in_dim] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList(
            SAGEConv(dims[i], dims[i + 1], aggr="mean") for i in range(num_layers)
        )
        # Built eagerly: a submodule created inside forward() is invisible to an
        # optimizer constructed before the first forward pass, so its weights
        # would never be updated.
        self.classifier = nn.Linear(hidden_dim * 2 + edge_dim, num_classes)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            h = F.relu(h)
            if i < len(self.convs) - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)

        src, dst = edge_index[0], edge_index[1]
        flow = torch.cat([h[src], h[dst], edge_attr], dim=1)
        return self.classifier(flow)
