from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


DEFAULT_IN_DIM = 10
DEFAULT_HIDDEN_DIM = 32
DEFAULT_EDGE_ATTR_DIM = 5
DEFAULT_NUM_CLASSES = 10
DEFAULT_HEADS = 8
DEFAULT_DROPOUT = 0.2
VARIANT_NAMES = ("temporal", "content", "behavioral")


EDGE_FOCUS_WEIGHTS = {
    # flow_count, total_bytes, avg_duration, protocol, dst_port
    "temporal": (1.25, 0.75, 1.50, 0.50, 0.25),
    "content": (0.50, 1.50, 0.50, 1.00, 1.25),
    "behavioral": (1.25, 0.75, 1.00, 0.75, 1.50),
}


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
    """
    Centralized FedGATSage-inspired edge classifier.

    FedGATSage uses three client-side GAT detector variants before the
    federated/community stage. This model keeps only that local GNN idea:
    temporal, content, and behavioral GAT branches produce detector-specific
    flow embeddings, then a learned fusion head emits one edge prediction.
    """

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
        self.hidden_dim = hidden_dim
        self.edge_attr_dim = edge_attr_dim
        self.dropout = dropout

        self.encoders = nn.ModuleDict(
            {
                name: GATStructuralEncoder(
                    in_dim=in_dim,
                    hidden_dim=hidden_dim,
                    edge_attr_dim=edge_attr_dim,
                    heads=heads,
                    dropout=dropout,
                )
                for name in VARIANT_NAMES
            }
        )
        self.variant_heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear((hidden_dim * 4) + edge_attr_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, num_classes),
                )
                for name in VARIANT_NAMES
            }
        )
        self.variant_embedders = nn.ModuleDict(
            {
                name: nn.Linear((hidden_dim * 4) + edge_attr_dim, hidden_dim)
                for name in VARIANT_NAMES
            }
        )
        self.fusion_hidden = nn.Linear(hidden_dim * len(VARIANT_NAMES), hidden_dim)
        self.fusion_out = nn.Linear(hidden_dim, num_classes)

        for name, weights in EDGE_FOCUS_WEIGHTS.items():
            self.register_buffer(
                f"{name}_edge_focus",
                torch.tensor(weights, dtype=torch.float).view(1, edge_attr_dim),
            )

    def _focused_edge_attr(self, name: str, edge_attr: torch.Tensor) -> torch.Tensor:
        focus = getattr(self, f"{name}_edge_focus")
        return edge_attr * focus.to(dtype=edge_attr.dtype, device=edge_attr.device)

    @staticmethod
    def _build_flow_repr(
        node_emb: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        src_h = node_emb[src]
        dst_h = node_emb[dst]
        return torch.cat(
            [
                src_h,
                dst_h,
                src_h * dst_h,
                torch.abs(src_h - dst_h),
                edge_attr,
            ],
            dim=1,
        )

    def encode_edges_by_variant(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        edge_reprs: dict[str, torch.Tensor] = {}
        for name, encoder in self.encoders.items():
            focused_edge_attr = self._focused_edge_attr(name, edge_attr)
            node_emb = encoder(x, edge_index, focused_edge_attr)
            edge_reprs[name] = self._build_flow_repr(
                node_emb, edge_index, focused_edge_attr
            )
        return edge_reprs

    def encode_edges(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Return fused 64-dim edge embeddings for downstream fusion."""
        edge_reprs = self.encode_edges_by_variant(x, edge_index, edge_attr)
        edge_emb, _ = self._fuse_variant_outputs(edge_reprs)
        return edge_emb

    def _fuse_variant_outputs(
        self, edge_reprs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        variant_embs = []
        aux_logits = {}
        for name in VARIANT_NAMES:
            emb = self.variant_embedders[name](edge_reprs[name])
            emb = F.relu(emb)
            variant_embs.append(emb)
            aux_logits[name] = self.variant_heads[name](edge_reprs[name])

        edge_emb = self.fusion_hidden(torch.cat(variant_embs, dim=1))
        edge_emb = F.relu(edge_emb)
        edge_emb = F.dropout(edge_emb, p=self.dropout, training=self.training)
        return edge_emb, aux_logits

    def classify_repr(self, edge_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Classifier head for already-fused edge embeddings."""
        logits = self.fusion_out(edge_emb)
        return logits, edge_emb

    def variant_logits(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        edge_reprs = self.encode_edges_by_variant(x, edge_index, edge_attr)
        return {
            name: self.variant_heads[name](edge_reprs[name])
            for name in VARIANT_NAMES
        }

    def forward_with_aux(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        edge_reprs = self.encode_edges_by_variant(x, edge_index, edge_attr)
        edge_emb, aux_logits = self._fuse_variant_outputs(edge_reprs)
        logits, edge_emb = self.classify_repr(edge_emb)
        return logits, edge_emb, aux_logits

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, edge_emb, _ = self.forward_with_aux(x, edge_index, edge_attr)
        return logits, edge_emb
