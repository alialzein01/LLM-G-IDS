from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_GNN_DIM = 64
DEFAULT_LLM_DIM = 768
DEFAULT_PROJ_DIM = 128
DEFAULT_HIDDEN_DIM = 128
DEFAULT_NUM_CLASSES = 10
DEFAULT_DROPOUT = 0.2
DEFAULT_GATE_TEMPERATURE = 1.0


class ModalityProjector(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, proj_dim)
        self.bn = nn.BatchNorm1d(proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        h = self.bn(h)
        return F.relu(h)


class FusionEdgeClassifier(nn.Module):
    """
    Gated attention fusion of GNN structural embeddings (low-dim, large-magnitude)
    and CySecBERT semantic embeddings (high-dim, small-magnitude).

    Inputs must already be StandardScaler-normalized in the trainer — scaling is
    deliberately kept outside the module so state_dict remains pure parameters
    and scaler state can be persisted as a side-car.
    """

    def __init__(
        self,
        gnn_dim: int = DEFAULT_GNN_DIM,
        llm_dim: int = DEFAULT_LLM_DIM,
        proj_dim: int = DEFAULT_PROJ_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        num_classes: int = DEFAULT_NUM_CLASSES,
        dropout: float = DEFAULT_DROPOUT,
        gate_temperature: float = DEFAULT_GATE_TEMPERATURE,
    ) -> None:
        super().__init__()
        self.gnn_proj = ModalityProjector(gnn_dim, proj_dim)
        self.llm_proj = ModalityProjector(llm_dim, proj_dim)
        self.gate = nn.Linear(2 * proj_dim, 2)
        self.gate_temperature = gate_temperature

        self.mlp_hidden = nn.Linear(proj_dim, hidden_dim)
        self.mlp_out = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def fuse(
        self, g_proj: torch.Tensor, l_proj: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate_logits = self.gate(torch.cat([g_proj, l_proj], dim=1))
        attn = F.softmax(gate_logits / self.gate_temperature, dim=1)
        fused = attn[:, 0:1] * g_proj + attn[:, 1:2] * l_proj
        return fused, attn

    def forward(
        self, gnn_emb: torch.Tensor, llm_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g = self.gnn_proj(gnn_emb)
        l = self.llm_proj(llm_emb)
        fused, attn = self.fuse(g, l)

        edge_emb = self.mlp_hidden(fused)
        edge_emb = F.relu(edge_emb)
        edge_emb = F.dropout(edge_emb, p=self.dropout, training=self.training)
        logits = self.mlp_out(edge_emb)
        return logits, edge_emb, attn


class AGAFFusionEdgeClassifier(nn.Module):
    """
    AGAF: Adaptive Gated Attention Fusion.

    Projects structural and semantic edge embeddings into a shared space,
    combines them with a feature-wise gate, then applies feature-wise attention
    before classification.
    """

    def __init__(
        self,
        gnn_dim: int = DEFAULT_GNN_DIM,
        llm_dim: int = DEFAULT_LLM_DIM,
        proj_dim: int = DEFAULT_PROJ_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        num_classes: int = DEFAULT_NUM_CLASSES,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.gnn_proj = ModalityProjector(gnn_dim, proj_dim)
        self.llm_proj = ModalityProjector(llm_dim, proj_dim)
        self.gate = nn.Linear(4 * proj_dim, proj_dim)
        self.feature_attention = nn.Linear(proj_dim, proj_dim)

        self.mlp_hidden = nn.Linear(proj_dim, hidden_dim)
        self.mlp_out = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def fuse(
        self, h_proj: torch.Tensor, s_proj: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate_input = torch.cat(
            [h_proj, s_proj, torch.abs(h_proj - s_proj), h_proj * s_proj], dim=1
        )
        gate = torch.sigmoid(self.gate(gate_input))
        fused = gate * h_proj + (1.0 - gate) * s_proj
        return fused, gate

    def forward(
        self, gnn_emb: torch.Tensor, llm_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        h = self.gnn_proj(gnn_emb)
        s = self.llm_proj(llm_emb)
        fused, gate = self.fuse(h, s)

        feature_attention = F.softmax(self.feature_attention(fused), dim=1)
        attended = feature_attention * fused

        edge_emb = self.mlp_hidden(attended)
        edge_emb = F.relu(edge_emb)
        edge_emb = F.dropout(edge_emb, p=self.dropout, training=self.training)
        logits = self.mlp_out(edge_emb)

        diagnostics = {
            "gate": gate,
            "feature_attention": feature_attention,
            "gate_gnn_mean": gate.mean(dim=1),
            "gate_llm_mean": 1.0 - gate.mean(dim=1),
            "feature_attention_entropy": -(
                feature_attention * (feature_attention + 1e-12).log()
            ).sum(dim=1),
        }
        return logits, edge_emb, diagnostics


class UnimodalEdgeClassifier(nn.Module):
    """
    MLP classifier for one frozen embedding modality.

    This shares the projection/head shape used by fusion so frozen GNN and LLM
    embedding baselines are comparable to the fused model.
    """

    def __init__(
        self,
        in_dim: int,
        proj_dim: int = DEFAULT_PROJ_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        num_classes: int = DEFAULT_NUM_CLASSES,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.projector = ModalityProjector(in_dim, proj_dim)
        self.mlp_hidden = nn.Linear(proj_dim, hidden_dim)
        self.mlp_out = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.projector(emb)
        edge_emb = self.mlp_hidden(h)
        edge_emb = F.relu(edge_emb)
        edge_emb = F.dropout(edge_emb, p=self.dropout, training=self.training)
        logits = self.mlp_out(edge_emb)
        return logits, edge_emb
