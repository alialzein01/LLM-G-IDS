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

# Fusion mechanisms surveyed in docs/, all reduced to a common interface so they
# can be swapped on the same folds:
#   feature_gate — the existing AGAF gate (GMU-style, dimension-wise sigmoid)
#   concat       — GMLM: no gate, no attention; concat then MLP
#   fixed        — BertGCN: one blend weight, identical for every edge
#   scalar       — Moorthy et al.: one learned weight per modality per edge
#   selfattn     — RAGFormer: the two embeddings as a 2-token sequence
FUSION_MODES = ("feature_gate", "concat", "fixed", "scalar", "selfattn")


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
        head_fusion: bool = False,
        fusion_mode: str = "feature_gate",
        fixed_lambda: float = 0.5,
    ) -> None:
        super().__init__()
        if fusion_mode not in FUSION_MODES:
            raise ValueError(f"fusion_mode must be one of {FUSION_MODES}, got {fusion_mode!r}")
        self.fusion_mode = fusion_mode
        self.fixed_lambda = fixed_lambda
        self.proj_dim = proj_dim

        self.gnn_proj = ModalityProjector(gnn_dim, proj_dim)
        self.llm_proj = ModalityProjector(llm_dim, proj_dim)

        # Only the mechanism in use is allocated, so parameter counts stay
        # honest when variants are compared on a 1275-edge training split.
        if fusion_mode == "feature_gate":
            self.gate = nn.Linear(4 * proj_dim, proj_dim)
        elif fusion_mode == "concat":
            self.concat_proj = nn.Linear(2 * proj_dim, proj_dim)
        elif fusion_mode == "scalar":
            # Shared scorer applied to each modality separately, softmaxed over
            # the two — one interpretable weight per edge, per modality.
            self.modality_score = nn.Sequential(
                nn.Linear(proj_dim, proj_dim // 4),
                nn.Tanh(),
                nn.Linear(proj_dim // 4, 1),
            )
        elif fusion_mode == "selfattn":
            self.modality_attn = nn.MultiheadAttention(
                proj_dim, num_heads=4, batch_first=True
            )

        self.feature_attention = nn.Linear(proj_dim, proj_dim)

        self.mlp_hidden = nn.Linear(proj_dim, hidden_dim)
        self.mlp_out = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

        # M3: confidence-gated late fusion of the trained LLM head logits. When
        # enabled, the semantic modality is fed as the LLM head's per-fold OOF
        # logits (llm_dim == num_classes) and those logits are re-injected at the
        # output through a learned per-edge gate, so the fusion's floor is the
        # LLM head itself and the GNN branch only has to add lift. Mirrors the
        # output fusion the feedback loop already uses.
        self.head_fusion = head_fusion
        self.num_classes = num_classes
        if head_fusion:
            # Per-class, per-example gate over the two views of an edge: the fused
            # GNN+semantic correction and the LLM head's own verdict. Its inputs
            # are the fused edge embedding and the head logits, matching what
            # forward() passes in.
            self.head_gate = nn.Linear(hidden_dim + num_classes, num_classes)
            # Init so the gate starts ~0.95 on the LLM head (its floor) and only
            # shifts mass to the fused view where that reduces loss.
            nn.init.zeros_(self.head_gate.weight)
            nn.init.constant_(self.head_gate.bias, 3.0)

    def fuse(
        self, h_proj: torch.Tensor, s_proj: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return (fused, gate). `gate` is always [E, proj_dim] and always reads as
        "share given to the GNN branch", so the per-class diagnostics and the
        pooled artifacts mean the same thing across every mechanism.
        """
        mode = self.fusion_mode

        if mode == "feature_gate":
            gate_input = torch.cat(
                [h_proj, s_proj, torch.abs(h_proj - s_proj), h_proj * s_proj], dim=1
            )
            gate = torch.sigmoid(self.gate(gate_input))
            fused = gate * h_proj + (1.0 - gate) * s_proj
            return fused, gate

        if mode == "concat":
            # GMLM: both modalities pass through whole; nothing decides a share.
            fused = F.relu(self.concat_proj(torch.cat([h_proj, s_proj], dim=1)))
            gate = torch.full_like(h_proj, 0.5)
            return fused, gate

        if mode == "fixed":
            # BertGCN: one blend weight, the same for every edge in the dataset.
            gate = torch.full_like(h_proj, self.fixed_lambda)
            fused = self.fixed_lambda * h_proj + (1.0 - self.fixed_lambda) * s_proj
            return fused, gate

        if mode == "scalar":
            # Moorthy et al.: score each modality, softmax over the two.
            scores = torch.cat(
                [self.modality_score(h_proj), self.modality_score(s_proj)], dim=1
            )
            alpha = F.softmax(scores, dim=1)
            fused = alpha[:, 0:1] * h_proj + alpha[:, 1:2] * s_proj
            gate = alpha[:, 0:1].expand(-1, self.proj_dim)
            return fused, gate

        # RAGFormer: two modality tokens, self-attention, residual.
        seq = torch.stack([h_proj, s_proj], dim=1)
        attended, weights = self.modality_attn(seq, seq, seq)
        seq = seq + attended
        fused = seq.mean(dim=1)
        # Attention mass the GNN token receives, averaged over queries.
        gate = weights[:, :, 0].mean(dim=1, keepdim=True).expand(-1, self.proj_dim)
        return fused, gate

    def forward(
        self,
        gnn_emb: torch.Tensor,
        llm_emb: torch.Tensor,
        head_logits: torch.Tensor | None = None,
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
            # Kept for the optional InfoNCE alignment term, which needs the two
            # projected views of the same edge before they are combined.
            "h_proj": h,
            "s_proj": s,
            "gate_gnn_mean": gate.mean(dim=1),
            "gate_llm_mean": 1.0 - gate.mean(dim=1),
            "feature_attention_entropy": -(
                feature_attention * (feature_attention + 1e-12).log()
            ).sum(dim=1),
        }

        # M3 late fusion: blend the GNN-informed logits with the LLM head's own
        # verdict via a learned per-edge gate g in [0, 1]. g -> 1 recovers the
        # LLM head exactly (the floor); g -> 0 trusts the fused GNN+semantic head.
        if self.head_fusion and head_logits is not None:
            g = torch.sigmoid(self.head_gate(torch.cat([edge_emb, head_logits], dim=1)))
            logits = (1.0 - g) * logits + g * head_logits
            diagnostics["head_gate_mean"] = g.mean(dim=1)

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
