"""Modules for the Step 4 bidirectional GNN↔LLM feedback loop.

Each module implements one sub-phase of the loop, built and verified
independently before composition:

    Phase 4.1  UncertaintySelector        — flag high-entropy edges.
    Phase 4.3  LiveCySecBERTScorer        — live LLM re-encoding with cache.
               WhitenedPrototypeScorer    — whitened cosine → semantic logits.
    Phase 4.4  BiasedGATv2Layer           — GATv2 with additive attention bias.
               SemanticAttentionBias      — project semantic logits → bias.
    Phase 4.5  FeedbackLoopClassifier     — outer loop wrapper.

Do not compose downstream phases until the current phase's dashboard has
passed its acceptance test.
"""

from __future__ import annotations

import math
import os
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import softmax


class UncertaintySelector(nn.Module):
    """Phase 4.1 — flag edges the GNN is uncertain about.

    Uncertainty = Shannon entropy of the per-edge softmax distribution.
    The top-k% highest-entropy edges are flagged. The threshold is
    calibrated once per fold on train-fold predictions and stored in a
    persistent buffer so inference reuses it without re-quantiling.

    Parameters
    ----------
    top_k_percent : float in (0, 100)
        Fraction of edges to flag. Default 16.
    """

    def __init__(self, top_k_percent: float = 16.0) -> None:
        super().__init__()
        if not 0.0 < top_k_percent < 100.0:
            raise ValueError(
                f"top_k_percent must be in (0, 100), got {top_k_percent}"
            )
        self.top_k_percent = float(top_k_percent)
        self.register_buffer(
            "calibrated_threshold",
            torch.tensor(float("nan")),
            persistent=True,
        )

    @staticmethod
    def entropy(probs: torch.Tensor) -> torch.Tensor:
        """Shannon entropy per row, in nats.

        Parameters
        ----------
        probs : [E, C] softmax probabilities.
        """
        eps = 1e-12
        return -(probs * (probs + eps).log()).sum(dim=-1)

    def calibrate(self, probs: torch.Tensor) -> torch.Tensor:
        """Set the entropy threshold to the top-k% quantile of `probs`."""
        h = self.entropy(probs)
        q = 1.0 - self.top_k_percent / 100.0
        threshold = torch.quantile(h, q)
        self.calibrated_threshold = threshold.detach()
        return self.calibrated_threshold

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        """Return `[E]` bool mask; True where entropy exceeds threshold.

        If `calibrate` has not been called, computes the threshold on the
        fly from `probs` itself (used by dashboard sweeps).
        """
        h = self.entropy(probs)
        if torch.isnan(self.calibrated_threshold):
            q = 1.0 - self.top_k_percent / 100.0
            threshold = torch.quantile(h, q)
        else:
            threshold = self.calibrated_threshold
        return h > threshold


CYSECBERT_MODEL_ID = "markusbayer/CySecBERT"
CYSECBERT_MAX_LENGTH = 128
CYSECBERT_EMBED_DIM = 768


def _auto_device() -> torch.device:
    # Set IDS_FORCE_CPU=1 to pin every stage to CPU. MPS/CPU produce slightly
    # different floats, so cross-machine reproducibility requires one device.
    if os.environ.get("IDS_FORCE_CPU") == "1":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    return (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


class LiveCySecBERTScorer(nn.Module):
    """Phase 4.3 — live CySecBERT re-encoding with per-edge cache.

    Wraps `AutoTokenizer` + `AutoModel` for the CySecBERT checkpoint (the
    same one Step 3 precomputed embeddings from). Mean-pools the last
    hidden state, matching `src.pipeline.step3.encode_kg.encode`.

    Cache key = `(fold_index, edge_row_id)`. NL text is invariant per
    edge, so after every edge has been encoded once the cache serves all
    subsequent iterations of the loop for free.
    """

    def __init__(
        self,
        model_id: str = CYSECBERT_MODEL_ID,
        max_length: int = CYSECBERT_MAX_LENGTH,
        device: torch.device | None = None,
        hf_token: str | None = None,
    ) -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.model_id = model_id
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        self.encoder = AutoModel.from_pretrained(model_id, token=hf_token)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self._device = device or _auto_device()
        self.encoder.to(self._device)
        self._cache: dict[tuple[int, int], torch.Tensor] = {}
        self._hits = 0
        self._misses = 0

    @property
    def embed_dim(self) -> int:
        return CYSECBERT_EMBED_DIM

    @torch.no_grad()
    def _encode_batch(self, sentences: list[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self._device) for k, v in encoded.items()}
        output = self.encoder(**encoded)
        emb = _mean_pool(output.last_hidden_state, encoded["attention_mask"])
        return emb.cpu()

    def encode(
        self,
        edge_ids: Iterable[int],
        sentences: list[str],
        fold_index: int,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Return `[k, 768]` CySecBERT embeddings for the given edges.

        Uses the (fold_index, edge_id) cache; encodes only misses.
        """
        edge_ids = [int(e) for e in edge_ids]
        if len(edge_ids) != len(sentences):
            raise ValueError(
                f"len(edge_ids)={len(edge_ids)} != len(sentences)={len(sentences)}"
            )

        results = torch.zeros(len(edge_ids), self.embed_dim)
        miss_positions: list[int] = []
        miss_sentences: list[str] = []

        for pos, (eid, sent) in enumerate(zip(edge_ids, sentences)):
            key = (int(fold_index), eid)
            if key in self._cache:
                results[pos] = self._cache[key]
                self._hits += 1
            else:
                miss_positions.append(pos)
                miss_sentences.append(sent)

        for start in range(0, len(miss_sentences), batch_size):
            batch_sents = miss_sentences[start : start + batch_size]
            batch_positions = miss_positions[start : start + batch_size]
            batch_emb = self._encode_batch(batch_sents)
            for i, pos in enumerate(batch_positions):
                key = (int(fold_index), edge_ids[pos])
                self._cache[key] = batch_emb[i]
                results[pos] = batch_emb[i]
                self._misses += 1

        return results

    def cache_stats(self) -> dict[str, int]:
        return {"hits": self._hits, "misses": self._misses, "size": len(self._cache)}

    def clear_cache(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0


class WhitenedPrototypeScorer(nn.Module):
    """Phase 4.3 — whitened cosine prototype scorer.

    Consumes CySecBERT embeddings `[k, 768]`, subtracts the fold's mean,
    applies the fold's ZCA whitener, computes cosine similarity to each
    of the `C` per-class prototypes, and divides by a learnable per-class
    temperature.

    Whitener / mean / prototypes are non-parameter buffers loaded by
    `load_fold_state`. Only `log_temperature` is learned.

    Output shape: `[k, C]` — the per-class semantic logits that Phase 4.4
    projects into an attention bias.
    """

    def __init__(
        self,
        num_classes: int,
        embed_dim: int = CYSECBERT_EMBED_DIM,
        initial_log_temperature: float = math.log(10.0),
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.register_buffer("mean", torch.zeros(embed_dim), persistent=True)
        self.register_buffer(
            "whitener", torch.eye(embed_dim), persistent=True
        )
        self.register_buffer(
            "prototypes", torch.zeros(num_classes, embed_dim), persistent=True
        )
        self.log_temperature = nn.Parameter(
            torch.full((num_classes,), float(initial_log_temperature))
        )

    def load_fold_state(
        self,
        mean: torch.Tensor,
        whitener: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            self.mean.copy_(mean.to(self.mean.dtype))
            self.whitener.copy_(whitener.to(self.whitener.dtype))
            self.prototypes.copy_(prototypes.to(self.prototypes.dtype))

    def whiten(self, emb: torch.Tensor) -> torch.Tensor:
        return (emb - self.mean) @ self.whitener

    def cosine(self, whitened_emb: torch.Tensor) -> torch.Tensor:
        z = F.normalize(whitened_emb, dim=-1)
        p = F.normalize(self.prototypes, dim=-1)
        return z @ p.T

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        z = self.whiten(emb)
        cos = self.cosine(z)
        temperature = torch.exp(self.log_temperature).clamp(min=1e-4)
        return cos / temperature.unsqueeze(0)


class SemanticAttentionBias(nn.Module):
    """Phase 4.4 — project per-class semantic logits into an edge-level
    attention bias (in nats).

    Consumes the `[k, C]` semantic logits from `WhitenedPrototypeScorer`
    for the `k` flagged edges and produces a full-graph bias tensor of
    shape `[E, bias_dim]`, exactly zero on the `E - k` unflagged edges.
    `BiasedGATv2Conv` adds this bias to GATv2's *unnormalised* attention
    logits, before the per-edge softmax — the true "attention bias" the
    spec calls for, not an `edge_attr` concat.

    `bias_dim == 1` (default) broadcasts one bias value across all heads,
    which keeps a single module usable by GAT layers with different head
    counts. `bias_dim == num_heads` would give a per-head bias.

    The projection is zero-initialised so a fresh model's first forward
    pass reproduces the unbiased GATv2 output exactly — this is both the
    Phase 4.4 equivalence invariant and the Phase 4.5 cold start. A
    learnable scalar `log_bias_strength` scales the projected bias; it
    starts at `log(1.5)` so that once the projection moves off zero the
    effective bias lands on the ~1.5-nat order the spec specifies.
    """

    def __init__(
        self,
        num_classes: int,
        bias_dim: int = 1,
        initial_log_bias_strength: float = math.log(1.5),
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.bias_dim = bias_dim
        self.projection = nn.Linear(num_classes, bias_dim)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self.log_bias_strength = nn.Parameter(
            torch.tensor(float(initial_log_bias_strength))
        )

    def forward(
        self,
        semantic_logits: torch.Tensor,
        flagged_edge_indices: torch.Tensor,
        num_edges: int,
    ) -> torch.Tensor:
        strength = torch.exp(self.log_bias_strength)
        flagged_bias = self.projection(semantic_logits) * strength
        full_bias = torch.zeros(
            num_edges,
            self.bias_dim,
            dtype=flagged_bias.dtype,
            device=flagged_bias.device,
        )
        if flagged_edge_indices.numel() > 0:
            full_bias[flagged_edge_indices] = flagged_bias
        return full_bias


class BiasedGATv2Conv(GATv2Conv):
    """GATv2 that adds a per-edge bias to the attention logits *before*
    the per-edge softmax.

    `edge_update` is copied verbatim from PyG 2.7.0's `GATv2Conv` with a
    single added line: the pending `edge_attn_bias` is summed into `alpha`
    before `softmax`. `edge_attr` keeps its original dimension — the
    semantic signal enters only through the bias, never through a concat.

    `edge_attn_bias` may be `[E]`, `[E, 1]` (broadcast over heads) or
    `[E, heads]`. With `edge_attn_bias=None` or an all-zero bias the
    output is bit-for-bit the stock `GATv2Conv` (see
    `tests/test_biased_gat.py`).

    Requires `add_self_loops=False` when a bias is supplied: self-loops
    would append rows to the attention tensor and break edge alignment.
    """

    _pending_bias: torch.Tensor | None = None

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        edge_attn_bias: torch.Tensor | None = None,
        return_attention_weights=None,
    ):
        if edge_attn_bias is not None and self.add_self_loops:
            raise ValueError(
                "BiasedGATv2Conv requires add_self_loops=False when "
                "edge_attn_bias is supplied (self-loops break edge alignment)."
            )
        self._pending_bias = edge_attn_bias
        try:
            return super().forward(
                x,
                edge_index,
                edge_attr=edge_attr,
                return_attention_weights=return_attention_weights,
            )
        finally:
            self._pending_bias = None

    def edge_update(  # type: ignore[override]
        self,
        x_j: torch.Tensor,
        x_i: torch.Tensor,
        edge_attr: torch.Tensor | None,
        index: torch.Tensor,
        ptr: torch.Tensor | None,
        dim_size: int | None,
    ) -> torch.Tensor:
        x = x_i + x_j

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x = x + edge_attr

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        bias = self._pending_bias
        if bias is not None:
            if bias.dim() == 1:
                bias = bias.unsqueeze(-1)
            alpha = alpha + bias  # <-- the only change vs. stock GATv2Conv
        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha


class BiasedGATv2Layer(nn.Module):
    """Phase 4.4 — thin wrapper around `BiasedGATv2Conv`.

    `edge_attr` keeps its original `edge_attr_dim` (no concat). The
    semantic signal enters only through `edge_attn_bias`, which
    `BiasedGATv2Conv` adds to the attention logits before softmax.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_attr_dim: int,
        bias_dim: int = 1,
        heads: int = 8,
        concat: bool = True,
        dropout: float = 0.0,
        add_self_loops: bool = False,
    ) -> None:
        super().__init__()
        self.edge_attr_dim = edge_attr_dim
        self.bias_dim = bias_dim
        self.heads = heads
        self.gat = BiasedGATv2Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=concat,
            edge_dim=edge_attr_dim,
            dropout=dropout,
            add_self_loops=add_self_loops,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_attn_bias: torch.Tensor,
        return_attention_weights: bool = False,
    ):
        if return_attention_weights:
            return self.gat(
                x,
                edge_index,
                edge_attr=edge_attr,
                edge_attn_bias=edge_attn_bias,
                return_attention_weights=True,
            )
        return self.gat(
            x, edge_index, edge_attr=edge_attr, edge_attn_bias=edge_attn_bias
        )


class BiasedGATStructuralEncoder(nn.Module):
    """Structural encoder mirroring `gnn_classifier.GATStructuralEncoder`,
    but built from `BiasedGATv2Layer` so an `edge_attn_bias` can be threaded
    through both message-passing rounds.

    With `edge_attn_bias = 0` this is numerically identical to the Step-3
    encoder — the feedback loop's iter-1 cold start therefore reproduces
    the plain GNN.
    """

    def __init__(
        self,
        in_dim: int = 10,
        hidden_dim: int = 64,
        edge_attr_dim: int = 5,
        heads: int = 8,
        dropout: float = 0.2,
        bias_dim: int = 1,
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})"
            )
        self.dropout = dropout
        self.gat1 = BiasedGATv2Layer(
            in_dim,
            hidden_dim // heads,
            edge_attr_dim,
            bias_dim=bias_dim,
            heads=heads,
            concat=True,
            add_self_loops=False,
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.res1 = nn.Linear(in_dim, hidden_dim)
        self.gat2 = BiasedGATv2Layer(
            hidden_dim,
            hidden_dim,
            edge_attr_dim,
            bias_dim=bias_dim,
            heads=1,
            concat=False,
            add_self_loops=False,
        )
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_attn_bias: torch.Tensor,
    ) -> torch.Tensor:
        h = self.gat1(x, edge_index, edge_attr, edge_attn_bias)
        h = self.bn1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + self.res1(x)

        h_in = h
        h = self.gat2(h, edge_index, edge_attr, edge_attn_bias)
        h = self.bn2(h)
        h = F.elu(h)
        h = h + h_in
        return h


FEEDBACK_MODES = ("real", "random", "head_only")


class FeedbackLoopClassifier(nn.Module):
    """Phase 4.5 — the bidirectional GNN↔LLM feedback loop.

    One iteration:
      1. run the biased GAT (iter 1 uses `edge_attn_bias = 0`);
      2. per-edge softmax → entropy → top-k% `uncertain_mask` (Phase 4.1);
      3. take the LLM's `[E, C]` semantic logits (Phase 4.3), keep only the
         flagged rows;
      4. project them to an attention bias (Phase 4.4) for the next round.
    Repeat until the predicted-class churn drops below `churn_tol` or
    `max_iterations` is reached.

    The three-variant (`temporal`/`content`/`behavioral`) fusion mirrors
    `gnn_classifier.GATEdgeClassifier` so capacity is comparable to the
    Step-3 baseline. The LLM embeddings are supplied to `forward`; they are
    fixed inputs, so the per-class semantic logits are computed once per
    forward and only the *flagged subset* changes across iterations.

    `feedback_mode`:
      - ``"real"``      — semantic logits from `WhitenedPrototypeScorer`;
      - ``"random"``    — semantic logits replaced by fixed per-edge noise
        (the random-feedback ablation the reverted attempt failed);
      - ``"head_only"`` — bias forced to zero every iteration (the loop is
        disabled; equivalent to the plain GNN run `max_iterations` times).
    """

    def __init__(
        self,
        in_dim: int = 10,
        hidden_dim: int = 64,
        edge_attr_dim: int = 5,
        num_classes: int = 10,
        heads: int = 8,
        dropout: float = 0.2,
        embed_dim: int = CYSECBERT_EMBED_DIM,
        top_k_percent: float = 16.0,
        max_iterations: int = 3,
        churn_tol: float = 0.01,
        bias_dim: int = 1,
        random_feedback_seed: int = 12345,
        bias_confidence_frac: float = 1.0,
        use_output_fusion: bool = True,
    ) -> None:
        super().__init__()
        from src.models.gnn_classifier import EDGE_FOCUS_WEIGHTS, VARIANT_NAMES

        self.variant_names = VARIANT_NAMES
        self.hidden_dim = hidden_dim
        self.edge_attr_dim = edge_attr_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.max_iterations = max_iterations
        self.churn_tol = churn_tol
        self.random_feedback_seed = random_feedback_seed
        self.bias_confidence_frac = bias_confidence_frac
        self.use_output_fusion = use_output_fusion

        self.encoders = nn.ModuleDict(
            {
                name: BiasedGATStructuralEncoder(
                    in_dim=in_dim,
                    hidden_dim=hidden_dim,
                    edge_attr_dim=edge_attr_dim,
                    heads=heads,
                    dropout=dropout,
                    bias_dim=bias_dim,
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

        self.selector = UncertaintySelector(top_k_percent=top_k_percent)
        self.scorer = WhitenedPrototypeScorer(
            num_classes=num_classes, embed_dim=embed_dim
        )
        self.bias_module = SemanticAttentionBias(num_classes, bias_dim=bias_dim)

        # Decision-level fusion (AGAF-capacity, symmetric). A learned head
        # over the FULL LLM embedding gives an LLM branch as strong as
        # AGAF's, and a learned per-edge gate mixes it with the loop-refined
        # GNN branch — so the fusion can defer entirely to whichever branch
        # is more reliable on each edge, then the loop's attention refinement
        # pushes above a static fusion. LayerNorm puts both logit vectors on
        # the same scale before mixing.
        self.embed_dim = embed_dim
        fusion_dim = hidden_dim
        # Embedding-level fusion (AGAF-capacity, but the loop's OWN, and the
        # GNN branch is the end-to-end-trained, LLM-refined edge embedding —
        # not a frozen precomputed one like AGAF's). A per-edge gate driven by
        # GNN uncertainty decides how much of the LLM branch to admit.
        self.fusion_gnn_proj = nn.Linear(hidden_dim, fusion_dim)
        self.fusion_llm_proj = nn.Linear(embed_dim, fusion_dim)
        # Projection for a trained LLM head's class logits (strong consultant).
        # When head logits are supplied the loop fuses this instead of the raw
        # CySecBERT embedding, so it leverages an already-strong LLM classifier
        # rather than re-learning one from frozen embeddings.
        self.fusion_head_proj = nn.Linear(num_classes, fusion_dim)
        self.fusion_gate = nn.Linear(2, 1)  # from [gnn_entropy, gnn_confidence]
        # Confidence-aware router (GLANCE-style): sees BOTH modalities' entropy
        # and confidence so it can route each edge to whichever is more reliable.
        # Used when a trained LLM head is supplied.
        self.fusion_gate_conf = nn.Linear(4, 1)
        self.fusion_classifier = nn.Sequential(
            nn.Linear(fusion_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def load_fold_state(self, mean, whitener, prototypes) -> None:
        self.scorer.load_fold_state(mean, whitener, prototypes)

    def _backbone_modules(self) -> list[nn.Module]:
        """The GAT baseline: everything that produces the graph-only logits.
        The feedback modules (scorer temperature, attention-bias projection,
        fusion head) are excluded — those stay trainable in frozen mode."""
        return [
            self.encoders,
            self.variant_heads,
            self.variant_embedders,
            self.fusion_hidden,
            self.fusion_out,
        ]

    def freeze_backbone(self) -> None:
        """Lock the GAT baseline so only the feedback modules learn on top.
        This makes the loop a residual correction over one canonical, frozen
        GNN — the same weights used as the standalone GNN baseline."""
        for module in self._backbone_modules():
            for p in module.parameters():
                p.requires_grad_(False)

    def backbone_eval(self) -> None:
        """Put the frozen backbone (incl. its BatchNorm/Dropout) in eval mode
        so its running stats don't drift while the correction trains."""
        for module in self._backbone_modules():
            module.eval()

    def _focused_edge_attr(self, name: str, edge_attr: torch.Tensor) -> torch.Tensor:
        focus = getattr(self, f"{name}_edge_focus")
        return edge_attr * focus.to(dtype=edge_attr.dtype, device=edge_attr.device)

    @staticmethod
    def _build_flow_repr(node_emb, edge_index, edge_attr) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        src_h = node_emb[src]
        dst_h = node_emb[dst]
        return torch.cat(
            [src_h, dst_h, src_h * dst_h, torch.abs(src_h - dst_h), edge_attr],
            dim=1,
        )

    def _one_pass(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_attn_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        variant_embs = []
        aux_logits: dict[str, torch.Tensor] = {}
        for name in self.variant_names:
            focused = self._focused_edge_attr(name, edge_attr)
            node_emb = self.encoders[name](x, edge_index, focused, edge_attn_bias)
            repr_ = self._build_flow_repr(node_emb, edge_index, focused)
            emb = F.relu(self.variant_embedders[name](repr_))
            variant_embs.append(emb)
            aux_logits[name] = self.variant_heads[name](repr_)

        edge_emb = F.relu(self.fusion_hidden(torch.cat(variant_embs, dim=1)))
        edge_emb = F.dropout(edge_emb, p=self.dropout, training=self.training)
        logits = self.fusion_out(edge_emb)
        return logits, aux_logits, edge_emb

    def _output_fusion(
        self,
        edge_emb: torch.Tensor,
        gnn_logits: torch.Tensor,
        llm_embeddings: torch.Tensor | None,
        head_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embedding-level, uncertainty-gated fusion of the loop's GNN branch
        and the LLM branch.

        g_e   = sigmoid(gate([H_gnn,e, conf_gnn,e]))     # per-edge, opens when
                                                          # the GNN is unsure
        fused = concat[(1−g_e)·proj_gnn(edge_emb), g_e·proj_llm(llm_emb)]
        final = classifier(fused)

        The GNN branch is the end-to-end-trained, LLM-refined edge embedding
        (distinct from AGAF's frozen precomputed one); the gate routes more of
        the LLM branch onto exactly the edges the GNN is uncertain about.
        """
        if not self.use_output_fusion or (llm_embeddings is None and head_logits is None):
            return gnn_logits
        probs = gnn_logits.softmax(dim=-1)
        h_gnn = self.selector.entropy(probs)
        conf_gnn = probs.max(dim=-1).values
        gnn_p = self.fusion_gnn_proj(edge_emb)
        # Prefer the trained LLM head's class logits (strong) over raw embeddings.
        if head_logits is not None:
            lprobs = head_logits.softmax(dim=-1)
            h_llm = self.selector.entropy(lprobs)
            conf_llm = lprobs.max(dim=-1).values
            g = torch.sigmoid(self.fusion_gate_conf(
                torch.stack([h_gnn, conf_gnn, h_llm, conf_llm], dim=-1)))
            llm_p = self.fusion_head_proj(head_logits)
        else:
            g = torch.sigmoid(self.fusion_gate(torch.stack([h_gnn, conf_gnn], dim=-1)))
            llm_p = self.fusion_llm_proj(llm_embeddings)
        fused = torch.cat([(1.0 - g) * gnn_p, g * llm_p], dim=-1)
        logits = self.fusion_classifier(fused)
        # Direct confidence-routed residual over the two strong classifiers —
        # pushes the fused output toward the per-edge oracle (route to whichever
        # modality is confident). Only when a trained LLM head is available.
        if head_logits is not None:
            logits = logits + (1.0 - g) * gnn_logits + g * head_logits
        return logits

    def _gate_by_confidence(
        self, flagged: torch.Tensor, semantic_logits: torch.Tensor
    ) -> torch.Tensor:
        """Keep only the most-confident flagged edges (top
        `bias_confidence_frac` by the LLM's max class probability).

        A weak consultant verdict is worse than none — biasing attention on
        edges the LLM is unsure about was regressing Generic/Reconnaissance
        and letting the loop overfit. Restricting to confident verdicts
        keeps the high-value nudges and drops the noisy ones.
        """
        if self.bias_confidence_frac >= 1.0 or flagged.numel() == 0:
            return flagged
        conf = semantic_logits[flagged].softmax(dim=-1).max(dim=-1).values
        k = max(1, int(round(self.bias_confidence_frac * flagged.numel())))
        top = torch.topk(conf, k).indices
        return flagged[top]

    def _semantic_logits(
        self,
        llm_embeddings: torch.Tensor,
        feedback_mode: str,
        generator: torch.Generator | None = None,
        head_logits: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if feedback_mode == "head_only":
            return None
        if feedback_mode == "real":
            # Prefer a strong trained LLM head over the whitened-prototype scorer.
            return head_logits if head_logits is not None else self.scorer(llm_embeddings)
        if feedback_mode == "random":
            # Fixed per-edge noise: sampled with a constant seed so it is
            # identical on every forward (train and eval) — a fixed
            # alternative "consultant", the fair control for real feedback.
            # (Fresh-per-step noise would instead act as an attention-noise
            # regulariser, which is not what this ablation is testing.)
            e = llm_embeddings.shape[0]
            g = generator or torch.Generator(device=llm_embeddings.device).manual_seed(
                self.random_feedback_seed
            )
            noise = torch.rand(
                e, self.num_classes, generator=g, device=llm_embeddings.device
            )
            probs = noise / noise.sum(dim=-1, keepdim=True)
            return (probs + 1e-9).log()
        raise ValueError(f"Unknown feedback_mode: {feedback_mode!r}")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        llm_embeddings: torch.Tensor,
        feedback_mode: str = "real",
        collect_trace: bool = False,
        generator: torch.Generator | None = None,
        head_logits: torch.Tensor | None = None,
    ):
        num_edges = edge_index.shape[1]
        bias = torch.zeros(
            num_edges, self.bias_module.bias_dim, device=x.device, dtype=x.dtype
        )
        semantic_logits = self._semantic_logits(
            llm_embeddings, feedback_mode, generator, head_logits
        )

        logits: torch.Tensor
        aux_logits: dict[str, torch.Tensor] = {}
        prev_pred: torch.Tensor | None = None
        trace: list[dict[str, float]] = []

        edge_emb: torch.Tensor | None = None
        for it in range(self.max_iterations):
            logits, aux_logits, edge_emb = self._one_pass(x, edge_index, edge_attr, bias)
            probs = logits.softmax(dim=-1)
            pred = probs.argmax(dim=-1)

            churn = float("nan")
            if prev_pred is not None:
                churn = float((pred != prev_pred).float().mean())
            if collect_trace:
                ent = self.selector.entropy(probs.detach())
                trace.append(
                    {
                        "iter": it + 1,
                        "churn": churn,
                        "mean_entropy": float(ent.mean()),
                        "logits": logits.detach().clone(),
                    }
                )
            prev_pred = pred

            if prev_pred is not None and not math.isnan(churn) and churn < self.churn_tol:
                break
            if semantic_logits is None or it == self.max_iterations - 1:
                continue

            mask = self.selector(probs)
            flagged = mask.nonzero(as_tuple=False).squeeze(-1)
            flagged = self._gate_by_confidence(flagged, semantic_logits)
            bias = self.bias_module(semantic_logits[flagged], flagged, num_edges)

        fusion_head: torch.Tensor | None = None
        if semantic_logits is None:
            fusion_emb = None  # head_only: fusion disabled
        elif feedback_mode == "random":
            # shuffle edges → real-looking but uninformative LLM branch (control)
            perm = torch.randperm(
                num_edges,
                generator=torch.Generator().manual_seed(self.random_feedback_seed + 7),
            )
            fusion_emb = llm_embeddings[perm]
            if head_logits is not None:
                fusion_head = head_logits[perm]
        else:
            fusion_emb = llm_embeddings
            fusion_head = head_logits
        logits = self._output_fusion(edge_emb, logits, fusion_emb, fusion_head)

        if collect_trace:
            if trace:
                trace[-1]["logits"] = logits.detach().clone()
            return logits, aux_logits, trace
        return logits, aux_logits
