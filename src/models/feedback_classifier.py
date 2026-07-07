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
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintySelector(nn.Module):
    """Phase 4.1 — flag edges the GNN is uncertain about.

    Uncertainty = Shannon entropy of the per-edge softmax distribution.
    The top-k% highest-entropy edges are flagged. The threshold is
    calibrated once per fold on train-fold predictions and stored in a
    persistent buffer so inference reuses it without re-quantiling.

    Parameters
    ----------
    top_k_percent : float in (0, 100)
        Fraction of edges to flag. Default 30.
    """

    def __init__(self, top_k_percent: float = 30.0) -> None:
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
    """Phase 4.4 — project per-class semantic logits into an edge-level bias.

    Consumes the `[k, C]` semantic logits from `WhitenedPrototypeScorer`
    for the `k` flagged edges and produces a full-graph bias tensor of
    shape `[E, bias_dim]` where the `E - k` unflagged edges are exact
    zero. The bias is fed to `BiasedGATv2Layer` as an extra channel of
    `edge_attr`.

    The projection is zero-initialised so the very first forward pass of
    a fresh model reproduces the unbiased GATv2 output exactly. This is
    the invariant the Phase 4.4 dashboard's acceptance test (c) relies
    on.
    """

    def __init__(self, num_classes: int, bias_dim: int = 1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.bias_dim = bias_dim
        self.projection = nn.Linear(num_classes, bias_dim)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(
        self,
        semantic_logits: torch.Tensor,
        flagged_edge_indices: torch.Tensor,
        num_edges: int,
    ) -> torch.Tensor:
        flagged_bias = self.projection(semantic_logits)
        full_bias = torch.zeros(
            num_edges,
            self.bias_dim,
            dtype=flagged_bias.dtype,
            device=flagged_bias.device,
        )
        if flagged_edge_indices.numel() > 0:
            full_bias[flagged_edge_indices] = flagged_bias
        return full_bias


class BiasedGATv2Layer(nn.Module):
    """Phase 4.4 — GATv2 layer that consumes a semantic-bias channel.

    Wraps `torch_geometric.nn.GATv2Conv` and augments `edge_attr` by
    concatenating a `bias_dim` channel produced by `SemanticAttentionBias`.
    The augmented tensor `[E, edge_attr_dim + bias_dim]` is passed to
    GATv2 with `edge_dim` set accordingly.

    This is the "edge_attr perturbation" injection path (option 2 in the
    Phase 4.4 spec). It does not modify the attention softmax directly;
    it changes the `edge_attr` GATv2 already reads. Whether that channel
    actually shifts attention weights on flagged edges is measured by
    Phase 4.4's dashboard.
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
        from torch_geometric.nn import GATv2Conv

        self.edge_attr_dim = edge_attr_dim
        self.bias_dim = bias_dim
        self.heads = heads
        self.gat = GATv2Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=concat,
            edge_dim=edge_attr_dim + bias_dim,
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
        augmented = torch.cat([edge_attr, edge_attn_bias], dim=-1)
        if return_attention_weights:
            return self.gat(
                x, edge_index, augmented, return_attention_weights=True
            )
        return self.gat(x, edge_index, augmented)
