from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from src.models.gnn_classifier import (
    DEFAULT_DROPOUT,
    DEFAULT_EDGE_ATTR_DIM,
    DEFAULT_HEADS,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_IN_DIM,
    DEFAULT_NUM_CLASSES,
    EDGE_FOCUS_WEIGHTS,
    VARIANT_NAMES,
    GATEdgeClassifier,
)

DEFAULT_FEEDBACK_DIM = 8
DEFAULT_LLM_DIM = 768
DEFAULT_UNCERTAIN_FRACTION = 0.30
DEFAULT_PROTOTYPE_TEMPERATURE = 10.0
DEFAULT_CHURN_TOL = 0.005
DEFAULT_ENTROPY_TOL = 1e-3
DEFAULT_MAX_ITERATIONS = 3
DEFAULT_SPARSE_LAMBDA = 1e-3
DEFAULT_ALPHA_INIT = 0.10


class UncertaintyDetector(nn.Module):
    """
    Select edges whose current prediction is uncertain.

    Entropy is a measure of how spread out a probability distribution is. If a
    model gives one class probability near 1.0, entropy is low. If it spreads
    probability across many classes, entropy is high.
    """

    def __init__(
        self,
        num_classes: int = DEFAULT_NUM_CLASSES,
        uncertain_fraction: float = DEFAULT_UNCERTAIN_FRACTION,
    ) -> None:
        super().__init__()
        if not 0.0 <= uncertain_fraction <= 1.0:
            raise ValueError("uncertain_fraction must be in [0, 1]")
        self.num_classes = num_classes
        self.uncertain_fraction = uncertain_fraction

    def entropy(self, logits: torch.Tensor) -> torch.Tensor:
        probs = logits.softmax(dim=1)
        log_probs = logits.log_softmax(dim=1)
        return -(probs * log_probs).sum(dim=1) / math.log(self.num_classes)

    def class_stratified_topp(
        self, entropy: torch.Tensor, predicted_class: torch.Tensor
    ) -> torch.Tensor:
        mask = torch.zeros_like(entropy, dtype=torch.bool)
        if self.uncertain_fraction <= 0.0 or entropy.numel() == 0:
            return mask

        for class_idx in range(self.num_classes):
            class_mask = predicted_class == class_idx
            class_count = int(class_mask.sum().item())
            if class_count == 0:
                continue

            k = max(1, math.ceil(class_count * self.uncertain_fraction))
            class_positions = class_mask.nonzero(as_tuple=False).flatten()
            class_entropy = entropy[class_positions]
            selected = torch.topk(class_entropy, k=k, largest=True).indices
            mask[class_positions[selected]] = True

        return mask

    def random_class_stratified_topp(
        self,
        entropy: torch.Tensor,
        predicted_class: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        mask = torch.zeros_like(entropy, dtype=torch.bool)
        if self.uncertain_fraction <= 0.0 or entropy.numel() == 0:
            return mask

        for class_idx in range(self.num_classes):
            class_mask = predicted_class == class_idx
            class_count = int(class_mask.sum().item())
            if class_count == 0:
                continue

            k = max(1, math.ceil(class_count * self.uncertain_fraction))
            class_positions = class_mask.nonzero(as_tuple=False).flatten()
            order = torch.randperm(
                class_count,
                generator=generator,
                device=class_positions.device,
            )
            mask[class_positions[order[:k]]] = True

        return mask

    def forward(
        self,
        gnn_logits: torch.Tensor,
        *,
        selection_mode: str = "entropy",
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if selection_mode not in {"entropy", "random"}:
            raise ValueError("selection_mode must be 'entropy' or 'random'")
        entropy = self.entropy(gnn_logits)
        predicted_class = gnn_logits.argmax(dim=1)
        if selection_mode == "entropy":
            uncertain_mask = self.class_stratified_topp(entropy, predicted_class)
        else:
            uncertain_mask = self.random_class_stratified_topp(
                entropy, predicted_class, generator=generator
            )
        return entropy, uncertain_mask


class SemanticFeedbackScorer(nn.Module):
    """
    Convert frozen LLM embeddings into class-level semantic feedback.

    A prototype is the average embedding for one class, built from train-fold
    edges only. This module compares each uncertain edge to those prototypes and
    returns a soft class vote.
    """

    def __init__(
        self,
        temperature: float = DEFAULT_PROTOTYPE_TEMPERATURE,
        num_classes: int = DEFAULT_NUM_CLASSES,
        pooling: str = "max",
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if pooling not in {"max", "logsumexp"}:
            raise ValueError("pooling must be 'max' or 'logsumexp'")
        self.temperature = temperature
        self.num_classes = num_classes
        self.pooling = pooling

    @staticmethod
    def build_prototypes(
        llm_emb: torch.Tensor,
        labels: torch.Tensor,
        train_mask: torch.Tensor,
        num_classes: int = DEFAULT_NUM_CLASSES,
    ) -> torch.Tensor:
        prototypes = []
        normalized = F.normalize(llm_emb, dim=1)
        for class_idx in range(num_classes):
            class_mask = train_mask & (labels == class_idx)
            if bool(class_mask.any()):
                prototype = normalized[class_mask].mean(dim=0)
            else:
                prototype = normalized[train_mask].mean(dim=0)
            prototypes.append(F.normalize(prototype, dim=0))
        return torch.stack(prototypes, dim=0)

    def _unpack_prototypes(
        self, prototypes: torch.Tensor | dict[str, torch.Tensor | str]
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        if isinstance(prototypes, torch.Tensor):
            prototype_classes = torch.arange(
                prototypes.size(0), device=prototypes.device, dtype=torch.long
            )
            return prototypes, prototype_classes, self.pooling

        prototype_tensor = prototypes["prototypes"]
        prototype_classes = prototypes.get("prototype_classes")
        pooling = prototypes.get("pooling", self.pooling)
        if not isinstance(prototype_tensor, torch.Tensor):
            raise TypeError("prototypes['prototypes'] must be a tensor")
        if prototype_classes is None:
            prototype_classes = torch.arange(
                prototype_tensor.size(0), device=prototype_tensor.device, dtype=torch.long
            )
        if not isinstance(prototype_classes, torch.Tensor):
            raise TypeError("prototypes['prototype_classes'] must be a tensor")
        if not isinstance(pooling, str):
            raise TypeError("prototypes['pooling'] must be a string")
        if pooling not in {"max", "logsumexp"}:
            raise ValueError("pooling must be 'max' or 'logsumexp'")
        return prototype_tensor, prototype_classes.long(), pooling

    def _aggregate_to_classes(
        self,
        prototype_logits: torch.Tensor,
        prototype_classes: torch.Tensor,
        pooling: str,
    ) -> torch.Tensor:
        class_logits = prototype_logits.new_full(
            (prototype_logits.size(0), self.num_classes), -torch.inf
        )
        for class_idx in range(self.num_classes):
            class_mask = prototype_classes == class_idx
            if not bool(class_mask.any()):
                continue
            class_values = prototype_logits[:, class_mask]
            if pooling == "max":
                class_logits[:, class_idx] = class_values.max(dim=1).values
            else:
                class_logits[:, class_idx] = torch.logsumexp(class_values, dim=1)
        return class_logits

    def forward(
        self,
        llm_emb_subset: torch.Tensor,
        prototypes: torch.Tensor | dict[str, torch.Tensor | str],
    ) -> torch.Tensor:
        prototype_tensor, prototype_classes, pooling = self._unpack_prototypes(prototypes)
        if llm_emb_subset.numel() == 0:
            return llm_emb_subset.new_zeros((0, self.num_classes))

        llm_norm = F.normalize(llm_emb_subset, dim=1)
        proto_norm = F.normalize(prototype_tensor, dim=1).to(llm_norm.device)
        prototype_classes = prototype_classes.to(llm_norm.device)
        prototype_logits = self.temperature * (llm_norm @ proto_norm.T)
        class_logits = self._aggregate_to_classes(
            prototype_logits, prototype_classes, pooling
        )
        return F.softmax(class_logits, dim=1)


class AttentionBiasGenerator(nn.Module):
    """
    Project semantic feedback into extra edge channels for the biased GAT.

    Confident edges receive exact zeros, so feedback only changes the selected
    uncertain edges.
    """

    def __init__(
        self,
        num_classes: int = DEFAULT_NUM_CLASSES,
        feedback_dim: int = DEFAULT_FEEDBACK_DIM,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.feedback_dim = feedback_dim
        self.proj = nn.Linear(num_classes, feedback_dim)

    def scatter_feedback(
        self,
        feedback_subset: torch.Tensor,
        uncertain_mask: torch.Tensor,
    ) -> torch.Tensor:
        full = feedback_subset.new_zeros(
            (uncertain_mask.numel(), self.num_classes)
        )
        if feedback_subset.numel() > 0:
            full[uncertain_mask] = feedback_subset
        return full

    def forward(
        self,
        feedback_subset: torch.Tensor,
        uncertain_mask: torch.Tensor,
    ) -> torch.Tensor:
        feedback_full = self.scatter_feedback(feedback_subset, uncertain_mask)
        edge_feedback = self.proj(feedback_full)
        return edge_feedback * uncertain_mask.unsqueeze(1).to(edge_feedback.dtype)


class BiasedGATStructuralEncoder(nn.Module):
    """
    GAT encoder that can append a semantic feedback channel to edge_attr.

    When edge_feedback is None, zeros are appended. That keeps the public forward
    call compatible with the old encoder while allowing the new GATv2 layers to
    have edge_dim = original edge attributes + feedback channels.
    """

    def __init__(
        self,
        in_dim: int = DEFAULT_IN_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        edge_attr_dim: int = DEFAULT_EDGE_ATTR_DIM,
        feedback_dim: int = DEFAULT_FEEDBACK_DIM,
        heads: int = DEFAULT_HEADS,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})"
            )

        self.dropout = dropout
        self.edge_attr_dim = edge_attr_dim
        self.feedback_dim = feedback_dim
        augmented_edge_dim = edge_attr_dim + feedback_dim

        self.gat1 = GATv2Conv(
            in_channels=in_dim,
            out_channels=hidden_dim // heads,
            heads=heads,
            concat=True,
            edge_dim=augmented_edge_dim,
            add_self_loops=False,
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.res1 = nn.Linear(in_dim, hidden_dim)

        self.gat2 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=1,
            concat=False,
            edge_dim=augmented_edge_dim,
            add_self_loops=False,
        )
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def _augment_edge_attr(
        self,
        edge_attr: torch.Tensor,
        edge_feedback: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_feedback is None:
            edge_feedback = edge_attr.new_zeros((edge_attr.size(0), self.feedback_dim))
        return torch.cat([edge_attr, edge_feedback], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_feedback: torch.Tensor | None = None,
    ) -> torch.Tensor:
        edge_attr_aug = self._augment_edge_attr(edge_attr, edge_feedback)

        h = self.gat1(x, edge_index, edge_attr=edge_attr_aug)
        h = self.bn1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + self.res1(x)

        h_in = h
        h = self.gat2(h, edge_index, edge_attr=edge_attr_aug)
        h = self.bn2(h)
        h = F.elu(h)
        h = h + h_in

        return h


class BiasedGATEdgeClassifier(GATEdgeClassifier):
    """
    Edge classifier variant whose encoders accept semantic feedback channels.
    """

    def __init__(
        self,
        in_dim: int = DEFAULT_IN_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        edge_attr_dim: int = DEFAULT_EDGE_ATTR_DIM,
        feedback_dim: int = DEFAULT_FEEDBACK_DIM,
        num_classes: int = DEFAULT_NUM_CLASSES,
        heads: int = DEFAULT_HEADS,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            edge_attr_dim=edge_attr_dim,
            num_classes=num_classes,
            heads=heads,
            dropout=dropout,
        )
        self.feedback_dim = feedback_dim
        self.encoders = nn.ModuleDict(
            {
                name: BiasedGATStructuralEncoder(
                    in_dim=in_dim,
                    hidden_dim=hidden_dim,
                    edge_attr_dim=edge_attr_dim,
                    feedback_dim=feedback_dim,
                    heads=heads,
                    dropout=dropout,
                )
                for name in VARIANT_NAMES
            }
        )
        self.variant_feedback_scale = nn.ParameterDict(
            {name: nn.Parameter(torch.ones(1, feedback_dim)) for name in VARIANT_NAMES}
        )

    def _focused_edge_attr(self, name: str, edge_attr: torch.Tensor) -> torch.Tensor:
        focus = getattr(self, f"{name}_edge_focus")
        original = edge_attr[:, : self.edge_attr_dim] * focus.to(
            dtype=edge_attr.dtype, device=edge_attr.device
        )
        if edge_attr.size(1) == self.edge_attr_dim:
            return original

        feedback = edge_attr[:, self.edge_attr_dim :]
        return torch.cat([original, feedback], dim=1)

    def _variant_feedback(
        self,
        name: str,
        edge_feedback: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if edge_feedback is None:
            return None
        scale = self.variant_feedback_scale[name].to(
            dtype=edge_feedback.dtype, device=edge_feedback.device
        )
        return edge_feedback * scale

    def encode_edges_by_variant(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_feedback: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        edge_reprs: dict[str, torch.Tensor] = {}
        for name, encoder in self.encoders.items():
            focused_edge_attr = self._focused_edge_attr(name, edge_attr)
            focused_feedback = self._variant_feedback(name, edge_feedback)
            node_emb = encoder(x, edge_index, focused_edge_attr, focused_feedback)
            edge_reprs[name] = self._build_flow_repr(
                node_emb, edge_index, focused_edge_attr
            )
        return edge_reprs

    def encode_edges(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_feedback: torch.Tensor | None = None,
    ) -> torch.Tensor:
        edge_reprs = self.encode_edges_by_variant(
            x, edge_index, edge_attr, edge_feedback
        )
        edge_emb, _ = self._fuse_variant_outputs(edge_reprs)
        return edge_emb

    def variant_logits(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_feedback: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        edge_reprs = self.encode_edges_by_variant(
            x, edge_index, edge_attr, edge_feedback
        )
        return {
            name: self.variant_heads[name](edge_reprs[name])
            for name in VARIANT_NAMES
        }

    def forward_with_aux(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_feedback: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        edge_reprs = self.encode_edges_by_variant(
            x, edge_index, edge_attr, edge_feedback
        )
        edge_emb, aux_logits = self._fuse_variant_outputs(edge_reprs)
        logits, edge_emb = self.classify_repr(edge_emb)
        return logits, edge_emb, aux_logits

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_feedback: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, edge_emb, _ = self.forward_with_aux(
            x, edge_index, edge_attr, edge_feedback
        )
        return logits, edge_emb


class ResidualCorrectionHead(nn.Module):
    """
    Learn a small correction added to frozen strong-modality logits.

    The residual form means alpha=0 exactly recovers the strong baseline.
    """

    def __init__(
        self,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        feedback_dim: int = DEFAULT_FEEDBACK_DIM,
        num_classes: int = DEFAULT_NUM_CLASSES,
        dropout: float = DEFAULT_DROPOUT,
        initial_alpha: float = DEFAULT_ALPHA_INIT,
    ) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(initial_alpha)))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + feedback_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        gnn_logits: torch.Tensor,
        edge_emb: torch.Tensor,
        edge_feedback: torch.Tensor,
        strong_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        correction_input = torch.cat([edge_emb, edge_feedback, gnn_logits], dim=1)
        correction = self.mlp(correction_input)
        final_logits = strong_logits + self.alpha * correction
        return final_logits, correction


@dataclass
class IterationState:
    iteration: int = 0
    prev_pred: torch.Tensor | None = None
    prev_entropy: torch.Tensor | None = None
    best_train_macro_f1: float | None = None


class IterativeController:
    """
    Plain Python helper that decides when the feedback loop has converged.
    """

    def __init__(
        self,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        churn_tol: float = DEFAULT_CHURN_TOL,
        entropy_tol: float = DEFAULT_ENTROPY_TOL,
    ) -> None:
        self.max_iterations = max_iterations
        self.churn_tol = churn_tol
        self.entropy_tol = entropy_tol
        self.state = IterationState()

    def reset(self) -> None:
        self.state = IterationState()

    def should_stop(
        self, pred: torch.Tensor, entropy: torch.Tensor
    ) -> tuple[bool, dict[str, float]]:
        metrics: dict[str, float] = {}
        if self.state.iteration >= self.max_iterations:
            metrics["hit_max_iterations"] = 1.0
            return True, metrics

        if self.state.prev_pred is None or self.state.prev_entropy is None:
            self.state.prev_pred = pred.detach()
            self.state.prev_entropy = entropy.detach()
            self.state.iteration += 1
            return False, metrics

        churn = (pred != self.state.prev_pred).float().mean()
        entropy_delta = (entropy - self.state.prev_entropy).abs().mean()
        metrics = {
            "prediction_churn": float(churn.detach().cpu()),
            "entropy_delta": float(entropy_delta.detach().cpu()),
        }

        self.state.prev_pred = pred.detach()
        self.state.prev_entropy = entropy.detach()
        self.state.iteration += 1
        should_stop = churn < self.churn_tol and entropy_delta < self.entropy_tol
        return bool(should_stop), metrics


class FeedbackFusionClassifier(nn.Module):
    """
    Top-level Phase 2 model for selective semantic feedback.

    The training script supplies frozen strong_logits. This module learns only
    the feedback-aware GNN path and a residual correction on top of that strong
    baseline.
    """

    def __init__(
        self,
        in_dim: int = DEFAULT_IN_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        edge_attr_dim: int = DEFAULT_EDGE_ATTR_DIM,
        feedback_dim: int = DEFAULT_FEEDBACK_DIM,
        llm_dim: int = DEFAULT_LLM_DIM,
        num_classes: int = DEFAULT_NUM_CLASSES,
        heads: int = DEFAULT_HEADS,
        dropout: float = DEFAULT_DROPOUT,
        uncertain_fraction: float = DEFAULT_UNCERTAIN_FRACTION,
        prototype_temperature: float = DEFAULT_PROTOTYPE_TEMPERATURE,
        initial_alpha: float = DEFAULT_ALPHA_INIT,
    ) -> None:
        super().__init__()
        self.feedback_dim = feedback_dim
        self.llm_dim = llm_dim
        self.num_classes = num_classes

        self.encoder = BiasedGATEdgeClassifier(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            edge_attr_dim=edge_attr_dim,
            feedback_dim=feedback_dim,
            num_classes=num_classes,
            heads=heads,
            dropout=dropout,
        )
        self.uncertainty_detector = UncertaintyDetector(
            num_classes=num_classes,
            uncertain_fraction=uncertain_fraction,
        )
        self.feedback_scorer = SemanticFeedbackScorer(
            temperature=prototype_temperature,
            num_classes=num_classes,
        )
        self.bias_generator = AttentionBiasGenerator(
            num_classes=num_classes,
            feedback_dim=feedback_dim,
        )
        self.correction_head = ResidualCorrectionHead(
            hidden_dim=hidden_dim,
            feedback_dim=feedback_dim,
            num_classes=num_classes,
            dropout=dropout,
            initial_alpha=initial_alpha,
        )

    @property
    def alpha(self) -> torch.nn.Parameter:
        return self.correction_head.alpha

    def encoder_forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_feedback: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x, edge_index, edge_attr, edge_feedback)

    def build_next_feedback(
        self,
        gnn_logits: torch.Tensor,
        llm_emb: torch.Tensor,
        prototypes: torch.Tensor | dict[str, Any],
        *,
        selection_mode: str = "entropy",
        feedback_mode: str = "prototype",
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if feedback_mode not in {"prototype", "random", "zero"}:
            raise ValueError("feedback_mode must be 'prototype', 'random', or 'zero'")

        entropy, uncertain_mask = self.uncertainty_detector(
            gnn_logits,
            selection_mode=selection_mode,
            generator=generator,
        )
        if feedback_mode == "prototype":
            feedback_subset = self.feedback_scorer(llm_emb[uncertain_mask], prototypes)
        elif feedback_mode == "random":
            selected_count = int(uncertain_mask.sum().item())
            random_logits = torch.randn(
                (selected_count, self.num_classes),
                dtype=llm_emb.dtype,
                device=llm_emb.device,
                generator=generator,
            )
            feedback_subset = F.softmax(random_logits, dim=1)
        else:
            feedback_subset = llm_emb.new_zeros(
                (int(uncertain_mask.sum().item()), self.num_classes)
            )
        edge_feedback = self.bias_generator(feedback_subset, uncertain_mask)
        return edge_feedback, entropy, uncertain_mask

    def sparse_correction_loss(self, correction: torch.Tensor) -> torch.Tensor:
        """Penalize correction magnitude so confident edges stay near the skip path."""
        return (self.alpha * correction).abs().mean()

    def correction_from(
        self,
        gnn_logits: torch.Tensor,
        edge_emb: torch.Tensor,
        edge_feedback: torch.Tensor,
        strong_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.correction_head(
            gnn_logits=gnn_logits,
            edge_emb=edge_emb,
            edge_feedback=edge_feedback,
            strong_logits=strong_logits,
        )

    def _trace_iteration(
        self,
        iteration: int,
        logits: torch.Tensor,
        entropy: torch.Tensor,
        uncertain_mask: torch.Tensor,
        stop_metrics: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        selected_entropy = entropy[uncertain_mask]
        predicted = logits.argmax(dim=1)
        selected_hist = torch.bincount(
            predicted[uncertain_mask], minlength=self.num_classes
        )
        payload: dict[str, Any] = {
            "iteration": iteration,
            "selected_count": int(uncertain_mask.sum().item()),
            "selected_fraction": float(uncertain_mask.float().mean().item()),
            "predicted_class_histogram": selected_hist.detach().cpu().tolist(),
            "entropy_mean": float(entropy.detach().mean().item()),
        }
        if selected_entropy.numel() > 0:
            payload.update(
                {
                    "selected_entropy_mean": float(
                        selected_entropy.detach().mean().item()
                    ),
                    "selected_entropy_min": float(
                        selected_entropy.detach().min().item()
                    ),
                    "selected_entropy_max": float(
                        selected_entropy.detach().max().item()
                    ),
                }
            )
        else:
            payload.update(
                {
                    "selected_entropy_mean": None,
                    "selected_entropy_min": None,
                    "selected_entropy_max": None,
                }
            )
        if stop_metrics:
            payload.update(stop_metrics)
        return payload

    def run_feedback_loop(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        llm_emb: torch.Tensor,
        prototypes: torch.Tensor | dict[str, Any],
        strong_logits: torch.Tensor,
        *,
        inject_feedback: bool,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        damping: float = 0.5,
        selection_mode: str = "entropy",
        feedback_mode: str = "prototype",
        generator: torch.Generator | None = None,
        controller: IterativeController | None = None,
        return_trace: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """
        Run the selective feedback loop.

        If inject_feedback is False, this is Stage A: feedback influences only the
        residual correction head. If True, feedback is damped and re-entered into
        the GAT edge channels for the next iteration.
        """
        if not 0.0 <= damping <= 1.0:
            raise ValueError("damping must be in [0, 1]")

        edge_feedback = edge_attr.new_zeros((edge_attr.size(0), self.feedback_dim))
        loop_controller = controller or IterativeController(max_iterations=max_iterations)
        loop_controller.reset()

        final_logits: torch.Tensor | None = None
        final_correction: torch.Tensor | None = None
        final_entropy: torch.Tensor | None = None
        final_uncertain_mask: torch.Tensor | None = None
        final_next_feedback: torch.Tensor | None = None
        final_gnn_logits: torch.Tensor | None = None
        final_edge_emb: torch.Tensor | None = None
        iterations: list[dict[str, Any]] = []

        loop_count = 1 if not inject_feedback else max_iterations
        for iteration in range(loop_count):
            gnn_logits, edge_emb = self.encoder_forward(
                x, edge_index, edge_attr, edge_feedback
            )
            next_feedback, entropy, uncertain_mask = self.build_next_feedback(
                gnn_logits,
                llm_emb,
                prototypes,
                selection_mode=selection_mode,
                feedback_mode=feedback_mode,
                generator=generator,
            )
            final_logits, final_correction = self.correction_from(
                gnn_logits, edge_emb, next_feedback, strong_logits
            )

            stop = False
            stop_metrics: dict[str, float] = {}
            if inject_feedback:
                stop, stop_metrics = loop_controller.should_stop(
                    final_logits.argmax(dim=1), entropy
                )

            if return_trace:
                iterations.append(
                    self._trace_iteration(
                        iteration, final_logits, entropy, uncertain_mask, stop_metrics
                    )
                )

            final_entropy = entropy
            final_uncertain_mask = uncertain_mask
            final_next_feedback = next_feedback
            final_gnn_logits = gnn_logits
            final_edge_emb = edge_emb

            if not inject_feedback or stop:
                break
            edge_feedback = damping * next_feedback + (1.0 - damping) * edge_feedback

        if final_logits is None or final_correction is None:
            raise RuntimeError("Feedback loop did not execute.")

        trace: dict[str, Any] = {
            "iterations": iterations,
            "edge_feedback": edge_feedback,
            "next_feedback": final_next_feedback,
            "entropy": final_entropy,
            "uncertain_mask": final_uncertain_mask,
            "gnn_logits": final_gnn_logits,
            "edge_emb": final_edge_emb,
            "alpha": self.alpha.detach().view(1),
            "selection_mode": selection_mode,
            "feedback_mode": feedback_mode,
        }
        return final_logits, final_correction, trace

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        llm_emb: torch.Tensor,
        prototypes: torch.Tensor | dict[str, Any],
        strong_logits: torch.Tensor,
        edge_feedback: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if edge_feedback is None:
            edge_feedback = edge_attr.new_zeros((edge_attr.size(0), self.feedback_dim))

        gnn_logits, edge_emb = self.encoder_forward(
            x, edge_index, edge_attr, edge_feedback
        )
        final_logits, correction = self.correction_from(
            gnn_logits, edge_emb, edge_feedback, strong_logits
        )
        next_feedback, entropy, uncertain_mask = self.build_next_feedback(
            gnn_logits, llm_emb, prototypes
        )

        diagnostics = {
            "gnn_logits": gnn_logits,
            "edge_emb": edge_emb,
            "correction": correction,
            "edge_feedback": edge_feedback,
            "next_feedback": next_feedback,
            "entropy": entropy,
            "uncertain_mask": uncertain_mask,
            "alpha": self.alpha.detach().view(1),
        }
        return final_logits, diagnostics
