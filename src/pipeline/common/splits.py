from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.data import Data


SPLITS_OUTPUT_DIR = "data/ton_iot/processed/splits"
SPLITS_OUTPUT_FILE = "folds.pt"

NUM_CLASSES = 10
DEFAULT_K = 5
DEFAULT_SEED = 42
VAL_FRACTION_OF_TRAINVAL = 0.25


def _print_split_distribution(
    edge_label: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    fold_idx: int,
) -> None:
    print(f"Fold {fold_idx} class distribution:")
    header = f"  {'class':>6} {'train':>8} {'val':>8} {'test':>8}"
    print(header)
    for cls in range(NUM_CLASSES):
        cls_mask = edge_label == cls
        train_count = int((cls_mask & train_mask).sum())
        val_count = int((cls_mask & val_mask).sum())
        test_count = int((cls_mask & test_mask).sum())
        print(f"  {cls:>6d} {train_count:>8d} {val_count:>8d} {test_count:>8d}")


def create_edge_splits(
    data: Data, k: int = DEFAULT_K, seed: int = DEFAULT_SEED
) -> list[dict[str, torch.Tensor]]:
    edge_label = data.edge_label
    num_edges = edge_label.shape[0]
    labels_np = edge_label.numpy()
    indices = np.arange(num_edges)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    folds: list[dict[str, torch.Tensor]] = []

    for fold_idx, (trainval_idx, test_idx) in enumerate(skf.split(indices, labels_np)):
        trainval_labels = labels_np[trainval_idx]
        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=VAL_FRACTION_OF_TRAINVAL,
            random_state=seed + fold_idx,
            stratify=trainval_labels,
        )

        train_mask = torch.zeros(num_edges, dtype=torch.bool)
        val_mask = torch.zeros(num_edges, dtype=torch.bool)
        test_mask = torch.zeros(num_edges, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        folds.append(
            {"train_mask": train_mask, "val_mask": val_mask, "test_mask": test_mask}
        )

        _print_split_distribution(edge_label, train_mask, val_mask, test_mask, fold_idx)

    output_path = Path(SPLITS_OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    torch.save(folds, output_path / SPLITS_OUTPUT_FILE)

    return folds


def get_class_weights(
    edge_label: torch.Tensor, train_mask: torch.Tensor
) -> torch.Tensor:
    train_labels = edge_label[train_mask]
    counts = torch.bincount(train_labels, minlength=NUM_CLASSES).float()
    safe_counts = torch.where(counts > 0, counts, torch.ones_like(counts))
    inverse = 1.0 / safe_counts
    inverse = torch.where(counts > 0, inverse, torch.zeros_like(inverse))
    weights = inverse * (NUM_CLASSES / inverse.sum())
    return weights


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification with class imbalance.

    Standard cross-entropy treats every mistake equally. This loss adds a
    (1 - p_t)^gamma factor that suppresses the loss for easy (already-correct)
    examples and focuses gradient on hard (uncertain or wrong) ones — exactly
    what we need when rare attack classes are swamped by Benign traffic.

    Parameters
    ----------
    alpha : class weight vector [num_classes], same inverse-frequency weights
            from get_class_weights(). Rare classes already get a higher base weight.
    gamma : focus parameter (default 2.0). Higher = more focus on hard examples.
            gamma=0 recovers standard weighted cross-entropy.
    """

    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0) -> None:
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal_weight = self.alpha[targets] * (1.0 - pt) ** self.gamma  # type: ignore[index]
        return (focal_weight * ce).mean()


def oversample_minority_embeddings(
    edge_repr: torch.Tensor,
    edge_label: torch.Tensor,
    target_ratio: float = 0.10,
    seed: int = DEFAULT_SEED,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    SMOTE-style oversampling at the edge representation level.

    For each class with fewer than (target_ratio * majority_class_count) training
    samples, create synthetic samples by linearly interpolating between random
    pairs of same-class representations. This is done AFTER the GNN encoder runs,
    so we never corrupt the graph topology — we only augment the MLP input.

    Parameters
    ----------
    edge_repr  : [T, D] tensor of edge representations for training edges only.
    edge_label : [T]    integer class labels for those training edges.
    target_ratio : fraction of the majority class to target for each minority class.
    seed       : random seed (vary per epoch for diversity).

    Returns
    -------
    aug_repr   : [M, D] synthetic edge representations (additions only, not originals).
    aug_labels : [M]    corresponding class labels for synthetic samples.
    """
    rng = np.random.default_rng(seed)
    counts = torch.bincount(edge_label, minlength=NUM_CLASSES)
    majority_count = int(counts.max())
    target_count = max(int(target_ratio * majority_count), 2)

    aug_reprs: list[torch.Tensor] = []
    aug_labels: list[torch.Tensor] = []

    for cls in range(NUM_CLASSES):
        cls_indices = (edge_label == cls).nonzero(as_tuple=True)[0]
        n_cls = cls_indices.shape[0]

        if n_cls == 0 or n_cls >= target_count:
            continue

        # Cap: don't generate more than 5x the real count (interpolations between
        # very few real samples become meaninglessly redundant beyond this)
        n_to_generate = min(target_count - n_cls, n_cls * 5)
        cls_repr = edge_repr[cls_indices]  # [n_cls, D]

        idx_i = rng.integers(0, n_cls, size=n_to_generate)
        idx_j = rng.integers(0, n_cls, size=n_to_generate)
        lam = torch.tensor(
            rng.uniform(0.0, 1.0, size=(n_to_generate, 1)), dtype=edge_repr.dtype
        )

        synthetic = (1.0 - lam) * cls_repr[idx_i] + lam * cls_repr[idx_j]
        aug_reprs.append(synthetic)
        aug_labels.append(
            torch.full((n_to_generate,), cls, dtype=torch.long)
        )

    if not aug_reprs:
        D = edge_repr.shape[1]
        return torch.zeros(0, D, dtype=edge_repr.dtype), torch.zeros(0, dtype=torch.long)

    return torch.cat(aug_reprs, dim=0), torch.cat(aug_labels, dim=0)
