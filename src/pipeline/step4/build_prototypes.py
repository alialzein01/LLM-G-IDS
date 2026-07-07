"""Phase 4.3 setup — per-fold ZCA whitener + class prototypes.

For each of the 5 CV folds, fits a ZCA whitener on the train-fold
CySecBERT embeddings (from `step3_llm/edge_embeddings.pt`, which are
row-aligned to the edge index), then computes the mean whitened
embedding per class as that class's prototype.

The reverted Step 4 skipped whitening — it L2-normalized, averaged, then
re-normalized. That collapses per-class spread and lets random softmax
beat prototype softmax. This script does the fix: subtract the fold's
mean, apply ZCA, then take the class-wise mean in the whitened space.

Rank-deficient cov (N_train < embed_dim = 768) is handled by clamping
eigenvalues to a small floor. Classes with zero training examples in a
fold are left as zero prototypes (flagged in the manifest).

Output: `data/{dataset}/processed/step4_feedback/prototypes.pt`, a dict
    {
        "folds": [
            {"mean": [D], "whitener": [D, D],
             "prototypes_whitened": [C, D], "prototypes_raw": [C, D],
             "class_counts": [C]},
            ...
        ],
        "num_classes": int, "embed_dim": int, "model_id": str,
        "empty_class_folds": [(fold_idx, class_idx), ...],
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.models.feedback_classifier import CYSECBERT_MODEL_ID
from src.pipeline.common.datasets import get_dataset_config
from src.pipeline.common.splits import NUM_CLASSES


EIGENVALUE_FLOOR = 1e-4


def _zca_whitener(centered: torch.Tensor) -> torch.Tensor:
    """Return `W` such that `centered @ W` has approximately unit covariance.

    Uses eigendecomposition of the covariance matrix with a floor on
    small eigenvalues to keep the inversion stable when N < D.
    """
    n = centered.shape[0]
    cov = centered.T @ centered / max(n - 1, 1)
    cov = cov.double()
    eig_vals, eig_vecs = torch.linalg.eigh(cov)
    inv_sqrt_vals = (eig_vals.clamp(min=EIGENVALUE_FLOOR)) ** -0.5
    whitener = eig_vecs @ torch.diag(inv_sqrt_vals) @ eig_vecs.T
    return whitener.float()


def build_prototypes(dataset: str) -> Path:
    config = get_dataset_config(dataset)
    llm_emb_path = Path(config.llm_embedding_path)
    if not llm_emb_path.exists():
        raise FileNotFoundError(
            f"Precomputed CySecBERT embeddings not found at {llm_emb_path}. "
            f"Run: python -m src.pipeline.step3.encode_kg --dataset {dataset}"
        )

    embeddings = torch.load(llm_emb_path, weights_only=False)
    data = torch.load(config.graph_path, weights_only=False)
    folds = torch.load(config.splits_path, weights_only=False)
    labels = data.edge_label

    if embeddings.shape[0] != labels.shape[0]:
        raise RuntimeError(
            f"CySecBERT embeddings ({embeddings.shape[0]}) not row-aligned to "
            f"edges ({labels.shape[0]})."
        )

    embed_dim = embeddings.shape[1]
    fold_states: list[dict] = []
    empty_class_folds: list[tuple[int, int]] = []

    for fold_idx, fold in enumerate(folds):
        train_mask = fold["train_mask"]
        train_emb = embeddings[train_mask].float()
        train_labels = labels[train_mask]

        mean = train_emb.mean(dim=0)
        centered = train_emb - mean
        whitener = _zca_whitener(centered)
        whitened = centered @ whitener

        prototypes_raw = torch.zeros(NUM_CLASSES, embed_dim)
        prototypes_whitened = torch.zeros(NUM_CLASSES, embed_dim)
        class_counts = torch.zeros(NUM_CLASSES, dtype=torch.long)

        for c in range(NUM_CLASSES):
            cls_mask = train_labels == c
            n = int(cls_mask.sum())
            class_counts[c] = n
            if n == 0:
                empty_class_folds.append((fold_idx, c))
                continue
            prototypes_raw[c] = train_emb[cls_mask].mean(dim=0)
            prototypes_whitened[c] = whitened[cls_mask].mean(dim=0)

        fold_states.append(
            {
                "mean": mean,
                "whitener": whitener,
                "prototypes_whitened": prototypes_whitened,
                "prototypes_raw": prototypes_raw,
                "class_counts": class_counts,
            }
        )
        print(
            f"fold {fold_idx}: N_train={int(train_mask.sum())} "
            f"empty_classes={[c for f, c in empty_class_folds if f == fold_idx]}"
        )

    output_dir = Path(f"data/{dataset}/processed/step4_feedback")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "prototypes.pt"
    torch.save(
        {
            "folds": fold_states,
            "num_classes": NUM_CLASSES,
            "embed_dim": embed_dim,
            "model_id": CYSECBERT_MODEL_ID,
            "empty_class_folds": empty_class_folds,
        },
        output_path,
    )

    manifest = {
        "dataset": dataset,
        "num_folds": len(fold_states),
        "num_classes": NUM_CLASSES,
        "embed_dim": embed_dim,
        "empty_class_folds": [
            {"fold": f, "class": c} for f, c in empty_class_folds
        ],
        "output_path": str(output_path),
    }
    with open(output_dir / "prototypes_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved prototypes to {output_path}")
    if empty_class_folds:
        print(f"Note: {len(empty_class_folds)} (fold, class) combos have zero training examples")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="unsw_nb15", choices=["ton_iot", "unsw_nb15"])
    args = parser.parse_args()
    build_prototypes(args.dataset)


if __name__ == "__main__":
    main()
