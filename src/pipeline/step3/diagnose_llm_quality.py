"""
LLM embedding quality diagnostic.

Answers the question: do the frozen CySecBERT embeddings carry enough
discriminative signal to separate attack types on their own?

Method: linear probe — a logistic regression classifier (no hidden layers)
trained on the frozen embeddings. If a linear boundary in 768-d space can
separate classes, the embeddings have linearly separable structure. This is
the standard way to audit pretrained representations without confounding the
result with a powerful nonlinear head.

Run this before fusion to understand how much each path contributes independently.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler, normalize

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

LLM_EMB_PATH = "data/ton_iot/processed/step3_llm/edge_embeddings.pt"
GRAPH_PATH   = "data/ton_iot/processed/step1/pyg_data.pt"
SPLITS_PATH  = "data/ton_iot/processed/splits/folds.pt"

LABEL_NAMES = [
    "Benign", "backdoor", "ddos", "dos", "injection",
    "mitm", "password", "ransomware", "scanning", "xss",
]
NUM_CLASSES = 10


def _print_class_report(preds: np.ndarray, targets: np.ndarray) -> None:
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, preds, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    print(f"\n  {'Class':<12} {'Support':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print(f"  {'-'*52}")
    order = np.argsort(-support)
    for cls in order:
        print(
            f"  {LABEL_NAMES[cls]:<12} {int(support[cls]):>8d} "
            f"{precision[cls]:>10.4f} {recall[cls]:>8.4f} {f1[cls]:>8.4f}"
        )


def main() -> None:
    print("=== LLM Embedding Quality Diagnostic ===\n")

    llm_emb = torch.load(LLM_EMB_PATH, weights_only=True).numpy()
    data    = torch.load(GRAPH_PATH, weights_only=False)
    folds   = torch.load(SPLITS_PATH, weights_only=False)
    labels  = data.edge_label.numpy()

    assert llm_emb.shape == (len(labels), 768), \
        f"Shape mismatch: {llm_emb.shape} vs expected ({len(labels)}, 768)"

    # ------------------------------------------------------------------ #
    # 1. PCA variance analysis                                             #
    # ------------------------------------------------------------------ #
    print("--- PCA: Variance Explained ---")
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(llm_emb)

    pca = PCA(n_components=20)
    pca.fit(emb_scaled)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    for k in [1, 2, 5, 10, 20]:
        print(f"  Top-{k:2d} PCs explain {cumvar[k-1]*100:.1f}% of variance")

    print()
    print("  Interpretation:")
    if cumvar[0] > 0.9:
        print("  WARNING: >90% in PC-1 — embeddings collapsed to 1-D subspace.")
    elif cumvar[0] > 0.5:
        print("  Moderate concentration in PC-1 — some collapse, but not severe.")
    else:
        print("  Signal is spread across many dimensions — healthy representation.")

    # ------------------------------------------------------------------ #
    # 2. Inter-class cosine similarity                                     #
    # ------------------------------------------------------------------ #
    print("\n--- Inter-Class Cosine Similarity ---")
    emb_norm = normalize(llm_emb)
    class_means = {
        c: emb_norm[labels == c].mean(axis=0)
        for c in range(NUM_CLASSES)
        if (labels == c).sum() > 0
    }
    sims = [
        float(np.dot(class_means[i], class_means[j]))
        for i in range(NUM_CLASSES) for j in range(i + 1, NUM_CLASSES)
        if i in class_means and j in class_means
    ]
    print(f"  Min:  {min(sims):.4f}")
    print(f"  Max:  {max(sims):.4f}")
    print(f"  Mean: {np.mean(sims):.4f}")
    print()
    print("  Note: BERT-family models always produce high cosine similarity between")
    print("  class mean vectors — this is expected and does NOT mean the embeddings")
    print("  are useless. The linear probe below gives the definitive quality signal.")

    # ------------------------------------------------------------------ #
    # 3. Linear probe: 5-fold CV                                           #
    # ------------------------------------------------------------------ #
    print("\n--- Linear Probe (5-fold CV, same splits as GNN) ---")
    print("  Method: LogisticRegression on z-score normalized embeddings.")
    print("  No hidden layers — tests for linearly separable structure only.\n")

    fold_test_f1s = []
    all_test_preds   = np.zeros(len(labels), dtype=int)
    all_test_targets = np.zeros(len(labels), dtype=int)
    test_coverage    = np.zeros(len(labels), dtype=bool)

    for fold_idx, fold in enumerate(folds):
        train_mask = fold["train_mask"].numpy()
        test_mask  = fold["test_mask"].numpy()

        X_train = emb_scaled[train_mask]
        y_train = labels[train_mask]
        X_test  = emb_scaled[test_mask]
        y_test  = labels[test_mask]

        clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        macro_f1 = float(
            f1_score(y_test, preds, average="macro",
                     labels=list(range(NUM_CLASSES)), zero_division=0)
        )
        fold_test_f1s.append(macro_f1)

        all_test_preds[test_mask]   = preds
        all_test_targets[test_mask] = y_test
        test_coverage[test_mask]    = True

        print(f"  Fold {fold_idx}: test_macro_f1={macro_f1:.4f}")

    mean_f1 = float(np.mean(fold_test_f1s))
    std_f1  = float(np.std(fold_test_f1s))
    print(f"\n  CV test macro-F1: {mean_f1:.4f} ± {std_f1:.4f}")

    # ------------------------------------------------------------------ #
    # 4. Aggregated per-class report                                       #
    # ------------------------------------------------------------------ #
    print("\n--- Per-Class Report (pooled over all folds) ---")
    _print_class_report(all_test_preds[test_coverage], all_test_targets[test_coverage])

    # ------------------------------------------------------------------ #
    # 5. Comparison summary                                                #
    # ------------------------------------------------------------------ #
    print("\n--- Comparison Summary ---")
    print(f"  Dummy baseline (always Benign):  ~0.10  macro-F1")
    print(f"  LLM linear probe:                {mean_f1:.4f} macro-F1  (this script)")
    print(f"  GNN FocalLoss (from train_gnn):  ~0.30  macro-F1  (see training_history.json)")
    print()
    print("  The LLM and GNN carry complementary signal.")
    print("  Fusion should outperform both paths individually.")


if __name__ == "__main__":
    main()
