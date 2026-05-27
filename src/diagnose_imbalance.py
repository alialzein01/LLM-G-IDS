from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch_geometric.data import Data

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.splits import NUM_CLASSES

DATA_PATH = "data/processed/step1/pyg_data.pt"
METRICS_PATH = "data/processed/step3_gnn/metrics.json"

LABEL_NAMES = [
    "Benign", "backdoor", "ddos", "dos", "injection",
    "mitm", "password", "ransomware", "scanning", "xss",
]


def main() -> None:
    print(f"Loading graph from {DATA_PATH}")
    data: Data = torch.load(DATA_PATH, weights_only=False)
    labels = data.edge_label.numpy()
    num_edges = len(labels)

    print(f"\n{'='*60}")
    print("STEP 1: WHAT DOES THE DATA LOOK LIKE?")
    print(f"{'='*60}")
    print(f"Total edges (network flows): {num_edges}")
    print(f"Total nodes (unique IPs):    {data.x.shape[0]}")
    print(f"\nClass distribution:")
    print(f"  {'Class':<12} {'Count':>8} {'Percent':>9}   (bar = 1 block per 2%)")
    print(f"  {'-'*50}")
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    for cls_id, name in enumerate(LABEL_NAMES):
        pct = 100.0 * counts[cls_id] / num_edges
        bar = "█" * max(1, int(pct / 2)) if counts[cls_id] > 0 else ""
        print(f"  {name:<12} {counts[cls_id]:>8d} {pct:>8.1f}%  {bar}")

    print(f"\n  Key insight: Benign is {100*counts[0]/num_edges:.0f}% of all traffic.")
    print(f"  The rarest class (ransomware) has only {counts[7]} total samples.")
    print(f"  With 5-fold cross-validation, that's ~{counts[7]*3//5} training examples per fold.")
    print(f"  No model can reliably learn a class from {counts[7]*3//5} examples.")

    print(f"\n{'='*60}")
    print("STEP 2: WHY IS ACCURACY A BAD METRIC FOR IDS?")
    print(f"{'='*60}")
    dummy_preds = np.zeros(num_edges, dtype=np.int64)
    dummy_accuracy = float((dummy_preds == labels).mean())
    dummy_macro_f1 = float(
        f1_score(labels, dummy_preds, average="macro",
                 labels=list(range(NUM_CLASSES)), zero_division=0)
    )
    print(f"\n  Imagine a 'model' that ALWAYS predicts 'Benign' — no learning at all:")
    print(f"  Accuracy:  {dummy_accuracy:.1%}  ← sounds great, but it's useless!")
    print(f"  Macro-F1:  {dummy_macro_f1:.4f}  ← correctly exposes it as terrible")
    print(f"\n  Macro-F1 averages the F1 score across ALL 10 classes equally.")
    print(f"  If you miss ransomware entirely, that class gets F1=0.0,")
    print(f"  which drags macro-F1 down regardless of how good you are on Benign.")
    print(f"  This is why macro-F1 is the right metric for intrusion detection.")

    print(f"\n{'='*60}")
    print("STEP 3: WHAT ARE THE CURRENT GNN RESULTS?")
    print(f"{'='*60}")
    metrics_file = Path(METRICS_PATH)
    if not metrics_file.exists():
        print(f"\n  No metrics found at {METRICS_PATH}")
        print(f"  Run training first:  python src/train_gnn.py")
        return

    with open(metrics_file) as f:
        metrics = json.load(f)

    print(f"\n  Overall metrics (final model trained on all data):")
    print(f"  Accuracy:    {metrics['overall_accuracy']:.4f}")
    print(f"  Macro-F1:    {metrics['overall_macro_f1']:.4f}   ← primary metric")
    print(f"  Weighted-F1: {metrics['overall_weighted_f1']:.4f}")

    print(f"\n  Per-class results (sorted by sample count, most common first):")
    print(f"  {'Class':<12} {'Count':>8} {'F1':>8}  Diagnosis")
    print(f"  {'-'*60}")
    per_class = sorted(metrics["per_class"], key=lambda x: -x["support"])
    for cls in per_class:
        f1 = cls["f1"]
        support = cls["support"]
        if support == 0:
            diag = "no test samples in final model eval"
        elif f1 >= 0.70:
            diag = "good"
        elif f1 >= 0.40:
            diag = "mediocre — needs improvement"
        elif f1 >= 0.10:
            diag = "POOR — class imbalance likely culprit"
        else:
            diag = "FAILING — model essentially ignores this class"
        print(f"  {cls['class_name']:<12} {support:>8d} {f1:>8.4f}  {diag}")

    print(f"\n  Comparison:")
    print(f"  Dummy 'always-Benign' macro-F1: {dummy_macro_f1:.4f}")
    print(f"  GNN macro-F1:                   {metrics['overall_macro_f1']:.4f}")
    gap = metrics["overall_macro_f1"] - dummy_macro_f1
    if gap < 0.05:
        print(f"  Gap: {gap:+.4f} — GNN is barely better than doing nothing!")
    elif gap < 0.20:
        print(f"  Gap: {gap:+.4f} — Some improvement, but still weak on rare classes.")
    else:
        print(f"  Gap: {gap:+.4f} — Meaningful improvement over dummy baseline.")


if __name__ == "__main__":
    main()
