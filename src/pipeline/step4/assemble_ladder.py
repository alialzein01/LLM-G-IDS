"""Assemble the official comparison ladder for a dataset.

Computes pooled out-of-fold macro-F1 for the four rungs on a single, honest
footing and writes both a JSON summary and a readable markdown report:

    GNN-alone  <  LLM-alone  <  AGAF  <  Feedback loop      (the target order)

All numbers are scored over ``config.eval_classes`` (the 8 well-supported
classes for ToN-IoT; all 10 for UNSW-NB15). Ultra-rare classes listed in
``config.dropped_classes`` are masked out of every method's logits before
argmax so no method can emit a phantom prediction into a class no one can
learn — see ``DatasetConfig`` and ``splits.mask_dropped_logits``.

Rungs and their sources (all leak-free OOF):
  - GNN-alone : step4_feedback/oof_logits.pt        (build_oof_predictions)
  - LLM-alone : per-fold WhitenedPrototypeScorer     (_llm_alone_oof)
  - AGAF      : step3_fusion/metrics.json predictions (leak-free train_fusion)
  - Loop      : step4_feedback/feedback_oof_real.pt   (train_feedback, real)

Run:
    python -m src.pipeline.step4.assemble_ladder --dataset ton_iot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.pipeline.common.datasets import get_dataset_config
from src.pipeline.common.splits import eval_macro_f1, mask_dropped_logits
from src.pipeline.step4.train_feedback import _llm_alone_oof

BOOTSTRAP_ITERS = 2000
SEED = 42


def _bootstrap_ci(labels, preds_a, preds_b, eval_classes, iters=BOOTSTRAP_ITERS, seed=SEED):
    """Bootstrap CI of macro-F1(a) - macro-F1(b) over resampled edges, scored on
    `eval_classes` only."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    preds_a = np.asarray(preds_a)
    preds_b = np.asarray(preds_b)
    n = len(labels)
    labs = list(eval_classes)
    diffs = np.empty(iters)
    for i in range(iters):
        idx = rng.integers(0, n, size=n)
        fa = f1_score(labels[idx], preds_a[idx], average="macro", labels=labs, zero_division=0)
        fb = f1_score(labels[idx], preds_b[idx], average="macro", labels=labs, zero_division=0)
        diffs[i] = fa - fb
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "prob_positive": float((diffs > 0).mean()),
    }


def _masked_preds(logits: torch.Tensor, dropped) -> torch.Tensor:
    return mask_dropped_logits(logits, dropped).argmax(dim=1)


def assemble_ladder(dataset: str) -> Path:
    config = get_dataset_config(dataset)
    ec = config.eval_classes
    dropped = config.dropped_classes
    root = Path(f"data/{dataset}/processed/step4_feedback")

    data = torch.load(config.graph_path, weights_only=False)
    labels = data.edge_label
    labels_np = labels.numpy()
    folds = torch.load(config.splits_path, weights_only=False)
    emb = torch.load(config.llm_embedding_path, weights_only=False).float()
    protos = torch.load(root / "prototypes.pt", weights_only=False)

    # --- rung predictions (all masked to eval classes) ---------------------
    gnn_logits = torch.load(root / "oof_logits.pt", weights_only=False)
    gnn_pred = _masked_preds(gnn_logits, dropped)

    # LLM-alone: ALWAYS the whitened-prototype scorer.
    #
    # This must be the same LLM the feedback loop consults, or the ladder is
    # not a like-for-like comparison. `train_feedback` defaults to the
    # prototype scorer (`use_llm_head=False`), so the prototype scorer is the
    # rung. Scoring this rung with the trained MLP head instead — while the
    # loop consults the prototype — is what produced the misleading
    # "loop beats LLM by +0.09" reading: the loop was being compared against an
    # LLM it never uses. Report the trained head separately as its own
    # baseline if you want it; do not substitute it here.
    llm_proto_logits = _llm_alone_oof(data, emb, folds, protos)
    llm_proto_pred = _masked_preds(llm_proto_logits, dropped)
    llm_pred = llm_proto_pred
    llm_head_used = "prototype_scorer"

    agaf_metrics = json.loads(
        (Path(f"data/{dataset}/processed/step3_fusion/metrics.json")).read_text()
    )
    agaf_pred = torch.tensor(agaf_metrics["predictions"], dtype=torch.long)

    loop_logits = torch.load(root / "feedback_oof_real.pt", weights_only=False)
    loop_pred = _masked_preds(loop_logits, dropped)

    def macro(pred):
        return eval_macro_f1(labels, pred, ec)

    def weighted(pred):
        return float(
            f1_score(labels_np, pred.numpy(), average="weighted", labels=list(ec), zero_division=0)
        )

    rungs = {
        "gnn_alone": macro(gnn_pred),
        "llm_alone": macro(llm_pred),
        "agaf": macro(agaf_pred),
        "feedback_loop": macro(loop_pred),
    }
    weighted_f1 = {
        "gnn_alone": weighted(gnn_pred),
        "llm_alone": weighted(llm_pred),
        "agaf": weighted(agaf_pred),
        "feedback_loop": weighted(loop_pred),
    }

    bootstraps = {
        "agaf_vs_gnn": _bootstrap_ci(labels_np, agaf_pred.numpy(), gnn_pred.numpy(), ec),
        "agaf_vs_llm": _bootstrap_ci(labels_np, agaf_pred.numpy(), llm_pred.numpy(), ec),
        "loop_vs_agaf": _bootstrap_ci(labels_np, loop_pred.numpy(), agaf_pred.numpy(), ec),
        "loop_vs_llm": _bootstrap_ci(labels_np, loop_pred.numpy(), llm_pred.numpy(), ec),
        "loop_vs_gnn": _bootstrap_ci(labels_np, loop_pred.numpy(), gnn_pred.numpy(), ec),
    }

    ladder_ok = rungs["gnn_alone"] < rungs["agaf"] and rungs["llm_alone"] < rungs["agaf"] \
        and rungs["agaf"] < rungs["feedback_loop"]

    summary = {
        "dataset": dataset,
        "eval_classes": list(ec),
        "dropped_classes": list(dropped),
        "metric": "pooled_oof_macro_f1",
        "llm_head": llm_head_used,
        "llm_alone_prototype": macro(llm_proto_pred),
        "ladder": rungs,
        "weighted_f1": weighted_f1,
        "bootstraps": bootstraps,
        "ladder_order_holds": bool(ladder_ok),
    }
    out_json = root / "ladder_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))

    # --- per-class F1 table for the loop -----------------------------------
    per_class = f1_score(
        labels_np, loop_pred.numpy(), average=None, labels=list(range(config.num_classes)),
        zero_division=0,
    )
    counts = torch.bincount(labels, minlength=config.num_classes).tolist()

    md = []
    md.append(f"# {config.display_name} — Comparison Ladder\n")
    md.append(f"Pooled out-of-fold macro-F1 over {len(ec)} classes "
              f"(dropped: {[config.label_names[c] for c in dropped] or 'none'}).\n")
    md.append("| Rung | Macro-F1 | Weighted-F1 |")
    md.append("| --- | ---: | ---: |")
    for name, key in [("GNN-alone", "gnn_alone"), ("LLM-alone", "llm_alone"),
                      ("AGAF", "agaf"), ("**Feedback loop**", "feedback_loop")]:
        md.append(f"| {name} | {rungs[key]:.4f} | {weighted_f1[key]:.4f} |")
    md.append("")
    md.append(f"**Ladder order (GNN & LLM < AGAF < loop): "
              f"{'HOLDS ✅' if ladder_ok else 'DOES NOT HOLD ❌'}**\n")

    md.append("## Significance (bootstrap, 2000 iters)\n")
    md.append("| Comparison | Δ macro-F1 | 95% CI | P(>0) |")
    md.append("| --- | ---: | :---: | ---: |")
    for label, key in [("AGAF − GNN", "agaf_vs_gnn"), ("AGAF − LLM", "agaf_vs_llm"),
                       ("loop − AGAF", "loop_vs_agaf"), ("loop − LLM", "loop_vs_llm"),
                       ("loop − GNN", "loop_vs_gnn")]:
        b = bootstraps[key]
        md.append(f"| {label} | {b['mean_diff']:+.4f} | "
                  f"[{b['ci_low']:+.4f}, {b['ci_high']:+.4f}] | {b['prob_positive']:.3f} |")
    md.append("")

    md.append("## Loop per-class F1\n")
    md.append("| Class | Support | F1 |")
    md.append("| --- | ---: | ---: |")
    for c in range(config.num_classes):
        tag = " (dropped)" if c in dropped else ""
        md.append(f"| {config.label_names[c]}{tag} | {counts[c]} | {per_class[c]:.3f} |")
    md.append("")

    out_md = Path(f"data/{dataset}/processed/step4_feedback/STEP4_{dataset.upper()}_RESULTS.md")
    out_md.write_text("\n".join(md))

    print(f"\n=== {config.display_name} LADDER (pooled OOF macro-F1, {len(ec)} classes) ===")
    for name, key in [("GNN-alone", "gnn_alone"), ("LLM-alone", "llm_alone"),
                      ("AGAF     ", "agaf"), ("Loop     ", "feedback_loop")]:
        print(f"  {name}: {rungs[key]:.4f}")
    print(f"  Ladder order holds: {ladder_ok}")
    for label, key in [("loop-AGAF", "loop_vs_agaf"), ("AGAF-LLM", "agaf_vs_llm"),
                       ("AGAF-GNN", "agaf_vs_gnn")]:
        b = bootstraps[key]
        print(f"  {label}: Δ={b['mean_diff']:+.4f} CI[{b['ci_low']:+.4f},{b['ci_high']:+.4f}] P={b['prob_positive']:.3f}")
    print(f"\nWrote {out_json}\nWrote {out_md}")
    return out_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ton_iot", choices=["ton_iot", "unsw_nb15"])
    args = parser.parse_args()
    assemble_ladder(args.dataset)


if __name__ == "__main__":
    main()
