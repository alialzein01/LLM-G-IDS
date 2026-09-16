#!/usr/bin/env python3
"""Bound what the NF-ToN-IoT scoring asymmetry could be worth.

Two NF-ToN-IoT classes are excluded from the metric, `dos` with 4 edges and
`ransomware` with 3. Rows whose true label is one of those are not scored. The
asymmetry is on the other side: until 2026-09-16 the fusion stage and the
re-trained baselines could *predict* one of those classes for an edge that is
scored, and the GNN, semantic and feedback stages could not, because they drove
those logit columns to -inf before every argmax. An edge lost that way is a miss
the other three stages were structurally unable to make.

The code is fixed now, but the published numbers were produced under the old
rule and no rung is being retrained. So this script bounds the effect instead,
from the predictions already on disk. For each affected model and seed it
reports:

  * how many scored edges were predicted into a dropped class;
  * the macro-F1 actually published;
  * an UPPER BOUND, obtained by replacing every one of those predictions with
    the true label. That is the most generous possible repair: it assumes the
    mask would have sent each of those edges to exactly the right class, which
    no mask can guarantee. The real corrected value lies between the two.

The loop's macro-F1 at the same seed sits beside each row, because the question
the bound has to answer is whether any comparison in the report could turn over.

Writes `results/ton_iot_dropped_class_bound.json`.

Run:
    OMP_NUM_THREADS=1 IDS_FORCE_CPU=1 python scripts/ton_dropped_class_bound.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.common.datasets import get_dataset_config
from src.pipeline.common.splits import eval_macro_f1, mask_dropped_logits

DATASET = "ton_iot"
SEEDS = (42, 1, 2)
CAPTURES = ROOT / "results" / "multiseed_v2" / "head"
BASELINE_PREDS = ROOT / "data" / DATASET / "processed" / "baselines"
OUT = ROOT / "results" / "ton_iot_dropped_class_bound.json"

# The six re-trained baseline variants, in the order `tab_baselines` prints them.
BASELINES = [
    ("e_graphsage", "as_published"),
    ("e_graphsage", "refit"),
    ("e_graphsage", "plus_node_features"),
    ("te_g_sage", "as_published"),
    ("te_g_sage", "plus_node_features"),
    ("te_g_sage", "refit"),
]
DISPLAY = {
    ("e_graphsage", "as_published"): "E-GraphSAGE as published",
    ("e_graphsage", "refit"): "E-GraphSAGE refit",
    ("e_graphsage", "plus_node_features"): "E-GraphSAGE + our node features",
    ("te_g_sage", "as_published"): "TE-G-SAGE as published",
    ("te_g_sage", "plus_node_features"): "TE-G-SAGE + our node features",
    ("te_g_sage", "refit"): "TE-G-SAGE refit",
}


def _capture(seed: int) -> Path:
    return CAPTURES / f"{DATASET}_seed{seed}"


def _masked_argmax(logits, dropped) -> np.ndarray:
    return mask_dropped_logits(logits, dropped).argmax(1).cpu().numpy()


def fusion_preds(seed: int) -> np.ndarray:
    """The fusion stage's pooled out-of-fold predictions, exactly as published.

    Read unmasked on purpose: these are the predictions the contract records,
    and the point of this script is to measure what the missing mask cost.
    """
    d = _capture(seed)
    slim = d / "agaf_predictions.json"
    src = slim if slim.exists() else d / "metrics.json"
    return np.asarray(json.loads(src.read_text())["predictions"])


def loop_preds(seed: int, config) -> np.ndarray:
    logits = torch.load(_capture(seed) / "feedback_oof_real.pt", weights_only=False)
    return _masked_argmax(logits, config.dropped_classes)


def baseline_preds(model: str, mode: str) -> np.ndarray:
    return np.load(BASELINE_PREDS / f"{model}_{mode}_oof_preds.npy")


def bound_row(labels, preds, scored, dropped, eval_classes) -> dict:
    """Counts and the two macro-F1 values for one model at one seed."""
    into_dropped = bool_into_dropped = np.isin(preds, list(dropped)) & scored
    n_into = int(bool_into_dropped.sum())

    repaired = preds.copy()
    repaired[into_dropped] = labels[into_dropped]

    actual = float(eval_macro_f1(labels, preds, eval_classes))
    bound = float(eval_macro_f1(labels, repaired, eval_classes))
    return {
        "predictions_into_dropped_classes_on_scored_rows": n_into,
        "macro_f1": actual,
        "macro_f1_upper_bound": bound,
        "headroom": bound - actual,
    }


def main() -> None:
    config = get_dataset_config(DATASET)
    data = torch.load(config.graph_path, weights_only=False)
    labels = data.edge_label.cpu().numpy()
    dropped = config.dropped_classes
    eval_classes = config.eval_classes
    scored = np.isin(labels, list(eval_classes))

    loop = {s: loop_preds(s, config) for s in SEEDS}
    loop_f1 = {
        str(s): float(eval_macro_f1(labels, loop[s], eval_classes)) for s in SEEDS
    }

    models: dict[str, dict] = {}

    fusion = {s: fusion_preds(s) for s in SEEDS}
    models["fusion"] = {
        "display_name": "Fusion model (AGAF)",
        "source": "results/multiseed_v2/head/ton_iot_seed<S>/metrics.json:predictions",
        "per_seed": {
            str(s): bound_row(labels, fusion[s], scored, dropped, eval_classes)
            for s in SEEDS
        },
    }

    for model, mode in BASELINES:
        preds = baseline_preds(model, mode)
        models[f"{model}_{mode}"] = {
            "display_name": DISPLAY[(model, mode)],
            "source": f"data/{DATASET}/processed/baselines/{model}_{mode}_oof_preds.npy",
            "per_seed": {
                str(s): bound_row(labels, preds[i], scored, dropped, eval_classes)
                for i, s in enumerate(SEEDS)
            },
        }

    # The only thing a reader needs from all of this: does any bound reach the
    # loop it is compared against, at any seed?
    exceeds = [
        {"model": key, "seed": s,
         "macro_f1_upper_bound": row["macro_f1_upper_bound"],
         "feedback_macro_f1": loop_f1[s]}
        for key, block in models.items()
        for s, row in block["per_seed"].items()
        if row["macro_f1_upper_bound"] > loop_f1[s]
    ]

    payload = {
        "schema_version": 1,
        "dataset": config.display_name,
        "dataset_key": DATASET,
        "metric": "pooled_oof_macro_f1",
        "seeds": list(SEEDS),
        "n_edges": int(labels.shape[0]),
        "n_scored_edges": int(scored.sum()),
        "dropped_classes": list(dropped),
        "dropped_class_names": [config.label_names[c] for c in dropped],
        "dropped_class_edges": {
            config.label_names[c]: int((labels == c).sum()) for c in dropped
        },
        "what_this_measures": (
            "Scored edges that the fusion model or a re-trained baseline predicted "
            "into a class the metric excludes. The GNN, semantic and feedback "
            "models could not make that prediction because their dropped-class "
            "logits were masked before argmax. The upper bound replaces every such "
            "prediction with the true label, which is the most generous repair "
            "available and is therefore an upper bound, not an estimate."
        ),
        "feedback_macro_f1_per_seed": loop_f1,
        "models": models,
        "bounds_that_exceed_the_feedback_model": exceeds,
        "code_fixed": (
            "src/pipeline/step3/train_fusion.py, src/pipeline/step3/train_gnn.py and "
            "src/pipeline/baselines/harness.py now mask dropped classes before every "
            "argmax; tests/test_dropped_class_masking.py pins it. The numbers here "
            "describe the published runs, which predate that fix and were not re-run."
        ),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")

    width = max(len(b["display_name"]) for b in models.values())
    print(f"{'model':<{width}}  seed  into  actual -> bound   feedback")
    for key, block in models.items():
        for s in SEEDS:
            row = block["per_seed"][str(s)]
            flag = " <-- above the feedback model" if (
                row["macro_f1_upper_bound"] > loop_f1[str(s)]
            ) else ""
            print(
                f"{block['display_name']:<{width}}  {s:>4}  {row['predictions_into_dropped_classes_on_scored_rows']:>4}  "
                f"{row['macro_f1']:.4f} -> {row['macro_f1_upper_bound']:.4f}   "
                f"{loop_f1[str(s)]:.4f}{flag}"
            )
    print(f"\n-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
