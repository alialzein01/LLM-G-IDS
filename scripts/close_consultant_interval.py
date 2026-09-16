#!/usr/bin/env python3
"""Interval for the consultant change, closing the report's first [GAP].

The report states that swapping the loop's consultant, from the embedding-only
whitened prototype scorer to a per-fold classification head trained on the same
frozen semantic embeddings, moved pooled out-of-fold macro-F1 by +0.0697 on
NF-UNSW-NB15 and +0.0523 on NF-ToN-IoT. Those were point estimates: no bootstrap
was ever run between the two loop configurations, while every other difference in
the report carries one.

Nothing needs retraining. Both configurations already wrote their pooled
out-of-fold predictions at all three seeds:

    results/multiseed_v2/head/<dataset>_seed<S>/feedback_oof_real.pt      trained head
    results/multiseed_v2/legacy/<dataset>_seed<S>/feedback_oof_real.pt    prototype

so this is a bootstrap over tensors that are already on disk. It reuses the exact
procedures the rest of the report uses, imported from `aggregate_multiseed`, so
the resulting interval is comparable with every other interval in the report:
the two-level bootstrap over edges and training seed as primary, with the seed
drawn independently for each arm, and the seed-matched bootstrap over edges
alone as secondary, both at 2,000 iterations from bootstrap seed 42.

**The comparison is confounded and stays confounded.** The two configurations
differ in the consultant AND in the selected entropy percentile, 31 to 29 on
NF-UNSW-NB15 and 25 to 16 on NF-ToN-IoT. The injection scale is the same in both.
An interval makes the difference measurable; it does not make it attributable, and
every statement of this number must keep saying so.

Writes `results/consultant_change_interval.json`.

Run:
    OMP_NUM_THREADS=1 python scripts/close_consultant_interval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.common.datasets import get_dataset_config
from src.pipeline.common.splits import eval_macro_f1, mask_dropped_logits
from src.pipeline.step4.aggregate_multiseed import (
    BOOTSTRAP_ITERS,
    BOOTSTRAP_SEED,
    SEEDS,
    _seed_matched_bootstrap,
    independent_seed_two_level,
)

ROOT = Path(__file__).resolve().parents[1]
CAPTURES = ROOT / "results" / "multiseed_v2"
OUT = ROOT / "results" / "consultant_change_interval.json"

# capture directory -> the name this configuration carries in the report
ARMS = {"head": "loop_trained_head", "legacy": "loop_prototype"}

# Reported in the contracts' `configuration` and `supersedes`; repeated here so the
# artifact carries the confound rather than relying on the reader to look it up.
KNOBS = {
    "unsw_nb15": {"loop_trained_head": {"top_k_percent": 29.0, "injection_scale": 2.0},
                  "loop_prototype": {"top_k_percent": 31.0, "injection_scale": 2.0}},
    "ton_iot": {"loop_trained_head": {"top_k_percent": 16.0, "injection_scale": 20.0},
                "loop_prototype": {"top_k_percent": 25.0, "injection_scale": 20.0}},
}


def load_arm(dataset: str, arm_dir: str, seed: int, config) -> np.ndarray:
    path = CAPTURES / arm_dir / f"{dataset}_seed{seed}" / "feedback_oof_real.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    logits = torch.load(path, weights_only=False)
    return mask_dropped_logits(logits, config.dropped_classes).argmax(1).cpu().numpy()


def run(dataset: str) -> dict:
    config = get_dataset_config(dataset)
    data = torch.load(config.graph_path, weights_only=False)
    labels = data.edge_label.cpu().numpy()

    per_seed: dict[int, dict[str, np.ndarray]] = {}
    for seed in SEEDS:
        per_seed[seed] = {
            name: load_arm(dataset, arm_dir, seed, config)
            for arm_dir, name in ARMS.items()
        }

    per_seed_scores = {
        name: {
            str(seed): float(eval_macro_f1(labels, per_seed[seed][name], config.eval_classes))
            for seed in SEEDS
        }
        for name in ARMS.values()
    }
    means = {n: float(np.mean(list(v.values()))) for n, v in per_seed_scores.items()}
    point = means["loop_trained_head"] - means["loop_prototype"]

    two_level = independent_seed_two_level(
        labels, per_seed, "loop_trained_head", "loop_prototype", config.eval_classes
    )
    seed_matched = _seed_matched_bootstrap(
        labels, per_seed, "loop_trained_head", "loop_prototype", config.eval_classes
    )
    signs = [
        per_seed_scores["loop_trained_head"][str(s)]
        - per_seed_scores["loop_prototype"][str(s)]
        for s in SEEDS
    ]
    return {
        "dataset_key": dataset,
        "dataset": config.display_name if hasattr(config, "display_name") else dataset,
        "n_eval_classes": len(config.eval_classes),
        "n_edges": int(labels.shape[0]),
        "per_seed_macro_f1": per_seed_scores,
        "mean_macro_f1": means,
        "point_difference": point,
        "per_seed_difference": {str(s): d for s, d in zip(SEEDS, signs)},
        "sign_stable_across_seeds": bool(all(d > 0 for d in signs) or all(d < 0 for d in signs)),
        "two_level": two_level,
        "seed_matched": seed_matched,
        "separated_two_level": bool(two_level["ci_low"] > 0 or two_level["ci_high"] < 0),
        "knobs": KNOBS[dataset],
        "confounded": (
            "The two configurations differ in the consultant AND in the selected "
            "entropy percentile (top_k_percent). The injection scale is identical. "
            "The interval measures the change as a whole and does not attribute it "
            "to the consultant alone."
        ),
    }


def main() -> None:
    out = {
        "schema_version": 1,
        "metric": "pooled_oof_macro_f1",
        "comparison": "loop_trained_head minus loop_prototype",
        "seeds": list(SEEDS),
        "bootstrap_iters": BOOTSTRAP_ITERS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "protocol": (
            "Fold partition held fixed at the seed-42 stratified split; only training "
            "seeds vary. Primary interval resamples edges and draws a training seed "
            "independently for each arm; the "
            "seed-matched interval resamples edges only, paired within a seed. A "
            "difference counts as separated only when the primary interval excludes zero."
        ),
        "source_artifacts": {
            name: f"results/multiseed_v2/{arm}/<dataset>_seed<S>/feedback_oof_real.pt"
            for arm, name in ARMS.items()
        },
        "closes": "report [GAP] in sections/04_results.tex, the consultant-change interval",
    }
    for dataset in ("unsw_nb15", "ton_iot"):
        out[dataset] = run(dataset)
        r = out[dataset]
        t = r["two_level"]
        print(
            f"{dataset:10s} point {r['point_difference']:+.4f}  "
            f"two-level {t['mean_diff']:+.4f} [{t['ci_low']:+.4f}, {t['ci_high']:+.4f}]  "
            f"P={t['prob_positive']:.4f}  separated={r['separated_two_level']}"
        )
        s = r["seed_matched"]
        print(
            f"{'':10s} seed-matched {s['mean_diff']:+.4f} "
            f"[{s['ci_low']:+.4f}, {s['ci_high']:+.4f}]  P={s['prob_positive']:.4f}"
        )
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
