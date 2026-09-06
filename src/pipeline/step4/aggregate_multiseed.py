"""Aggregate a 3-seed re-run of the ladder into one artifact.

Consumes the per-seed captures written by the multi-seed sweep
(`results/multiseed/{dataset}_seed{S}/`) and reports, for each dataset:

  * per-rung pooled OOF macro-F1 at every seed, plus mean and standard deviation;
  * whether each pairwise comparison keeps its sign at every seed;
  * a TWO-LEVEL bootstrap CI that resamples BOTH edges and the training seed, so
    the interval carries rung-side seed variance rather than assuming it away;
  * a seed-matched bootstrap (edges only, paired within a seed, averaged across
    seeds) as the secondary, narrower estimate.

Why two levels. The single-seed contracts bootstrap edges at a fixed seed, which
answers "would this gap survive a different sample of edges?" but not "would it
survive a different training run?". `results/unsw_nb15_current.json` records a
conversation-only note that AGAF and the loop swap order across seeds; that claim
was never artifact-backed. This module produces the artifact either way.

Scope. The LLM rung is deterministic given the folds and the frozen CySecBERT
embeddings -- its prototypes come from `build_prototypes`, which contains no
training randomness -- so it carries no seed variance and is reported without a
two-level interval. Comparisons involving it keep their single-seed CIs.

Run:
    python -m src.pipeline.step4.aggregate_multiseed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.pipeline.common.datasets import DATASETS, get_dataset_config
from src.pipeline.common.splits import eval_macro_f1, mask_dropped_logits

SEEDS = (42, 1, 2)
BOOTSTRAP_ITERS = 2000
BOOTSTRAP_SEED = 42
CAPTURE_ROOT = Path("results/multiseed")
# Seed-varying rungs only. See the module docstring on the LLM rung.
RUNGS = ("gnn", "agaf", "loop")
COMPARISONS = (("loop", "agaf"), ("loop", "gnn"), ("agaf", "gnn"))


# Knobs that must agree across the seeds being pooled. They are read from each
# capture's own benchmark_summary.json — the value the run recorded, not the value a
# config file claims — because that is exactly where the two diverged in the sweep
# behind results/multiseed_ladder.json.
PINNED_KNOBS = ("injection_scale", "top_k_percent")


def _capture_dir(dataset: str, seed: int, capture_root: Path | None = None) -> Path:
    return (capture_root or CAPTURE_ROOT) / f"{dataset}_seed{seed}"


def _load_seed_knobs(dataset: str, seed: int, capture_root: Path | None = None) -> dict:
    """The knobs one capture actually ran at, or None where it did not record them."""
    path = _capture_dir(dataset, seed, capture_root) / "benchmark_summary.json"
    if not path.exists():
        return {k: None for k in PINNED_KNOBS}
    payload = json.loads(path.read_text())
    return {k: payload.get(k) for k in PINNED_KNOBS}


def _verify_knobs(dataset: str, seeds, capture_root: Path | None = None) -> dict:
    per_seed = {s: _load_seed_knobs(dataset, s, capture_root) for s in seeds}
    resolved: dict = {}
    for knob in PINNED_KNOBS:
        values = {s: v[knob] for s, v in per_seed.items()}
        distinct = {v for v in values.values() if v is not None}
        if len(distinct) > 1:
            raise RuntimeError(
                f"[{dataset}] cannot pool seeds that ran at different {knob}: "
                + ", ".join(f"seed {s}={values[s]}" for s in sorted(values))
                + ". Re-run the disagreeing seeds at one value."
            )
        resolved[knob] = next(iter(distinct)) if distinct else None
    resolved["verified"] = all(
        all(v[k] is not None for k in PINNED_KNOBS) for v in per_seed.values()
    )
    resolved["per_seed"] = {str(s): per_seed[s] for s in seeds}
    return resolved


def _load_seed_preds(
    dataset: str, seed: int, capture_root: Path | None = None
) -> dict[str, np.ndarray]:
    """Predictions for one dataset at one training seed, scored the ladder's way."""
    config = get_dataset_config(dataset)
    d = _capture_dir(dataset, seed, capture_root)
    if not d.is_dir():
        raise FileNotFoundError(f"missing capture directory {d}")

    def from_logits(name: str) -> np.ndarray:
        logits = torch.load(d / name, weights_only=False)
        if isinstance(logits, dict):
            logits = logits.get("logits", next(iter(logits.values())))
        return mask_dropped_logits(logits, config.dropped_classes).argmax(1).cpu().numpy()

    agaf = np.asarray(json.loads((d / "metrics.json").read_text())["predictions"])
    return {
        "gnn": from_logits("oof_logits.pt"),
        "loop": from_logits("feedback_oof_real.pt"),
        "agaf": agaf,
    }


def _two_level_bootstrap(
    labels: np.ndarray,
    per_seed: dict[int, dict[str, np.ndarray]],
    a: str,
    b: str,
    eval_classes,
    iters: int = BOOTSTRAP_ITERS,
) -> dict:
    """Resample edges AND the training seed.

    Each iteration draws one seed uniformly from the completed runs and one
    bootstrap sample of edges, then scores both rungs on that same draw. The
    resulting interval answers "would this gap survive a different training run
    and a different sample of edges?", which is the question a reader asks of a
    single-seed headline.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    seeds = sorted(per_seed)
    n = len(labels)
    diffs = np.empty(iters)
    for i in range(iters):
        s = seeds[rng.integers(len(seeds))]
        idx = rng.integers(0, n, n)
        preds = per_seed[s]
        diffs[i] = eval_macro_f1(
            labels[idx], preds[a][idx], eval_classes
        ) - eval_macro_f1(labels[idx], preds[b][idx], eval_classes)
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "prob_positive": float((diffs > 0).mean()),
        "resampled": "edges_and_training_seed",
        "n_seeds": len(seeds),
    }


def _seed_matched_bootstrap(
    labels: np.ndarray,
    per_seed: dict[int, dict[str, np.ndarray]],
    a: str,
    b: str,
    eval_classes,
    iters: int = BOOTSTRAP_ITERS,
) -> dict:
    """Edges only, paired within each seed, then averaged over seeds."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    seeds = sorted(per_seed)
    n = len(labels)
    diffs = np.empty(iters)
    for i in range(iters):
        idx = rng.integers(0, n, n)
        per = [
            eval_macro_f1(labels[idx], per_seed[s][a][idx], eval_classes)
            - eval_macro_f1(labels[idx], per_seed[s][b][idx], eval_classes)
            for s in seeds
        ]
        diffs[i] = float(np.mean(per))
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "prob_positive": float((diffs > 0).mean()),
        "resampled": "edges_only_seed_matched",
        "n_seeds": len(seeds),
    }


def aggregate(dataset: str, seeds=SEEDS, capture_root: Path | None = None) -> dict:
    config = get_dataset_config(dataset)
    data = torch.load(config.graph_path, weights_only=False)
    labels = data.edge_label.cpu().numpy()

    knobs = _verify_knobs(dataset, seeds, capture_root)
    per_seed = {s: _load_seed_preds(dataset, s, capture_root) for s in seeds}
    scores = {
        rung: {s: eval_macro_f1(labels, per_seed[s][rung], config.eval_classes) for s in seeds}
        for rung in RUNGS
    }

    # Only `llm_alone_prototype` is read from the ladder summary. That rung is an
    # argmax over the frozen prototype scorer, so it is invariant to both the training
    # seed and the consultant temperature — nothing else in that file is trusted here.
    llm = json.loads(
        (
            _capture_dir(dataset, seeds[0], capture_root) / "ladder_summary.json"
        ).read_text()
    )["llm_alone_prototype"]

    per_rung = {
        rung: {
            "per_seed": {str(s): scores[rung][s] for s in seeds},
            "mean": float(np.mean(list(scores[rung].values()))),
            "std": float(np.std(list(scores[rung].values()), ddof=1)),
        }
        for rung in RUNGS
    }
    per_rung["llm"] = {
        "per_seed": {str(s): llm for s in seeds},
        "mean": llm,
        "std": 0.0,
        "note": "deterministic given folds and frozen embeddings; carries no training-seed variance",
    }

    comparisons = {}
    for a, b in COMPARISONS:
        signs = {str(s): float(scores[a][s] - scores[b][s]) for s in seeds}
        comparisons[f"{a}_vs_{b}"] = {
            "per_seed_diff": signs,
            "sign_stable_across_seeds": len({v > 0 for v in signs.values()}) == 1,
            "two_level": _two_level_bootstrap(
                labels, per_seed, a, b, config.eval_classes
            ),
            "seed_matched": _seed_matched_bootstrap(
                labels, per_seed, a, b, config.eval_classes
            ),
        }

    order = sorted(
        [(r, per_rung[r]["mean"]) for r in list(RUNGS) + ["llm"]],
        key=lambda kv: kv[1],
        reverse=True,
    )
    return {
        "schema_version": 1,
        "dataset": config.display_name,
        "dataset_key": dataset,
        "metric": "pooled_oof_macro_f1",
        "n_eval_classes": len(config.eval_classes),
        "seeds": list(seeds),
        "capture_root": str(capture_root or CAPTURE_ROOT),
        "feedback_configuration": knobs,
        "protocol": (
            "Fold partition held fixed at the seed-42 stratified split; only training "
            "seeds vary. The pooled OOF edge set is therefore identical across seeds, "
            "which is what makes the seed-matched bootstrap a paired comparison."
        ),
        "rungs": per_rung,
        "comparisons": comparisons,
        "highest_mean_rung": order[0][0],
        "mean_order": " > ".join(f"{r} {v:.4f}" for r, v in order),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=sorted(DATASETS))
    parser.add_argument("--output", default="results/multiseed_ladder.json")
    parser.add_argument(
        "--capture-root", default=None,
        help="Directory holding the per-seed captures (default: results/multiseed).",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()

    capture_root = Path(args.capture_root) if args.capture_root else None
    out = {
        d: aggregate(d, seeds=tuple(args.seeds), capture_root=capture_root)
        for d in args.datasets
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n")

    for d, r in out.items():
        print(f"\n=== {r['dataset']} ({r['n_eval_classes']} classes) ===")
        k = r["feedback_configuration"]
        print(
            f"  knobs: top_k={k['top_k_percent']} injection_scale={k['injection_scale']}"
            f" verified={k['verified']}"
        )
        for rung, v in r["rungs"].items():
            per = "  ".join(f"{s}:{x:.4f}" for s, x in v["per_seed"].items())
            print(f"  {rung:<5} mean {v['mean']:.4f} +/- {v['std']:.4f}   [{per}]")
        print(f"  order: {r['mean_order']}")
        for name, c in r["comparisons"].items():
            t = c["two_level"]
            print(
                f"  {name:<14} two-level {t['mean_diff']:+.4f} "
                f"[{t['ci_low']:+.4f}, {t['ci_high']:+.4f}] P={t['prob_positive']:.3f}"
                f"  sign_stable={c['sign_stable_across_seeds']}"
            )
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
