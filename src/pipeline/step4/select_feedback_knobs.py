"""Re-select the two per-dataset feedback knobs on validation folds, 3 seeds.

`top_k_percent` was originally selected against the entropy of a head that was never
trained as a classifier (it had nonzero F1 on 1-2 of 10 classes), and `injection_scale`
was selected against a consultant whose softmax was uniform. Both signals changed in
Tasks 1-2, and PROJECT_NOTES.md requires `injection_scale` to be re-selected whenever the
injection path changes, so neither old value carries.

Protocol, per dataset, never mixed across datasets:
  stage `top_k`  — 15..35, at the dataset's currently selected scale, one run per seed;
  stage `scale`  — {0.5, 1, 2, 5, 10, 20} at the selected k, one run per seed;
  stage `finalize` — mean `best_val_macro_f1` across seeds picks each knob. Test-fold
                     numbers are recorded by the sweep but never consulted here.

Run (see scripts/reselect_knobs.sh for the parallel driver):
    python -m src.pipeline.step4.select_feedback_knobs --dataset unsw_nb15 \
        --stage top_k --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.pipeline.common.datasets import DATASETS
from src.pipeline.step4 import sweep_top_k as sweeper
from src.pipeline.step4.feedback_config import (
    load_feedback_config,
    selected_config_path,
)

SELECTION_SEEDS = (42, 1, 2)
TOP_K_RANGE = tuple(range(15, 36))
SCALE_CANDIDATES = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
SELECTION_SOURCE = "multiseed_validation_sweep_v2_fixed_signals"
RESULT_ROOT = Path("results/knob_selection_v2")


def run_root(dataset: str) -> Path:
    return RESULT_ROOT / dataset


def select_by_mean_validation(per_seed: dict[int, dict[float, float]]) -> float:
    """Argmax of the across-seed mean validation macro-F1; smaller value wins a tie.

    `per_seed` maps seed -> {candidate: mean_best_val_macro_f1}. Every seed must
    have scored every candidate, otherwise the mean is over different populations.
    """
    if not per_seed:
        raise ValueError("no seeds to select from")
    candidate_sets = {frozenset(v) for v in per_seed.values()}
    if len(candidate_sets) != 1:
        raise ValueError(
            "seeds scored different candidate sets: "
            + "; ".join(f"seed {s}: {sorted(v)}" for s, v in sorted(per_seed.items()))
        )
    candidates = sorted(next(iter(candidate_sets)))
    means = {
        c: sum(per_seed[s][c] for s in per_seed) / len(per_seed) for c in candidates
    }
    return min(candidates, key=lambda c: (-means[c], c))


def _sweep(dataset: str, seed: int, candidates, injection_scale: float, tag: str) -> dict:
    out_dir = run_root(dataset) / f"{tag}_seed{seed}"
    summary_path = sweeper.run_top_k_sweep(
        dataset,
        candidates=tuple(candidates),
        output_dir=out_dir,
        injection_scale=injection_scale,
        seed=seed,
        write_config=False,
    )
    return json.loads(Path(summary_path).read_text())


def stage_top_k(dataset: str, seed: int) -> Path:
    """Sweep top_k at the dataset's current scale — the scale is re-picked after."""
    current = load_feedback_config(dataset)
    scale = float(current["injection_scale"])
    summary = _sweep(dataset, seed, TOP_K_RANGE, scale, "top_k")
    payload = {
        "dataset": dataset,
        "seed": seed,
        "stage": "top_k",
        "injection_scale_held_at": scale,
        "validation_by_candidate": {
            str(row["top_k_percent"]): row["mean_best_val_macro_f1"]
            for row in summary["candidates"]
        },
        "test_by_candidate_not_used_for_selection": {
            str(row["top_k_percent"]): row["pooled_oof_test_macro_f1"]
            for row in summary["candidates"]
        },
    }
    path = run_root(dataset) / f"top_k_seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def stage_scale(dataset: str, seed: int, top_k: float) -> Path:
    """Sweep injection_scale at the freshly selected k."""
    validation: dict[str, float] = {}
    test: dict[str, float] = {}
    for scale in SCALE_CANDIDATES:
        summary = _sweep(
            dataset, seed, (int(top_k),), scale, f"scale_{scale:g}"
        )
        row = summary["candidates"][0]
        validation[str(scale)] = row["mean_best_val_macro_f1"]
        test[str(scale)] = row["pooled_oof_test_macro_f1"]
    payload = {
        "dataset": dataset,
        "seed": seed,
        "stage": "scale",
        "top_k_held_at": top_k,
        "validation_by_candidate": validation,
        "test_by_candidate_not_used_for_selection": test,
    }
    path = run_root(dataset) / f"scale_seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _collect(dataset: str, stage: str, seeds=SELECTION_SEEDS) -> dict[int, dict[float, float]]:
    out: dict[int, dict[float, float]] = {}
    for seed in seeds:
        path = run_root(dataset) / f"{stage}_seed{seed}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing {stage} run for seed {seed}: {path}")
        payload = json.loads(path.read_text())
        out[seed] = {
            float(k): float(v)
            for k, v in payload["validation_by_candidate"].items()
        }
    return out


def selected_top_k(dataset: str, seeds=SELECTION_SEEDS) -> float:
    return select_by_mean_validation(_collect(dataset, "top_k", seeds))


def finalize(dataset: str, seeds=SELECTION_SEEDS) -> Path:
    """Write the new knobs into selected_feedback_config.json, old ones superseded."""
    previous = load_feedback_config(dataset)
    top_k = selected_top_k(dataset, seeds)
    scale = select_by_mean_validation(_collect(dataset, "scale", seeds))

    top_k_curves = _collect(dataset, "top_k", seeds)
    scale_curves = _collect(dataset, "scale", seeds)

    payload = dict(previous)
    payload["superseded"] = {
        "top_k_percent": previous.get("top_k_percent"),
        "injection_scale": previous.get("injection_scale"),
        "source": previous.get("source"),
        "reason": (
            "top_k was selected against the entropy of an unsupervised selector head "
            "(1-2 of 10 classes with nonzero F1) and injection_scale against a uniform "
            "consultant softmax (mean_disagreement 0.900). Both signals changed."
        ),
    }
    payload.update({
        "source": SELECTION_SOURCE,
        "selection_uses_test_labels": False,
        "selection_metric": "mean_best_val_macro_f1",
        "selection_seeds": list(seeds),
        "selection_parameter": "top_k_percent+injection_scale",
        "top_k_percent": float(top_k),
        "injection_scale": float(scale),
        "effective_feedback_percent": float(top_k)
        * float(payload["bias_confidence_fraction"]),
        "selection_curves": {
            "top_k": {
                str(s): {str(k): v for k, v in sorted(top_k_curves[s].items())}
                for s in sorted(top_k_curves)
            },
            "injection_scale": {
                str(s): {str(k): v for k, v in sorted(scale_curves[s].items())}
                for s in sorted(scale_curves)
            },
        },
    })
    path = selected_config_path(dataset)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"[{dataset}] top_k {previous.get('top_k_percent')} -> {top_k}; "
        f"injection_scale {previous.get('injection_scale')} -> {scale}  ({path})"
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--stage", required=True, choices=("top_k", "scale", "finalize"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--top-k", type=float)
    args = parser.parse_args()

    if args.stage == "top_k":
        print(stage_top_k(args.dataset, args.seed))
    elif args.stage == "scale":
        top_k = args.top_k if args.top_k is not None else selected_top_k(args.dataset)
        print(stage_scale(args.dataset, args.seed, top_k))
    else:
        finalize(args.dataset)


if __name__ == "__main__":
    main()
