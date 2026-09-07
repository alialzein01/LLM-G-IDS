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
HEAD_SELECTION_SOURCE = "multiseed_validation_sweep_v2_trained_head_consultant"
RESULT_ROOT = Path("results/knob_selection_v2")
HEAD_RESULT_ROOT = Path("results/knob_selection_head")


def run_root(dataset: str, use_llm_head: bool = False) -> Path:
    """Head-consultant sweeps write to their own tree: the two consultants pick
    different knobs, and mixing their curves in one directory is how a stale
    candidate score gets read as the current one."""
    return (HEAD_RESULT_ROOT if use_llm_head else RESULT_ROOT) / dataset


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


def _sweep(dataset: str, seed: int, candidates, injection_scale: float, tag: str,
           use_llm_head: bool = False) -> dict:
    out_dir = run_root(dataset, use_llm_head) / f"{tag}_seed{seed}"
    summary_path = sweeper.run_top_k_sweep(
        dataset,
        candidates=tuple(candidates),
        output_dir=out_dir,
        injection_scale=injection_scale,
        seed=seed,
        write_config=False,
        use_llm_head=use_llm_head,
    )
    return json.loads(Path(summary_path).read_text())


def _starting_scale(dataset: str, use_llm_head: bool) -> float:
    """Scale to hold top_k at during stage 1. For the head consultant the
    prototype's selected scale is the only value on file, so it is the starting
    point; stage 2 then re-picks it for the head."""
    current = load_feedback_config(dataset)
    if use_llm_head:
        block = current.get("consultant_trained_llm_head")
        if block and block.get("injection_scale") is not None:
            return float(block["injection_scale"])
    return float(current["injection_scale"])


def stage_top_k(dataset: str, seed: int, use_llm_head: bool = False) -> Path:
    """Sweep top_k at the dataset's current scale — the scale is re-picked after."""
    scale = _starting_scale(dataset, use_llm_head)
    summary = _sweep(dataset, seed, TOP_K_RANGE, scale, "top_k", use_llm_head)
    payload = {
        "dataset": dataset,
        "seed": seed,
        "stage": "top_k",
        "consultant": "trained_llm_head" if use_llm_head else "whitened_prototype_scorer",
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
    path = run_root(dataset, use_llm_head) / f"top_k_seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def stage_scale(dataset: str, seed: int, top_k: float,
                use_llm_head: bool = False) -> Path:
    """Sweep injection_scale at the freshly selected k."""
    validation: dict[str, float] = {}
    test: dict[str, float] = {}
    for scale in SCALE_CANDIDATES:
        summary = _sweep(
            dataset, seed, (int(top_k),), scale, f"scale_{scale:g}", use_llm_head
        )
        row = summary["candidates"][0]
        validation[str(scale)] = row["mean_best_val_macro_f1"]
        test[str(scale)] = row["pooled_oof_test_macro_f1"]
    payload = {
        "dataset": dataset,
        "seed": seed,
        "stage": "scale",
        "consultant": "trained_llm_head" if use_llm_head else "whitened_prototype_scorer",
        "top_k_held_at": top_k,
        "validation_by_candidate": validation,
        "test_by_candidate_not_used_for_selection": test,
    }
    path = run_root(dataset, use_llm_head) / f"scale_seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _collect(dataset: str, stage: str, seeds=SELECTION_SEEDS,
             use_llm_head: bool = False) -> dict[int, dict[float, float]]:
    out: dict[int, dict[float, float]] = {}
    for seed in seeds:
        path = run_root(dataset, use_llm_head) / f"{stage}_seed{seed}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing {stage} run for seed {seed}: {path}")
        payload = json.loads(path.read_text())
        out[seed] = {
            float(k): float(v)
            for k, v in payload["validation_by_candidate"].items()
        }
    return out


def selected_top_k(dataset: str, seeds=SELECTION_SEEDS,
                   use_llm_head: bool = False) -> float:
    return select_by_mean_validation(
        _collect(dataset, "top_k", seeds, use_llm_head)
    )


def _finalize_head(dataset, previous, top_k, scale, seeds, curves) -> Path:
    """Promote the trained-head consultant to the active knobs.

    The prototype's knobs are kept verbatim under `prototype_superseded`: they are
    what `results/*_current.json` schema 4/6 was measured under, and the ladder in
    that contract cannot be re-derived without them.
    """
    payload = dict(previous)
    prototype_block = {
        k: previous.get(k)
        for k in (
            "source", "top_k_percent", "injection_scale",
            "effective_feedback_percent", "mean_validation_macro_f1",
            "validation_macro_f1_std", "pooled_oof_test_macro_f1",
            "selection_seeds", "active_condition", "active_condition_note",
        )
        if previous.get(k) is not None
    }
    prototype_block["semantic_consultant"] = "whitened_prototype_scorer"
    prototype_block["note"] = (
        "Condition A with the whitened-prototype consultant — the configuration "
        "results/*_current.json schema 4 (UNSW) / 6 (ToN) was measured under. "
        "Superseded 2026-09-07 by the trained-head consultant; reachable by "
        "running train_feedback without --use-llm-head at these knobs."
    )
    for key in ("condition_c",):
        if key in previous:
            prototype_block[key] = previous[key]
            payload.pop(key, None)

    head_block = {
        "source": HEAD_SELECTION_SOURCE,
        "selection_uses_test_labels": False,
        "selection_metric": "mean_best_val_macro_f1",
        "selection_seeds": list(seeds),
        "selection_parameter": "top_k_percent+injection_scale",
        "top_k_percent": float(top_k),
        "injection_scale": float(scale),
        "effective_feedback_percent": float(top_k)
        * float(previous["bias_confidence_fraction"]),
        "top_k_range": [float(min(TOP_K_RANGE)), float(max(TOP_K_RANGE))],
        "scale_candidates": [float(c) for c in SCALE_CANDIDATES],
        "top_k_on_range_boundary": float(top_k) in
        (float(min(TOP_K_RANGE)), float(max(TOP_K_RANGE))),
        "scale_on_range_boundary": float(scale) in
        (float(min(SCALE_CANDIDATES)), float(max(SCALE_CANDIDATES))),
        "selection_curves": curves,
    }
    payload.update({
        "source": HEAD_SELECTION_SOURCE,
        "selection_uses_test_labels": False,
        "selection_metric": "mean_best_val_macro_f1",
        "selection_seeds": list(seeds),
        "selection_parameter": "top_k_percent+injection_scale",
        "semantic_consultant": "trained_llm_head",
        "trained_llm_head": True,
        "consultant": "trained_llm_head",
        "top_k_percent": float(top_k),
        "injection_scale": float(scale),
        "effective_feedback_percent": head_block["effective_feedback_percent"],
        "consultant_trained_llm_head": head_block,
        "prototype_superseded": prototype_block,
        "selection_curves": curves,
    })
    payload.pop("superseded", None)
    path = selected_config_path(dataset)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"[{dataset}] CONSULTANT -> trained_llm_head | top_k "
        f"{prototype_block.get('top_k_percent')} -> {top_k}; injection_scale "
        f"{prototype_block.get('injection_scale')} -> {scale}  ({path})"
    )
    return path


def finalize(dataset: str, seeds=SELECTION_SEEDS, use_llm_head: bool = False) -> Path:
    """Write the new knobs into selected_feedback_config.json, old ones superseded."""
    previous = load_feedback_config(dataset)
    top_k = selected_top_k(dataset, seeds, use_llm_head)
    scale = select_by_mean_validation(
        _collect(dataset, "scale", seeds, use_llm_head)
    )

    top_k_curves = _collect(dataset, "top_k", seeds, use_llm_head)
    scale_curves = _collect(dataset, "scale", seeds, use_llm_head)
    curves = {
        "top_k": {
            str(s): {str(k): v for k, v in sorted(top_k_curves[s].items())}
            for s in sorted(top_k_curves)
        },
        "injection_scale": {
            str(s): {str(k): v for k, v in sorted(scale_curves[s].items())}
            for s in sorted(scale_curves)
        },
    }

    if use_llm_head:
        return _finalize_head(dataset, previous, top_k, scale, seeds, curves)

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
    parser.add_argument(
        "--use-llm-head", action="store_true",
        help="Select knobs for the trained-head consultant. Sweeps write to "
             "results/knob_selection_head/ and finalize promotes the head block, "
             "keeping the prototype knobs under `prototype_superseded`.",
    )
    args = parser.parse_args()

    if args.stage == "top_k":
        print(stage_top_k(args.dataset, args.seed, args.use_llm_head))
    elif args.stage == "scale":
        top_k = (
            args.top_k if args.top_k is not None
            else selected_top_k(args.dataset, use_llm_head=args.use_llm_head)
        )
        print(stage_scale(args.dataset, args.seed, top_k, args.use_llm_head))
    else:
        finalize(args.dataset, use_llm_head=args.use_llm_head)


if __name__ == "__main__":
    main()
