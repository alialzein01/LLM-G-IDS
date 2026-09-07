"""Rung-vs-baseline intervals with BOTH sides at three seeds.

`compare_to_ladder.py` compared a seed-42-only rung against three baseline
seeds. Its own caveat said so: "no rung-side training stochasticity enters these
intervals ... they are therefore NARROWER than a symmetric multi-seed comparison
would give". Both sides now have three seeds on disk, so this module replaces
that asymmetry.

Two intervals per (rung x baseline x config), both from
`aggregate_multiseed`'s existing bootstrap functions rather than a third
implementation:

  * **two-level** — a rung seed and a baseline seed drawn INDEPENDENTLY and
    uniformly, plus a bootstrap resample of edges. Encoded by handing
    `_two_level_bootstrap` a `per_seed` dict keyed by the nine (rung seed,
    baseline seed) pairs: drawing one of those nine uniformly *is* drawing the
    two seeds independently. Answers "would this gap survive a different
    training run on either side, and a different sample of edges?".
  * **seed-matched** — 42<->42, 1<->1, 2<->2, edges resampled, differences
    averaged over the three pairs. The pooled OOF edge set is identical across
    seeds, so this is a paired comparison and is the tighter of the two.

Both sides' predictions are re-scored and checked against the committed
contracts before any interval is computed: a baseline row against
`baselines[model][mode].seeds[*].macro_f1` in `*_sota_baselines.json`, a rung
against `multi_seed.rungs.<rung>.macro_f1_per_seed` in `*_current.json`. A
mismatch raises rather than quietly producing intervals for predictions that are
not the ones the contracts describe.

The LLM rung is an argmax over the frozen prototype scorer, so it is identical
at every training seed; it is expanded to three identical rows so it flows
through the same code path. Its blocks still say `rung_seeds: 3` — three
captures entered the draw — and carry `rung_is_seed_invariant: true`.

Run:
    python -m src.pipeline.baselines.compare_to_ladder_multiseed --dataset unsw_nb15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.pipeline.common.datasets import get_dataset_config
from src.pipeline.common.splits import eval_macro_f1
from src.pipeline.step4.aggregate_multiseed import (
    _seed_matched_bootstrap,
    _two_level_bootstrap,
)
from src.pipeline.step4.assemble_ladder import _masked_preds
from src.pipeline.step4.train_feedback import _llm_alone_oof

SEEDS = (42, 1, 2)
MODELS = ("e_graphsage", "te_g_sage")
MODES = ("as_published", "refit", "plus_node_features")
RUNGS = ("gnn", "llm", "agaf", "feedback", "head_alone")
CAPTURE_ROOT = Path("results/multiseed_v2/head")
SUPERSEDED_NOTE = "rung_seeds was 1; superseded 2026-09-07"

COMPARISON_CAVEAT_3SEED = (
    "Both sides are three training seeds (42, 1, 2) with the fold partition "
    "fixed at the seed-42 split. `statistical_comparisons_3seed` draws the rung "
    "seed and the baseline seed independently; `seed_matched_comparisons_3seed` "
    "pairs them. The seed-matched interval is the tighter of the two and is the "
    "right one to quote for a paired claim; the two-level one is the right one "
    "for 'would this survive a rerun'. `separated` means the 95% interval "
    "excludes zero. These supersede the schema-1 blocks, which held the rung at "
    "seed 42 only and were therefore too narrow."
)


def _load_baseline_seeds(dataset: str, model: str, mode: str, n_edges: int) -> np.ndarray:
    path = Path(
        f"data/{dataset}/processed/baselines/{model}_{mode}_oof_preds.npy"
    )
    preds = np.load(path)
    if preds.shape != (len(SEEDS), n_edges):
        raise ValueError(
            f"{path}: expected shape {(len(SEEDS), n_edges)}, got {preds.shape}"
        )
    return preds


def _load_rung_seeds(
    dataset: str, capture_root: Path | None = None
) -> dict[str, np.ndarray]:
    """`[3, E]` predictions per rung, rows ordered as `SEEDS`."""
    config = get_dataset_config(dataset)
    root = capture_root or CAPTURE_ROOT
    data = torch.load(config.graph_path, weights_only=False)
    dropped = config.dropped_classes

    # The LLM rung is seed-invariant by construction (see the module docstring),
    # so it is computed once and broadcast, exactly as `assemble_ladder` treats it.
    folds = torch.load(config.splits_path, weights_only=False)
    embeddings = torch.load(config.llm_embedding_path, weights_only=False).float()
    prototypes = torch.load(
        Path(f"data/{dataset}/processed/step4_feedback/prototypes.pt"),
        weights_only=False,
    )
    llm = _masked_preds(
        _llm_alone_oof(data, embeddings, folds, prototypes), dropped
    ).cpu().numpy()

    # Rung predictions come from `aggregate_multiseed._load_seed_preds`, the same
    # loader the ladder aggregate uses, so the two can never disagree about what a
    # capture contains (it accepts either the slim agaf_predictions.json or the
    # full fusion metrics.json).
    from src.pipeline.step4.aggregate_multiseed import (
        _head_alone_preds,
        _load_seed_preds,
    )

    per_rung: dict[str, list[np.ndarray]] = {r: [] for r in RUNGS}
    for seed in SEEDS:
        if not (root / f"{dataset}_seed{seed}").is_dir():
            raise FileNotFoundError(
                f"missing capture directory {root / f'{dataset}_seed{seed}'}"
            )
        preds = _load_seed_preds(dataset, seed, root)
        per_rung["gnn"].append(preds["gnn"])
        per_rung["agaf"].append(preds["agaf"])
        per_rung["feedback"].append(preds["loop"])
        per_rung["llm"].append(llm)
        head_alone = _head_alone_preds(dataset, seed, config)
        if head_alone is None:
            raise FileNotFoundError(
                f"missing head-alone logits for {dataset} seed {seed}; build them "
                f"with `build_llm_heads --dataset {dataset} --seed {seed}`"
            )
        per_rung["head_alone"].append(head_alone)
    return {r: np.stack(rows) for r, rows in per_rung.items()}


def _verify_against_contracts(
    dataset: str,
    labels: np.ndarray,
    eval_classes,
    rung_preds: dict[str, np.ndarray],
    baseline_preds: dict[tuple[str, str], np.ndarray],
) -> dict:
    """Re-score every row and match it to the committed contract, to full float
    precision. Raises on any mismatch — intervals must describe the predictions
    the contracts describe, or they describe nothing."""
    current = json.loads(Path(f"results/{dataset}_current.json").read_text())
    sota = json.loads(Path(f"results/{dataset}_sota_baselines.json").read_text())
    checked: dict[str, dict] = {"rungs": {}, "baselines": {}}
    problems: list[str] = []

    for rung, preds in rung_preds.items():
        want = current["multi_seed"]["rungs"][rung]["macro_f1_per_seed"]
        for i, seed in enumerate(SEEDS):
            got = float(eval_macro_f1(labels, preds[i], eval_classes))
            expected = float(want[str(seed)])
            checked["rungs"][f"{rung}_seed{seed}"] = got
            if repr(got) != repr(expected):
                problems.append(
                    f"{dataset} rung {rung} seed {seed}: re-scored {got!r} but the "
                    f"contract says {expected!r}"
                )

    for (model, mode), preds in baseline_preds.items():
        rows = sota["baselines"][model][mode]["seeds"]
        if [r["seed"] for r in rows] != list(SEEDS):
            problems.append(
                f"{dataset} {model}.{mode}: seed rows are "
                f"{[r['seed'] for r in rows]}, expected {list(SEEDS)}"
            )
            continue
        for i, row in enumerate(rows):
            got = float(eval_macro_f1(labels, preds[i], eval_classes))
            checked["baselines"][f"{model}_{mode}_seed{row['seed']}"] = got
            if repr(got) != repr(float(row["macro_f1"])):
                problems.append(
                    f"{dataset} {model}.{mode} seed {row['seed']}: re-scored "
                    f"{got!r} but the contract says {row['macro_f1']!r}"
                )

    if problems:
        raise ValueError(
            "prediction/contract mismatch — refusing to compute intervals:\n  "
            + "\n  ".join(problems)
        )
    return checked


def _compare(
    labels: np.ndarray,
    rung: np.ndarray,
    baseline: np.ndarray,
    eval_classes,
) -> tuple[dict, dict]:
    """Two-level and seed-matched intervals for one (rung, baseline) pair.

    `per_seed` for the two-level case is keyed by the nine (rung seed, baseline
    seed) combinations, so `_two_level_bootstrap`'s uniform draw over its keys is
    an independent uniform draw on each side. The seed-matched case is keyed by
    the three real seeds, which is what makes it paired.
    """
    n_r, n_b = rung.shape[0], baseline.shape[0]
    cross = {
        i * n_b + j: {"rung": rung[i], "baseline": baseline[j]}
        for i in range(n_r)
        for j in range(n_b)
    }
    matched = {
        s: {"rung": rung[i], "baseline": baseline[i]} for i, s in enumerate(SEEDS)
    }

    two_level = _two_level_bootstrap(
        labels, cross, "rung", "baseline", eval_classes
    )
    seed_matched = _seed_matched_bootstrap(
        labels, matched, "rung", "baseline", eval_classes
    )

    per_pair_diff = {
        str(s): float(eval_macro_f1(labels, rung[i], eval_classes))
        - float(eval_macro_f1(labels, baseline[i], eval_classes))
        for i, s in enumerate(SEEDS)
    }
    sign_stable = len({d > 0 for d in per_pair_diff.values()}) == 1

    # Three rung captures always enter the draw. The LLM rung is an argmax over
    # the frozen prototype scorer, so its three rows are identical — recorded as
    # `rung_is_seed_invariant` rather than by writing `rung_seeds: 1`, which is
    # exactly what the superseded blocks said for every rung and would be read
    # as the old asymmetry coming back.
    seed_invariant = all(
        np.array_equal(rung[0], rung[i]) for i in range(1, n_r)
    )
    for block, resampled in ((two_level, "edges_and_both_seeds_independently"),
                             (seed_matched, "edges_only_seed_matched")):
        block["resampled"] = resampled
        block["rung_seeds"] = int(n_r)
        block["rung_is_seed_invariant"] = bool(seed_invariant)
        block["baseline_seeds"] = int(n_b)
        block["separated"] = bool(
            block["ci_low"] > 0.0 or block["ci_high"] < 0.0
        )
        block["sign_stable_across_seed_pairs"] = sign_stable
        block["per_seed_pair_diff"] = per_pair_diff
        block.pop("n_seeds", None)
    return two_level, seed_matched


def compare_to_ladder_multiseed(
    dataset: str, capture_root: Path | None = None, write: bool = True
) -> dict:
    config = get_dataset_config(dataset)
    data = torch.load(config.graph_path, weights_only=False)
    labels = data.edge_label.cpu().numpy()
    eval_classes = config.eval_classes

    rung_preds = _load_rung_seeds(dataset, capture_root)
    baseline_preds = {
        (m, mode): _load_baseline_seeds(dataset, m, mode, len(labels))
        for m in MODELS
        for mode in MODES
    }
    verification = _verify_against_contracts(
        dataset, labels, eval_classes, rung_preds, baseline_preds
    )

    two_level: dict[str, dict] = {}
    matched: dict[str, dict] = {}
    for (model, mode), bpred in baseline_preds.items():
        for rung in RUNGS:
            key = f"{rung}_vs_{model}_{mode}"
            two_level[key], matched[key] = _compare(
                labels, rung_preds[rung], bpred, eval_classes
            )

    if not write:
        return {"two_level": two_level, "seed_matched": matched,
                "verification": verification}

    path = Path("results") / f"{dataset}_sota_baselines.json"
    payload = json.loads(path.read_text())
    superseded = payload.setdefault("superseded", {})
    if "statistical_comparisons" in payload:
        superseded["statistical_comparisons"] = payload.pop("statistical_comparisons")
        superseded["seed_matched_comparisons"] = payload.pop(
            "seed_matched_comparisons"
        )
        superseded["comparison_caveat"] = payload.pop("comparison_caveat", None)
        superseded["schema_version"] = payload["schema_version"]
        superseded["note"] = SUPERSEDED_NOTE
    # A previous 3-seed block is history too, not scratch space. Re-running this
    # module after the loop's consultant changed would otherwise overwrite the
    # earlier consultant's intervals in place and lose them: the first rerun did
    # exactly that to the prototype-consultant blocks, which had to be recovered
    # from git. File them under the consultant they were computed for.
    if "statistical_comparisons_3seed" in payload:
        prior_consultant = (
            payload.get("comparison_sources_3seed", {}).get("loop_consultant")
            or "whitened_prototype_scorer"
        )
        key = f"{prior_consultant}_3seed"
        superseded[key] = {
            "statistical_comparisons_3seed":
                payload.pop("statistical_comparisons_3seed"),
            "seed_matched_comparisons_3seed":
                payload.pop("seed_matched_comparisons_3seed"),
            "comparison_sources_3seed": payload.pop("comparison_sources_3seed", None),
            "generated": payload.pop("generated_3seed", None),
            "note": (
                f"Both sides at 3 seeds, with the loop consulting the "
                f"{prior_consultant}. Superseded when the loop's canonical "
                f"consultant changed; the rungs other than the loop are unaffected "
                f"by that change and are reproduced in the current block."
            ),
        }
    payload["schema_version"] = 2
    payload["generated_3seed"] = "2026-09-07"
    payload["statistical_comparisons_3seed"] = two_level
    payload["seed_matched_comparisons_3seed"] = matched
    payload["comparison_caveat"] = COMPARISON_CAVEAT_3SEED
    payload["comparison_sources_3seed"] = {
        "loop_consultant": "trained_llm_head",
        "rung_captures": str(capture_root or CAPTURE_ROOT),
        "baseline_predictions": f"data/{dataset}/processed/baselines/",
        "seeds": list(SEEDS),
        "bootstrap": (
            "src.pipeline.step4.aggregate_multiseed._two_level_bootstrap / "
            "._seed_matched_bootstrap, 2000 iterations, RNG seed 42"
        ),
        "reverified_macro_f1": verification,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--capture-root", default=None)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute and print without writing the contract.",
    )
    args = parser.parse_args()
    root = Path(args.capture_root) if args.capture_root else None
    result = compare_to_ladder_multiseed(
        args.dataset, capture_root=root, write=not args.dry_run
    )
    blocks = (
        (result["two_level"], result["seed_matched"])
        if args.dry_run
        else (result["statistical_comparisons_3seed"],
              result["seed_matched_comparisons_3seed"])
    )
    for name, block in zip(("two-level", "seed-matched"), blocks):
        print(f"\n=== {args.dataset} — {name} ===")
        for key, v in block.items():
            print(f"  {key:<44} {v['mean_diff']:+.4f} "
                  f"[{v['ci_low']:+.4f}, {v['ci_high']:+.4f}] "
                  f"P={v['prob_positive']:.3f} "
                  f"{'separated' if v['separated'] else 'NOT separated'}")
    if not args.dry_run:
        print(f"\nWrote results/{args.dataset}_sota_baselines.json")


if __name__ == "__main__":
    main()
