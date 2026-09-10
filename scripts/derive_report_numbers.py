#!/usr/bin/env python3
"""Compute the quantities the report derives from the contracts.

Some numbers the Results section needs are arithmetic on committed values rather
than fields in a contract: the steps of the UNSW decomposition, the range of a
flat selection curve, the share of edges the loop consults. Writing those by
hand puts unverifiable numbers in the manuscript, which is exactly what this
project got burned by before. So they are computed here, written to
`docs/research_report/derived_numbers.json` with their inputs and formula, and
picked up by `check_report_numbers.py` as an allowed source.

    python scripts/derive_report_numbers.py          # rewrite the JSON
    python scripts/derive_report_numbers.py --print   # show it
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "research_report" / "derived_numbers.json"

DATASETS = ("unsw_nb15", "ton_iot")


def load(path: str):
    return json.loads((ROOT / path).read_text())


def decomposition() -> dict:
    """The UNSW staircase: published baseline -> tuning -> features -> ours."""
    sota = load("results/unsw_nb15_sota_baselines.json")["baselines"]["te_g_sage"]
    ladder = load("results/multiseed_ladder_v2_head.json")["unsw_nb15"]["rungs"]
    steps = {
        "te_g_sage_as_published": sota["as_published"]["macro_f1"],
        "te_g_sage_refit": sota["refit"]["macro_f1"],
        "te_g_sage_plus_node_features": sota["plus_node_features"]["macro_f1"],
        "agaf": ladder["agaf"]["mean"],
        "loop": ladder["loop"]["mean"],
    }
    tuning = steps["te_g_sage_refit"] - steps["te_g_sage_as_published"]
    features = steps["te_g_sage_plus_node_features"] - steps["te_g_sage_refit"]
    out = {
        "levels": {k: round(v, 4) for k, v in steps.items()},
        "step_tuning": round(tuning, 4),
        "step_node_features": round(features, 4),
        "note": (
            "Point estimates only. Every level is a 3-seed mean; no step carries an "
            "interval and no step is a separation claim."
        ),
        "inputs": [
            "results/unsw_nb15_sota_baselines.json:baselines.te_g_sage",
            "results/multiseed_ladder_v2_head.json:unsw_nb15.rungs",
        ],
    }
    for endpoint in ("agaf", "loop"):
        architecture = steps[endpoint] - steps["te_g_sage_plus_node_features"]
        total = steps[endpoint] - steps["te_g_sage_as_published"]
        out[f"endpoint_{endpoint}"] = {
            "step_architecture": round(architecture, 4),
            "total_gap": round(total, 4),
            "share_tuning_percent": round(100.0 * tuning / total, 1),
            "share_node_features_percent": round(100.0 * features / total, 1),
            "share_architecture_percent": round(100.0 * architecture / total, 1),
        }
    return out


def knob_curves() -> dict:
    """Range of each selection curve, averaged over the three selection seeds."""
    out: dict[str, dict] = {}
    for ds in DATASETS:
        out[ds] = {}
        for knob in ("top_k", "scale"):
            per_seed = {
                seed: load(f"results/knob_selection_head/{ds}/{knob}_seed{seed}.json")[
                    "validation_by_candidate"
                ]
                for seed in (42, 1, 2)
            }
            candidates = sorted(per_seed[42], key=float)
            means = {c: statistics.fmean(per_seed[s][c] for s in per_seed) for c in candidates}
            lo, hi = min(means.values()), max(means.values())
            argmax = max(means, key=means.get)
            out[ds][knob] = {
                "n_candidates": len(candidates),
                "candidate_min": float(candidates[0]),
                "candidate_max": float(candidates[-1]),
                "validation_min": round(lo, 4),
                "validation_max": round(hi, 4),
                "span": round(hi - lo, 4),
                "argmax_candidate": float(argmax),
                "argmax_on_range_boundary": argmax in (candidates[0], candidates[-1]),
                "source": f"results/knob_selection_head/{ds}/{knob}_seed*.json",
            }
    out["note"] = (
        "Mean validation macro-F1 over selection seeds 42/1/2, per candidate. The "
        "selection_curve_note string in the *_current.json contracts records "
        "0.8241-0.8277 (UNSW top_k) and 0.5168-0.5232 (ToN top_k); recomputed from "
        "the sweep artifacts the endpoints are 0.8238-0.8277 and 0.5170-0.5232."
    )
    return out


def consultation_rates() -> dict:
    """How much of the graph the loop actually consults, at seed 42."""
    stats = json.loads((ROOT / "docs" / "research_report" / "dataset_stats.json").read_text())
    crossfit = load("results/dev/crossfit/task2_summary.json")
    out: dict[str, dict] = {}
    for ds in DATASETS:
        edges = stats[ds]["num_edges"]
        flagged = crossfit[ds]["n_flagged"]
        out[ds] = {
            "edges": edges,
            "flagged": flagged,
            "flagged_percent": round(100.0 * flagged / edges, 1),
            "gated": flagged // 2,
            "gated_percent": round(100.0 * (flagged // 2) / edges, 1),
            "source": f"results/dev/crossfit/task2_summary.json:{ds}.n_flagged, seed 42",
        }
    out["note"] = (
        "The confidence gate keeps the top half of the flagged set by consultant "
        "confidence (bias_confidence_fraction 0.5), so gated = flagged // 2."
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--print", action="store_true", help="print the JSON as well")
    args = ap.parse_args()

    payload = {
        "generated_by": "scripts/derive_report_numbers.py",
        "decomposition_unsw": decomposition(),
        "knob_curves": knob_curves(),
        "consultation_rates": consultation_rates(),
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"-> {OUT.relative_to(ROOT)}")
    if args.print:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
