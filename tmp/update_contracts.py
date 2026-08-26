"""Regenerate the authoritative contracts from freshly produced Step 4 artifacts.

Reads ladder_summary.json / ablation_summary.json / selected_feedback_config.json for
each dataset and rewrites results/{unsw_nb15,ton_iot}_current.json plus
results/cross_dataset_comparison.json, preserving the curated prose fields.

Run from the repo root AFTER a canonical pipeline run:
    python tmp/update_contracts.py
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

GENERATED = date.today().isoformat()
BACKUP = Path("tmp/contract_backup_20260825")


def art(dataset: str, name: str) -> dict:
    return json.loads(Path(f"data/{dataset}/processed/step4_feedback/{name}").read_text())


def sweep(dataset: str) -> dict:
    return json.loads(
        Path(f"data/{dataset}/processed/step4_feedback/top_k_sweep/summary.json").read_text()
    )


def configuration(dataset: str, ladder: dict, sel: dict, sw: dict) -> dict:
    cfg = ladder["configuration"]
    top_k = float(cfg["top_k_percent"])
    frac = float(cfg["bias_confidence_fraction"])
    out = {
        "top_k_percent": top_k,
        "top_k_source": "validation_sweep 15-35, prototype consultant, no test labels",
        "semantic_confidence_fraction": frac,
        "effective_feedback_percent": top_k * frac,
        "semantic_consultant": "whitened_prototype",
        "trained_llm_head": False,
        "agaf_head_fusion": False,
        # The mechanism. Recorded explicitly because the default changed on 2026-08-25
        # and a contract that omits it gets misread as whatever the default is today.
        "injection_mode": cfg.get("injection_mode") or sel.get("injection_mode"),
        "injection_scale": sw.get("injection_scale", 10.0),
        "max_feedback_iterations": sel.get("max_feedback_iterations", 3),
        "churn_tolerance": sel.get("churn_tolerance", 0.01),
        "seed": 42,
        "deterministic_cpu": True,
    }
    if dataset == "unsw_nb15":
        out.pop("agaf_head_fusion")
    return out


def rungs(ladder: dict) -> dict:
    acc, f1 = ladder["accuracy"], ladder["ladder"]
    keys = [("gnn", "gnn_alone"), ("llm", "llm_alone"), ("agaf", "agaf"),
            ("feedback", "feedback_loop")]
    return {name: {"accuracy": acc[k], "macro_f1": f1[k]} for name, k in keys}


def order_string(ladder: dict) -> str:
    f1 = ladder["ladder"]
    pairs = sorted(
        [("GNN", f1["gnn_alone"]), ("LLM", f1["llm_alone"]),
         ("AGAF", f1["agaf"]), ("Loop", f1["feedback_loop"])],
        key=lambda kv: kv[1],
    )
    return " < ".join(f"{n} {v:.4f}" for n, v in pairs)


def build_unsw() -> dict:
    ladder = art("unsw_nb15", "ladder_summary.json")
    abl = art("unsw_nb15", "ablation_summary.json")
    sel = art("unsw_nb15", "selected_feedback_config.json")
    prev = json.loads((BACKUP / "unsw_nb15_current.json").read_text())
    b = ladder["bootstraps"]
    per_mode = abl["per_mode_pooled_macro_f1"]

    return {
        "schema_version": 2,
        "dataset": "UNSW-NB15",
        "dataset_key": "unsw_nb15",
        "metric_protocol": "pooled five-fold out-of-fold evaluation",
        "primary_metric": "macro_f1",
        "generated": GENERATED,
        "architecture": "shared canonical architecture (identical to NF-ToN-IoT)",
        "configuration": configuration("unsw_nb15", ladder, sel, sweep("unsw_nb15")),
        "results": rungs(ladder),
        "weighted_f1": ladder["weighted_f1"],
        "ladder_order_holds": ladder["ladder_order_holds"],
        "observed_order": order_string(ladder),
        "feedback_ablations": {
            "random": {"macro_f1": per_mode["random"]},
            "head_only": {"macro_f1": per_mode["head_only"]},
            "real_vs_random": {
                "mean_macro_f1_difference": abl["real_vs_random"]["mean_diff"],
                "ci_95": [abl["real_vs_random"]["ci_low"], abl["real_vs_random"]["ci_high"]],
            },
            "real_vs_head_only": {
                "mean_macro_f1_difference": abl["real_vs_head_only"]["mean_diff"],
                "ci_95": [
                    abl["real_vs_head_only"]["ci_low"],
                    abl["real_vs_head_only"]["ci_high"],
                ],
            },
        },
        "selection": {
            "criterion": "mean validation macro_f1",
            "selected_top_k_percent": sel["top_k_percent"],
            "mean_validation_macro_f1": sel["mean_validation_macro_f1"],
            "validation_macro_f1_std": sel["validation_macro_f1_std"],
            "sweep_range": [15, 35],
            "swept_under_injection_mode": sel.get("injection_mode"),
        },
        "statistical_comparisons": {
            k: {
                "mean_macro_f1_difference": b[k]["mean_diff"],
                "ci_95": [b[k]["ci_low"], b[k]["ci_high"]],
                "bootstrap_probability_positive": b[k]["prob_positive"],
            }
            for k in ("loop_vs_agaf", "agaf_vs_llm", "agaf_vs_gnn", "loop_vs_gnn")
        },
        "source_artifacts": {
            "ladder": "data/unsw_nb15/processed/step4_feedback/ladder_summary.json",
            "ablation": "data/unsw_nb15/processed/step4_feedback/ablation_summary.json",
            "top_k_sweep": "data/unsw_nb15/processed/step4_feedback/top_k_sweep/summary.json",
            "command": (
                "python -m src.pipeline.common.run_pipeline --dataset unsw_nb15 "
                "--only fusion oof prototypes top_k_sweep feedback ladder"
            ),
        },
        "supersedes": {
            "schema_version": 1,
            "reason": (
                "Schema 1 was produced under injection_mode='attention', the mechanism "
                "measured inert (churn 0.0000). The default is now 'edge' injection, so "
                "a default canonical run no longer reproduced schema 1. Schema 2 records "
                "injection_mode explicitly. Schema 1 values are preserved below."
            ),
            "schema_1_values": prev["results"],
        },
        "limitations": prev["limitations"],
    }


def build_ton() -> dict:
    ladder = art("ton_iot", "ladder_summary.json")
    sel = art("ton_iot", "selected_feedback_config.json")
    prev = json.loads((BACKUP / "ton_iot_current.json").read_text())
    b = ladder["bootstraps"]
    f1 = ladder["ladder"]

    return {
        "schema_version": 4,
        "dataset": "NF-ToN-IoT",
        "dataset_key": "ton_iot",
        "graph_scope": "aggregated",
        "feature_profile": "structural10",
        "metric_protocol": "pooled five-fold out-of-fold evaluation",
        "primary_metric": "macro_f1",
        "generated": GENERATED,
        "eval_classes": ladder["eval_classes"],
        "dropped_classes": ladder["dropped_classes"],
        "n_eval_classes": len(ladder["eval_classes"]),
        "architecture": "shared canonical architecture (identical to UNSW-NB15)",
        "configuration": configuration("ton_iot", ladder, sel, sweep("ton_iot")),
        "results": rungs(ladder),
        "weighted_f1": ladder["weighted_f1"],
        "ladder_order_holds": ladder["ladder_order_holds"],
        "observed_order": order_string(ladder),
        "llm_below_gnn": f1["llm_alone"] < f1["gnn_alone"],
        "selection": {
            "criterion": "mean validation macro_f1",
            "selected_top_k_percent": sel["top_k_percent"],
            "mean_validation_macro_f1": sel["mean_validation_macro_f1"],
            "validation_macro_f1_std": sel["validation_macro_f1_std"],
            "sweep_range": [15, 35],
            "swept_under_injection_mode": sel.get("injection_mode"),
        },
        "statistical_comparisons": {
            k: {
                "mean_diff": b[k]["mean_diff"],
                "ci_low": b[k]["ci_low"],
                "ci_high": b[k]["ci_high"],
                "prob_positive": b[k]["prob_positive"],
            }
            for k in ("agaf_vs_gnn", "agaf_vs_llm", "loop_vs_agaf", "loop_vs_gnn",
                      "loop_vs_llm")
        },
        "source_artifacts": prev["source_artifacts"],
        "supersedes": {
            "schema_version": 3,
            "reason": (
                "Schema 3 was produced under injection_mode='attention'. The default is "
                "now 'edge' injection; schema 4 records injection_mode explicitly. The "
                "earlier non-reproducing prototype ladder note is retained below."
            ),
            "schema_3_values": prev["results"],
            "earlier": prev["supersedes"],
        },
        "limitations": prev["limitations"],
    }


def build_comparison(unsw: dict, ton: dict) -> dict:
    prev = json.loads((BACKUP / "cross_dataset_comparison.json").read_text())
    out = dict(prev)
    out["schema_version"] = 3
    out["generated"] = GENERATED
    out["architecture"] = {
        "shared": True,
        "semantic_consultant": "whitened_prototype",
        "trained_llm_head": False,
        "agaf_head_fusion": False,
        "injection_mode": unsw["configuration"]["injection_mode"],
        "injection_scale": unsw["configuration"]["injection_scale"],
        "note": (
            "Both datasets run the identical architecture, including the injection "
            "mechanism. The only per-dataset quantity is top_k_percent, which is "
            "selected on validation folds under that same mechanism and must never be "
            "carried across datasets."
        ),
    }
    out["notes"] = [
        "Absolute macro-F1 levels are not interchangeable across datasets because "
        "eval-class counts differ (UNSW 10 vs ToN 8).",
        "The ladder holds on UNSW-NB15 and does NOT hold on ToN-IoT. On ToN the loop "
        "sits below AGAF - now significantly (P(>0)=0.025) - and the LLM rung sits "
        "below the GNN rung.",
        "Trained-head LLM-only baselines are reported separately in "
        "results/{unsw_nb15,ton_iot}_head_baseline.json. On both datasets that baseline "
        "matches or beats the full system, which must be stated in any write-up.",
        "Schema 3 re-runs both datasets under injection_mode='edge', the current "
        "default. Schema 2 was produced under the inert 'attention' mechanism and its "
        "top_k values (UNSW 16.0, ToN 23.0) were selected against a mechanism that "
        "never fired.",
    ]
    for key, payload in (("unsw_nb15", unsw), ("ton_iot_aggregated", ton)):
        entry = dict(prev["datasets"][key])
        entry["top_k_percent"] = payload["configuration"]["top_k_percent"]
        entry["ladder_order_holds"] = payload["ladder_order_holds"]
        entry["macro_f1"] = {k: v["macro_f1"] for k, v in payload["results"].items()}
        out["datasets"][key] = entry
    return out


def main() -> None:
    unsw = build_unsw()
    ton = build_ton()
    comparison = build_comparison(unsw, ton)
    for path, payload in [
        ("results/unsw_nb15_current.json", unsw),
        ("results/ton_iot_current.json", ton),
        ("results/cross_dataset_comparison.json", comparison),
    ]:
        Path(path).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {path}")
    for name, payload in [("UNSW", unsw), ("ToN", ton)]:
        cfg = payload["configuration"]
        print(
            f"{name}: mode={cfg['injection_mode']} top_k={cfg['top_k_percent']} "
            f"holds={payload['ladder_order_holds']} | {payload['observed_order']}"
        )


if __name__ == "__main__":
    main()
