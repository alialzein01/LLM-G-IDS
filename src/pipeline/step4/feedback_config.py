from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_TOP_K_PERCENT = 16.0
DEFAULT_BIAS_CONFIDENCE_FRAC = 0.5
DEFAULT_MAX_ITERATIONS = 3
DEFAULT_CHURN_TOLERANCE = 0.01
SELECTED_CONFIG_FILE = "selected_feedback_config.json"


def feedback_root(dataset: str) -> Path:
    return Path(f"data/{dataset}/processed/step4_feedback")


def selected_config_path(dataset: str, root: str | Path | None = None) -> Path:
    return Path(root) / SELECTED_CONFIG_FILE if root is not None else feedback_root(dataset) / SELECTED_CONFIG_FILE


def default_feedback_config(dataset: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dataset": dataset,
        "source": "default",
        "selection_uses_test_labels": False,
        "selection_metric": "mean_best_val_macro_f1",
        "top_k_percent": DEFAULT_TOP_K_PERCENT,
        "bias_confidence_fraction": DEFAULT_BIAS_CONFIDENCE_FRAC,
        "effective_feedback_percent": DEFAULT_TOP_K_PERCENT * DEFAULT_BIAS_CONFIDENCE_FRAC,
        "max_feedback_iterations": DEFAULT_MAX_ITERATIONS,
        "churn_tolerance": DEFAULT_CHURN_TOLERANCE,
        "semantic_consultant": "whitened_prototype_scorer",
        "trained_llm_head": False,
    }


def write_selected_feedback_config(
    dataset: str,
    selected: dict[str, Any],
    root: str | Path | None = None,
) -> Path:
    top_k = float(selected["top_k_percent"])
    payload = default_feedback_config(dataset)
    payload.update(
        {
            "source": "validation_sweep",
            "top_k_percent": top_k,
            "effective_feedback_percent": top_k * DEFAULT_BIAS_CONFIDENCE_FRAC,
            "mean_validation_macro_f1": float(selected["mean_best_val_macro_f1"]),
            "validation_macro_f1_std": float(selected["std_best_val_macro_f1"]),
            "pooled_oof_test_macro_f1": float(selected["pooled_oof_test_macro_f1"]),
            "sweep_summary_path": selected.get("sweep_summary_path"),
        }
    )
    path = selected_config_path(dataset, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def load_feedback_config(
    dataset: str,
    root: str | Path | None = None,
    *,
    require_selected: bool = False,
) -> dict[str, Any]:
    path = selected_config_path(dataset, root)
    if not path.exists():
        if require_selected:
            raise FileNotFoundError(
                f"Missing selected feedback config at {path}. Run "
                f"`python -m src.pipeline.step4.sweep_top_k --dataset {dataset} "
                "--min-percent 15 --max-percent 35` first."
            )
        return default_feedback_config(dataset)

    payload = json.loads(path.read_text())
    if payload.get("dataset") != dataset:
        raise ValueError(
            f"Selected feedback config at {path} is for {payload.get('dataset')!r}, "
            f"not {dataset!r}."
        )
    if payload.get("selection_uses_test_labels") is not False:
        raise ValueError(f"Selected feedback config at {path} used test labels.")
    return payload
