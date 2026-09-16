from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_TOP_K_PERCENT = 16.0
DEFAULT_BIAS_CONFIDENCE_FRAC = 0.5
DEFAULT_MAX_ITERATIONS = 3
DEFAULT_CHURN_TOLERANCE = 0.01
DEFAULT_GATE_MODE = "confidence"
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
        "gate_mode": DEFAULT_GATE_MODE,
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
    *,
    semantic_consultant: str = "whitened_prototype_scorer",
    trained_llm_head: bool = False,
    bias_confidence_fraction: float = DEFAULT_BIAS_CONFIDENCE_FRAC,
    selection_parameter: str = "top_k_percent",
    injection_mode: str = "edge",
    gate_mode: str = DEFAULT_GATE_MODE,
    injection_scale: float | None = None,
) -> Path:
    """Write the config a sweep selected.

    `injection_scale` is written when given. `resolve_injection_scale` refuses to
    guess one, so a sweep-written config that omitted it made the very next
    `train_feedback` raise unless the caller repeated the flag by hand. The
    sweep already resolved a scale in order to rank its candidates; that is the
    value to record, because ranking candidates under one strength and training
    under another is what the no-default rule exists to prevent.
    """
    top_k = float(selected["top_k_percent"])
    payload = default_feedback_config(dataset)
    payload.update(
        {
            "source": "validation_sweep",
            "selection_parameter": selection_parameter,
            "top_k_percent": top_k,
            "bias_confidence_fraction": bias_confidence_fraction,
            "effective_feedback_percent": top_k * bias_confidence_fraction,
            "mean_validation_macro_f1": float(selected["mean_best_val_macro_f1"]),
            "validation_macro_f1_std": float(selected["std_best_val_macro_f1"]),
            "pooled_oof_test_macro_f1": float(selected["pooled_oof_test_macro_f1"]),
            "sweep_summary_path": selected.get("sweep_summary_path"),
            "semantic_consultant": semantic_consultant,
            "trained_llm_head": trained_llm_head,
            # The mechanism this top_k was selected under. A top_k chosen against the
            # inert attention path does not transfer to an edge-injection run.
            "injection_mode": injection_mode,
            "gate_mode": gate_mode,
        }
    )
    if injection_scale is not None:
        payload["injection_scale"] = float(injection_scale)
    path = selected_config_path(dataset, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def resolve_injection_scale(
    explicit: float | None,
    selected_config: dict[str, Any],
    dataset: str,
) -> float:
    """The injection scale, from the flag if given, else from the selected config.

    There is deliberately NO numeric fallback. `injection_scale` used to be a plain
    argparse default of 10.0 that the config could not override, so a run that simply
    omitted the flag silently used one value for both datasets while its own artifacts
    reported the selected 2.0 / 20.0 beside it. Raising is what makes that impossible.
    """
    if explicit is not None:
        return float(explicit)
    value = selected_config.get("injection_scale")
    if value is None:
        raise ValueError(
            f"No injection_scale for {dataset!r}: pass --injection-scale explicitly, or "
            f"put one in {SELECTED_CONFIG_FILE}. There is no default — it is per-dataset "
            "and must be re-selected whenever the encoding or injection path changes."
        )
    return float(value)


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
