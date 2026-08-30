"""Gate 0 oracle-ceiling test, v2 encoding, both datasets, 3 seeds.

Writes the cross-dataset raw payload and both authoritative v2 oracle contracts.
Run from repo root:

    python -m src.pipeline.step4.mechanism_only_edge_injection

use_output_fusion=False kills the late-fusion door entirely, so the LLM reaches the
GNN ONLY through the consultative feedback path. Scored against `control_head_only`
-- the same model with the feedback structurally absent -- NEVER against the bare GNN
rung, which is a different model from a different training driver.

The deliberately leaky oracle arms replace the semantic consultant's class logits
with the true class at +/-4 nats.  Output fusion remains disabled, so even the oracle
can reach the GNN only through the feedback mechanism being measured.

NOT comparable to ladder numbers: fusion is off in every arm.

Requires prototypes.pt and folds.pt already built for both datasets
(step4.build_prototypes, splits) before running.
"""
import os

# Must precede the torch import: without single-threaded BLAS, runs drift ~0.012
# macro-F1 at a fixed seed, which is larger than every effect this experiment
# measures (real - control is +0.0064 on UNSW, +0.0026 on ToN).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import math
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch

torch.set_num_threads(1)
warnings.filterwarnings("ignore")

import src.pipeline.step4.train_feedback as FB
from src.pipeline.step4.sweep_top_k import _pooled_macro_f1
from src.pipeline.common.datasets import get_dataset_config

SEEDS = (42, 1, 2)
SCHEMA_VERSION = 2
ORACLE_LOGIT_MAGNITUDE = 4.0
PAIRED_BOOTSTRAP_ITERS = 10_000
RAW_OUTPUT_PATH = Path("results/raw/oracle_ceiling_v2_trained_head.json")
CONTRACT_PATHS = MappingProxyType(
    {
        "unsw_nb15": Path(
            "results/unsw_nb15_oracle_ceiling_v2_trained_head.json"
        ),
        "ton_iot": Path(
            "results/ton_iot_oracle_ceiling_v2_trained_head.json"
        ),
    }
)
SETUP = {
    "unsw_nb15": dict(top_k=31.0, scale=2.0),
    "ton_iot": dict(top_k=25.0, scale=20.0),
}


@dataclass(frozen=True)
class ArmSpec:
    """One prespecified Gate 0 arm.

    ``advice_source='oracle'`` is diagnostic label leakage by construction.  It
    is permitted only because every arm disables the output-fusion branch.
    """

    feedback_mode: str
    injection_mode: str
    advice_source: str


ARMS = MappingProxyType(
    {
        "control_head_only": ArmSpec("head_only", "edge", "none"),
        "real_prototype_edge": ArmSpec("real", "edge", "prototype"),
        "head_trained_edge": ArmSpec("real", "edge", "trained_head"),
        "oracle_edge": ArmSpec("real", "edge", "oracle"),
        "oracle_attention": ArmSpec("real", "attention", "oracle"),
    }
)


def _oracle_logits(
    labels: torch.Tensor,
    num_classes: int,
    magnitude: float = 4.0,
) -> torch.Tensor:
    """Return deliberately leaky true-label logits at ``-magnitude/+magnitude``."""
    if labels.ndim != 1 or labels.dtype != torch.long:
        raise ValueError("labels must be a one-dimensional torch.long tensor")
    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError("num_classes must be an integer >= 2")
    if not math.isfinite(magnitude) or magnitude <= 0.0:
        raise ValueError("magnitude must be finite and > 0")
    if labels.numel() == 0:
        raise ValueError("labels must not be empty")
    if int(labels.min()) < 0 or int(labels.max()) >= num_classes:
        raise ValueError("labels must lie in [0, num_classes)")

    logits = torch.full(
        (labels.shape[0], num_classes),
        -float(magnitude),
        device=labels.device,
        dtype=torch.get_default_dtype(),
    )
    return logits.scatter_(1, labels.unsqueeze(1), float(magnitude))


def _train_arm_fold(
    arm_name: str,
    *,
    labels: torch.Tensor,
    num_classes: int,
    fold_idx: int,
    trained_head_logits: torch.Tensor | None = None,
    **fold_kwargs,
):
    """Train one fold while enforcing the Gate 0 advice and injection route."""
    try:
        spec = ARMS[arm_name]
    except KeyError as exc:
        raise ValueError(f"Unknown Gate 0 arm: {arm_name!r}") from exc

    if spec.advice_source == "oracle":
        head_logits = _oracle_logits(
            labels, num_classes, ORACLE_LOGIT_MAGNITUDE
        )
    elif spec.advice_source == "trained_head":
        expected_tail = (labels.shape[0], num_classes)
        if (
            trained_head_logits is None
            or trained_head_logits.ndim != 3
            or tuple(trained_head_logits.shape[1:]) != expected_tail
        ):
            actual = (
                None
                if trained_head_logits is None
                else tuple(trained_head_logits.shape)
            )
            raise ValueError(
                "trained_head_logits shape must be [folds, E, C] with "
                f"(E, C)={expected_tail}; got {actual}"
            )
        if fold_idx < 0 or fold_idx >= trained_head_logits.shape[0]:
            raise ValueError(
                f"fold_idx must be in [0, {trained_head_logits.shape[0]}), "
                f"got {fold_idx}"
            )
        # This exact fold slice is load-bearing: its test rows are OOF for this
        # fold, whereas selecting another slice silently breaks OOF discipline.
        head_logits = trained_head_logits[fold_idx]
    else:
        head_logits = None
    call_kwargs = dict(fold_kwargs)
    call_kwargs.update(
        fold_idx=fold_idx,
        mode=spec.feedback_mode,
        injection_mode=spec.injection_mode,
        use_output_fusion=False,
        head_logits=head_logits,
    )
    return FB._train_one_fold(**call_kwargs)


def _paired_fold_ci(
    candidate: Sequence[Mapping],
    control: Sequence[Mapping],
    *,
    iters: int = PAIRED_BOOTSTRAP_ITERS,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap a CI over matched ``(seed, fold)`` macro-F1 differences."""
    if iters < 1:
        raise ValueError("iters must be >= 1")

    def indexed(rows: Sequence[Mapping]) -> dict[tuple[int, int], float]:
        result: dict[tuple[int, int], float] = {}
        for row in rows:
            key = (int(row["seed"]), int(row["fold"]))
            if key in result:
                raise ValueError(f"duplicate seed/fold pair: {key}")
            value = float(row["test_macro_f1"])
            if not math.isfinite(value):
                raise ValueError(f"non-finite fold score for {key}")
            result[key] = value
        return result

    candidate_by_pair = indexed(candidate)
    control_by_pair = indexed(control)
    if not candidate_by_pair or candidate_by_pair.keys() != control_by_pair.keys():
        raise ValueError("candidate and control must have identical seed/fold pairs")

    keys = sorted(candidate_by_pair)
    paired = np.asarray(
        [candidate_by_pair[key] - control_by_pair[key] for key in keys],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    sampled = paired[rng.integers(0, paired.size, size=(iters, paired.size))]
    bootstrap_means = sampled.mean(axis=1)
    return {
        "mean_diff": float(paired.mean()),
        "ci_low": float(np.percentile(bootstrap_means, 2.5)),
        "ci_high": float(np.percentile(bootstrap_means, 97.5)),
        "prob_positive": float((bootstrap_means > 0).mean()),
        "n_pairs": int(paired.size),
    }


def _json_safe(value):
    """Recursively convert numpy scalars and non-finite floats for strict JSON."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json_dump(payload: Mapping, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(_json_safe(payload), handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _flatten_folds(runs: Sequence[Mapping]) -> list[dict]:
    return [dict(fold) for run in runs for fold in run["folds"]]


def _iteration_evidence(fold_records: Sequence[Mapping]) -> dict[str, dict]:
    by_iteration: dict[int, list[Mapping]] = {}
    for fold in fold_records:
        for row in fold["iterations"]:
            by_iteration.setdefault(int(row["iter"]), []).append(row)

    summary: dict[str, dict] = {}
    for iteration, rows in sorted(by_iteration.items()):
        f1_values = np.asarray([float(row["test_macro_f1"]) for row in rows])
        baselines = []
        for fold in fold_records:
            iterations = {int(row["iter"]): row for row in fold["iterations"]}
            if iteration in iterations and 1 in iterations:
                baselines.append(
                    float(iterations[iteration]["test_macro_f1"])
                    - float(iterations[1]["test_macro_f1"])
                )
        jaccards = [
            float(row["selection_jaccard_previous"])
            for row in rows
            if row.get("selection_jaccard_previous") is not None
            and math.isfinite(float(row["selection_jaccard_previous"]))
        ]
        summary[str(iteration)] = {
            "n_fold_runs": len(rows),
            "mean_test_macro_f1": float(f1_values.mean()),
            "mean_change_from_iteration_1": float(np.mean(baselines)),
            "wrong_to_correct": int(
                sum(int(row["wrong_to_correct"]) for row in rows)
            ),
            "correct_to_wrong": int(
                sum(int(row["correct_to_wrong"]) for row in rows)
            ),
            "mean_selected_count": float(
                np.mean([row["selected_count"] for row in rows])
            ),
            "mean_selection_jaccard_previous": (
                float(np.mean(jaccards)) if jaccards else None
            ),
            "advice_recomputed": any(bool(row["advice_recomputed"]) for row in rows),
        }
    return summary


def _summarize_arm(runs: list[dict]) -> dict:
    pooled = np.asarray([float(run["pooled_macro_f1"]) for run in runs])
    folds = _flatten_folds(runs)
    churn = [
        float(row["churn"])
        for fold in folds
        for row in fold["iterations"]
        if row.get("churn") is not None and math.isfinite(float(row["churn"]))
    ]
    return {
        "mean": float(pooled.mean()),
        "std": float(pooled.std()),
        "runs": runs,
        "fold_records": folds,
        "mean_churn": float(np.mean(churn)) if churn else 0.0,
        "iteration_evidence": _iteration_evidence(folds),
    }


def _arm_contracts() -> dict[str, dict]:
    return {
        name: {
            "feedback_mode": spec.feedback_mode,
            "injection_mode": spec.injection_mode,
            "advice_source": spec.advice_source,
            "use_output_fusion": False,
            "diagnostic_label_leakage": spec.advice_source == "oracle",
            "diagnostic_only": spec.advice_source in {"trained_head", "oracle"},
            "eligible_for_ladder": False,
        }
        for name, spec in ARMS.items()
    }


def _shared_architecture(fold_count: int) -> dict:
    return {
        "edge_attr_encoding": "v2_log_cont_cat_idx",
        "seeds": list(SEEDS),
        "fold_count": fold_count,
        "max_iterations": FB.MAX_ITERATIONS,
        "churn_tolerance": FB.CHURN_TOL,
        "bias_confidence_fraction": FB.BIAS_CONFIDENCE_FRAC,
        "gate_mode": FB.DEFAULT_GATE_MODE,
        "real_arm_semantic_consultant": "whitened_prototype",
        "trained_llm_head": False,
        "trained_head_diagnostic_arm": True,
        "trained_head_artifact_regenerated": True,
        "trained_head_use_output_fusion": False,
        "use_output_fusion": False,
        "injection_modes": {
            name: spec.injection_mode for name, spec in ARMS.items()
        },
        "oracle_logit_magnitude": ORACLE_LOGIT_MAGNITUDE,
        "deterministic_cpu": True,
        "omp_num_threads": 1,
        "mkl_num_threads": 1,
        "torch_num_threads": 1,
    }


def _run_dataset(dataset: str, setup: Mapping) -> dict:
    cfg = get_dataset_config(dataset)
    root = Path(f"data/{dataset}/processed")
    data = torch.load(root / "step1/pyg_data.pt", weights_only=False)
    folds = torch.load(root / "splits/folds.pt", weights_only=False)
    emb = torch.load(cfg.llm_embedding_path, weights_only=False)
    protos = torch.load(root / "step4_feedback/prototypes.pt", weights_only=False)
    labels = data.edge_label
    eval_classes = tuple(cfg.eval_classes)
    dropped_classes = tuple(cfg.dropped_classes or ())
    num_classes = cfg.num_classes
    if len(folds) != 5 or len(protos.get("folds", ())) != len(folds):
        raise ValueError(
            f"Gate 0 requires five matching CV/prototype folds; got "
            f"{len(folds)} CV folds and {len(protos.get('folds', ()))} prototype folds"
        )
    trained_head_path = root / "step4_feedback/llm_head_logits.pt"
    trained_head_logits = torch.load(trained_head_path, weights_only=False)
    expected_head_shape = (len(folds), labels.shape[0], num_classes)
    if tuple(trained_head_logits.shape) != expected_head_shape:
        raise RuntimeError(
            f"Trained-head logits have shape {tuple(trained_head_logits.shape)}; "
            f"expected {expected_head_shape}."
        )
    source_paths = (
        Path(cfg.graph_path),
        Path(cfg.splits_path),
        Path(cfg.llm_embedding_path),
    )
    newest_source_mtime = max(path.stat().st_mtime_ns for path in source_paths)
    if trained_head_path.stat().st_mtime_ns <= newest_source_mtime:
        raise RuntimeError(
            f"Stale trained-head artifact {trained_head_path}; regenerate it after "
            "the current graph, folds, and LLM embeddings."
        )
    started = time.time()
    arm_results: dict[str, dict] = {}

    print(
        f"\n===== {cfg.display_name}  (top_k={setup['top_k']}, "
        f"scale={setup['scale']}, output_fusion=OFF) =====",
        flush=True,
    )
    for arm_name in ARMS:
        seed_runs: list[dict] = []
        for seed in SEEDS:
            FB.SEED = seed
            oof = torch.full((labels.shape[0], num_classes), float("nan"))
            fold_records: list[dict] = []
            for fold_idx, fold in enumerate(folds):
                result = _train_arm_fold(
                    arm_name,
                    labels=labels,
                    num_classes=num_classes,
                    trained_head_logits=trained_head_logits,
                    data=data,
                    emb=emb,
                    fold_state=protos["folds"][fold_idx],
                    fold=fold,
                    fold_idx=fold_idx,
                    top_k_percent=setup["top_k"],
                    eval_classes=eval_classes,
                    dropped_classes=dropped_classes,
                    injection_scale=setup["scale"],
                )
                oof[fold["test_mask"]] = result.logits[fold["test_mask"]]
                fold_records.append(
                    {
                        "seed": seed,
                        "fold": fold_idx,
                        "best_val_macro_f1": result.best_val_macro_f1,
                        "test_macro_f1": result.test_macro_f1,
                        "iterations": result.iterations,
                        "bias_diagnostics": result.bias_diagnostics,
                    }
                )
            seed_runs.append(
                {
                    "seed": seed,
                    "pooled_macro_f1": _pooled_macro_f1(
                        labels, oof, eval_classes, dropped_classes
                    ),
                    "folds": fold_records,
                }
            )
        arm_results[arm_name] = _summarize_arm(seed_runs)
        arm = arm_results[arm_name]
        print(
            f"  {arm_name:24s} {arm['mean']:.4f} +/-{arm['std']:.4f}  "
            f"[{time.time() - started:5.0f}s]",
            flush=True,
        )

    control_folds = arm_results["control_head_only"]["fold_records"]
    comparisons = {
        f"{name}_vs_control_head_only": _paired_fold_ci(
            arm_results[name]["fold_records"], control_folds
        )
        for name in (
            "real_prototype_edge",
            "head_trained_edge",
            "oracle_edge",
            "oracle_attention",
        )
    }
    control_mean = arm_results["control_head_only"]["mean"]
    prototype_gain = arm_results["real_prototype_edge"]["mean"] - control_mean
    trained_head_gain = arm_results["head_trained_edge"]["mean"] - control_mean
    oracle_headroom = arm_results["oracle_edge"]["mean"] - control_mean
    captured_share = (
        prototype_gain / oracle_headroom if oracle_headroom != 0.0 else None
    )
    trained_head_captured_share = (
        trained_head_gain / oracle_headroom if oracle_headroom != 0.0 else None
    )
    return {
        "dataset": cfg.display_name,
        "dataset_key": dataset,
        "configuration": {
            **_shared_architecture(len(folds)),
            "top_k_percent": float(setup["top_k"]),
            "injection_scale": float(setup["scale"]),
            "trained_head_artifact": str(trained_head_path),
            "trained_head_artifact_shape": list(trained_head_logits.shape),
            "trained_head_artifact_regenerated": True,
            "trained_head_artifact_newer_than_sources": True,
        },
        "arms": arm_results,
        "paired_comparisons": comparisons,
        "decision": {
            "oracle_edge_headroom": oracle_headroom,
            "prototype_gain": prototype_gain,
            "prototype_captured_share": captured_share,
            "trained_head_gain": trained_head_gain,
            "trained_head_captured_share": trained_head_captured_share,
            "headroom_below_0_02": oracle_headroom < 0.02,
        },
        "elapsed_seconds": time.time() - started,
    }


def _dataset_contract(raw: Mapping, dataset: str) -> dict:
    result = raw["datasets"][dataset]
    return {
        "schema_version": raw["schema_version"],
        "dataset": result["dataset"],
        "dataset_key": dataset,
        "experiment": raw["experiment"],
        "generated": raw["generated"],
        "status": raw["status"],
        "question": raw["question"],
        "primary_metric": raw["primary_metric"],
        "configuration": result["configuration"],
        "arms": raw["arms"],
        "results": result["arms"],
        "paired_comparisons": result["paired_comparisons"],
        "decision": result["decision"],
        "program_decision": raw["program_decision"],
        "reproduce": raw["reproduce"],
        "raw_output": raw["raw_output"],
        "extends": f"results/{dataset}_oracle_ceiling_v2.json",
        "supersedes": None,
    }


def _write_outputs(
    raw: Mapping,
    output_path: str | Path,
    contract_paths: Mapping[str, str | Path],
) -> None:
    _atomic_json_dump(raw, output_path)
    for dataset, path in contract_paths.items():
        _atomic_json_dump(_dataset_contract(raw, dataset), path)


def run(
    output_path: str | Path = RAW_OUTPUT_PATH,
    contract_paths: Mapping[str, str | Path] | None = None,
) -> dict:
    original_seed = FB.SEED
    datasets: dict[str, dict] = {}
    try:
        for dataset, setup in SETUP.items():
            datasets[dataset] = _run_dataset(dataset, setup)
    finally:
        FB.SEED = original_seed

    stop = all(
        result["decision"]["headroom_below_0_02"]
        for result in datasets.values()
    )
    raw = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "oracle_consultant_ceiling_v2_trained_head",
        "generated": date.today().isoformat(),
        "status": (
            "DIAGNOSTIC ONLY - DELIBERATE LABEL LEAKAGE; NEVER REPORTABLE "
            "AS DEPLOYABLE PERFORMANCE"
        ),
        "question": (
            "Under the Gate 0 v2 mechanism-only protocol, how much oracle "
            "headroom can a leakage-free per-fold trained-head consultant capture?"
        ),
        "primary_metric": "pooled_5_fold_oof_macro_f1",
        "seeds": list(SEEDS),
        "raw_output": str(output_path),
        "arms": _arm_contracts(),
        "datasets": datasets,
        "program_decision": {
            "rule": "stop P1'-P4 if oracle_edge headroom < 0.02 on both datasets",
            "stop_mechanism_surgery": stop,
            "outcome": "stop" if stop else "continue",
            "gate05_is_diagnostic_only": True,
            "trained_head_is_ladder_rung": False,
        },
        "reproduce": (
            "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m "
            "src.pipeline.step4.mechanism_only_edge_injection"
        ),
    }
    _write_outputs(raw, output_path, contract_paths or CONTRACT_PATHS)
    print(
        "\nGATE 0 DECISION: "
        + ("STOP P1'-P4" if stop else "HEADROOM REMAINS; REVIEW NEXT OPTION"),
        flush=True,
    )
    return raw


if __name__ == "__main__":
    run()
