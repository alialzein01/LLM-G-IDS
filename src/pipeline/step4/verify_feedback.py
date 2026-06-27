from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.models.feedback_classifier import FeedbackFusionClassifier
from src.models.fusion_classifier import UnimodalEdgeClassifier
from src.pipeline.common.datasets import DATASETS, DatasetConfig, get_dataset_config
from src.pipeline.common.reports import report_dir, utc_now_iso, write_json
from src.pipeline.common.splits import NUM_CLASSES
from src.pipeline.step4.train_feedback import (
    ALPHA_INIT,
    DAMPING,
    DROPOUT,
    FEEDBACK_DIM,
    FEEDBACK_HIDDEN_DIM,
    MAX_ITERATIONS,
    PROJ_DIM,
    REENTRY_ADVANTAGE_LAMBDA,
    STRONG_HIDDEN_DIM,
    UNCERTAIN_FRACTION,
)


ALPHA_INERTIA_TOL = 1e-3
BENIGN_SELECTION_WARNING_FRACTION = 0.60


def _status(name: str, status: str, detail: str, evidence: str = "") -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "passed": status != "fail",
        "detail": detail,
        "evidence": evidence,
    }


def _load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _strong_embedding(config: DatasetConfig) -> tuple[torch.Tensor, int, str]:
    if config.strong_modality == "gnn":
        return (
            torch.load(config.gnn_embedding_path, weights_only=True).float(),
            config.gnn_dim,
            config.gnn_embedding_path,
        )
    if config.strong_modality == "llm":
        return (
            torch.load(config.llm_embedding_path, weights_only=True).float(),
            config.llm_dim,
            config.llm_embedding_path,
        )
    raise ValueError(f"Unsupported strong modality: {config.strong_modality}")


def _apply_scaler(emb: torch.Tensor, scaler: Any) -> torch.Tensor:
    return torch.from_numpy(scaler.transform(emb.numpy())).float()


def _build_strong_head(in_dim: int) -> UnimodalEdgeClassifier:
    return UnimodalEdgeClassifier(
        in_dim=in_dim,
        proj_dim=PROJ_DIM,
        hidden_dim=STRONG_HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    )


def _build_feedback_model(
    config: DatasetConfig,
    prototype_temperature: float,
    uncertain_fraction: float,
) -> FeedbackFusionClassifier:
    return FeedbackFusionClassifier(
        hidden_dim=FEEDBACK_HIDDEN_DIM,
        feedback_dim=FEEDBACK_DIM,
        llm_dim=config.llm_dim,
        num_classes=NUM_CLASSES,
        uncertain_fraction=uncertain_fraction,
        prototype_temperature=prototype_temperature,
        initial_alpha=ALPHA_INIT,
    )


def _artifact_paths(config: DatasetConfig) -> dict[str, Path]:
    out = Path(config.feedback_output_dir)
    return {
        "model": out / "model.pt",
        "strong_head": out / "strong_head.pt",
        "scalers": out / "scalers.pt",
        "feedback_weights": out / "feedback_weights.pt",
        "metrics": out / "metrics.json",
        "benchmark_summary": out / "benchmark_summary.json",
        "training_history": out / "training_history.json",
        "prototypes": Path(config.prototypes_path),
        "redundancy": out / "redundancy_check.json",
        "agaf_summary": Path(config.fusion_output_dir) / "benchmark_summary.json",
        "strong_summary": Path(config.baseline_output_dir)
        / f"{config.strong_modality}_embedding"
        / "benchmark_summary.json",
    }


def _load_runtime(config: DatasetConfig) -> dict[str, Any]:
    paths = _artifact_paths(config)
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required artifacts: "
            + ", ".join(f"{name}={paths[name]}" for name in missing)
        )

    data = torch.load(config.graph_path, weights_only=False)
    llm_emb = torch.load(config.llm_embedding_path, weights_only=True).float()
    strong_emb, strong_dim, strong_path = _strong_embedding(config)
    prototypes = torch.load(paths["prototypes"], weights_only=False)
    metrics = _load_json(paths["metrics"])
    history = _load_json(paths["training_history"])
    benchmark = _load_json(paths["benchmark_summary"])
    redundancy = _load_json(paths["redundancy"])
    agaf_summary = _load_json(paths["agaf_summary"])
    strong_summary = _load_json(paths["strong_summary"])

    scalers = torch.load(paths["scalers"], weights_only=False)
    strong_scaler = scalers["strong"]
    strong_model = _build_strong_head(strong_dim)
    strong_model.load_state_dict(torch.load(paths["strong_head"], weights_only=True))
    strong_model.eval()

    strong_t = _apply_scaler(strong_emb, strong_scaler)
    with torch.no_grad():
        strong_logits, _ = strong_model(strong_t)

    model = _build_feedback_model(
        config,
        prototype_temperature=float(prototypes.get("prototype_temperature", 10.0)),
        uncertain_fraction=float(history.get("uncertain_fraction", UNCERTAIN_FRACTION)),
    )
    model.load_state_dict(torch.load(paths["model"], weights_only=True))
    model.eval()

    return {
        "paths": paths,
        "data": data,
        "llm_emb": llm_emb,
        "strong_emb": strong_emb,
        "strong_dim": strong_dim,
        "strong_path": strong_path,
        "strong_logits": strong_logits.detach(),
        "prototypes": prototypes,
        "model": model,
        "metrics": metrics,
        "history": history,
        "benchmark": benchmark,
        "redundancy": redundancy,
        "agaf_summary": agaf_summary,
        "strong_summary": strong_summary,
    }


def _source_checks(repo_root: Path, runtime: dict[str, Any]) -> list[dict[str, Any]]:
    feedback_src = (repo_root / "src/models/feedback_classifier.py").read_text()
    build_src = (repo_root / "src/pipeline/step4/build_prototypes.py").read_text()
    train_src = (repo_root / "src/pipeline/step4/train_feedback.py").read_text()
    datasets_src = (repo_root / "src/pipeline/common/datasets.py").read_text()

    model = runtime["model"]
    data = runtime["data"]
    prototype = runtime["prototypes"]
    checks = [
        _status(
            "Uncertainty selection is entropy + class-stratified top-p",
            "pass"
            if all(
                token in feedback_src
                for token in ["def entropy", "class_stratified_topp", "torch.topk"]
            )
            and abs(model.uncertainty_detector.uncertain_fraction - 0.30) < 1e-9
            else "fail",
            f"uncertain_fraction={model.uncertainty_detector.uncertain_fraction}",
            "UncertaintyDetector.entropy + class_stratified_topp",
        ),
        _status(
            "Semantic feedback uses prototype soft-vote and train-fold grouping",
            "pass"
            if all(
                token in feedback_src
                for token in ["SemanticFeedbackScorer", "F.softmax", "prototype_logits"]
            )
            and "train_mask & (labels == class_idx)" in build_src
            else "fail",
            f"prototype_mode={prototype.get('prototype_mode', 'single')}",
            "SemanticFeedbackScorer + build_prototypes train_mask",
        ),
        _status(
            "Feedback is appended to edge_attr with edge_dim = 5 + k",
            "pass"
            if "torch.cat([edge_attr, edge_feedback], dim=1)" in feedback_src
            and model.encoder.feedback_dim == FEEDBACK_DIM
            and data.edge_attr.shape[1] == 5
            else "fail",
            f"edge_attr_dim={data.edge_attr.shape[1]}, feedback_dim={model.encoder.feedback_dim}",
            "BiasedGATStructuralEncoder._augment_edge_attr",
        ),
        _status(
            "Feedback can re-enter message passing",
            "pass"
            if "inject_feedback" in feedback_src
            and "edge_feedback = damping * next_feedback" in feedback_src
            else "fail",
            "run_feedback_loop has inject_feedback=True path with damping",
            "FeedbackFusionClassifier.run_feedback_loop",
        ),
        _status(
            "Strong-modality residual skip is configured per dataset",
            "pass"
            if "strong_modality" in datasets_src
            and "strong_logits + self.alpha * correction" in feedback_src
            else "fail",
            f"strong_modality={runtime['strong_summary'].get('model_name')} via config",
            "ResidualCorrectionHead.forward + DatasetConfig.strong_modality",
        ),
        _status(
            "Iterative stopping uses churn and entropy delta with hard cap",
            "pass"
            if all(
                token in feedback_src
                for token in ["prediction_churn", "entropy_delta", "max_iterations"]
            )
            else "fail",
            f"max_iterations={runtime['history'].get('max_iterations', MAX_ITERATIONS)}, damping={runtime['history'].get('damping', DAMPING)}",
            "IterativeController.should_stop",
        ),
        _status(
            "Loss is focal + sparse correction, not AGAF gate entropy",
            "pass"
            if "FocalLoss" in train_src
            and "lambda_sparse * sparse_loss" in train_src
            and "gate_entropy_lambda" not in train_src
            else "fail",
            f"lambda_sparse={runtime['history'].get('lambda_sparse')}",
            "_feedback_loss in train_feedback.py",
        ),
        _status(
            "Stage B has a re-entry advantage objective",
            "pass"
            if "reentry_advantage_lambda * reentry_loss" in train_src
            and "no_reentry_logits" in train_src
            else "fail",
            f"reentry_advantage_lambda={runtime['history'].get('reentry_advantage_lambda', REENTRY_ADVANTAGE_LAMBDA)}",
            "_feedback_loss compares inject_feedback=True against no-reentry logits",
        ),
        _status(
            "Selection analysis is saved in metrics",
            "pass"
            if "selection_stats" in runtime["metrics"]
            and "predicted_class_histogram" in json.dumps(runtime["metrics"]["selection_stats"])
            else "fail",
            "metrics.json contains selection_stats with predicted_class_histogram",
            "metrics.json",
        ),
        _status(
            "Redundancy pre-flight exists and passed",
            "pass" if runtime["redundancy"].get("is_redundant") is False else "fail",
            f"pooled_pearson={runtime['redundancy'].get('pooled_pearson')}",
            "redundancy_check.json",
        ),
    ]
    return checks


def _prediction_churn(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    changed = (a != b).sum().item()
    total = a.numel()
    return {"changed": int(changed), "total": int(total), "fraction": float(changed / total)}


def _selection_summary(trace: dict[str, Any], benign_class: int = 0) -> dict[str, Any]:
    iterations = trace.get("iterations", [])
    if not iterations:
        return {"iteration_count": 0, "benign_fractions": [], "max_benign_fraction": None}
    benign_fractions = []
    for row in iterations:
        selected = row.get("selected_count", 0)
        hist = row.get("predicted_class_histogram", [])
        benign = hist[benign_class] if selected and len(hist) > benign_class else 0
        benign_fractions.append(float(benign / selected) if selected else 0.0)
    return {
        "iteration_count": len(iterations),
        "benign_fractions": benign_fractions,
        "max_benign_fraction": max(benign_fractions) if benign_fractions else None,
    }


def _stage_delta(history: dict[str, Any]) -> dict[str, Any]:
    stage_a = []
    stage_b = []
    for fold in history.get("folds", []):
        stages = fold.get("stage_results", [])
        for stage in stages:
            if stage.get("stage") == "stage_a_no_reentry":
                stage_a.append(stage.get("best_val_macro_f1"))
            elif stage.get("stage") == "stage_b_feedback_loop":
                stage_b.append(stage.get("best_val_macro_f1"))
    if not stage_a or not stage_b:
        return {"stage_a_mean": None, "stage_b_mean": None, "delta": None}
    a = float(np.mean(stage_a))
    b = float(np.mean(stage_b))
    return {"stage_a_mean": a, "stage_b_mean": b, "delta": b - a}


def _macro_f1_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float(
        f1_score(
            labels.cpu().numpy(),
            logits.argmax(dim=1).cpu().numpy(),
            average="macro",
            labels=list(range(NUM_CLASSES)),
            zero_division=0,
        )
    )


def _efficacy_checks(runtime: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model: FeedbackFusionClassifier = runtime["model"]
    data = runtime["data"]
    llm_emb = runtime["llm_emb"]
    prototypes = runtime["prototypes"]["all_edges"]
    strong_logits = runtime["strong_logits"]
    labels = data.edge_label.long()
    alpha = float(model.alpha.detach().item())

    with torch.no_grad():
        logits_off, _, trace_off = model.run_feedback_loop(
            data.x,
            data.edge_index,
            data.edge_attr,
            llm_emb,
            prototypes,
            strong_logits,
            inject_feedback=False,
            max_iterations=int(runtime["history"].get("max_iterations", MAX_ITERATIONS)),
            damping=float(runtime["history"].get("damping", DAMPING)),
            return_trace=True,
        )
        logits_on, _, trace_on = model.run_feedback_loop(
            data.x,
            data.edge_index,
            data.edge_attr,
            llm_emb,
            prototypes,
            strong_logits,
            inject_feedback=True,
            max_iterations=int(runtime["history"].get("max_iterations", MAX_ITERATIONS)),
            damping=float(runtime["history"].get("damping", DAMPING)),
            return_trace=True,
        )

        random_feedback_generator = torch.Generator().manual_seed(20260201)
        logits_random_feedback, _, _ = model.run_feedback_loop(
            data.x,
            data.edge_index,
            data.edge_attr,
            llm_emb,
            prototypes,
            strong_logits,
            inject_feedback=True,
            max_iterations=int(runtime["history"].get("max_iterations", MAX_ITERATIONS)),
            damping=float(runtime["history"].get("damping", DAMPING)),
            feedback_mode="random",
            generator=random_feedback_generator,
            return_trace=False,
        )

        random_selection_generator = torch.Generator().manual_seed(20260202)
        logits_random_selection, _, _ = model.run_feedback_loop(
            data.x,
            data.edge_index,
            data.edge_attr,
            llm_emb,
            prototypes,
            strong_logits,
            inject_feedback=True,
            max_iterations=int(runtime["history"].get("max_iterations", MAX_ITERATIONS)),
            damping=float(runtime["history"].get("damping", DAMPING)),
            selection_mode="random",
            generator=random_selection_generator,
            return_trace=False,
        )

        original_alpha = model.alpha.detach().clone()
        model.alpha.zero_()
        skip_logits, _, _ = model.run_feedback_loop(
            data.x,
            data.edge_index,
            data.edge_attr,
            llm_emb,
            prototypes,
            strong_logits,
            inject_feedback=False,
            return_trace=False,
        )
        model.alpha.copy_(original_alpha)

    strong_pred = strong_logits.argmax(dim=1)
    off_pred = logits_off.argmax(dim=1)
    on_pred = logits_on.argmax(dim=1)
    residual_exact = bool(torch.allclose(skip_logits, strong_logits, atol=1e-6))
    churn_on_off = _prediction_churn(on_pred, off_pred)
    churn_on_strong = _prediction_churn(on_pred, strong_pred)
    selection = _selection_summary(trace_on)
    stage = _stage_delta(runtime["history"])
    final_real_f1 = _macro_f1_from_logits(logits_on, labels)
    final_no_reentry_f1 = _macro_f1_from_logits(logits_off, labels)
    final_random_feedback_f1 = _macro_f1_from_logits(logits_random_feedback, labels)
    final_random_selection_f1 = _macro_f1_from_logits(logits_random_selection, labels)

    feedback_f1 = float(runtime["benchmark"]["pooled_cv_macro_f1"])
    agaf_f1 = float(runtime["agaf_summary"]["pooled_cv_macro_f1"])
    strong_f1 = float(runtime["strong_summary"]["pooled_cv_macro_f1"])

    checks = [
        _status(
            "Residual skip invariant holds",
            "pass" if residual_exact else "fail",
            "alpha=0 reproduces strong_logits exactly" if residual_exact else "alpha=0 changed logits",
            "ResidualCorrectionHead final_logits = strong_logits + alpha * correction",
        ),
        _status(
            "Feedback loop changes predictions relative to no-reentry",
            "pass" if churn_on_off["changed"] > 0 else "fail",
            f"{churn_on_off['changed']}/{churn_on_off['total']} edges changed ({churn_on_off['fraction']:.4f})",
            "run_feedback_loop inject_feedback=True vs False",
        ),
        _status(
            "Feedback loop changes predictions relative to strong head",
            "pass" if churn_on_strong["changed"] > 0 else "fail",
            f"{churn_on_strong['changed']}/{churn_on_strong['total']} edges changed ({churn_on_strong['fraction']:.4f})",
            "feedback final predictions vs frozen strong-head predictions",
        ),
        _status(
            "Residual correction scale is active",
            "pass" if abs(alpha) > ALPHA_INERTIA_TOL else "fail",
            f"alpha={alpha:.8f}",
            "ResidualCorrectionHead.alpha",
        ),
        _status(
            "Loop runs multiple iterations before convergence/cap",
            "pass" if selection["iteration_count"] > 1 else "note",
            f"iterations={selection['iteration_count']}",
            "trace.iterations from run_feedback_loop",
        ),
        _status(
            "Selection is not dominated by Benign/Normal",
            "pass"
            if selection["max_benign_fraction"] is not None
            and selection["max_benign_fraction"] <= BENIGN_SELECTION_WARNING_FRACTION
            else "note",
            f"max class-0 selected fraction={selection['max_benign_fraction']}",
            "selected predicted-class histogram",
        ),
        _status(
            "Stage B improves over Stage A",
            "pass" if stage["delta"] is not None and stage["delta"] > 0 else "fail",
            f"stage_a_mean={stage['stage_a_mean']}, stage_b_mean={stage['stage_b_mean']}, delta={stage['delta']}",
            "training_history stage_results",
        ),
        _status(
            "Real prototype feedback beats random feedback in final model",
            "pass" if final_real_f1 > final_random_feedback_f1 else "fail",
            f"real={final_real_f1:.4f}, random_feedback={final_random_feedback_f1:.4f}",
            "feedback_mode='prototype' vs feedback_mode='random'",
        ),
        _status(
            "Entropy selection beats random selection in final model",
            "pass" if final_real_f1 > final_random_selection_f1 else "fail",
            f"real={final_real_f1:.4f}, random_selection={final_random_selection_f1:.4f}",
            "selection_mode='entropy' vs selection_mode='random'",
        ),
        _status(
            "Feedback benchmark beats AGAF",
            "pass" if feedback_f1 > agaf_f1 else "fail",
            f"feedback={feedback_f1:.4f}, agaf={agaf_f1:.4f}",
            "benchmark_summary.json",
        ),
        _status(
            "Feedback benchmark beats strong unimodal baseline",
            "pass" if feedback_f1 > strong_f1 else "fail",
            f"feedback={feedback_f1:.4f}, strong={strong_f1:.4f}",
            "benchmark_summary.json",
        ),
    ]
    metrics = {
        "alpha": alpha,
        "alpha_abs": abs(alpha),
        "alpha_inert": abs(alpha) <= ALPHA_INERTIA_TOL,
        "feedback_vs_no_reentry_churn": churn_on_off,
        "feedback_vs_strong_churn": churn_on_strong,
        "selection_summary": selection,
        "stage_delta": stage,
        "final_real_macro_f1": final_real_f1,
        "final_no_reentry_macro_f1": final_no_reentry_f1,
        "final_random_feedback_macro_f1": final_random_feedback_f1,
        "final_random_selection_macro_f1": final_random_selection_f1,
        "feedback_pooled_macro_f1": feedback_f1,
        "agaf_pooled_macro_f1": agaf_f1,
        "strong_pooled_macro_f1": strong_f1,
    }
    return checks, metrics


def _deviation_notes(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    notes = [
        _status(
            "Architecture note: frozen prototype lookup replaces live LLM call",
            "note",
            "todo.md says route uncertain flows to the LLM; implementation follows PHASE2 plan constraint: zero live LLM calls, use frozen CySecBERT prototypes.",
            "todo.md Step 4 vs PHASE2_FEEDBACK_LOOP_PLAN.md section 3",
        ),
        _status(
            "Architecture note: Stage A encoder is trainable",
            "note",
            "PHASE2 plan describes Stage 1 with frozen encoder; implemented Stage A uses the same model and differs mainly by inject_feedback=False.",
            "train_feedback.py _train_feedback_stage call path",
        ),
    ]
    if metrics["alpha_inert"] or metrics["feedback_vs_no_reentry_churn"]["changed"] == 0:
        notes.append(
            _status(
                "Efficacy note: feedback correction may be inert",
                "note",
                f"alpha={metrics['alpha']:.8f}, feedback-vs-no-reentry changed {metrics['feedback_vs_no_reentry_churn']['changed']} edges.",
                "final model efficacy check",
            )
        )
    return notes


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 2 Feedback Loop Verification",
        "",
        f"- Dataset: `{payload['dataset']}`",
        f"- Status: `{payload['status']}`",
        f"- Generated at: `{payload['generated_at']}`",
        "",
        "## Metrics",
    ]
    for key, value in payload["metrics"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Checks"])
    for check in payload["checks"]:
        lines.append(
            f"- `{check['status'].upper()}` {check['name']}: {check['detail']}"
        )
        if check.get("evidence"):
            lines.append(f"  Evidence: `{check['evidence']}`")
    lines.extend(["", "## Artifacts"])
    for key, value in payload["artifacts"].items():
        lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines) + "\n")


def verify(dataset: str) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    config = get_dataset_config(dataset)
    runtime = _load_runtime(config)

    structural = _source_checks(repo_root, runtime)
    efficacy, metrics = _efficacy_checks(runtime)
    deviations = _deviation_notes(metrics)
    checks = structural + efficacy + deviations

    fail_count = sum(1 for item in checks if item["status"] == "fail")
    note_count = sum(1 for item in checks if item["status"] == "note")
    status = "failed" if fail_count else ("passed_with_notes" if note_count else "passed")

    paths = runtime["paths"]
    artifacts = {name: str(path) for name, path in paths.items()}
    payload = {
        "dataset": dataset,
        "status": status,
        "generated_at": utc_now_iso(),
        "checks": checks,
        "metrics": metrics,
        "artifacts": artifacts,
    }

    out = report_dir(config, "phase2_feedback_verification")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "verification_report.json", payload)
    _write_markdown(out / "verification_report.md", payload)

    print(f"Dataset: {dataset}")
    print(f"Status: {status}")
    print(f"Failures: {fail_count}; notes: {note_count}")
    print(
        "Macro-F1: "
        f"feedback={metrics['feedback_pooled_macro_f1']:.4f}, "
        f"AGAF={metrics['agaf_pooled_macro_f1']:.4f}, "
        f"strong={metrics['strong_pooled_macro_f1']:.4f}"
    )
    print(
        "Feedback churn: "
        f"{metrics['feedback_vs_no_reentry_churn']['changed']}/"
        f"{metrics['feedback_vs_no_reentry_churn']['total']} "
        "vs no-reentry"
    )
    print(f"Reports written to {out}")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify Phase 2 feedback-loop architecture conformance."
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="ton_iot")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    verify(args.dataset)
