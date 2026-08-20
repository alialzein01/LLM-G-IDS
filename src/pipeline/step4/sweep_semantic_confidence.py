"""Validation-select the semantic-confidence fraction for trained-head feedback.

The top-k percentage is held fixed. For each candidate, only the requested
fraction of entropy-flagged edges with the highest semantic-head confidence
receives semantic attention bias. Selection uses mean best-validation macro-F1;
held-out OOF test scores are reported but never used for candidate selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

import src.pipeline.step4.train_feedback as feedback_training
from src.pipeline.common.datasets import DATASETS, get_dataset_config
from src.pipeline.common.splits import NUM_CLASSES
from src.pipeline.step4.feedback_config import write_selected_feedback_config
from src.pipeline.step4.sweep_top_k import _pooled_macro_f1


DEFAULT_CANDIDATES = tuple(round(value / 10, 1) for value in range(1, 11))


def _validate_candidates(candidates: tuple[float, ...]) -> None:
    if not candidates:
        raise ValueError("At least one semantic-confidence fraction is required.")
    if len(set(candidates)) != len(candidates):
        raise ValueError(f"Candidate fractions must be unique: {candidates}")
    invalid = [value for value in candidates if not 0.0 < value <= 1.0]
    if invalid:
        raise ValueError(f"Candidate fractions must be in (0, 1]: {invalid}")


def select_best_candidate(rows: list[dict]) -> dict:
    """Select by validation score only; smaller fraction wins an exact tie."""
    if not rows:
        raise ValueError("Cannot select from an empty sweep.")
    return min(
        rows,
        key=lambda row: (
            -float(row["mean_best_val_macro_f1"]),
            float(row["bias_confidence_fraction"]),
        ),
    )


def _write_csv(rows: list[dict], path: Path) -> None:
    columns = [
        "bias_confidence_fraction",
        "top_k_percent",
        "effective_feedback_percent",
        "mean_best_val_macro_f1",
        "std_best_val_macro_f1",
        "pooled_oof_test_macro_f1",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in columns})


def run_semantic_confidence_sweep(
    dataset: str,
    top_k_percent: float,
    candidates: tuple[float, ...] = DEFAULT_CANDIDATES,
    output_dir: str | Path | None = None,
    head_logits_path: str | Path | None = None,
) -> Path:
    _validate_candidates(candidates)
    config = get_dataset_config(dataset)
    root = Path(
        output_dir
        or f"data/{dataset}/processed/step4_feedback_trained_head/semantic_confidence_sweep"
    )
    root.mkdir(parents=True, exist_ok=True)
    feedback_root = root.parent
    canonical_root = Path(f"data/{dataset}/processed/step4_feedback")

    data: Data = torch.load(config.graph_path, weights_only=False)
    embeddings = torch.load(config.llm_embedding_path, weights_only=False).float()
    folds = torch.load(config.splits_path, weights_only=False)
    prototypes = torch.load(canonical_root / "prototypes.pt", weights_only=False)
    resolved_head_path = (
        Path(head_logits_path)
        if head_logits_path is not None
        else canonical_root / "llm_head_logits.pt"
    )
    head_logits_all = torch.load(resolved_head_path, weights_only=False)

    expected_head_shape = (len(folds), data.edge_label.shape[0], NUM_CLASSES)
    if head_logits_all.shape != expected_head_shape:
        raise RuntimeError(
            f"Trained-head logits have shape {tuple(head_logits_all.shape)}; "
            f"expected {expected_head_shape}."
        )

    feedback_training.IN_DIM = data.x.shape[1]
    labels = data.edge_label
    num_edges = labels.shape[0]
    rows: list[dict] = []

    print(
        "Semantic-confidence sweep: "
        f"dataset={dataset}, top_k={top_k_percent}, candidates={list(candidates)}"
    )
    for candidate in candidates:
        print(f"\n===== BIAS_CONFIDENCE_FRACTION={candidate:.1f} =====")
        candidate_dir = root / f"f_{int(round(candidate * 100)):03d}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        oof = torch.full((num_edges, NUM_CLASSES), float("nan"))
        fold_rows: list[dict] = []

        for fold_idx, fold in enumerate(folds):
            result = feedback_training._train_one_fold(
                data=data,
                emb=embeddings,
                fold=fold,
                fold_state=prototypes["folds"][fold_idx],
                fold_idx=fold_idx,
                mode="real",
                head_logits=head_logits_all[fold_idx],
                top_k_percent=top_k_percent,
                bias_confidence_fraction=candidate,
                eval_classes=config.eval_classes,
                dropped_classes=config.dropped_classes,
            )
            oof[fold["test_mask"]] = result.logits[fold["test_mask"]]
            fold_rows.append(
                {
                    "fold": fold_idx,
                    "best_val_macro_f1": result.best_val_macro_f1,
                    "test_macro_f1": result.test_macro_f1,
                    "iterations": result.iterations,
                }
            )

        if torch.isnan(oof).any():
            raise RuntimeError(
                f"Confidence fraction {candidate} left OOF edges uncovered."
            )

        validation_scores = np.asarray(
            [row["best_val_macro_f1"] for row in fold_rows], dtype=float
        )
        row = {
            "bias_confidence_fraction": candidate,
            "top_k_percent": top_k_percent,
            "effective_feedback_percent": top_k_percent * candidate,
            "mean_best_val_macro_f1": float(validation_scores.mean()),
            "std_best_val_macro_f1": float(validation_scores.std()),
            "pooled_oof_test_macro_f1": _pooled_macro_f1(
                labels,
                oof,
                config.eval_classes,
                config.dropped_classes,
            ),
            "folds": fold_rows,
            "oof_logits_path": str(candidate_dir / "oof_logits.pt"),
        }
        rows.append(row)
        torch.save(oof, candidate_dir / "oof_logits.pt")
        (candidate_dir / "metrics.json").write_text(json.dumps(row, indent=2))
        print(
            f"[fraction={candidate:.1f}] val="
            f"{row['mean_best_val_macro_f1']:.4f}; "
            f"pooled OOF test={row['pooled_oof_test_macro_f1']:.4f}"
        )

    selected = dict(select_best_candidate(rows))
    selected["sweep_summary_path"] = str(root / "summary.json")
    summary = {
        "dataset": dataset,
        "top_k_percent": top_k_percent,
        "eval_classes": list(config.eval_classes),
        "dropped_classes": list(config.dropped_classes),
        "selection_metric": "mean_best_val_macro_f1",
        "selection_uses_test_labels": False,
        "semantic_consultant": "trained_oof_head",
        "trained_llm_head": True,
        "selected": selected,
        "candidates": rows,
        "caveat": (
            "Development-time validation selection on rotating folds; use nested "
            "CV or an untouched dataset for an unbiased publication estimate."
        ),
    }
    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    _write_csv(rows, root / "summary.csv")
    selected_config_path = write_selected_feedback_config(
        dataset,
        selected,
        root=feedback_root,
        semantic_consultant="trained_oof_head",
        trained_llm_head=True,
        bias_confidence_fraction=float(selected["bias_confidence_fraction"]),
        selection_parameter="bias_confidence_fraction",
    )

    print("\n=== VALIDATION RANKING ===")
    for rank, row in enumerate(
        sorted(
            rows,
            key=lambda item: (
                -item["mean_best_val_macro_f1"],
                item["bias_confidence_fraction"],
            ),
        ),
        start=1,
    ):
        print(
            f"{rank:2d}. fraction={row['bias_confidence_fraction']:.1f} "
            f"val={row['mean_best_val_macro_f1']:.4f}±"
            f"{row['std_best_val_macro_f1']:.4f} "
            f"test={row['pooled_oof_test_macro_f1']:.4f}"
        )
    print(
        "\nSelected semantic-confidence fraction="
        f"{selected['bias_confidence_fraction']:.1f} by validation macro-F1; "
        f"pooled OOF test={selected['pooled_oof_test_macro_f1']:.4f}"
    )
    print(f"Selected feedback config -> {selected_config_path}")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ton_iot", choices=sorted(DATASETS))
    parser.add_argument("--top-k-percent", type=float, required=True)
    parser.add_argument("--min-tenths", type=int, default=1)
    parser.add_argument("--max-tenths", type=int, default=10)
    parser.add_argument("--output-dir")
    parser.add_argument("--head-logits-path")
    args = parser.parse_args()
    if not 1 <= args.min_tenths <= args.max_tenths <= 10:
        parser.error("Require 1 <= --min-tenths <= --max-tenths <= 10")

    os.environ.setdefault("IDS_FORCE_CPU", "1")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    candidates = tuple(
        round(value / 10, 1)
        for value in range(args.min_tenths, args.max_tenths + 1)
    )
    run_semantic_confidence_sweep(
        args.dataset,
        top_k_percent=args.top_k_percent,
        candidates=candidates,
        output_dir=args.output_dir,
        head_logits_path=args.head_logits_path,
    )


if __name__ == "__main__":
    main()
