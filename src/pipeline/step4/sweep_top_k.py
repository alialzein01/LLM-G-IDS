"""Validation-selected sweep of feedback-loop entropy percentages.

Each candidate is retrained independently on the existing five folds with the
prototype semantic consultant. Candidate selection uses mean best-validation
macro-F1 only; held-out OOF test F1 is reported after selection.

Run:
    python -m src.pipeline.step4.sweep_top_k --dataset unsw_nb15
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
from src.pipeline.common.splits import NUM_CLASSES, eval_macro_f1, mask_dropped_logits
from src.pipeline.step4.feedback_config import write_selected_feedback_config


DEFAULT_CANDIDATES = tuple(range(15, 36))


def _validate_candidates(candidates: tuple[int, ...]) -> None:
    if not candidates:
        raise ValueError("At least one top-k percentage is required.")
    if len(set(candidates)) != len(candidates):
        raise ValueError(f"Candidate percentages must be unique: {candidates}")
    invalid = [value for value in candidates if not isinstance(value, int) or not 0 < value < 100]
    if invalid:
        raise ValueError(f"Candidate percentages must be integers in (0, 100): {invalid}")


def select_best_candidate(rows: list[dict]) -> dict:
    """Select by validation score only; smaller percentage wins an exact tie."""
    if not rows:
        raise ValueError("Cannot select from an empty sweep.")
    return min(
        rows,
        key=lambda row: (
            -float(row["mean_best_val_macro_f1"]),
            int(row["top_k_percent"]),
        ),
    )


def _pooled_macro_f1(
    labels: torch.Tensor,
    logits: torch.Tensor,
    eval_classes: tuple[int, ...] | list[int],
    dropped_classes: tuple[int, ...] | list[int] = (),
) -> float:
    preds = mask_dropped_logits(logits, dropped_classes).argmax(dim=1)
    return eval_macro_f1(labels, preds, eval_classes)


def _write_csv(rows: list[dict], path: Path) -> None:
    columns = [
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


def run_top_k_sweep(
    dataset: str,
    candidates: tuple[int, ...] = DEFAULT_CANDIDATES,
    output_dir: str | Path | None = None,
    use_llm_head: bool = False,
) -> Path:
    """Train, persist, and validation-rank all requested entropy percentages."""
    _validate_candidates(candidates)
    config = get_dataset_config(dataset)
    root = Path(output_dir or f"data/{dataset}/processed/step4_feedback/top_k_sweep")
    root.mkdir(parents=True, exist_ok=True)
    feedback_root = root.parent

    data: Data = torch.load(config.graph_path, weights_only=False)
    embeddings = torch.load(config.llm_embedding_path, weights_only=False).float()
    folds = torch.load(config.splits_path, weights_only=False)
    prototype_path = Path(f"data/{dataset}/processed/step4_feedback/prototypes.pt")
    prototypes = torch.load(prototype_path, weights_only=False)

    # Tune top-k against the same semantic consultant the final run will use;
    # the prototype scorer and the trained head flag different edges.
    head_logits_all = None
    if use_llm_head:
        head_path = Path(f"data/{dataset}/processed/step4_feedback/llm_head_logits.pt")
        if not head_path.exists():
            raise FileNotFoundError(
                f"--use-llm-head needs {head_path}; run build_llm_heads first."
            )
        head_logits_all = torch.load(head_path, weights_only=False)

    if embeddings.shape[0] != data.edge_label.shape[0]:
        raise RuntimeError(
            f"LLM embeddings ({embeddings.shape[0]}) are not row-aligned with "
            f"graph edges ({data.edge_label.shape[0]})."
        )
    if len(folds) != len(prototypes["folds"]):
        raise RuntimeError(
            f"Fold count ({len(folds)}) does not match prototype states "
            f"({len(prototypes['folds'])})."
        )

    feedback_training.IN_DIM = data.x.shape[1]
    labels = data.edge_label
    num_edges = labels.shape[0]
    eval_classes = config.eval_classes
    dropped_classes = config.dropped_classes
    rows: list[dict] = []

    print(
        f"Top-k entropy sweep: dataset={dataset}, candidates={list(candidates)}, "
        f"semantic_consultant={'trained_llm_head' if use_llm_head else 'whitened_prototypes'}"
    )
    for candidate in candidates:
        print(f"\n===== TOP_K_PERCENT={candidate} =====")
        candidate_dir = root / f"n_{candidate:02d}"
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
                head_logits=(
                    head_logits_all[fold_idx] if head_logits_all is not None else None
                ),
                top_k_percent=float(candidate),
                eval_classes=eval_classes,
                dropped_classes=dropped_classes,
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
                f"Candidate {candidate} left some OOF edges without predictions."
            )

        validation_scores = np.asarray(
            [row["best_val_macro_f1"] for row in fold_rows], dtype=float
        )
        row = {
            "top_k_percent": candidate,
            "effective_feedback_percent": candidate
            * feedback_training.BIAS_CONFIDENCE_FRAC,
            "mean_best_val_macro_f1": float(validation_scores.mean()),
            "std_best_val_macro_f1": float(validation_scores.std()),
            "pooled_oof_test_macro_f1": _pooled_macro_f1(
                labels, oof, eval_classes, dropped_classes
            ),
            "folds": fold_rows,
            "oof_logits_path": str(candidate_dir / "oof_logits.pt"),
        }
        rows.append(row)
        torch.save(oof, candidate_dir / "oof_logits.pt")
        with (candidate_dir / "metrics.json").open("w") as handle:
            json.dump(row, handle, indent=2)
        print(
            f"[n={candidate}] mean best-val macro-F1="
            f"{row['mean_best_val_macro_f1']:.4f}; "
            f"pooled OOF test macro-F1={row['pooled_oof_test_macro_f1']:.4f}"
        )

    selected = dict(select_best_candidate(rows))
    selected["sweep_summary_path"] = str(root / "summary.json")
    baseline = next(
        (row for row in rows if row["top_k_percent"] == 30),
        None,
    )
    selected["test_f1_delta_vs_n30"] = (
        selected["pooled_oof_test_macro_f1"]
        - baseline["pooled_oof_test_macro_f1"]
        if baseline is not None
        else None
    )
    summary = {
        "dataset": dataset,
        "eval_classes": list(eval_classes),
        "dropped_classes": list(dropped_classes),
        "selection_metric": "mean_best_val_macro_f1",
        "selection_uses_test_labels": False,
        "semantic_consultant": "whitened_prototype_scorer",
        "bias_confidence_fraction": feedback_training.BIAS_CONFIDENCE_FRAC,
        "selected": selected,
        "candidates": rows,
        "caveat": (
            "Development-time validation selection on rotating folds; use nested "
            "CV or an untouched dataset for an unbiased publication estimate."
        ),
    }
    summary_path = root / "summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    _write_csv(rows, root / "summary.csv")
    selected_config_path = write_selected_feedback_config(
        dataset, selected, root=feedback_root
    )

    print("\n=== VALIDATION RANKING ===")
    for rank, row in enumerate(
        sorted(
            rows,
            key=lambda item: (
                -item["mean_best_val_macro_f1"],
                item["top_k_percent"],
            ),
        ),
        start=1,
    ):
        print(
            f"{rank:2d}. n={row['top_k_percent']:2d} "
            f"val={row['mean_best_val_macro_f1']:.4f}±"
            f"{row['std_best_val_macro_f1']:.4f} "
            f"test={row['pooled_oof_test_macro_f1']:.4f}"
        )
    print(
        f"\nSelected n={selected['top_k_percent']} by validation macro-F1; "
        f"pooled OOF test macro-F1={selected['pooled_oof_test_macro_f1']:.4f}"
    )
    print(f"Selected feedback config -> {selected_config_path}")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="unsw_nb15", choices=sorted(DATASETS))
    parser.add_argument("--min-percent", type=int, default=25)
    parser.add_argument("--max-percent", type=int, default=35)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--use-llm-head", action="store_true",
        help="Tune against the trained per-fold MLP head instead of the "
             "whitened-prototype scorer.",
    )
    args = parser.parse_args()
    if args.min_percent > args.max_percent:
        parser.error("--min-percent must be less than or equal to --max-percent")

    os.environ.setdefault("IDS_FORCE_CPU", "1")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    candidates = tuple(range(args.min_percent, args.max_percent + 1))
    run_top_k_sweep(
        args.dataset,
        candidates=candidates,
        output_dir=args.output_dir,
        use_llm_head=args.use_llm_head,
    )


if __name__ == "__main__":
    main()
