"""
Run the surveyed fusion mechanisms on one dataset, on identical folds, and
report them side by side.

Every variant consumes the same frozen GNN and LLM edge embeddings and the same
saved splits, so the only thing that differs between rows is the fusion
mechanism and its capacity. Results land in results/<dataset>_fusion_variants.json.

    python -m src.pipeline.step3.sweep_fusion_variants --dataset ton_iot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.pipeline.common.datasets import DATASETS, get_dataset_config
from src.pipeline.step3 import train_fusion


# (name, kwargs) — kwargs are passed straight through to train_fusion.main().
# The first row reproduces the current AGAF so every other row has a reference.
VARIANTS: list[tuple[str, dict]] = [
    ("agaf_baseline", {}),
    ("agaf_no_entropy_reg", {"gate_entropy_lambda": 0.0}),
    ("concat_gmlm", {"fusion_mode": "concat", "gate_entropy_lambda": 0.0}),
    ("fixed_bertgcn", {"fusion_mode": "fixed", "gate_entropy_lambda": 0.0}),
    ("scalar_moorthy", {"fusion_mode": "scalar", "gate_entropy_lambda": 0.0}),
    ("selfattn_ragformer", {"fusion_mode": "selfattn", "gate_entropy_lambda": 0.0}),
    ("agaf_small", {"gate_entropy_lambda": 0.0, "proj_dim": 32, "hidden_dim": 32}),
    ("scalar_small", {"fusion_mode": "scalar", "gate_entropy_lambda": 0.0,
                      "proj_dim": 32, "hidden_dim": 32}),
    ("agaf_align", {"gate_entropy_lambda": 0.0, "align_lambda": 0.1}),
    ("scalar_small_align", {"fusion_mode": "scalar", "gate_entropy_lambda": 0.0,
                            "proj_dim": 32, "hidden_dim": 32, "align_lambda": 0.1}),
]


def _gate_spread(diagnostics_summary: list[dict], eval_classes: tuple[int, ...]) -> dict:
    """
    How much the learned modality weight actually moves across attack classes.

    A flat gate makes the interpretability claim untestable, so this is reported
    next to macro-F1 rather than derived later.
    """
    vals = [
        row["gate_gnn_mean"]
        for row in diagnostics_summary
        if row["class_id"] in eval_classes and not np.isnan(row["gate_gnn_mean"])
    ]
    if not vals:
        return {"gate_min": float("nan"), "gate_max": float("nan"), "gate_spread": float("nan")}
    return {
        "gate_min": float(min(vals)),
        "gate_max": float(max(vals)),
        "gate_spread": float(max(vals) - min(vals)),
    }


def main(dataset: str, use_oof: bool, output: str | None) -> None:
    config = get_dataset_config(dataset)
    scratch = Path("data") / dataset / "processed" / "step3_fusion_variants"
    scratch.mkdir(parents=True, exist_ok=True)

    gnn_emb_path = config.gnn_embedding_path
    if use_oof:
        oof = Path(gnn_emb_path).with_name("edge_embeddings_oof.pt")
        if not oof.exists():
            raise FileNotFoundError(
                f"--use-oof requested but {oof} is missing. Run "
                f"src.pipeline.step3.build_oof_gnn_embeddings first."
            )
        gnn_emb_path = str(oof)
    print(f"GNN embeddings: {gnn_emb_path}")

    rows: list[dict] = []
    for name, overrides in VARIANTS:
        print(f"\n{'=' * 70}\n=== {name} ===\n{'=' * 70}")
        out_dir = scratch / name
        train_fusion.main(
            data_path=config.graph_path,
            gnn_emb_path=gnn_emb_path,
            llm_emb_path=config.llm_embedding_path,
            splits_path=config.splits_path,
            output_dir=str(out_dir),
            dataset=dataset,
            **overrides,
        )

        metrics = json.loads((out_dir / "metrics.json").read_text())
        summary = json.loads((out_dir / "benchmark_summary.json").read_text())
        model_params = sum(
            p.numel() for p in torch.load(out_dir / "model.pt", weights_only=True).values()
        )
        rows.append(
            {
                "variant": name,
                "overrides": overrides,
                "pooled_macro_f1": metrics["overall_macro_f1"],
                "pooled_accuracy": metrics["overall_accuracy"],
                "pooled_weighted_f1": metrics["overall_weighted_f1"],
                "cv_test_macro_f1_mean": summary["cv_test_macro_f1_mean"],
                "cv_test_macro_f1_std": summary["cv_test_macro_f1_std"],
                "n_parameters": int(model_params),
                **_gate_spread(metrics["fusion_diagnostics_summary"], config.eval_classes),
            }
        )

    rows.sort(key=lambda r: r["pooled_macro_f1"], reverse=True)
    print(f"\n{'=' * 100}")
    print(
        f"{'variant':<22}{'macroF1':>9}{'fold_std':>10}{'acc':>8}"
        f"{'params':>10}{'gate_min':>10}{'gate_max':>10}{'spread':>9}"
    )
    print("-" * 100)
    for r in rows:
        print(
            f"{r['variant']:<22}{r['pooled_macro_f1']:>9.4f}"
            f"{r['cv_test_macro_f1_std']:>10.4f}{r['pooled_accuracy']:>8.4f}"
            f"{r['n_parameters']:>10,}{r['gate_min']:>10.4f}"
            f"{r['gate_max']:>10.4f}{r['gate_spread']:>9.4f}"
        )

    out_path = Path(output or f"results/{dataset}_fusion_variants.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "gnn_embeddings": gnn_emb_path,
                "leak_free_gnn_embeddings": use_oof,
                "eval_classes": list(config.eval_classes),
                "metric_protocol": "pooled five-fold out-of-fold evaluation",
                "variants": rows,
            },
            indent=2,
        )
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="ton_iot")
    parser.add_argument(
        "--use-oof",
        action="store_true",
        help="Consume the leak-free per-fold GNN embeddings instead of the "
        "fit-on-all-edges artifact.",
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    main(args.dataset, args.use_oof, args.output)
