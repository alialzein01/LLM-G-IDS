from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.pipeline.common.datasets import DATASETS, get_dataset_config


def _load_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main(dataset: str = "ton_iot") -> None:
    config = get_dataset_config(dataset)
    candidates = [
        ("end_to_end_gnn", Path(config.gnn_output_dir) / "benchmark_summary.json"),
        (
            "frozen_gnn_embedding_mlp",
            Path(config.baseline_output_dir) / "gnn_embedding" / "benchmark_summary.json",
        ),
        (
            "frozen_llm_embedding_mlp",
            Path(config.baseline_output_dir) / "llm_embedding" / "benchmark_summary.json",
        ),
        ("entropy_regularized_fusion", Path(config.fusion_output_dir) / "benchmark_summary.json"),
    ]

    print(f"=== Benchmark comparison: {config.display_name} ===")
    print(f"{'Model':<32} {'CV mean':>10} {'CV std':>10} {'Pooled':>10}")
    print("-" * 66)
    missing = []
    for label, path in candidates:
        summary = _load_summary(path)
        if summary is None:
            missing.append(str(path))
            continue
        print(
            f"{label:<32} "
            f"{summary['cv_test_macro_f1_mean']:>10.4f} "
            f"{summary['cv_test_macro_f1_std']:>10.4f} "
            f"{summary['pooled_cv_macro_f1']:>10.4f}"
        )

    if missing:
        print("\nMissing summaries:")
        for path in missing:
            print(f"  {path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Step 3 benchmark summaries.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="ton_iot")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(dataset=args.dataset)
