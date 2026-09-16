#!/usr/bin/env python3
"""Regenerate `results/multiseed_head_per_class.json` from the seed captures.

The per-class artifact was assembled ad hoc in commit `b03cd0d` and has had no
writer since, so the table and heatmap in the report rested on numbers nothing
could reproduce. This script rebuilds it from the same tensors the ladder
aggregate reads, through the same loaders, so per-class F1 and pooled macro-F1
can never describe different predictions.

Per-class F1 is computed on the rows the metric scores, exactly as
`eval_macro_f1` defines them: rows whose TRUE label is an evaluated class,
scored against `labels=eval_classes`. The macro-F1 in the ladder artifact is the
unweighted mean of these values, which the script asserts before writing.

Two keys are new beside the committed `eval_classes` and `per_class_f1_3seed`:
`class_names` and `class_counts`, read from `docs/research_report/dataset_stats.json`,
so a reader of the artifact can see which class a column is and how many edges
carry it. On NF-ToN-IoT the smallest evaluated class holds 12 edges, and a
per-class F1 there moves substantially on one edge.

**This changed the NF-ToN-IoT numbers.** The committed arrays were computed over
ALL 2,127 rows, so a `dos` or `ransomware` edge predicted as `ddos` counted as a
false positive against `ddos`, which the headline metric does not do. Their
unweighted mean was therefore 0.4305 where the ladder printed 0.4336, and the
same gap appeared at every rung. Scoring the rows the metric scores closes it
exactly at all five rungs. NF-UNSW-NB15 drops no class and is unchanged. The
script asserts the identity on every run so the two can never diverge again.

It also writes `per_class_f1_3seed` into `results/{dataset}_current.json`, which
carried its own copy of the same arrays.

Run:
    OMP_NUM_THREADS=1 IDS_FORCE_CPU=1 python scripts/build_per_class_artifact.py
    OMP_NUM_THREADS=1 IDS_FORCE_CPU=1 python scripts/build_per_class_artifact.py --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.common.datasets import get_dataset_config
from src.pipeline.common.splits import eval_macro_f1
from src.pipeline.step4.aggregate_multiseed import (
    SEEDS,
    _head_alone_preds,
    _llm_alone_preds,
    _load_seed_preds,
)

CAPTURE_ROOT = ROOT / "results" / "multiseed_v2" / "head"
STATS = ROOT / "docs" / "research_report" / "dataset_stats.json"
OUT = ROOT / "results" / "multiseed_head_per_class.json"
CONTRACTS = {
    d: ROOT / "results" / f"{d}_current.json" for d in ("unsw_nb15", "ton_iot")
}
DATASETS = ("unsw_nb15", "ton_iot")
RUNGS = ("gnn", "agaf", "loop", "head_alone", "llm")


def per_class_f1(labels: np.ndarray, preds: np.ndarray, eval_classes) -> list[float]:
    from sklearn.metrics import f1_score

    labs = list(eval_classes)
    row_mask = np.isin(labels, labs)
    return [
        float(v)
        for v in f1_score(
            labels[row_mask], preds[row_mask],
            average=None, labels=labs, zero_division=0,
        )
    ]


def build(dataset: str) -> dict:
    config = get_dataset_config(dataset)
    data = torch.load(config.graph_path, weights_only=False)
    labels = data.edge_label.cpu().numpy()
    eval_classes = list(config.eval_classes)

    per_seed = {s: _load_seed_preds(dataset, s, CAPTURE_ROOT) for s in SEEDS}
    llm = _llm_alone_preds(dataset, config)
    for s in SEEDS:
        per_seed[s]["llm"] = llm
        head = _head_alone_preds(dataset, s, config)
        if head is None:
            raise FileNotFoundError(
                f"[{dataset}] seed {s} has no llm_head_logits; the head-alone rung "
                "cannot be rebuilt and the artifact would silently lose a rung."
            )
        per_seed[s]["head_alone"] = head

    block: dict[str, dict] = {}
    for rung in RUNGS:
        rows = np.array(
            [per_class_f1(labels, per_seed[s][rung], eval_classes) for s in SEEDS]
        )
        block[rung] = {
            "mean": [float(v) for v in rows.mean(axis=0)],
            "std": [float(v) for v in rows.std(axis=0, ddof=1)],
        }
        # The per-class values and the headline macro-F1 must describe the same
        # predictions. Macro-F1 is their unweighted mean by definition, so a
        # disagreement means the two were computed over different rows.
        pooled = np.mean(
            [eval_macro_f1(labels, per_seed[s][rung], eval_classes) for s in SEEDS]
        )
        if abs(pooled - np.mean(block[rung]["mean"])) > 1e-9:
            raise RuntimeError(
                f"[{dataset}] {rung}: per-class mean {np.mean(block[rung]['mean'])} "
                f"disagrees with pooled macro-F1 {pooled}"
            )

    # Names come from the dataset config, which is indexed by class id. The
    # `class_counts` dict in dataset_stats.json is written with sorted keys, and
    # on NF-UNSW-NB15 alphabetical order is NOT index order: class 0 is `Normal`,
    # not `Analysis`. Reading names out of that dict by position would relabel
    # seven of the ten columns.
    counts = json.loads(STATS.read_text())[dataset]["class_counts"]
    names = [config.label_names[c] for c in eval_classes]
    return {
        "eval_classes": eval_classes,
        "class_names": names,
        "class_counts": [counts[n] for n in names],
        "per_class_f1_3seed": block,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="Compare against the committed file without writing it. Exits "
             "non-zero if any `mean` array differs.",
    )
    args = parser.parse_args()

    payload = {d: build(d) for d in DATASETS}

    if OUT.exists():
        committed = json.loads(OUT.read_text())
        drifted = []
        for dataset in DATASETS:
            old = committed.get(dataset, {}).get("per_class_f1_3seed", {})
            new = payload[dataset]["per_class_f1_3seed"]
            for rung in RUNGS:
                if rung not in old:
                    drifted.append(f"{dataset}.{rung}: absent from the committed file")
                    continue
                gap = max(
                    abs(a - b) for a, b in zip(old[rung]["mean"], new[rung]["mean"])
                )
                if gap > 5e-9:
                    drifted.append(f"{dataset}.{rung}: mean differs by up to {gap:.2e}")
        if drifted:
            print("per-class means do NOT reproduce:")
            for line in drifted:
                print(f"  {line}")
        else:
            print("per-class means reproduce the committed file exactly")
        if args.check and drifted:
            sys.exit(1)

    if args.check:
        return

    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    for dataset, path in CONTRACTS.items():
        contract = json.loads(path.read_text())
        contract["per_class_f1_3seed"] = payload[dataset]["per_class_f1_3seed"]
        path.write_text(json.dumps(contract, indent=2) + "\n")
        print(f"-> {path.relative_to(ROOT)}:per_class_f1_3seed")
    for dataset in DATASETS:
        block = payload[dataset]
        print(f"\n=== {dataset} ({len(block['eval_classes'])} evaluated classes) ===")
        header = "  ".join(f"{n[:6]:>6}" for n in block["class_names"])
        print(f"  {'rung':<11}{header}")
        for rung in RUNGS:
            row = "  ".join(f"{v:6.3f}" for v in block["per_class_f1_3seed"][rung]["mean"])
            print(f"  {rung:<11}{row}")
    print(f"\n-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
