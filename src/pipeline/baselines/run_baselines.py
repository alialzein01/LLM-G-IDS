from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import torch

from src.models.baselines.e_graphsage import (
    EGraphSAGE,
    PUBLISHED_DROPOUT as EGRAPH_DROPOUT,
    PUBLISHED_HIDDEN_DIM as EGRAPH_HIDDEN_DIM,
    PUBLISHED_LR as EGRAPH_LR,
    PUBLISHED_NUM_LAYERS as EGRAPH_NUM_LAYERS,
)
from src.models.baselines.te_g_sage import (
    PUBLISHED_DROPOUT as TEG_DROPOUT,
    PUBLISHED_EDGE_MLP_HIDDEN as TEG_EDGE_MLP_HIDDEN,
    PUBLISHED_EPOCHS as TEG_EPOCHS,
    PUBLISHED_HIDDEN_DIM as TEG_HIDDEN_DIM,
    PUBLISHED_LR as TEG_LR,
    PUBLISHED_NUM_LAYERS as TEG_NUM_LAYERS,
    PUBLISHED_WEIGHT_DECAY as TEG_WEIGHT_DECAY,
    TEGSage,
)
from src.pipeline.baselines.harness import pooled_scores, run_out_of_fold
from src.pipeline.baselines.preprocess import (
    egraphsage_edge_features,
    load_aligned_edges,
    te_g_sage_edge_features,
)
from src.pipeline.common.datasets import get_dataset_config

SCHEMA_VERSION = 1

DEVIATIONS = [
    {
        "code": "D1",
        "reason": "TE-G-SAGE publishes a 60/30/10 chronological split; aggregation "
                  "collapsed per-flow timestamps, so no chronological order exists.",
        "resolution": "Our stratified 5-fold folds.pt is used for all models.",
    },
    {
        "code": "D2",
        "reason": "TE-G-SAGE's published fanout (25, 15) exceeds the node degree "
                  "available in a 656-edge (UNSW) / 2127-edge (ToN) graph.",
        "resolution": "Full-neighbourhood, full-batch training. TE-G-SAGE's "
                      "batch_size 4096 also exceeds the entire graph.",
    },
    {
        "code": "D3",
        "reason": "Both released implementations train for a fixed epoch count with no "
                  "validation split and no early stopping -- E-GraphSAGE runs "
                  "range(1, 5000) (4999 epochs), TE-G-SAGE 20. A fixed schedule with no "
                  "validation cannot be used here: our protocol selects on validation "
                  "folds, and 4999 full-batch epochs on a 656-edge graph is unbounded "
                  "overfitting.",
        "resolution": "Max 200 epochs, patience 25 on validation macro-F1, best state "
                      "restored before test prediction -- identical to the schedule our "
                      "own rungs use, so neither side is advantaged.",
    },
    {
        "code": "D4",
        "reason": "TE-G-SAGE's rare_min_freq=50 removes almost every port category "
                  "at our scale, where the whole graph has fewer than 2200 edges.",
        "resolution": "Applied as published in as_published; swept in refit.",
    },
    {
        "code": "D5",
        "reason": "E-GraphSAGE's paper Eq. 4 aggregates edge features alone, but the "
                  "authors' released implementation builds the message as "
                  "W_msg([h_u || e_uv]), concatenating the source node state. The "
                  "published numbers came from the code, not the equation.",
        "resolution": "The released implementation's form is used. Source: "
                      "github.com/waimorris/E-GraphSAGE, "
                      "E-GraphSAGE/netflow/ton-iot/Unsw_ton_iot_multiclass_mean_agg.ipynb",
    },
]


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def empty_payload(dataset: str) -> dict:
    config = get_dataset_config(dataset)
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": config.display_name,
        "dataset_key": config.key,
        "metric_protocol": "pooled five-fold out-of-fold evaluation",
        "primary_metric": "macro_f1",
        "generated": date.today().isoformat(),
        "claim_boundary": (
            "Published architectures re-trained on our aggregated flow-graph "
            "representation. NOT a comparison against their published numbers, "
            "which were obtained on per-flow graphs ~300x larger."
        ),
        "data_provenance": {
            "aggregated_edges_path": config.aggregated_edges_path,
            "aggregated_edges_sha256": _sha256(config.aggregated_edges_path),
            "splits_path": config.splits_path,
            "splits_sha256": _sha256(config.splits_path),
            "eval_classes": list(config.eval_classes),
            "excluded_classes": list(config.dropped_classes),
        },
        "preprocessing": {
            "ours_v2": "log1p+Z on cols 0-2; learned embeddings on protocol/port",
            "e_graphsage": "all five NetFlow columns standard-scaled",
            "te_g_sage": "log1p + StandardScaler + corr-prune@0.995; one-hot "
                         "categoricals with rare-category folding",
        },
        "deviations": list(DEVIATIONS),
        "baselines": {},
    }


def _mean_per_class(seed_scores: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = seed_scores[0]["per_class_f1"]
    return [
        {
            "class_id": row["class_id"],
            "class_name": row["class_name"],
            "f1": float(
                np.mean(
                    [score["per_class_f1"][idx]["f1"] for score in seed_scores]
                )
            ),
        }
        for idx, row in enumerate(rows)
    ]


def _result_entry(
    seed_scores: list[dict[str, object]], hyperparameters: dict[str, object]
) -> dict[str, object]:
    macro_f1 = np.array([score["macro_f1"] for score in seed_scores], dtype=float)
    accuracy = np.array([score["accuracy"] for score in seed_scores], dtype=float)
    weighted_f1 = np.array(
        [score["weighted_f1"] for score in seed_scores], dtype=float
    )
    return {
        "macro_f1": float(macro_f1.mean()),
        "macro_f1_std": float(macro_f1.std()),
        "accuracy": float(accuracy.mean()),
        "weighted_f1": float(weighted_f1.mean()),
        "per_class_f1": _mean_per_class(seed_scores),
        "seeds": seed_scores,
        "hyperparameters": hyperparameters,
        "is_faithful_to_paper": True,
    }


def run(dataset: str, mode: str, seeds: list[int]) -> Path:
    if mode != "as_published":
        raise ValueError(f"mode {mode!r} is implemented in a later task")

    config = get_dataset_config(dataset)
    df, data = load_aligned_edges(config)
    folds = torch.load(config.splits_path, weights_only=False)
    x = data.x.float()
    edge_index = data.edge_index
    labels = data.edge_label

    egraph_features = torch.from_numpy(egraphsage_edge_features(df)).float()
    egraph_hyperparameters: dict[str, object] = {
        "hidden_dim": EGRAPH_HIDDEN_DIM,
        "num_layers": EGRAPH_NUM_LAYERS,
        "dropout": EGRAPH_DROPOUT,
        "optimizer": "adam",
        "lr": EGRAPH_LR,
        "weight_decay": 0.0,
        "class_weighting": "inverse_frequency",
        "published_epochs": 4999,
        "max_epochs": 200,
        "patience": 25,
        "node_init": "ones",
    }

    def egraph_factory() -> EGraphSAGE:
        return EGraphSAGE(
            edge_dim=egraph_features.shape[1],
            hidden_dim=EGRAPH_HIDDEN_DIM,
            num_layers=EGRAPH_NUM_LAYERS,
            num_classes=config.num_classes,
            dropout=EGRAPH_DROPOUT,
            node_init="ones",
        )

    egraph_scores: list[dict[str, object]] = []
    for seed in seeds:
        preds = run_out_of_fold(
            egraph_factory,
            x,
            edge_index,
            egraph_features,
            labels,
            folds,
            seed=seed,
            class_weighting="inverse_frequency",
            eval_classes=config.eval_classes,
            optimizer="adam",
            lr=EGRAPH_LR,
            weight_decay=0.0,
        )
        egraph_scores.append({"seed": seed, **pooled_scores(labels, preds, config)})

    te_features_np, _ = te_g_sage_edge_features(df, rare_min_freq=50)
    te_features = torch.from_numpy(te_features_np).float()
    te_hyperparameters: dict[str, object] = {
        "hidden_dim": TEG_HIDDEN_DIM,
        "num_layers": TEG_NUM_LAYERS,
        "dropout": TEG_DROPOUT,
        "edge_mlp_hidden": TEG_EDGE_MLP_HIDDEN,
        "rare_min_freq": 50,
        "optimizer": "adam",
        "lr": TEG_LR,
        "weight_decay": TEG_WEIGHT_DECAY,
        "class_weighting": "inverse_frequency",
        "published_epochs": TEG_EPOCHS,
        "max_epochs": 200,
        "patience": 25,
        "node_init": "learned_constant",
    }

    def te_factory() -> TEGSage:
        return TEGSage(
            edge_dim=te_features.shape[1],
            hidden_dim=TEG_HIDDEN_DIM,
            num_layers=TEG_NUM_LAYERS,
            num_classes=config.num_classes,
            dropout=TEG_DROPOUT,
            edge_mlp_hidden=TEG_EDGE_MLP_HIDDEN,
            node_init="learned_constant",
        )

    te_scores: list[dict[str, object]] = []
    for seed in seeds:
        preds = run_out_of_fold(
            te_factory,
            x,
            edge_index,
            te_features,
            labels,
            folds,
            seed=seed,
            class_weighting="inverse_frequency",
            eval_classes=config.eval_classes,
            optimizer="adam",
            lr=TEG_LR,
            weight_decay=TEG_WEIGHT_DECAY,
        )
        te_scores.append({"seed": seed, **pooled_scores(labels, preds, config)})

    output_path = Path("results") / f"{dataset}_sota_baselines.json"
    if output_path.exists():
        payload = json.loads(output_path.read_text())
    else:
        payload = empty_payload(dataset)
    baselines = payload.setdefault("baselines", {})
    baselines.setdefault("e_graphsage", {})["as_published"] = _result_entry(
        egraph_scores, egraph_hyperparameters
    )
    baselines.setdefault("te_g_sage", {})["as_published"] = _result_entry(
        te_scores, te_hyperparameters
    )
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--mode", choices=["as_published"], default="as_published"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    args = parser.parse_args()
    output_path = run(args.dataset, args.mode, args.seeds)
    print(output_path)


if __name__ == "__main__":
    main()
