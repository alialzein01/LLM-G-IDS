from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from itertools import product
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

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
from src.pipeline.common.splits import (
    eval_macro_f1,
    get_class_weights,
    mask_dropped_logits,
)

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
        "resolution": "Max 300 epochs, patience 25 on validation macro-F1, best state "
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


def _save_oof_predictions(
    dataset: str,
    model_name: str,
    mode: str,
    predictions: list[np.ndarray],
) -> Path:
    output_dir = Path(f"data/{dataset}/processed/baselines")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_{mode}_oof_preds.npy"
    np.save(output_path, np.stack(predictions))
    return output_path


def refit_grid(model_name: str) -> list[dict[str, int | float]]:
    base = [
        {"hidden_dim": hidden_dim, "num_layers": num_layers, "lr": lr}
        for hidden_dim, num_layers, lr in product(
            (32, 64, 128), (1, 2), (1e-3, 5e-4, 3e-4)
        )
    ]
    if model_name == "e_graphsage":
        return base
    if model_name == "te_g_sage":
        return [
            {**config, "rare_min_freq": rare_min_freq}
            for config in base
            for rare_min_freq in (50, 2)
        ]
    raise ValueError(f"unknown model_name {model_name!r}")


def _validation_score(
    model_factory: Callable[[], nn.Module],
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    labels: torch.Tensor,
    folds: list[dict[str, torch.Tensor]],
    *,
    seed: int,
    lr: float,
    weight_decay: float,
    eval_classes: tuple[int, ...],
    dropped_classes: tuple[int, ...] = (),
) -> float:
    fold_scores: list[float] = []
    for fold_idx, fold in enumerate(folds):
        torch.manual_seed(seed + fold_idx)
        np.random.seed(seed + fold_idx)

        model = model_factory()
        opt = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        train_mask = fold["train_mask"]
        val_mask = fold["val_mask"]
        criterion = nn.CrossEntropyLoss(
            weight=get_class_weights(labels, train_mask)
        )

        best_f1, stale = -1.0, 0
        for _ in range(300):
            model.train()
            opt.zero_grad()
            loss = criterion(
                model(x, edge_index, edge_attr)[train_mask], labels[train_mask]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            model.eval()
            with torch.no_grad():
                val_preds = mask_dropped_logits(
                    model(x, edge_index, edge_attr)[val_mask], dropped_classes
                ).argmax(dim=1)
            val_f1 = eval_macro_f1(labels[val_mask], val_preds, eval_classes)
            if val_f1 > best_f1:
                best_f1, stale = val_f1, 0
            else:
                stale += 1
                if stale >= 25:
                    break
        fold_scores.append(best_f1)
    return float(np.mean(fold_scores))


def select_refit(dataset: str, model_name: str, seeds: list[int]) -> dict:
    config = get_dataset_config(dataset)
    df, data = load_aligned_edges(config)
    folds = torch.load(config.splits_path, weights_only=False)
    x = data.x.float()
    edge_index = data.edge_index
    labels = data.edge_label

    egraph_features = torch.from_numpy(egraphsage_edge_features(df)).float()
    te_features = {
        rare_min_freq: torch.from_numpy(
            te_g_sage_edge_features(df, rare_min_freq=rare_min_freq)[0]
        ).float()
        for rare_min_freq in (50, 2)
    }

    trace: list[dict[str, object]] = []
    for candidate in refit_grid(model_name):
        if model_name == "e_graphsage":
            edge_attr = egraph_features

            def model_factory() -> EGraphSAGE:
                return EGraphSAGE(
                    edge_dim=edge_attr.shape[1],
                    hidden_dim=int(candidate["hidden_dim"]),
                    num_layers=int(candidate["num_layers"]),
                    num_classes=config.num_classes,
                    dropout=EGRAPH_DROPOUT,
                    node_init="ones",
                )

            weight_decay = 0.0
        elif model_name == "te_g_sage":
            edge_attr = te_features[int(candidate["rare_min_freq"])]

            def model_factory() -> TEGSage:
                return TEGSage(
                    edge_dim=edge_attr.shape[1],
                    hidden_dim=int(candidate["hidden_dim"]),
                    num_layers=int(candidate["num_layers"]),
                    num_classes=config.num_classes,
                    dropout=TEG_DROPOUT,
                    edge_mlp_hidden=TEG_EDGE_MLP_HIDDEN,
                    node_init="learned_constant",
                )

            weight_decay = TEG_WEIGHT_DECAY
        else:
            raise ValueError(f"unknown model_name {model_name!r}")

        seed_scores = [
            {
                "seed": seed,
                "validation_macro_f1": _validation_score(
                    model_factory,
                    x,
                    edge_index,
                    edge_attr,
                    labels,
                    folds,
                    seed=seed,
                    lr=float(candidate["lr"]),
                    weight_decay=weight_decay,
                    eval_classes=config.eval_classes,
                    dropped_classes=config.dropped_classes,
                ),
            }
            for seed in seeds
        ]
        trace.append(
            {
                "hyperparameters": dict(candidate),
                "validation_macro_f1": float(
                    np.mean([row["validation_macro_f1"] for row in seed_scores])
                ),
                "seeds": seed_scores,
            }
        )

    winner = max(trace, key=lambda row: row["validation_macro_f1"])
    return {
        "hyperparameters": winner["hyperparameters"],
        "validation_macro_f1": winner["validation_macro_f1"],
        "grid_trace": trace,
    }


def _run_refit(dataset: str, seeds: list[int]) -> Path:
    config = get_dataset_config(dataset)
    df, data = load_aligned_edges(config)
    folds = torch.load(config.splits_path, weights_only=False)
    x = data.x.float()
    edge_index = data.edge_index
    labels = data.edge_label

    egraph_selection = select_refit(dataset, "e_graphsage", seeds)
    egraph_winner = egraph_selection["hyperparameters"]
    egraph_features = torch.from_numpy(egraphsage_edge_features(df)).float()

    def egraph_factory() -> EGraphSAGE:
        return EGraphSAGE(
            edge_dim=egraph_features.shape[1],
            hidden_dim=int(egraph_winner["hidden_dim"]),
            num_layers=int(egraph_winner["num_layers"]),
            num_classes=config.num_classes,
            dropout=EGRAPH_DROPOUT,
            node_init="ones",
        )

    egraph_scores: list[dict[str, object]] = []
    egraph_predictions: list[np.ndarray] = []
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
            dropped_classes=config.dropped_classes,
            optimizer="adam",
            lr=float(egraph_winner["lr"]),
            weight_decay=0.0,
        )
        egraph_predictions.append(preds)
        egraph_scores.append({"seed": seed, **pooled_scores(labels, preds, config)})

    egraph_hyperparameters = {
        **egraph_winner,
        "dropout": EGRAPH_DROPOUT,
        "optimizer": "adam",
        "weight_decay": 0.0,
        "class_weighting": "inverse_frequency",
        "max_epochs": 300,
        "patience": 25,
        "node_init": "ones",
    }
    egraph_entry = _result_entry(egraph_scores, egraph_hyperparameters)
    egraph_entry["validation_macro_f1"] = egraph_selection["validation_macro_f1"]
    egraph_entry["grid_trace"] = egraph_selection["grid_trace"]

    te_selection = select_refit(dataset, "te_g_sage", seeds)
    te_winner = te_selection["hyperparameters"]
    te_features = torch.from_numpy(
        te_g_sage_edge_features(
            df, rare_min_freq=int(te_winner["rare_min_freq"])
        )[0]
    ).float()

    def te_factory() -> TEGSage:
        return TEGSage(
            edge_dim=te_features.shape[1],
            hidden_dim=int(te_winner["hidden_dim"]),
            num_layers=int(te_winner["num_layers"]),
            num_classes=config.num_classes,
            dropout=TEG_DROPOUT,
            edge_mlp_hidden=TEG_EDGE_MLP_HIDDEN,
            node_init="learned_constant",
        )

    te_scores: list[dict[str, object]] = []
    te_predictions: list[np.ndarray] = []
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
            dropped_classes=config.dropped_classes,
            optimizer="adam",
            lr=float(te_winner["lr"]),
            weight_decay=TEG_WEIGHT_DECAY,
        )
        te_predictions.append(preds)
        te_scores.append({"seed": seed, **pooled_scores(labels, preds, config)})

    te_hyperparameters = {
        **te_winner,
        "dropout": TEG_DROPOUT,
        "edge_mlp_hidden": TEG_EDGE_MLP_HIDDEN,
        "optimizer": "adam",
        "weight_decay": TEG_WEIGHT_DECAY,
        "class_weighting": "inverse_frequency",
        "max_epochs": 300,
        "patience": 25,
        "node_init": "learned_constant",
    }
    te_entry = _result_entry(te_scores, te_hyperparameters)
    te_entry["validation_macro_f1"] = te_selection["validation_macro_f1"]
    te_entry["grid_trace"] = te_selection["grid_trace"]

    output_path = Path("results") / f"{dataset}_sota_baselines.json"
    if output_path.exists():
        payload = json.loads(output_path.read_text())
    else:
        payload = empty_payload(dataset)
    baselines = payload.setdefault("baselines", {})
    baselines.setdefault("e_graphsage", {})["refit"] = egraph_entry
    baselines.setdefault("te_g_sage", {})["refit"] = te_entry
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    _save_oof_predictions(
        dataset, "e_graphsage", "refit", egraph_predictions
    )
    _save_oof_predictions(dataset, "te_g_sage", "refit", te_predictions)
    return output_path


def _refit_hyperparameters(
    payload: dict, dataset: str, model_name: str
) -> dict[str, object]:
    try:
        return payload["baselines"][model_name]["refit"]["hyperparameters"]
    except KeyError as exc:
        raise RuntimeError(
            f"{dataset}: missing baselines.{model_name}.refit.hyperparameters; "
            "run Task 7 refit selection before plus_node_features"
        ) from exc


def _run_plus_node_features(dataset: str, seeds: list[int]) -> Path:
    config = get_dataset_config(dataset)
    output_path = Path("results") / f"{dataset}_sota_baselines.json"
    if not output_path.exists():
        raise RuntimeError(
            f"{dataset}: missing {output_path}; run Task 7 before plus_node_features"
        )
    payload = json.loads(output_path.read_text())
    egraph_refit = _refit_hyperparameters(payload, dataset, "e_graphsage")
    te_refit = _refit_hyperparameters(payload, dataset, "te_g_sage")

    df, data = load_aligned_edges(config)
    folds = torch.load(config.splits_path, weights_only=False)
    x = data.x.float()
    edge_index = data.edge_index
    labels = data.edge_label
    node_feat_dim = int(x.shape[1])

    egraph_features = torch.from_numpy(egraphsage_edge_features(df)).float()

    def egraph_factory() -> EGraphSAGE:
        return EGraphSAGE(
            edge_dim=egraph_features.shape[1],
            hidden_dim=int(egraph_refit["hidden_dim"]),
            num_layers=int(egraph_refit["num_layers"]),
            num_classes=config.num_classes,
            dropout=float(egraph_refit["dropout"]),
            node_init="node_features",
            node_feat_dim=node_feat_dim,
        )

    egraph_scores: list[dict[str, object]] = []
    egraph_predictions: list[np.ndarray] = []
    for seed in seeds:
        preds = run_out_of_fold(
            egraph_factory,
            x,
            edge_index,
            egraph_features,
            labels,
            folds,
            seed=seed,
            class_weighting=str(egraph_refit["class_weighting"]),
            eval_classes=config.eval_classes,
            dropped_classes=config.dropped_classes,
            optimizer=str(egraph_refit["optimizer"]),
            lr=float(egraph_refit["lr"]),
            weight_decay=float(egraph_refit["weight_decay"]),
            max_epochs=int(egraph_refit["max_epochs"]),
            patience=int(egraph_refit["patience"]),
        )
        egraph_predictions.append(preds)
        egraph_scores.append({"seed": seed, **pooled_scores(labels, preds, config)})

    egraph_hyperparameters = {
        **egraph_refit,
        "node_init": "node_features",
        "node_feat_dim": node_feat_dim,
    }
    egraph_entry = _result_entry(egraph_scores, egraph_hyperparameters)
    egraph_entry["is_faithful_to_paper"] = False
    egraph_entry["variant_label"] = "E-GraphSAGE + our node features"

    te_features = torch.from_numpy(
        te_g_sage_edge_features(
            df, rare_min_freq=int(te_refit["rare_min_freq"])
        )[0]
    ).float()

    def te_factory() -> TEGSage:
        return TEGSage(
            edge_dim=te_features.shape[1],
            hidden_dim=int(te_refit["hidden_dim"]),
            num_layers=int(te_refit["num_layers"]),
            num_classes=config.num_classes,
            dropout=float(te_refit["dropout"]),
            edge_mlp_hidden=int(te_refit["edge_mlp_hidden"]),
            node_init="node_features",
            node_feat_dim=node_feat_dim,
        )

    te_scores: list[dict[str, object]] = []
    te_predictions: list[np.ndarray] = []
    for seed in seeds:
        preds = run_out_of_fold(
            te_factory,
            x,
            edge_index,
            te_features,
            labels,
            folds,
            seed=seed,
            class_weighting=str(te_refit["class_weighting"]),
            eval_classes=config.eval_classes,
            dropped_classes=config.dropped_classes,
            optimizer=str(te_refit["optimizer"]),
            lr=float(te_refit["lr"]),
            weight_decay=float(te_refit["weight_decay"]),
            max_epochs=int(te_refit["max_epochs"]),
            patience=int(te_refit["patience"]),
        )
        te_predictions.append(preds)
        te_scores.append({"seed": seed, **pooled_scores(labels, preds, config)})

    te_hyperparameters = {
        **te_refit,
        "node_init": "node_features",
        "node_feat_dim": node_feat_dim,
    }
    te_entry = _result_entry(te_scores, te_hyperparameters)
    te_entry["is_faithful_to_paper"] = False
    te_entry["variant_label"] = "TE-G-SAGE + our node features"

    baselines = payload["baselines"]
    baselines["e_graphsage"]["plus_node_features"] = egraph_entry
    baselines["te_g_sage"]["plus_node_features"] = te_entry
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    _save_oof_predictions(
        dataset, "e_graphsage", "plus_node_features", egraph_predictions
    )
    _save_oof_predictions(
        dataset, "te_g_sage", "plus_node_features", te_predictions
    )
    return output_path


def run(dataset: str, mode: str, seeds: list[int]) -> Path:
    if mode == "plus_node_features":
        return _run_plus_node_features(dataset, seeds)
    if mode == "refit":
        return _run_refit(dataset, seeds)
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
        "max_epochs": 300,
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
    egraph_predictions: list[np.ndarray] = []
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
            dropped_classes=config.dropped_classes,
            optimizer="adam",
            lr=EGRAPH_LR,
            weight_decay=0.0,
        )
        egraph_predictions.append(preds)
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
        "max_epochs": 300,
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
    te_predictions: list[np.ndarray] = []
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
            dropped_classes=config.dropped_classes,
            optimizer="adam",
            lr=TEG_LR,
            weight_decay=TEG_WEIGHT_DECAY,
        )
        te_predictions.append(preds)
        te_scores.append({"seed": seed, **pooled_scores(labels, preds, config)})

    output_path = Path("results") / f"{dataset}_sota_baselines.json"
    if output_path.exists():
        payload = json.loads(output_path.read_text())
    else:
        payload = empty_payload(dataset)
    payload["deviations"] = list(DEVIATIONS)
    baselines = payload.setdefault("baselines", {})
    baselines.setdefault("e_graphsage", {})["as_published"] = _result_entry(
        egraph_scores, egraph_hyperparameters
    )
    baselines.setdefault("te_g_sage", {})["as_published"] = _result_entry(
        te_scores, te_hyperparameters
    )
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    _save_oof_predictions(
        dataset, "e_graphsage", "as_published", egraph_predictions
    )
    _save_oof_predictions(
        dataset, "te_g_sage", "as_published", te_predictions
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--mode",
        choices=["as_published", "refit", "plus_node_features"],
        default="as_published",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    args = parser.parse_args()
    output_path = run(args.dataset, args.mode, args.seeds)
    print(output_path)


if __name__ == "__main__":
    main()
