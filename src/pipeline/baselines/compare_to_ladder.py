from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from src.pipeline.common.datasets import get_dataset_config
from src.pipeline.step4.assemble_ladder import _bootstrap_ci, _masked_preds
from src.pipeline.step4.train_feedback import _llm_alone_oof

BOOTSTRAP_ITERS = 2000
BOOTSTRAP_SEED = 42
MODELS = ("e_graphsage", "te_g_sage")
MODES = ("as_published", "refit", "plus_node_features")

COMPARISON_CAVEAT = (
    "Our four rungs were run at seed 42 only, so no rung-side training "
    "stochasticity enters these intervals. They are therefore NARROWER than a "
    "symmetric multi-seed comparison would give. Re-running the ladder at seeds "
    "1 and 2 is the highest-value follow-up."
)


def bootstrap_ci_multiseed(
    labels, preds_rung, preds_baseline_seeds, eval_classes, iters=2000, seed=42
):
    """CI of macro-F1(rung) - macro-F1(baseline), resampling edges AND baseline seed.

    preds_rung:            [E]      our rung, seed 42 only (see the asymmetry note)
    preds_baseline_seeds:  [S, E]   one row per baseline seed
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    pr = np.asarray(preds_rung)
    pb = np.asarray(preds_baseline_seeds)
    labs = list(eval_classes)

    row = np.isin(labels, labs)
    labels, pr, pb = labels[row], pr[row], pb[:, row]
    n, n_seeds = len(labels), pb.shape[0]

    diffs = np.empty(iters)
    for i in range(iters):
        idx = rng.integers(0, n, size=n)
        s = rng.integers(0, n_seeds)
        fa = f1_score(labels[idx], pr[idx], average="macro", labels=labs, zero_division=0)
        fb = f1_score(labels[idx], pb[s][idx], average="macro", labels=labs, zero_division=0)
        diffs[i] = fa - fb

    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "prob_positive": float((diffs > 0).mean()),
        "resampled": "edges_and_baseline_seed",
        "rung_seeds": 1,
        "baseline_seeds": int(n_seeds),
    }


def _load_rung_predictions(dataset: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    config = get_dataset_config(dataset)
    root = Path(f"data/{dataset}/processed/step4_feedback")
    data = torch.load(config.graph_path, weights_only=False)
    folds = torch.load(config.splits_path, weights_only=False)
    embeddings = torch.load(config.llm_embedding_path, weights_only=False).float()
    prototypes = torch.load(root / "prototypes.pt", weights_only=False)

    gnn_logits = torch.load(root / "oof_logits.pt", weights_only=False)
    gnn_preds = _masked_preds(gnn_logits, config.dropped_classes)

    llm_logits = _llm_alone_oof(data, embeddings, folds, prototypes)
    llm_preds = _masked_preds(llm_logits, config.dropped_classes)

    agaf_metrics = json.loads(
        (Path(config.fusion_output_dir) / "metrics.json").read_text()
    )
    agaf_preds = np.asarray(agaf_metrics["predictions"], dtype=np.int64)

    feedback_logits = torch.load(
        root / "feedback_oof_real.pt", weights_only=False
    )
    feedback_preds = _masked_preds(feedback_logits, config.dropped_classes)

    return data.edge_label.cpu().numpy(), {
        "gnn": gnn_preds.cpu().numpy(),
        "llm": llm_preds.cpu().numpy(),
        "agaf": agaf_preds,
        "feedback": feedback_preds.cpu().numpy(),
    }


def compare_to_ladder(dataset: str) -> dict:
    config = get_dataset_config(dataset)
    labels, rung_predictions = _load_rung_predictions(dataset)
    result_path = Path("results") / f"{dataset}_sota_baselines.json"
    payload = json.loads(result_path.read_text())

    primary: dict[str, dict] = {}
    seed_matched: dict[str, dict] = {}
    for model in MODELS:
        for mode in MODES:
            prediction_path = Path(
                f"data/{dataset}/processed/baselines/"
                f"{model}_{mode}_oof_preds.npy"
            )
            baseline_predictions = np.load(prediction_path)
            if baseline_predictions.shape != (3, len(labels)):
                raise ValueError(
                    f"{prediction_path}: expected shape {(3, len(labels))}, "
                    f"got {baseline_predictions.shape}"
                )
            seed_rows = payload["baselines"][model][mode]["seeds"]
            if [row["seed"] for row in seed_rows] != [42, 1, 2]:
                raise ValueError(
                    f"{dataset} {model}.{mode}: predictions must be ordered "
                    "as seeds [42, 1, 2]"
                )

            for rung, rung_preds in rung_predictions.items():
                key = f"{rung}_vs_{model}_{mode}"
                primary[key] = bootstrap_ci_multiseed(
                    labels,
                    rung_preds,
                    baseline_predictions,
                    config.eval_classes,
                    iters=BOOTSTRAP_ITERS,
                    seed=BOOTSTRAP_SEED,
                )
                seed_matched[key] = _bootstrap_ci(
                    labels,
                    rung_preds,
                    baseline_predictions[0],
                    config.eval_classes,
                    iters=BOOTSTRAP_ITERS,
                    seed=BOOTSTRAP_SEED,
                )

    payload["statistical_comparisons"] = primary
    payload["seed_matched_comparisons"] = seed_matched
    payload["comparison_caveat"] = COMPARISON_CAVEAT
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    compare_to_ladder(args.dataset)
    print(Path("results") / f"{args.dataset}_sota_baselines.json")


if __name__ == "__main__":
    main()
