"""Mechanism-only edge-injection test, v2 encoding, both datasets, 3 seeds.

Reproduces results/unsw_nb15_edge_injection_v2.json. Run from repo root:

    python -m src.pipeline.step4.mechanism_only_edge_injection

use_output_fusion=False kills the late-fusion door entirely, so the LLM reaches the
GNN ONLY through the consultative feedback path. Scored against `control_head_only`
-- the same model with the feedback structurally absent -- NEVER against the bare GNN
rung, which is a different model from a different training driver.

Two advice controls decide whether any gain is semantic content or just added capacity:
  shuffled = real advice delivered to the WRONG edge
  random   = fixed uniform noise
If either matches `real`, the mechanism is not reading the advice.

NOT comparable to ladder numbers: fusion is off in every arm.

Requires prototypes.pt and folds.pt already built for both datasets
(step4.build_prototypes, splits) before running.
"""
import sys
import time
import json
import numpy as np
import torch
import warnings

warnings.filterwarnings("ignore")

import src.pipeline.step4.train_feedback as FB
from src.pipeline.step4.sweep_top_k import _pooled_macro_f1
from src.pipeline.common.datasets import get_dataset_config

SEEDS = (42, 1, 2)
SETUP = {
    "unsw_nb15": dict(top_k=31.0, scale=2.0),
    "ton_iot": dict(top_k=25.0, scale=20.0),
}
ARMS = ["control_head_only", "real", "control_shuffled", "control_random"]


def run(output_path: str = "results/raw/mechanism_only_v2.json") -> dict:
    _orig_seed = FB.SEED
    out: dict = {}

    for ds, cfgv in SETUP.items():
        cfg = get_dataset_config(ds)
        D = f"data/{ds}/processed"
        data = torch.load(f"{D}/step1/pyg_data.pt", weights_only=False)
        folds = torch.load(f"{D}/splits/folds.pt", weights_only=False)
        emb = torch.load(cfg.llm_embedding_path, weights_only=False)
        protos = torch.load(f"{D}/step4_feedback/prototypes.pt", weights_only=False)
        labels = data.edge_label
        ec = tuple(cfg.eval_classes)
        dc = tuple(cfg.dropped_classes) if cfg.dropped_classes else ()
        NC = cfg.num_classes
        out[ds] = {}
        print(
            f"\n===== {cfg.display_name}  (top_k={cfgv['top_k']}, scale={cfgv['scale']}, "
            f"output_fusion=OFF) =====",
            flush=True,
        )
        t0 = time.time()
        for arm in ARMS:
            mode = (
                "head_only" if arm == "control_head_only"
                else "shuffled" if arm == "control_shuffled"
                else "random" if arm == "control_random"
                else "real"
            )
            runs, churns = [], []
            for seed in SEEDS:
                FB.SEED = seed
                oof = torch.full((labels.shape[0], NC), float("nan"))
                ch = []
                for fi, f in enumerate(folds):
                    r = FB._train_one_fold(
                        data=data, emb=emb, fold_state=protos["folds"][fi], fold=f,
                        fold_idx=fi, mode=mode, top_k_percent=cfgv["top_k"],
                        eval_classes=ec, dropped_classes=dc, injection_mode="edge",
                        injection_scale=cfgv["scale"], use_output_fusion=False,
                    )
                    oof[f["test_mask"]] = r.logits[f["test_mask"]]
                    c = [
                        it["churn"] for it in r.iterations
                        if isinstance(it.get("churn"), float) and it["churn"] == it["churn"]
                    ]
                    ch.append(float(np.mean(c)) if c else 0.0)
                runs.append(_pooled_macro_f1(labels, oof, ec, dc))
                churns.append(float(np.mean(ch)))
            out[ds][arm] = dict(
                mean=float(np.mean(runs)), std=float(np.std(runs)),
                runs=[round(x, 4) for x in runs], churn=float(np.mean(churns)),
            )
            a = out[ds][arm]
            print(
                f"  {arm:20s} {a['mean']:.4f} +/-{a['std']:.4f}  churn={a['churn']:.4f}  "
                f"runs={a['runs']}  [{time.time() - t0:5.0f}s]",
                flush=True,
            )
        c = out[ds]["control_head_only"]["mean"]
        print(f"  --> real - control          = {out[ds]['real']['mean'] - c:+.4f}")
        print(f"      shuffled - control      = {out[ds]['control_shuffled']['mean'] - c:+.4f}")
        print(f"      random - control        = {out[ds]['control_random']['mean'] - c:+.4f}")

    FB.SEED = _orig_seed
    json.dump(out, open(output_path, "w"), indent=1)
    print("\nPASS CRITERION: real > control, AND both advice controls <= control.")
    return out


if __name__ == "__main__":
    run()
