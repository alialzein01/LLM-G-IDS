"""Reproduce the UNSW-NB15 comparison ladder from the saved graph and splits.

    python reproduce_ladder.py

Prints the four rungs and writes `ladder_summary.json` +
`STEP4_UNSW_NB15_RESULTS.md` under `data/unsw_nb15/processed/step4_feedback/`.

Two rules this script exists to enforce — both were violated by earlier runs and
both silently inflate the loop's apparent advantage:

  1. Every rung uses the SAME LLM: the whitened-prototype scorer. That is the
     LLM the feedback loop consults, so it is the LLM the loop is measured
     against. The trained MLP head is a separate, stronger baseline — report it
     on its own, never as this ladder's LLM rung.

  2. AGAF consumes the per-fold OOF GNN embeddings, never
     `step3_gnn/edge_embeddings.pt`. That tensor is written by a GNN trained on
     ALL edges, so using it lets AGAF see test-edge labels through its features.

Determinism: graph-attention message passing sums neighbours in thread
completion order, which moves results by ~0.01. The env settings below pin it.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("IDS_FORCE_CPU", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True, warn_only=True)

from src.pipeline.common.datasets import get_dataset_config
from src.pipeline.step3.train_fusion import main as train_fusion
from src.pipeline.step4.assemble_ladder import assemble_ladder
from src.pipeline.step4.train_feedback import train_feedback

DATASET = "unsw_nb15"
OOF_GNN_EMB = "data/unsw_nb15/processed/step3_gnn/edge_embeddings_oof.pt"

EXPECTED = {
    "gnn_alone": 0.5496,
    "llm_alone": 0.7353,
    "agaf": 0.7459,
    "feedback_loop": 0.7436,
}


def main() -> None:
    config = get_dataset_config(DATASET)

    if not os.path.exists(OOF_GNN_EMB):
        raise SystemExit(
            f"Missing {OOF_GNN_EMB}.\n"
            "Build it first:  python -m src.pipeline.step3.build_oof_gnn_embeddings "
            f"--dataset {DATASET}"
        )

    print("\n=== AGAF fusion (leak-free: per-fold OOF GNN embeddings) ===")
    train_fusion(
        data_path=config.graph_path,
        gnn_emb_path=OOF_GNN_EMB,
        llm_emb_path=config.llm_embedding_path,
        splits_path=config.splits_path,
        output_dir=config.fusion_output_dir,
        dataset=DATASET,
    )

    print("\n=== Feedback loop (prototype consultant) + ablations ===")
    # use_llm_head is left at its default False. Setting it True swaps in the
    # trained MLP head, which the loop then largely echoes -- see the module
    # docstring.
    train_feedback(DATASET, modes=["real", "random", "head_only"])

    print("\n=== Ladder ===")
    assemble_ladder(DATASET)

    import json

    summary = json.loads(
        open(f"data/{DATASET}/processed/step4_feedback/ladder_summary.json").read()
    )
    drift = {
        k: (round(v, 4), EXPECTED[k])
        for k, v in summary["ladder"].items()
        if abs(v - EXPECTED[k]) > 0.0005
    }
    if drift:
        print("\n!! Rungs differ from the recorded values (got, expected):")
        for k, (got, exp) in drift.items():
            print(f"     {k}: {got} vs {exp}")
    else:
        print("\nAll four rungs match the recorded values.")


if __name__ == "__main__":
    main()
