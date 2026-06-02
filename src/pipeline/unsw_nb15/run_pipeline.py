from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.pipeline.unsw_nb15.preprocess import LABEL_MAPPING, run_preprocess
from src.pipeline.step1 import run_step1
from src.pipeline.step2.knowledge_graph import run_step2
from src.pipeline.step3.train_gnn import main as train_gnn
from src.pipeline.step3.encode_kg import run_encode_kg
from src.pipeline.step3.train_fusion import main as train_fusion

RAW_DIR = "data/unsw_nb15/raw"
NORMALIZED_CSV = "data/unsw_nb15/processed/step0/NF-UNSW-NB15-normalized.csv"
STEP1_DIR = "data/unsw_nb15/processed/step1"
STEP2_DIR = "data/unsw_nb15/processed/step2"
SPLITS_PATH = "data/unsw_nb15/processed/splits/folds.pt"
STEP3_GNN_DIR = "data/unsw_nb15/processed/step3_gnn"
STEP3_LLM_DIR = "data/unsw_nb15/processed/step3_llm"
STEP3_FUSION_DIR = "data/unsw_nb15/processed/step3_fusion"


def run_all(
    skip_preprocess: bool = False,
    skip_gnn: bool = False,
    skip_encode: bool = False,
    skip_fusion: bool = False,
) -> None:
    if not skip_preprocess and not Path(NORMALIZED_CSV).exists():
        print("=== Step 0: Preprocessing ===")
        run_preprocess(RAW_DIR, NORMALIZED_CSV)
    else:
        print(f"Step 0: Using existing {NORMALIZED_CSV}")

    print("\n=== Step 1: Graph construction ===")
    run_step1(NORMALIZED_CSV, STEP1_DIR, label_mapping=LABEL_MAPPING)

    print("\n=== Step 2: Knowledge graph ===")
    run_step2(
        csv_path=f"{STEP1_DIR}/aggregated_edges.csv",
        output_dir=STEP2_DIR,
    )

    if not skip_gnn:
        print("\n=== Step 3a: GNN training ===")
        train_gnn(
            data_path=f"{STEP1_DIR}/pyg_data.pt",
            splits_path=SPLITS_PATH,
            output_dir=STEP3_GNN_DIR,
            dataset="unsw_nb15",
        )
    else:
        print(f"Step 3a: Using existing {STEP3_GNN_DIR}")

    if not skip_encode:
        print("\n=== Step 3b: LLM encoding ===")
        run_encode_kg(
            nl_path=f"{STEP2_DIR}/kg_triples_nl.txt",
            output_dir=STEP3_LLM_DIR,
        )
    else:
        print(f"Step 3b: Using existing {STEP3_LLM_DIR}")

    if not skip_fusion:
        print("\n=== Step 3c: Fusion training ===")
        train_fusion(
            data_path=f"{STEP1_DIR}/pyg_data.pt",
            gnn_emb_path=f"{STEP3_GNN_DIR}/edge_embeddings.pt",
            llm_emb_path=f"{STEP3_LLM_DIR}/edge_embeddings.pt",
            splits_path=SPLITS_PATH,
            output_dir=STEP3_FUSION_DIR,
            dataset="unsw_nb15",
        )
    else:
        print(f"Step 3c: Skipping fusion ({STEP3_FUSION_DIR})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run full pipeline on NF-UNSW-NB15")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip preprocessing if normalized CSV already exists")
    parser.add_argument("--skip-gnn", action="store_true",
                        help="Skip GNN training and reuse existing GNN artifacts")
    parser.add_argument("--skip-encode", action="store_true",
                        help="Skip LLM encoding (step 3b)")
    parser.add_argument("--skip-fusion", action="store_true",
                        help="Skip fusion training (step 3c)")
    args = parser.parse_args()

    run_all(
        skip_preprocess=args.skip_preprocess,
        skip_gnn=args.skip_gnn,
        skip_encode=args.skip_encode,
        skip_fusion=args.skip_fusion,
    )
