"""End-to-end ToN-IoT pipeline: graph construction through the Step 4 ladder.

Every stage runs from a single Step 1 graph, so the artifacts it produces are
mutually consistent by construction. Running stages by hand at different times is
what let `reports/phase4/validation_report.md` and `STEP4_TON_IOT_RESULTS.md`
disagree about the same quantities.

Two graph forms are supported, selected by `--dataset`:

  ton_iot         aggregated by (src, dst, attack) — the todo.md-mandated form
  ton_iot_capped  one edge per distinct flow signature, carrying its occurrence
                  count, capped at `--cap` rows per class

Architecture, hyperparameters and every downstream stage are identical between
them. Only the dataset differs.

    python -m src.pipeline.ton_iot.run_pipeline --dataset ton_iot_capped --cap 20000
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.pipeline.common.datasets import DATASETS, get_dataset_config
from src.pipeline.step1.graph_construction import LABEL_MAPPING, run_step1
from src.pipeline.step2.knowledge_graph import run_step2

STAGE_ORDER = [
    "step1",
    "prune",
    "augment",
    "step2",
    "splits",
    "verify",
    "gnn",
    "oof_emb",
    "encode",
    "fusion",
    "baselines",
    "oof",
    "llm_heads",
    "prototypes",
    "feedback",
    "ladder",
    "validate",
]


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def _run_module(module: str, *args: str) -> None:
    cmd = [sys.executable, "-u", "-m", module, *args]
    print(f"\n$ {' '.join(cmd)}", flush=True)
    # Match reproduce_ladder.py: attention message passing sums neighbours in
    # thread-completion order, which moves rungs by ~0.01 run to run.
    env = {**os.environ, "IDS_FORCE_CPU": "1", "OMP_NUM_THREADS": "1"}
    subprocess.run(cmd, check=True, env=env)


def run_all(
    dataset: str = "ton_iot_capped",
    cap: int | None = 20_000,
    seed: int = 42,
    only: list[str] | None = None,
    skip: list[str] | None = None,
) -> None:
    config = get_dataset_config(dataset)
    per_flow = dataset.endswith("_capped")
    stages = [s for s in STAGE_ORDER if (not only or s in only) and s not in (skip or [])]

    step1_dir = str(Path(config.graph_path).parent)
    step2_dir = str(Path(config.kg_csv_path).parent)
    llm_dir = str(Path(config.llm_embedding_path).parent)
    oof_gnn_emb = str(
        Path(config.gnn_embedding_path).with_name("edge_embeddings_oof.pt")
    )

    manifest: dict[str, object] = {
        "dataset": dataset,
        "per_flow": per_flow,
        "cap_per_class": cap if per_flow else None,
        "seed": seed,
        "git_sha": _git_sha(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stages": {},
    }

    def stage(name: str) -> bool:
        if name not in stages:
            print(f"\n--- skipping {name} ---", flush=True)
            return False
        print(f"\n{'=' * 72}\n=== {name.upper()}\n{'=' * 72}", flush=True)
        return True

    def done(name: str, t0: float) -> None:
        manifest["stages"][name] = round(time.perf_counter() - t0, 1)

    if stage("step1"):
        t0 = time.perf_counter()
        run_step1(
            config.phase1_input_path,
            step1_dir,
            label_mapping=LABEL_MAPPING,
            per_flow=per_flow,
            cap_per_class=cap if per_flow else None,
            seed=seed,
        )
        done("step1", t0)

    if stage("prune"):
        t0 = time.perf_counter()
        _run_module("src.pipeline.step1.prune_node_features", "--dataset", dataset)
        done("prune", t0)

    if stage("augment"):
        t0 = time.perf_counter()
        _run_module("src.pipeline.step1.augment_traffic_features", "--dataset", dataset)
        done("augment", t0)

    if stage("step2"):
        t0 = time.perf_counter()
        run_step2(csv_path=config.aggregated_edges_path, output_dir=step2_dir)
        done("step2", t0)

    if stage("splits"):
        t0 = time.perf_counter()
        splits = Path(config.splits_path)
        if splits.exists():
            splits.unlink()  # stale masks are sized to the previous edge count
        _run_module("src.pipeline.common.build_splits", "--dataset", dataset,
                    "--seed", str(seed))
        done("splits", t0)

    if stage("verify"):
        t0 = time.perf_counter()
        # Fails loudly on duplicate signatures or fold leakage — the defect that
        # made the first per-flow attempt score 0.7597 on memorised test edges.
        if per_flow:
            _run_module("src.pipeline.step1.verify_flow_graph", "--dataset", dataset)
        else:
            print("aggregated graph — signature leakage check not applicable")
        done("verify", t0)

    if stage("gnn"):
        t0 = time.perf_counter()
        _run_module("src.pipeline.step3.train_gnn", "--dataset", dataset,
                    "--device", "cpu")
        done("gnn", t0)

    if stage("oof_emb"):
        t0 = time.perf_counter()
        _run_module("src.pipeline.step3.build_oof_gnn_embeddings",
                    "--dataset", dataset)
        done("oof_emb", t0)

    if stage("encode"):
        t0 = time.perf_counter()
        # encode_kg has no CLI; call it directly with this dataset's paths.
        from src.pipeline.step3.encode_kg import run_encode_kg

        run_encode_kg(nl_path=config.kg_nl_path, output_dir=llm_dir)
        done("encode", t0)

    if stage("fusion"):
        t0 = time.perf_counter()
        # AGAF must consume per-fold OOF embeddings. step3_gnn/edge_embeddings.pt
        # comes from a GNN fitted on ALL edges, so feeding it here lets fusion see
        # test-edge labels through its features — the leak commit 2d13ff1 fixed
        # (UNSW AGAF fell 0.8085 -> 0.7459 once corrected).
        _run_module("src.pipeline.step3.train_fusion", "--dataset", dataset,
                    "--gnn-emb-path", oof_gnn_emb)
        done("fusion", t0)

    if stage("baselines"):
        t0 = time.perf_counter()
        _run_module("src.pipeline.step3.train_unimodal_baselines",
                    "--dataset", dataset, "--modality", "both")
        done("baselines", t0)

    for name, module, extra in [
        ("oof", "src.pipeline.step4.build_oof_predictions", []),
        ("llm_heads", "src.pipeline.step4.build_llm_heads", []),
        ("prototypes", "src.pipeline.step4.build_prototypes", []),
        # use_llm_head stays OFF. With it on, the loop consults the trained MLP
        # head and largely echoes it, while assemble_ladder scores the LLM rung
        # with the prototype scorer — comparing the loop against an LLM it never
        # used. reproduce_ladder.py enforces the same default.
        ("feedback", "src.pipeline.step4.train_feedback",
         ["--modes", "real", "random", "head_only"]),
        ("ladder", "src.pipeline.step4.assemble_ladder", []),
    ]:
        if stage(name):
            t0 = time.perf_counter()
            _run_module(module, "--dataset", dataset, *extra)
            done(name, t0)

    if stage("validate"):
        t0 = time.perf_counter()
        for phase in ("phase1_validate", "phase2_validate", "phase3_validate",
                      "phase4_validate"):
            try:
                _run_module(f"src.pipeline.{phase}", "--dataset", dataset)
            except subprocess.CalledProcessError as exc:
                # A validator returning non-zero is a verdict, not a crash.
                print(f"{phase} reported failures (exit {exc.returncode})", flush=True)
        done("validate", t0)

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    out = Path(f"data/{dataset}/processed/run_manifest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n{'=' * 72}\nPipeline complete. Manifest -> {out}", flush=True)
    print(json.dumps(manifest, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ton_iot_capped", choices=sorted(DATASETS))
    parser.add_argument("--cap", type=int, default=20_000,
                        help="Rows per class (capped datasets only). 0 disables.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", nargs="+", choices=STAGE_ORDER)
    parser.add_argument("--skip", nargs="+", choices=STAGE_ORDER)
    args = parser.parse_args()
    run_all(
        dataset=args.dataset,
        cap=args.cap or None,
        seed=args.seed,
        only=args.only,
        skip=args.skip,
    )


if __name__ == "__main__":
    main()
