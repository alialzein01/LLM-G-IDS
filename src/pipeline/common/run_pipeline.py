"""Dataset-parameterized pipeline runner for graph construction through Step 4.

The default feature profile is ``structural10``: the same ten centrality node
features used by the canonical UNSW-NB15 ladder. ToN-IoT's older
``enhanced12`` profile remains available for a later feature ablation, but it
must be requested explicitly so parity runs cannot reuse those artifacts by
accident.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from src.pipeline.common.datasets import DATASETS, get_dataset_config
from src.pipeline.step1.graph_construction import LABEL_MAPPING as TON_LABEL_MAPPING
from src.pipeline.step1.graph_construction import run_step1
from src.pipeline.step2.knowledge_graph import run_step2

STAGE_ORDER = [
    "preprocess",
    "step1",
    "features",
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
    "top_k_sweep",
    "feedback",
    "ladder",
    "validate",
]

FEATURE_PROFILES = ("structural10", "enhanced12")
DEFAULT_SEED = 42


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
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
    env = {**os.environ, "IDS_FORCE_CPU": "1", "OMP_NUM_THREADS": "1"}
    subprocess.run(cmd, check=True, env=env)


def _label_mapping(dataset: str) -> dict[str, int]:
    if dataset == "unsw_nb15":
        from src.pipeline.unsw_nb15.preprocess import LABEL_MAPPING

        return LABEL_MAPPING
    return TON_LABEL_MAPPING


def _maybe_preprocess(dataset: str, dry_run: bool) -> None:
    if dataset != "unsw_nb15":
        print("preprocess not applicable for this dataset")
        return
    from src.pipeline.unsw_nb15.preprocess import run_preprocess

    raw_dir = "data/unsw_nb15/raw"
    normalized = Path(get_dataset_config(dataset).phase1_input_path)
    if dry_run:
        print(f"would preprocess {raw_dir} -> {normalized}")
        return
    if normalized.exists():
        print(f"using existing {normalized}")
    else:
        run_preprocess(raw_dir, str(normalized))


def _apply_feature_profile(dataset: str, feature_profile: str, dry_run: bool) -> None:
    if feature_profile not in FEATURE_PROFILES:
        raise ValueError(
            f"Unknown feature profile {feature_profile!r}; valid choices: {FEATURE_PROFILES}"
        )
    config = get_dataset_config(dataset)
    graph_path = Path(config.graph_path)

    if feature_profile == "enhanced12":
        if not dataset.startswith("ton_iot"):
            raise ValueError("enhanced12 is currently defined only for ToN-IoT datasets.")
        if dry_run:
            print(f"would prune and augment {graph_path} to enhanced12")
            return
        _run_module("src.pipeline.step1.prune_node_features", "--dataset", dataset)
        _run_module("src.pipeline.step1.augment_traffic_features", "--dataset", dataset)

    if dry_run:
        print(f"would verify {graph_path} has feature profile {feature_profile}")
        return

    data = torch.load(graph_path, weights_only=False)
    expected_dim = 12 if feature_profile == "enhanced12" else 10
    if data.x.shape[1] != expected_dim:
        raise RuntimeError(
            f"{dataset} graph at {graph_path} has {data.x.shape[1]} node features, "
            f"but profile {feature_profile!r} expects {expected_dim}. Re-run step1 "
            "and the matching features stage before downstream training."
        )
    data.feature_profile = feature_profile
    torch.save(data, graph_path)


def _select_stages(
    only: list[str] | None,
    skip: list[str] | None,
) -> list[str]:
    return [s for s in STAGE_ORDER if (not only or s in only) and s not in (skip or [])]


def _processed_root(dataset: str) -> Path:
    return Path(get_dataset_config(dataset).graph_path).parents[1]


def run_all(
    dataset: str = "ton_iot",
    *,
    feature_profile: str = "structural10",
    seed: int = DEFAULT_SEED,
    only: list[str] | None = None,
    skip: list[str] | None = None,
    dry_run: bool = False,
    module_runner: Callable[..., None] = _run_module,
) -> dict[str, Any]:
    config = get_dataset_config(dataset)
    stages = _select_stages(only, skip)

    step1_dir = str(Path(config.graph_path).parent)
    step2_dir = str(Path(config.kg_csv_path).parent)
    llm_dir = str(Path(config.llm_embedding_path).parent)
    oof_gnn_emb = str(Path(config.gnn_embedding_path).with_name("edge_embeddings_oof.pt"))

    manifest: dict[str, Any] = {
        "dataset": dataset,
        "feature_profile": feature_profile,
        "seed": seed,
        "git_sha": _git_sha(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "stages": {},
    }

    def stage(name: str) -> bool:
        if name not in stages:
            print(f"\n--- skipping {name} ---", flush=True)
            return False
        print(f"\n{'=' * 72}\n=== {name.upper()}\n{'=' * 72}", flush=True)
        return True

    def done(name: str, t0: float) -> None:
        elapsed = round(time.perf_counter() - t0, 1)
        if isinstance(manifest["stages"].get(name), dict):
            manifest["stages"][name]["elapsed_seconds"] = elapsed
        else:
            manifest["stages"][name] = elapsed

    def run_stage_module(name: str, module: str, *args: str) -> None:
        if dry_run:
            manifest["stages"][name] = {"module": module, "args": list(args)}
            print(f"would run: {module} {' '.join(args)}")
        else:
            module_runner(module, *args)

    if stage("preprocess"):
        t0 = time.perf_counter()
        _maybe_preprocess(dataset, dry_run)
        done("preprocess", t0)

    if stage("step1"):
        t0 = time.perf_counter()
        if dry_run:
            print(f"would build graph {config.phase1_input_path} -> {step1_dir}")
        else:
            run_step1(
                config.phase1_input_path,
                step1_dir,
                label_mapping=_label_mapping(dataset),
            )
        done("step1", t0)

    if stage("features"):
        t0 = time.perf_counter()
        _apply_feature_profile(dataset, feature_profile, dry_run)
        done("features", t0)

    if stage("step2"):
        t0 = time.perf_counter()
        if dry_run:
            print(f"would build KG {config.aggregated_edges_path} -> {step2_dir}")
        else:
            run_step2(csv_path=config.aggregated_edges_path, output_dir=step2_dir)
        done("step2", t0)

    if stage("splits"):
        t0 = time.perf_counter()
        splits = Path(config.splits_path)
        if dry_run:
            print(f"would rebuild splits at {splits}")
        else:
            if splits.exists():
                splits.unlink()
            module_runner(
                "src.pipeline.common.build_splits",
                "--dataset", dataset,
                "--seed", str(seed),
            )
        done("splits", t0)

    if stage("verify"):
        t0 = time.perf_counter()
        print("aggregated graph: signature leakage check not applicable")
        done("verify", t0)

    for name, module, extra in [
        ("gnn", "src.pipeline.step3.train_gnn", ["--device", "cpu"]),
        ("oof_emb", "src.pipeline.step3.build_oof_gnn_embeddings", []),
    ]:
        if stage(name):
            t0 = time.perf_counter()
            run_stage_module(name, module, "--dataset", dataset, *extra)
            done(name, t0)

    if stage("encode"):
        t0 = time.perf_counter()
        if dry_run:
            print(f"would encode KG NL {config.kg_nl_path} -> {llm_dir}")
        else:
            from src.pipeline.step3.encode_kg import run_encode_kg

            run_encode_kg(nl_path=config.kg_nl_path, output_dir=llm_dir)
        done("encode", t0)

    if stage("fusion"):
        t0 = time.perf_counter()
        run_stage_module(
            "fusion",
            "src.pipeline.step3.train_fusion",
            "--dataset", dataset,
            "--gnn-emb-path", oof_gnn_emb,
        )
        done("fusion", t0)

    if stage("baselines"):
        t0 = time.perf_counter()
        run_stage_module(
            "baselines",
            "src.pipeline.step3.train_unimodal_baselines",
            "--dataset", dataset,
            "--modality", "both",
        )
        done("baselines", t0)

    for name, module, extra in [
        ("oof", "src.pipeline.step4.build_oof_predictions", []),
        ("llm_heads", "src.pipeline.step4.build_llm_heads", []),
        ("prototypes", "src.pipeline.step4.build_prototypes", []),
        (
            "top_k_sweep",
            "src.pipeline.step4.sweep_top_k",
            ["--min-percent", "15", "--max-percent", "35"],
        ),
        ("feedback", "src.pipeline.step4.train_feedback", ["--modes", "real", "random", "head_only"]),
        ("ladder", "src.pipeline.step4.assemble_ladder", []),
    ]:
        if stage(name):
            t0 = time.perf_counter()
            run_stage_module(name, module, "--dataset", dataset, *extra)
            done(name, t0)

    if stage("validate"):
        t0 = time.perf_counter()
        for phase in ("phase1_validate", "phase2_validate", "phase3_validate", "phase4_validate"):
            try:
                run_stage_module(
                    f"validate:{phase}", f"src.pipeline.{phase}", "--dataset", dataset
                )
            except subprocess.CalledProcessError as exc:
                print(f"{phase} reported failures (exit {exc.returncode})", flush=True)
        done("validate", t0)

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    if not dry_run:
        out = _processed_root(dataset) / f"run_manifest_{feature_profile}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2) + "\n")
        compat = _processed_root(dataset) / "run_manifest.json"
        compat.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"\n{'=' * 72}\nPipeline complete. Manifest -> {out}", flush=True)
        print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ton_iot", choices=sorted(DATASETS))
    parser.add_argument("--feature-profile", default="structural10", choices=FEATURE_PROFILES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--only", nargs="+", choices=STAGE_ORDER)
    parser.add_argument("--skip", nargs="+", choices=STAGE_ORDER)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_all(
        dataset=args.dataset,
        feature_profile=args.feature_profile,
        seed=args.seed,
        only=args.only,
        skip=args.skip,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
