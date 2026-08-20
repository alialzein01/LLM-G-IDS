"""Thin UNSW-NB15 wrapper around the shared leak-free pipeline runner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.pipeline.common.run_pipeline import DEFAULT_SEED, STAGE_ORDER
from src.pipeline.common.run_pipeline import run_all as _shared_run_all


def run_all(**kwargs):
    kwargs.setdefault("dataset", "unsw_nb15")
    kwargs.setdefault("feature_profile", "structural10")
    return _shared_run_all(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="unsw_nb15", choices=["unsw_nb15"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--only", nargs="+", choices=STAGE_ORDER)
    parser.add_argument("--skip", nargs="+", choices=STAGE_ORDER)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_all(
        dataset=args.dataset,
        seed=args.seed,
        only=args.only,
        skip=args.skip,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
