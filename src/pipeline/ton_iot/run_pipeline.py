"""Thin ToN-IoT wrapper around the shared pipeline runner.

Primary parity runs now default to the ``structural10`` feature profile, which
matches UNSW-NB15's ten centrality node features. Use
``--feature-profile enhanced12`` only for the later ToN-specific feature
ablation.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.pipeline.common.run_pipeline import main, run_all

__all__ = ["main", "run_all"]


if __name__ == "__main__":
    main()
