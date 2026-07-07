"""Step 4 training driver and shared helpers.

Right now this file only holds the NL loader used by Phase 4.2's
dashboard. The outer training loop for the feedback classifier lands
here in Phase 4.5.
"""

from __future__ import annotations

from pathlib import Path

from src.pipeline.common.datasets import get_dataset_config


def load_nl_lines(dataset: str) -> list[str]:
    """Load `kg_triples_nl.txt` for `dataset`, one line per edge.

    The file is row-aligned to `data.edge_index` (Step 2 writes it in
    the same order as `aggregated_edges.csv`, which Step 1 built by
    grouping the raw flow CSV in a stable order).
    """
    config = get_dataset_config(dataset)
    path = Path(config.kg_nl_path)
    if not path.exists():
        raise FileNotFoundError(
            f"KG NL file not found at {path}. Re-run Step 2 for {dataset}."
        )
    lines = path.read_text().splitlines()
    return [line for line in lines if line.strip()]
