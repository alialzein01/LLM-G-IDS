"""Build progressively simplified NL variants of the KG edge sentences.

Motivation (instructor request, 2026-08-18): the canonical `kg_triples_nl.txt`
verbalizes each edge with rich prose — absolute magnitude cues ("thousands of
flows", "several megabytes", "5-20 KB per flow") layered on top of the coarse
global levels. The question is what happens to the LLM / AGAF / feedback rungs
when that linguistic assistance is removed.

These variants are a deliberate handicap, not a leakage fix: the canonical text
is already label-free (`knowledge_graph.assert_label_free`), and the LLM already
sees strictly less than the GNN (no centrality, no structure). Levels:

  L0 rich           canonical file, unchanged (prose + magnitudes + levels)
  L1 levels_only    prose + coarse levels, absolute magnitude cues removed
  L2 bare_numbers   raw feature values, no prose at all
  L3 structure_only endpoints + protocol/port; traffic statistics removed

Every variant is row-aligned to `aggregated_edges.csv`, hence to
`data.edge_index`, hence to the canonical embeddings.

Run:
    python -m src.pipeline.step2.build_simplified_nl --dataset unsw_nb15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.pipeline.common.datasets import DATASETS, get_dataset_config
from src.pipeline.step2.knowledge_graph import (
    TRIPLE_COLS,
    assert_label_free,
    discretize,
)

LEVELS = ("levels_only", "bare_numbers", "structure_only")


def _protocol_phrase(protocol: str) -> str:
    if protocol == "TCP":
        return "over TCP sessions"
    if protocol == "UDP":
        return "over UDP datagrams"
    if protocol == "ICMP":
        return "over ICMP control messages"
    return f"over protocol {protocol}"


def _port_phrase(port: int) -> str:
    if port == 80 or port == 443:
        return "to a web service port"
    if port < 1024:
        return "to a well-known system port"
    return "to a high-numbered ephemeral port"


def _levels_only(row: pd.Series) -> str:
    """Prose retained, absolute magnitudes stripped — only the coarse levels."""
    return (
        f"Observed traffic from source {row['IPV4_SRC_ADDR']} "
        f"to destination {row['IPV4_DST_ADDR']} "
        f"with flow-count level {row['flow_count_level']}, "
        f"average-byte level {row['avg_bytes_level']}, "
        f"duration level {row['avg_duration_level']}, "
        f"{_protocol_phrase(row['protocol_name'])}, "
        f"{_port_phrase(int(row['most_common_port']))}."
    )


def _bare_numbers(row: pd.Series) -> str:
    """No prose: the five edge features as raw values."""
    return (
        f"src {row['IPV4_SRC_ADDR']} dst {row['IPV4_DST_ADDR']} "
        f"flows {int(row['flow_count'])} "
        f"bytes {int(row['total_bytes'])} "
        f"duration {float(row['avg_duration']):.2f} "
        f"protocol {row['most_common_protocol']} "
        f"port {int(row['most_common_port'])}"
    )


def _structure_only(row: pd.Series) -> str:
    """Traffic statistics removed entirely — endpoints and protocol/port only."""
    return (
        f"src {row['IPV4_SRC_ADDR']} dst {row['IPV4_DST_ADDR']} "
        f"protocol {row['most_common_protocol']} "
        f"port {int(row['most_common_port'])}"
    )


BUILDERS = {
    "levels_only": _levels_only,
    "bare_numbers": _bare_numbers,
    "structure_only": _structure_only,
}


def build_simplified_nl(dataset: str, output_dir: str | Path | None = None) -> dict[str, Path]:
    config = get_dataset_config(dataset)
    edges_csv = Path(f"data/{dataset}/processed/step1/aggregated_edges.csv")
    out = Path(output_dir) if output_dir else Path(f"data/{dataset}/processed/step2_simplified")
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(edges_csv, usecols=TRIPLE_COLS)
    df["Attack"] = df["Attack"].str.strip()
    attack_types = set(df["Attack"].unique())
    df = discretize(df)

    canonical = Path(config.kg_nl_path).read_text().strip().splitlines()
    if len(canonical) != len(df):
        raise RuntimeError(
            f"Canonical NL has {len(canonical)} lines but {edges_csv} has "
            f"{len(df)} edges — row alignment broken."
        )

    written: dict[str, Path] = {}
    for level, builder in BUILDERS.items():
        sentences = [builder(row) for _, row in df.iterrows()]
        assert_label_free(sentences, attack_types)
        if len(sentences) != len(df):
            raise RuntimeError(f"{level}: produced {len(sentences)} of {len(df)} lines.")
        path = out / f"kg_triples_nl_{level}.txt"
        path.write_text("\n".join(sentences) + "\n")
        written[level] = path
        chars = sum(len(s) for s in sentences) / len(sentences)
        print(f"{level:16s} -> {path}  ({len(sentences)} lines, {chars:.0f} chars/line)")
        print(f"                    e.g. {sentences[320]}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="unsw_nb15", choices=sorted(DATASETS))
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    build_simplified_nl(args.dataset, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
