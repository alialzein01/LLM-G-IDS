from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.pipeline.common.datasets import DATASETS, get_dataset_config
from src.pipeline.common.reports import check, finalize_report
from src.pipeline.step1.graph_construction import ATTACK_COL, DST_COL, SRC_COL
from src.pipeline.step2.knowledge_graph import DISCRETIZE_COLS, SEMANTIC_RELATIONS


LEVELS = {"low", "medium", "high"}


def main(dataset: str = "ton_iot") -> dict:
    config = get_dataset_config(dataset)
    agg = pd.read_csv(config.aggregated_edges_path)
    kg = pd.read_csv(config.kg_csv_path)
    sentences = Path(config.kg_nl_path).read_text().strip().splitlines()
    agg[ATTACK_COL] = agg[ATTACK_COL].str.strip()

    expected_relations = agg[ATTACK_COL].map(SEMANTIC_RELATIONS)
    relation_missing = sorted(agg.loc[expected_relations.isna(), ATTACK_COL].unique())
    leakage_terms = {
        name.lower()
        for name in config.label_names
        if name.lower() not in {"benign", "normal"}
    }
    leaks = [
        {"row": i, "term": term}
        for i, sentence in enumerate(sentences)
        for term in leakage_terms
        if term in sentence.lower()
    ]

    level_cols = [f"{col}_level" for col in DISCRETIZE_COLS]
    checks = [
        check("kg csv exists", Path(config.kg_csv_path).exists(), config.kg_csv_path),
        check("kg natural-language file exists", Path(config.kg_nl_path).exists(), config.kg_nl_path),
        check("kg row count equals aggregated edges", len(kg) == len(agg), f"kg={len(kg)}, agg={len(agg)}"),
        check("nl sentence count equals aggregated edges", len(sentences) == len(agg), f"nl={len(sentences)}, agg={len(agg)}"),
        check("source IP row alignment", bool((agg[SRC_COL].values == kg[SRC_COL].values).all())),
        check("destination IP row alignment", bool((agg[DST_COL].values == kg[DST_COL].values).all())),
        check("semantic relation mapping coverage", not relation_missing, str(relation_missing)),
        check(
            "semantic relation names match mapping",
            bool((kg["relation_name"].values == expected_relations.values).all()) if not relation_missing else False,
        ),
        check(
            "all discretization columns are present",
            all(col in kg.columns for col in level_cols),
            ", ".join(level_cols),
        ),
        check(
            "discretization values are low/medium/high",
            all(set(kg[col].dropna().unique()).issubset(LEVELS) for col in level_cols),
        ),
        check("protocol values present", bool(kg["protocol_name"].notna().all())),
        check("port values present", bool(kg["most_common_port"].notna().all())),
        check("natural-language triples are label-free", len(leaks) == 0, f"violations={len(leaks)}"),
    ]

    metrics = {
        "triple_count": len(kg),
        "relation_distribution": kg["relation_name"].value_counts().sort_index().astype(int).to_dict(),
        "level_distribution": {
            col: kg[col].value_counts().sort_index().astype(int).to_dict()
            for col in level_cols
            if col in kg.columns
        },
        "leakage_violations": leaks[:20],
    }
    artifacts = {
        "kg_csv": config.kg_csv_path,
        "kg_nl": config.kg_nl_path,
        "kg_visualization": str(Path(config.kg_csv_path).with_name("kg_graph.html")),
    }
    return finalize_report(
        config, "phase2", f"Phase 2 Validation - {config.display_name}", checks, artifacts, metrics
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Phase 2 KG artifacts.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="ton_iot")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    report = main(args.dataset)
    print(f"Phase 2 {args.dataset}: {report['status']}")
