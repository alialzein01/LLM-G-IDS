from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    display_name: str
    phase1_input_path: str
    graph_path: str
    aggregated_edges_path: str
    kg_csv_path: str
    kg_nl_path: str
    splits_path: str
    gnn_output_dir: str
    gnn_embedding_path: str
    llm_embedding_path: str
    fusion_output_dir: str
    baseline_output_dir: str
    label_names: tuple[str, ...]
    num_classes: int = 10
    gnn_dim: int = 64
    llm_dim: int = 768
    # Classes scored in the headline macro-F1. Ultra-rare classes (< 5 samples,
    # unlearnable under 5-fold CV) are excluded from the metric so the ladder is
    # not drowned by classes no method can learn. They remain in the graph for
    # message passing; they are simply not counted.
    eval_classes: tuple[int, ...] = tuple(range(10))
    dropped_classes: tuple[int, ...] = ()


DATASETS: dict[str, DatasetConfig] = {
    "ton_iot": DatasetConfig(
        key="ton_iot",
        display_name="NF-ToN-IoT",
        phase1_input_path="data/ton_iot/raw/NF-ToN-IoT.csv",
        graph_path="data/ton_iot/processed/step1/pyg_data.pt",
        aggregated_edges_path="data/ton_iot/processed/step1/aggregated_edges.csv",
        kg_csv_path="data/ton_iot/processed/step2/kg_triples.csv",
        kg_nl_path="data/ton_iot/processed/step2/kg_triples_nl.txt",
        splits_path="data/ton_iot/processed/splits/folds.pt",
        gnn_output_dir="data/ton_iot/processed/step3_gnn",
        gnn_embedding_path="data/ton_iot/processed/step3_gnn/edge_embeddings.pt",
        llm_embedding_path="data/ton_iot/processed/step3_llm/edge_embeddings.pt",
        fusion_output_dir="data/ton_iot/processed/step3_fusion",
        baseline_output_dir="data/ton_iot/processed/step3_baselines",
        label_names=(
            "Benign",
            "backdoor",
            "ddos",
            "dos",
            "injection",
            "mitm",
            "password",
            "ransomware",
            "scanning",
            "xss",
        ),
        # AGGREGATED (src,dst,attack) graph — todo.md-mandated form. dos (4 edges)
        # and ransomware (3 edges) fall below the 5-fold minimum and cannot be
        # evaluated, so they are dropped from the headline macro-F1 (kept in the
        # graph for message passing). The other 8 classes still span ~145:1
        # imbalance (xss 12 ... Benign 1746). Expanded-graph config reserved in
        # data/ton_iot/processed_expanded_reserve.tar.gz (eval_classes=range(10)).
        eval_classes=(0, 1, 2, 4, 5, 6, 8, 9),
        dropped_classes=(3, 7),
    ),
    "ton_iot_capped": DatasetConfig(
        key="ton_iot_capped",
        display_name="NF-ToN-IoT (capped)",
        phase1_input_path="data/ton_iot/raw/NF-ToN-IoT.csv",
        graph_path="data/ton_iot_capped/processed/step1/pyg_data_perflow.pt",
        aggregated_edges_path="data/ton_iot_capped/processed/step1/flow_edges.csv",
        kg_csv_path="data/ton_iot_capped/processed/step2/kg_triples.csv",
        kg_nl_path="data/ton_iot_capped/processed/step2/kg_triples_nl.txt",
        splits_path="data/ton_iot_capped/processed/splits/folds.pt",
        gnn_output_dir="data/ton_iot_capped/processed/step3_gnn",
        gnn_embedding_path="data/ton_iot_capped/processed/step3_gnn/edge_embeddings.pt",
        llm_embedding_path="data/ton_iot_capped/processed/step3_llm/edge_embeddings.pt",
        fusion_output_dir="data/ton_iot_capped/processed/step3_fusion",
        baseline_output_dir="data/ton_iot_capped/processed/step3_baselines",
        label_names=(
            "Benign",
            "backdoor",
            "ddos",
            "dos",
            "injection",
            "mitm",
            "password",
            "ransomware",
            "scanning",
            "xss",
        ),
        # SIGNATURE-GROUPED per-flow graph, capped per class. One edge per distinct
        # model-visible behaviour, carrying its occurrence count in flow_count. This
        # lifts dos (4 -> 814) and ransomware (3 -> 30) above the 5-fold minimum, so
        # all ten classes are scored here. Compare against the aggregated `ton_iot`
        # ladder on its eight eval classes for a like-for-like reading.
        eval_classes=tuple(range(10)),
        dropped_classes=(),
    ),
    "unsw_nb15": DatasetConfig(
        key="unsw_nb15",
        display_name="UNSW-NB15",
        phase1_input_path="data/unsw_nb15/processed/step0/NF-UNSW-NB15-normalized.csv",
        graph_path="data/unsw_nb15/processed/step1/pyg_data.pt",
        aggregated_edges_path="data/unsw_nb15/processed/step1/aggregated_edges.csv",
        kg_csv_path="data/unsw_nb15/processed/step2/kg_triples.csv",
        kg_nl_path="data/unsw_nb15/processed/step2/kg_triples_nl.txt",
        splits_path="data/unsw_nb15/processed/splits/folds.pt",
        gnn_output_dir="data/unsw_nb15/processed/step3_gnn",
        gnn_embedding_path="data/unsw_nb15/processed/step3_gnn/edge_embeddings.pt",
        llm_embedding_path="data/unsw_nb15/processed/step3_llm/edge_embeddings.pt",
        fusion_output_dir="data/unsw_nb15/processed/step3_fusion",
        baseline_output_dir="data/unsw_nb15/processed/step3_baselines",
        label_names=(
            "Normal",
            "Analysis",
            "Backdoors",
            "DoS",
            "Exploits",
            "Fuzzers",
            "Generic",
            "Reconnaissance",
            "Shellcode",
            "Worms",
        ),
    ),
}


def get_dataset_config(dataset: str) -> DatasetConfig:
    try:
        return DATASETS[dataset]
    except KeyError as exc:
        valid = ", ".join(sorted(DATASETS))
        raise ValueError(f"Unknown dataset '{dataset}'. Valid choices: {valid}") from exc
