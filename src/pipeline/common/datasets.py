from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    display_name: str
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


DATASETS: dict[str, DatasetConfig] = {
    "ton_iot": DatasetConfig(
        key="ton_iot",
        display_name="NF-ToN-IoT",
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
    ),
    "unsw_nb15": DatasetConfig(
        key="unsw_nb15",
        display_name="NF-UNSW-NB15",
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
