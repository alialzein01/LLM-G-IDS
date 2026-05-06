from __future__ import annotations

from pathlib import Path

import pandas as pd


CSV_PATH = "data/processed/step1/aggregated_edges.csv"
OUTPUT_DIR = "data/processed/step2"

TRIPLE_COLS = [
    "IPV4_SRC_ADDR",
    "IPV4_DST_ADDR",
    "Attack",
    "flow_count",
    "total_bytes",
    "avg_duration",
    "most_common_protocol",
    "most_common_port",
]

PROTOCOL_NAMES = {1: "ICMP", 2: "IGMP", 6: "TCP", 17: "UDP"}

RELATION_NAMES = {
    "Benign":     "communicated_with",
    "backdoor":   "established_backdoor",
    "ddos":       "launched_ddos",
    "dos":        "launched_dos",
    "injection":  "performed_injection",
    "mitm":       "intercepted_traffic",
    "password":   "attempted_brute_force",
    "ransomware": "deployed_ransomware",
    "scanning":   "initiated_scan",
    "xss":        "executed_xss",
}

# Attributes to discretize; each is binned per attack type independently
DISCRETIZE_COLS = ["avg_bytes", "flow_count", "avg_duration"]


def load_triples(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=TRIPLE_COLS)
    df["Attack"] = df["Attack"].str.strip()
    print(f"Loaded {len(df)} triples")
    print(f"Attack types present: {sorted(df['Attack'].unique())}")
    return df


def _discretize_per_attack(df: pd.DataFrame, col: str) -> pd.Series:
    """Bin a numerical column into low/medium/high within each attack type."""
    result = pd.Series("", index=df.index, dtype=str)
    for _, group in df.groupby("Attack", sort=False):
        p33 = group[col].quantile(1 / 3)
        p66 = group[col].quantile(2 / 3)
        bins = [-float("inf"), p33, p66, float("inf")]
        labels = pd.cut(group[col], bins=bins, labels=["low", "medium", "high"])
        result.loc[group.index] = labels.astype(str)
    return result


def discretize(df: pd.DataFrame) -> pd.DataFrame:
    # avg_bytes is more meaningful than total_bytes for per-flow comparison
    df = df.copy()
    df["avg_bytes"] = df["total_bytes"] / df["flow_count"]

    for col in DISCRETIZE_COLS:
        df[f"{col}_level"] = _discretize_per_attack(df, col)

    df["protocol_name"] = df["most_common_protocol"].map(PROTOCOL_NAMES).fillna(
        df["most_common_protocol"].astype(str)
    )
    return df


def add_relation_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["relation_name"] = df["Attack"].map(RELATION_NAMES)
    unmapped = df.loc[df["relation_name"].isna(), "Attack"].unique()
    if len(unmapped):
        raise ValueError(f"No relation name defined for: {sorted(unmapped)}")
    return df


def build_nl_triples(df: pd.DataFrame) -> list[str]:
    sentences = []
    for _, row in df.iterrows():
        s = (
            f"{row['IPV4_SRC_ADDR']} {row['relation_name']} {row['IPV4_DST_ADDR']}, "
            f"frequency: {row['flow_count_level']}, "
            f"bytes: {row['avg_bytes_level']}, "
            f"duration: {row['avg_duration_level']}, "
            f"protocol: {row['protocol_name']}, "
            f"port: {int(row['most_common_port'])}"
        )
        sentences.append(s)
    return sentences


def run_step2(csv_path: str = CSV_PATH, output_dir: str = OUTPUT_DIR) -> pd.DataFrame:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_triples(csv_path)
    df = discretize(df)
    df = add_relation_names(df)

    # --- Save structured CSV ---
    kg_cols = [
        "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "relation_name",
        "flow_count_level", "avg_bytes_level", "avg_duration_level",
        "protocol_name", "most_common_port",
    ]
    df[kg_cols].to_csv(out / "kg_triples.csv", index=False)
    print(f"Saved kg_triples.csv ({len(df)} rows)")

    # --- Save natural language triples ---
    sentences = build_nl_triples(df)
    (out / "kg_triples_nl.txt").write_text("\n".join(sentences))
    print(f"Saved kg_triples_nl.txt ({len(sentences)} sentences)")

    return df


if __name__ == "__main__":
    run_step2()
