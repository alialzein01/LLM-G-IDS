from __future__ import annotations

from pathlib import Path

import pandas as pd


CSV_PATH = "data/ton_iot/processed/step1/aggregated_edges.csv"
OUTPUT_DIR = "data/ton_iot/processed/step2"

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

# Attributes to discretize; each is binned per attack type independently
DISCRETIZE_COLS = ["avg_bytes", "flow_count", "avg_duration"]

SEMANTIC_RELATIONS = {
    "Benign": "communicated with",
    "backdoor": "opened backdoor connection to",
    "ddos": "launched ddos against",
    "dos": "launched denial of service against",
    "injection": "attempted injection against",
    "mitm": "intercepted traffic to",
    "password": "attempted password attack against",
    "ransomware": "delivered ransomware traffic to",
    "scanning": "initiated scan against",
    "xss": "attempted cross site scripting against",
    "Normal": "communicated with",
    "Analysis": "performed analysis traffic against",
    "Backdoors": "opened backdoor connection to",
    "DoS": "launched denial of service against",
    "Exploits": "attempted exploit against",
    "Fuzzers": "sent fuzzing traffic to",
    "Generic": "launched generic attack against",
    "Reconnaissance": "initiated reconnaissance against",
    "Shellcode": "delivered shellcode traffic to",
    "Worms": "propagated worm traffic to",
}


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
        if group[col].nunique(dropna=False) <= 1:
            result.loc[group.index] = "medium"
            continue

        p33 = group[col].quantile(1 / 3)
        p66 = group[col].quantile(2 / 3)
        if p33 >= p66:
            ranks = group[col].rank(method="first", pct=True)
            result.loc[group.index] = pd.cut(
                ranks,
                bins=[0.0, 1 / 3, 2 / 3, 1.0],
                labels=["low", "medium", "high"],
                include_lowest=True,
            ).astype(str)
            continue

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
    df["relation_name"] = df["Attack"].map(SEMANTIC_RELATIONS)
    if df["relation_name"].isna().any():
        missing = sorted(df.loc[df["relation_name"].isna(), "Attack"].unique())
        raise ValueError(f"Missing semantic relation mapping for attacks: {missing}")
    return df


def _frequency_phrase(level: str) -> str:
    return {
        "low": "sparse connection frequency (flow-count level low)",
        "medium": "steady connection frequency (flow-count level medium)",
        "high": "unusually high connection frequency (flow-count level high)",
    }[level]


def _bytes_phrase(level: str, protocol: str) -> str:
    if level == "high" and protocol == "TCP":
        return "transferring large TCP payloads (average-byte level high)"
    return {
        "low": "moving small per-flow byte volumes (average-byte level low)",
        "medium": "moving moderate per-flow byte volumes (average-byte level medium)",
        "high": "moving large per-flow byte volumes (average-byte level high)",
    }[level]


def _duration_phrase(level: str) -> str:
    return {
        "low": "in brief, short-lived connections (duration level low)",
        "medium": "across medium-duration connections (duration level medium)",
        "high": "across long-lived connections (duration level high)",
    }[level]


def _protocol_phrase(protocol: str) -> str:
    return {
        "ICMP": "using ICMP messaging",
        "TCP": "over TCP sessions",
        "UDP": "over UDP datagrams",
    }.get(protocol, f"over protocol {protocol}")


def _port_phrase(port: int) -> str:
    if port in (80, 443, 8080, 8443):
        return f"to a web service port {port}"
    if port in (22, 23, 3389):
        return f"to a remote-access port {port}"
    if port < 1024:
        return f"to a well-known system port {port}"
    if port >= 49152:
        return f"to a high-numbered ephemeral port {port}"
    return f"to a registered service port {port}"


def build_nl_triples(df: pd.DataFrame) -> list[str]:
    """Label-free NL sentences: no attack labels or relation verbs."""
    sentences = []
    for _, row in df.iterrows():
        protocol = row["protocol_name"]
        port = int(row["most_common_port"])
        s = (
            f"Observed traffic from source {row['IPV4_SRC_ADDR']} "
            f"to destination {row['IPV4_DST_ADDR']} "
            f"with {_frequency_phrase(row['flow_count_level'])}, "
            f"{_bytes_phrase(row['avg_bytes_level'], protocol)}, "
            f"and {_duration_phrase(row['avg_duration_level'])}, "
            f"{_protocol_phrase(protocol)}, "
            f"{_port_phrase(port)}."
        )
        sentences.append(s)
    return sentences


def assert_label_free(sentences: list[str], attack_types: set[str]) -> None:
    """Raise if any attack type label appears in NL text."""
    leakage_terms = {t.lower() for t in attack_types}
    for i, sentence in enumerate(sentences):
        lower = sentence.lower()
        for term in leakage_terms:
            if term in lower:
                raise ValueError(
                    f"Label leakage in sentence {i}: found '{term}' in: {sentence[:120]}..."
                )


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

    # --- Save label-free natural language triples ---
    sentences = build_nl_triples(df)
    assert_label_free(sentences, set(df["Attack"].unique()))
    (out / "kg_triples_nl.txt").write_text("\n".join(sentences) + "\n")
    print(f"Saved kg_triples_nl.txt ({len(sentences)} label-free sentences)")

    return df


if __name__ == "__main__":
    run_step2()
