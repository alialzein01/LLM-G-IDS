from __future__ import annotations

import glob
import json
from pathlib import Path

import networkx as nx
import pandas as pd
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community.quality import modularity

RAW_COLS = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes",
    "sttl", "dttl", "sloss", "dloss", "service", "Sload", "Dload", "Spkts", "Dpkts",
    "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz", "trans_depth", "res_bdy_len",
    "Sjit", "Djit", "Stime", "Ltime", "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat",
    "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd",
    "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "Label",
]

# Standard IANA protocol numbers for known protocol strings
_PROTO_MAP: dict[str, int] = {
    "tcp": 6, "udp": 17, "icmp": 1, "icmp6": 58, "igmp": 2, "ospf": 89,
    "sctp": 132, "gre": 47, "esp": 50, "ah": 51, "pim": 103, "vrrp": 112,
    "ipv6": 41, "ipv6-icmp": 58, "arp": 0, "rarp": 0,
}

LABEL_MAPPING: dict[str, int] = {
    "Normal": 0,
    "Analysis": 1,
    "Backdoors": 2,
    "DoS": 3,
    "Exploits": 4,
    "Fuzzers": 5,
    "Generic": 6,
    "Reconnaissance": 7,
    "Shellcode": 8,
    "Worms": 9,
}

LABEL_NAMES: list[str] = [
    "Normal", "Analysis", "Backdoors", "DoS", "Exploits",
    "Fuzzers", "Generic", "Reconnaissance", "Shellcode", "Worms",
]

_CENTRALITY_MEASURES = [
    "betweenness", "pagerank", "degree", "closeness", "eigenvector",
    "k_core", "k_truss", "global_betweenness", "global_pagerank", "modularity_vitality",
]


def _load_raw(raw_dir: str) -> pd.DataFrame:
    pattern = str(Path(raw_dir) / "UNSW-NB15_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No UNSW-NB15_*.csv files found in {raw_dir}")
    dfs = [
        pd.read_csv(f, header=None, names=RAW_COLS, encoding="latin-1", low_memory=False)
        for f in files
    ]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(files)} raw files → {df.shape[0]:,} rows")
    return df


_IP_RE = r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df["srcip"] = df["srcip"].str.lstrip("﻿").str.strip()
    df["dstip"] = df["dstip"].str.strip()
    df["attack_cat"] = (
        df["attack_cat"].fillna("Normal").astype(str).str.strip()
        .replace({"Backdoor": "Backdoors"})  # normalise across files
    )
    # Drop summary/metadata rows embedded at the end of some files
    valid_ips = df["srcip"].str.match(_IP_RE, na=False) & df["dstip"].str.match(_IP_RE, na=False)
    dropped = (~valid_ips).sum()
    if dropped:
        print(f"  Dropped {dropped} non-IP rows (embedded summary data)")
    df = df[valid_ips]
    return df.reset_index(drop=True)


def _encode_protocol(series: pd.Series) -> pd.Series:
    return series.str.lower().map(lambda p: _PROTO_MAP.get(p, 255)).astype(int)


def _parse_port(series: pd.Series) -> pd.Series:
    """Convert port values that may be hex strings (e.g. '0xcc09') to integers."""
    def _parse(v):
        try:
            s = str(v).strip()
            return int(s, 16) if s.startswith("0x") else int(float(s))
        except (ValueError, TypeError):
            return 0
    return series.map(_parse).astype(int)


def _compute_centrality(df: pd.DataFrame) -> dict[str, dict]:
    G = nx.DiGraph()
    G.add_edges_from(zip(df["srcip"], df["dstip"]))
    G.remove_edges_from(nx.selfloop_edges(G))
    G_u = G.to_undirected()
    G_u.remove_edges_from(nx.selfloop_edges(G_u))
    print(f"  IP graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges (self-loops removed)")

    betweenness = nx.betweenness_centrality(G, normalized=True)
    pagerank = nx.pagerank(G, max_iter=1000)
    degree = nx.in_degree_centrality(G)
    closeness = nx.closeness_centrality(G)
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        eigenvector = dict.fromkeys(G.nodes(), 0.0)

    k_core = nx.core_number(G_u)

    try:
        edge_truss = nx.truss_decomposition(G_u)
        k_truss: dict = {}
        for node in G.nodes():
            vals = [edge_truss.get((u, v), edge_truss.get((v, u), 2))
                    for u, v in G_u.edges(node)]
            k_truss[node] = max(vals) if vals else 2
    except Exception:
        k_truss = dict.fromkeys(G.nodes(), 2)

    try:
        communities = list(greedy_modularity_communities(G_u))
        base_mod = modularity(G_u, communities)
        mod_vitality: dict = {}
        for node in G.nodes():
            G_temp = G_u.copy()
            G_temp.remove_node(node)
            if G_temp.number_of_nodes() < 2:
                mod_vitality[node] = 0.0
                continue
            try:
                comms_temp = list(greedy_modularity_communities(G_temp))
                mod_vitality[node] = base_mod - modularity(G_temp, comms_temp)
            except Exception:
                mod_vitality[node] = 0.0
    except Exception:
        mod_vitality = dict.fromkeys(G.nodes(), 0.0)

    return {
        "betweenness": betweenness,
        "pagerank": pagerank,
        "degree": degree,
        "closeness": closeness,
        "eigenvector": eigenvector,
        "k_core": k_core,
        "k_truss": k_truss,
        "global_betweenness": betweenness,
        "global_pagerank": pagerank,
        "modularity_vitality": mod_vitality,
    }


STEP0_SUMMARY_NAME = "step0_summary.json"
STEP0_SAMPLE_NAME = "step0_sample.csv"
STEP0_SAMPLE_ROWS = 200


def write_step0_sidecar(df_out: pd.DataFrame, csv_path: str | Path) -> None:
    """Record shape, classes and a short preview beside the normalized CSV.

    The normalized CSV is ~1 GB. The sidecar lets a consumer describe Step 0's
    output without reading it, and lets a distribution omit it entirely.
    """
    directory = Path(csv_path).parent
    summary = {
        "csv_name": Path(csv_path).name,
        "rows": int(len(df_out)),
        "columns": list(df_out.columns),
        "attack_categories": sorted(df_out["Attack"].unique().tolist()),
        "attack_distribution": {
            str(name): int(count)
            for name, count in df_out["Attack"].value_counts().items()
        },
    }
    with open(directory / STEP0_SUMMARY_NAME, "w") as f:
        json.dump(summary, f, indent=2)
    df_out.head(STEP0_SAMPLE_ROWS).to_csv(directory / STEP0_SAMPLE_NAME, index=False)


def load_step0_sidecar(csv_path: str | Path) -> tuple[dict, pd.DataFrame] | None:
    """Read the sidecar written by `write_step0_sidecar`, or None if absent."""
    directory = Path(csv_path).parent
    summary_path = directory / STEP0_SUMMARY_NAME
    sample_path = directory / STEP0_SAMPLE_NAME
    if not (summary_path.exists() and sample_path.exists()):
        return None
    with open(summary_path) as f:
        summary = json.load(f)
    return summary, pd.read_csv(sample_path)


def run_preprocess(raw_dir: str, output_path: str) -> pd.DataFrame:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = _load_raw(raw_dir)
    df = _clean(df)

    print("Computing centrality measures on IP graph...")
    centrality = _compute_centrality(df)

    print("Attaching centrality to flows...")
    for measure, values in centrality.items():
        df[f"src_{measure}"] = df["srcip"].map(values).fillna(0.0)
        df[f"dst_{measure}"] = df["dstip"].map(values).fillna(0.0)

    df["PROTOCOL"] = _encode_protocol(df["proto"])
    df["FLOW_DURATION_MILLISECONDS"] = (
        df["dur"].fillna(0).clip(lower=0) * 1000
    ).round().astype(int)
    df["dsport"] = _parse_port(df["dsport"])
    df["sbytes"] = df["sbytes"].fillna(0).astype(int)

    df = df.rename(columns={
        "srcip": "IPV4_SRC_ADDR",
        "dstip": "IPV4_DST_ADDR",
        "attack_cat": "Attack",
        "sbytes": "IN_BYTES",
        "dsport": "L4_DST_PORT",
    })

    unknown = set(df["Attack"].unique()) - set(LABEL_MAPPING)
    if unknown:
        raise ValueError(f"Unknown attack labels in data: {unknown}")

    from src.pipeline.step1.graph_construction import SRC_CENTRALITY_COLUMNS, DST_CENTRALITY_COLUMNS
    output_cols = [
        "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Attack",
        "IN_BYTES", "FLOW_DURATION_MILLISECONDS", "PROTOCOL", "L4_DST_PORT",
        *SRC_CENTRALITY_COLUMNS,
        *DST_CENTRALITY_COLUMNS,
    ]
    df_out = df[output_cols]

    df_out.to_csv(out, index=False)
    write_step0_sidecar(df_out, out)
    print(f"Saved {len(df_out):,} rows → {out}")
    print("Attack distribution:")
    print(df_out["Attack"].value_counts().to_string())
    return df_out


if __name__ == "__main__":
    run_preprocess(
        "data/unsw_nb15/raw",
        "data/unsw_nb15/processed/step0/NF-UNSW-NB15-normalized.csv",
    )
