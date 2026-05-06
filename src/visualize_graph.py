from __future__ import annotations

from pathlib import Path

import pandas as pd
from pyvis.network import Network

CSV_PATH = "data/processed/step1/aggregated_edges.csv"
OUTPUT_HTML = "data/processed/step1/graph.html"

# One distinct color per attack type (edge color = attack on that flow)
ATTACK_COLORS = {
    "Benign":     "#4CAF50",  # green
    "backdoor":   "#9C27B0",  # purple
    "ddos":       "#F44336",  # red
    "dos":        "#FF5722",  # deep orange
    "injection":  "#FF9800",  # orange
    "mitm":       "#00BCD4",  # cyan
    "password":   "#3F51B5",  # indigo
    "ransomware": "#E91E63",  # pink
    "scanning":   "#FFC107",  # amber
    "xss":        "#009688",  # teal
}

DEFAULT_EDGE_COLOR = "#888888"


def _node_size(degree: int) -> int:
    """Scale node size by degree, clamped to a readable range."""
    return max(8, min(40, 8 + degree * 2))


def build_graph(df: pd.DataFrame) -> Network:
    net = Network(
        height="100vh",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="#ffffff",
        directed=True,
        notebook=False,
    )
    net.barnes_hut(
        gravity=-8000,
        central_gravity=0.3,
        spring_length=150,
        spring_strength=0.001,
        damping=0.09,
    )

    # Compute degree for node sizing
    src_deg = df["IPV4_SRC_ADDR"].value_counts()
    dst_deg = df["IPV4_DST_ADDR"].value_counts()
    degree = src_deg.add(dst_deg, fill_value=0).astype(int)

    # Add nodes
    unique_ips = pd.concat([df["IPV4_SRC_ADDR"], df["IPV4_DST_ADDR"]]).unique()
    for ip in unique_ips:
        deg = int(degree.get(ip, 1))
        net.add_node(
            ip,
            label=ip,
            title=f"{ip}\ndegree: {deg}",
            size=_node_size(deg),
            color="#e0e0e0",
            font={"size": 10},
        )

    # Add edges
    for _, row in df.iterrows():
        attack = row["Attack"]
        color = ATTACK_COLORS.get(attack, DEFAULT_EDGE_COLOR)
        title = (
            f"Src: {row['IPV4_SRC_ADDR']}\n"
            f"Dst: {row['IPV4_DST_ADDR']}\n"
            f"Attack: {attack}\n"
            f"Flows: {int(row['flow_count'])}\n"
            f"Bytes: {int(row['total_bytes'])}\n"
            f"Avg duration: {row['avg_duration']:.1f} ms"
        )
        net.add_edge(
            row["IPV4_SRC_ADDR"],
            row["IPV4_DST_ADDR"],
            color=color,
            title=title,
            width=max(1, min(6, row["flow_count"] / df["flow_count"].max() * 6)),
            arrows="to",
        )

    return net


def _freeze_after_stabilization(html_path: Path) -> None:
    html = html_path.read_text()
    snippet = (
        '\n        network.on("stabilizationIterationsDone", function () {'
        '\n            network.setOptions({ physics: { enabled: false } });'
        '\n        });'
    )
    html = html.replace(
        "var network = new vis.Network(container, data, options);",
        "var network = new vis.Network(container, data, options);" + snippet,
    )
    html_path.write_text(html)


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df["Attack"] = df["Attack"].str.strip()

    print(f"Loaded {len(df)} edges, {pd.concat([df['IPV4_SRC_ADDR'], df['IPV4_DST_ADDR']]).nunique()} unique nodes")

    net = build_graph(df)

    output_path = Path(OUTPUT_HTML)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    net.save_graph(str(output_path))
    _freeze_after_stabilization(output_path)
    print(f"Graph saved → {output_path.resolve()}")
    print("Open the HTML file in your browser to explore the graph.")
    print("\nLegend:")
    for attack, color in ATTACK_COLORS.items():
        print(f"  {color}  {attack}")


if __name__ == "__main__":
    main()
