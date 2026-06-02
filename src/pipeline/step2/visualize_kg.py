from __future__ import annotations

from pathlib import Path

import pandas as pd
from pyvis.network import Network


CSV_PATH = "data/ton_iot/processed/step2/kg_triples.csv"
NL_PATH = "data/ton_iot/processed/step2/kg_triples_nl.txt"
OUTPUT_HTML = "data/ton_iot/processed/step2/kg_graph.html"

RELATION_COLORS = {
    "communicated with": "#4CAF50",
    "opened backdoor connection to": "#9C27B0",
    "launched ddos against": "#F44336",
    "launched denial of service against": "#FF5722",
    "attempted injection against": "#FF9800",
    "intercepted traffic to": "#00BCD4",
    "attempted password attack against": "#3F51B5",
    "delivered ransomware traffic to": "#E91E63",
    "initiated scan against": "#FFC107",
    "attempted cross site scripting against": "#009688",
    "performed analysis traffic against": "#607D8B",
    "attempted exploit against": "#795548",
    "sent fuzzing traffic to": "#CDDC39",
    "launched generic attack against": "#673AB7",
    "initiated reconnaissance against": "#FFC107",
    "delivered shellcode traffic to": "#E91E63",
    "propagated worm traffic to": "#F44336",
}

LEVEL_WIDTH = {"low": 1, "medium": 3, "high": 5}


def _node_size(degree: int) -> int:
    return max(8, min(40, 8 + degree * 2))


def build_kg_graph(df: pd.DataFrame, nl_sentences: list[str]) -> Network:
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

    src_deg = df["IPV4_SRC_ADDR"].value_counts()
    dst_deg = df["IPV4_DST_ADDR"].value_counts()
    degree = src_deg.add(dst_deg, fill_value=0).astype(int)

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

    for (_, row), nl in zip(df.iterrows(), nl_sentences):
        relation = row["relation_name"]
        color = RELATION_COLORS.get(relation, "#888888")
        width = LEVEL_WIDTH.get(row["flow_count_level"], 1)
        net.add_edge(
            row["IPV4_SRC_ADDR"],
            row["IPV4_DST_ADDR"],
            color=color,
            title=nl,
            width=width,
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
    nl_sentences = Path(NL_PATH).read_text().splitlines()

    assert len(df) == len(nl_sentences), "Row count mismatch between CSV and NL file"

    unique_nodes = pd.concat([df["IPV4_SRC_ADDR"], df["IPV4_DST_ADDR"]]).nunique()
    print(f"Loaded {len(df)} triples, {unique_nodes} unique nodes")

    net = build_kg_graph(df, nl_sentences)

    output_path = Path(OUTPUT_HTML)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(output_path))
    _freeze_after_stabilization(output_path)

    print(f"KG graph saved → {output_path.resolve()}")
    print("\nLegend (edge color = relation):")
    for relation, color in RELATION_COLORS.items():
        print(f"  {color}  {relation}")


if __name__ == "__main__":
    main()
