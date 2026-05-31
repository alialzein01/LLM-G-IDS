# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LLM-G-IDS** — a Graph Neural Network-based Intrusion Detection System for IoT network traffic. The goal is to classify network flows as benign or one of 9 attack types using graph representations of IP-to-IP communication patterns.

Dataset: `data/ton_iot/raw/NF-ToN-IoT.csv` (532 MB, not in git). Processed artifacts live in `data/ton_iot/processed/step1/`. The parallel dataset NF-UNSW-NB15 lives under `data/unsw_nb15/` (raw parquet at `data/unsw_nb15/raw/`, processed outputs will go to `data/unsw_nb15/processed/`).

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.14.3 is used. The `.venv/` directory is local and not committed.

## Running the Pipeline

There is no CLI entry point yet. Run `run_step1` directly from a script or REPL:

```python
from src.pipeline.step1 import run_step1
data, edge_df = run_step1("data/ton_iot/raw/NF-ToN-IoT.csv", "data/ton_iot/processed/step1")
```

This reads the raw CSV, aggregates flows, builds a PyTorch Geometric `Data` object, and writes two files:

- `data/ton_iot/processed/step1/pyg_data.pt` — serialized graph
- `data/ton_iot/processed/step1/aggregated_edges.csv` — aggregated edge table

Loading a saved graph:
```python
import torch
data = torch.load("data/ton_iot/processed/step1/pyg_data.pt")
```

## Architecture

### Graph Representation

The network is modeled as a **directed multigraph** where:
- **Nodes** = unique IP addresses. Node features (`data.x`, shape `[N, 10]`) are the 10 centrality measures averaged across all edges incident to that node.
- **Edges** = aggregated flows between `(src_ip, dst_ip, attack_type)` triples. Each unique combination becomes one edge. Edge attributes (`data.edge_attr`, shape `[E, 5]`): `flow_count`, `total_bytes`, `avg_duration`, `most_common_protocol`, `most_common_port`. The first 3 are Z-score normalized; the last 2 are raw categorical integers.
- **Edge labels** (`data.edge_label`, shape `[E]`): integer class 0–9 from `LABEL_MAPPING`.

This means **classification is on edges, not nodes**. Any GNN model built on top must use edge-level prediction heads.

### Key Constants (src/pipeline/step1/graph_construction.py)

| Constant | Purpose |
|---|---|
| `CENTRALITY_MEASURES` | 10 centrality features used as node features |
| `EDGE_ATTR_COLUMNS` | 5 features stored per edge |
| `LABEL_MAPPING` | Maps attack name strings → int labels 0–9 |
| `SRC_COL`, `DST_COL`, `ATTACK_COL` | CSV column names for source IP, dest IP, attack label |

### Data Flow

```
NF-ToN-IoT.csv
  └─ df.groupby([src_ip, dst_ip, attack]) → edge_df
       ├─ _build_node_index()     → node_to_idx / idx_to_node dicts
       ├─ _build_node_features()  → data.x  [N × 10]
       └─ _build_edge_tensors()   → data.edge_index, data.edge_attr, data.edge_label
```

Normalization stats (`data.edge_attr_mean`, `data.edge_attr_std`) are stored on the `Data` object so they can be reused at inference time.

### Source Layout

```text
src/
  models/              # neural network modules
  pipeline/
    common/            # shared utilities (splits, loss, oversampling)
    step1/             # graph construction from CSV
    step2/             # knowledge graph + visualization
    step3/             # GNN training, LLM encoding, diagnostics
```

Run scripts from the repo root, e.g. `python -m src.pipeline.step3.train_gnn`.
