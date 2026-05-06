# LLM-G-IDS

A Graph Neural Network + LLM framework for network intrusion detection on IoT traffic.

## Dataset

**NF-ToN-IoT** — 532 MB of real IoT network flow records (CSV). Place it at:

```text
data/raw/NF-ToN-IoT.csv
```

10 traffic classes: Benign, backdoor, ddos, dos, injection, mitm, password, ransomware, scanning, xss.

## Setup

```bash
pip install -r requirements.txt
```

The project uses Python 3.14. All scripts are run from the repository root.

## Pipeline

### Step 1 — Graph Construction

Collapses raw flow records into a directed communication graph.

```python
from src.graph_construction import run_step1
data, edge_df = run_step1("data/raw/NF-ToN-IoT.csv", "data/processed/step1")
```

**Output:**

- `data/processed/step1/pyg_data.pt` — PyTorch Geometric `Data` object
- `data/processed/step1/aggregated_edges.csv` — 2,127 aggregated edges

**Graph:** 1,501 IP nodes × 2,127 directed edges. Node features = 10 centrality measures. Edge attributes = flow count, total bytes, avg duration, protocol, port. Edge labels = attack class (0–9).

Visualize:

```bash
python -m src.visualize_graph
# opens: data/processed/step1/graph.html
```

---

### Step 2 — Knowledge Graph Construction

Transforms the aggregated edges into a semantically enriched Knowledge Graph for LLM reasoning.

```python
from src.knowledge_graph import run_step2
df = run_step2("data/processed/step1/aggregated_edges.csv", "data/processed/step2")
```

**Output:**

- `data/processed/step2/kg_triples.csv` — structured triples with relation names and discretized attributes
- `data/processed/step2/kg_triples_nl.txt` — one natural language sentence per triple

**Example triple:**

```text
192.168.1.1 initiated_scan on 10.0.0.2, frequency: high, bytes: low, duration: low, protocol: TCP, port: 22
```

Numerical attributes (flow count, avg bytes, avg duration) are binned into `low / medium / high` **per attack type independently**, so scales are relative to each attack's natural range. Attack labels are replaced with descriptive relation verbs (e.g. `ddos → launched_ddos`).

Visualize:

```bash
python -m src.visualize_kg
# opens: data/processed/step2/kg_graph.html
```

## Project Structure

```text
src/
  graph_construction.py   # Step 1: raw CSV → PyG graph
  knowledge_graph.py      # Step 2: aggregated edges → KG triples
  visualize_graph.py      # Interactive HTML visualization for Step 1
  visualize_kg.py         # Interactive HTML visualization for Step 2
data/
  raw/                    # Place NF-ToN-IoT.csv here (not in git)
  processed/
    step1/                # PyG data object + aggregated edges
    step2/                # KG triples (CSV + natural language)
requirements/
  todo.pdf                # Project task specification
```
