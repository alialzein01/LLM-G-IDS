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
from src.pipeline.step1 import run_step1
data, edge_df = run_step1("data/raw/NF-ToN-IoT.csv", "data/processed/step1")
```

**Output:**

- `data/processed/step1/pyg_data.pt` — PyTorch Geometric `Data` object
- `data/processed/step1/aggregated_edges.csv` — 2,127 aggregated edges

**Graph:** 1,501 IP nodes × 2,127 directed edges. Node features = 10 centrality measures. Edge attributes = flow count, total bytes, avg duration, protocol, port. Edge labels = attack class (0–9).

Visualize:

```bash
python -m src.pipeline.step1.visualize_graph
# opens: data/processed/step1/graph.html
```

---

### Step 2 — Knowledge Graph Construction

Transforms the aggregated edges into a semantically enriched Knowledge Graph for LLM reasoning.

```python
from src.pipeline.step2 import run_step2
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
python -m src.pipeline.step2.visualize_kg
# opens: data/processed/step2/kg_graph.html
```

### Step 3 — GNN baseline

Train and verify the GAT edge classifier:

```bash
python -m src.pipeline.step3.train_gnn
python -m src.pipeline.step3.verify_gnn_baseline
```

Encode KG sentences with CySecBERT (LLM path):

```bash
python -m src.pipeline.step3.encode_kg
```

## Project Structure

```text
src/
  models/
    gnn_classifier.py       # GAT edge classifier
  pipeline/
    common/
      splits.py             # stratified k-fold splits, class weights, focal loss
    step1/
      graph_construction.py # raw CSV → PyG graph
      visualize_graph.py
    step2/
      knowledge_graph.py    # aggregated edges → KG triples
      visualize_kg.py
    step3/
      encode_kg.py          # NL triples → BERT embeddings
      train_gnn.py
      verify_gnn_baseline.py
      diagnose_imbalance.py
data/
  raw/                      # Place NF-ToN-IoT.csv here (not in git)
  processed/
    step1/                  # PyG data object + aggregated edges
    step2/                  # KG triples (CSV + natural language)
    step3_gnn/              # trained GNN artifacts
requirements/
  todo.pdf                  # Project task specification
```
