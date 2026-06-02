# LLM-G-IDS

A Graph Neural Network + LLM framework for edge-level intrusion detection on
NetFlow-style network traffic.

The project currently supports two datasets:

- **NF-ToN-IoT** under `data/ton_iot/`
- **NF-UNSW-NB15** under `data/unsw_nb15/`

Raw data is local and not committed. Processed artifacts are written under each
dataset's `processed/` directory.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run commands from the repository root.

## Pipeline

### Step 1: Graph Construction

Step 1 aggregates raw/normalized flow rows into a PyTorch Geometric edge
classification graph.

```python
from src.pipeline.step1 import run_step1

data, edge_df = run_step1(
    "data/ton_iot/raw/NF-ToN-IoT.csv",
    "data/ton_iot/processed/step1",
)
```

Outputs:

- `data/<dataset>/processed/step1/pyg_data.pt`
- `data/<dataset>/processed/step1/aggregated_edges.csv`

Graph semantics:

- Nodes are IP addresses.
- Edges are aggregated `(src_ip, dst_ip, attack_type)` flows.
- Classification is on edges, not nodes.
- Edge labels are the attack classes.

### Step 2: Label-Free KG Text

Step 2 builds structured KG rows and natural-language sentences from the same
aggregated edges.

```python
from src.pipeline.step2 import run_step2

df = run_step2(
    "data/ton_iot/processed/step1/aggregated_edges.csv",
    "data/ton_iot/processed/step2",
)
```

Outputs:

- `data/<dataset>/processed/step2/kg_triples.csv`
- `data/<dataset>/processed/step2/kg_triples_nl.txt`

Important: `kg_triples_nl.txt` is intentionally label-free. It must not contain
attack labels or attack relation verbs such as `launched_ddos`; otherwise the
LLM path can leak the target label.

### Step 3: Training and Evaluation

Build splits:

```bash
python -m src.pipeline.common.build_splits --dataset ton_iot
python -m src.pipeline.common.build_splits --dataset unsw_nb15
```

Train the end-to-end GNN baseline:

```bash
python -m src.pipeline.step3.train_gnn --dataset ton_iot
python -m src.pipeline.step3.train_gnn --dataset unsw_nb15
```

Encode label-free KG sentences with CySecBERT:

```bash
python -m src.pipeline.step3.encode_kg
```

For UNSW-NB15, use the dataset-specific paths:

```python
from src.pipeline.step3.encode_kg import run_encode_kg

run_encode_kg(
    nl_path="data/unsw_nb15/processed/step2/kg_triples_nl.txt",
    output_dir="data/unsw_nb15/processed/step3_llm",
)
```

Verify alignment before fusion:

```bash
python -m src.pipeline.step3.verify_alignment --dataset ton_iot
python -m src.pipeline.step3.verify_alignment --dataset unsw_nb15
```

Train frozen-embedding unimodal baselines:

```bash
python -m src.pipeline.step3.train_unimodal_baselines --dataset ton_iot
python -m src.pipeline.step3.train_unimodal_baselines --dataset unsw_nb15
```

Train AGAF fusion:

```bash
python -m src.pipeline.step3.train_fusion --dataset ton_iot
python -m src.pipeline.step3.train_fusion --dataset unsw_nb15
```

### AGAF Fusion

AGAF, Adaptive Gated Attention Fusion, is the Phase 1 fusion module. It combines
edge-level structural and semantic representations:

```text
h_e = structural edge embedding from the GNN path
s_e = semantic edge embedding from the label-free KG sentence encoder
```

Both embeddings are projected to the same dimension. AGAF then computes a
feature-wise gate from `[h_e; s_e; |h_e - s_e|; h_e * s_e]`, fuses the two
modalities dimension by dimension, and applies feature-wise attention before
classification. The gate summaries show how the model used structural versus
semantic evidence, but those summaries are diagnostic rather than causal proof.

Compare results:

```bash
python -m src.pipeline.step3.compare_results --dataset ton_iot
python -m src.pipeline.step3.compare_results --dataset unsw_nb15
```

### Phase Validation and Dashboard

Each phase has a dataset-aware validator that writes JSON, Markdown, and approval
manifest files under `data/<dataset>/processed/reports/phase<N>/`.

```bash
python -m src.pipeline.phase1_validate --dataset ton_iot
python -m src.pipeline.phase1_validate --dataset unsw_nb15
python -m src.pipeline.phase2_validate --dataset ton_iot
python -m src.pipeline.phase2_validate --dataset unsw_nb15
python -m src.pipeline.phase3_validate --dataset ton_iot
python -m src.pipeline.phase3_validate --dataset unsw_nb15
python -m src.pipeline.phase4_validate --dataset ton_iot
python -m src.pipeline.phase4_validate --dataset unsw_nb15
```

Phase 4 is intentionally strict: fusion is approved only when pooled macro-F1
beats both the GATv2 and CySecBERT baselines on the same dataset. If the stored
metrics do not satisfy that condition, the validator writes a failed approval
manifest instead of approving the phase.

Generate the local dashboard:

```bash
python -m src.pipeline.dashboard
```

This writes `data/dashboard/index.html`. To serve it locally:

```bash
python -m src.pipeline.dashboard --serve --port 8765
```

## Reading Results

Use cross-validation macro-F1 for research claims. In each output directory:

- `training_history.json` contains fold metrics and CV mean/std.
- `metrics.json` contains pooled out-of-fold test metrics.
- `benchmark_summary.json` is the compact comparison artifact.
- `model.pt` and embedding files come from the final model trained on all edges.

Final all-edge model metrics, when present, are fit checks only. They are not
generalization performance.

## UNSW-NB15 Pipeline

The UNSW runner normalizes raw UNSW files into the same schema used by Step 1,
then reuses the shared graph, KG, GNN, LLM, and fusion stages.

```bash
python -m src.pipeline.unsw_nb15.run_pipeline
```

Useful flags:

```bash
python -m src.pipeline.unsw_nb15.run_pipeline --skip-preprocess --skip-gnn --skip-encode
python -m src.pipeline.unsw_nb15.run_pipeline --skip-fusion
```

## Source Layout

```text
src/
  models/
    gnn_classifier.py
    fusion_classifier.py
  pipeline/
    common/
      datasets.py
      splits.py
      metrics.py
      build_splits.py
    step1/
      graph_construction.py
    step2/
      knowledge_graph.py
    step3/
      train_gnn.py
      encode_kg.py
      verify_alignment.py
      train_unimodal_baselines.py
      train_fusion.py
      compare_results.py
    unsw_nb15/
      preprocess.py
      run_pipeline.py
```
