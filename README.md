# LLM-G-IDS

A Graph Neural Network + LLM framework for edge-level intrusion detection on
NetFlow-style network traffic.

The project currently supports two datasets:

- **NF-ToN-IoT** under `data/ton_iot/`
- **NF-UNSW-NB15** under `data/unsw_nb15/`

Raw data is local and not committed. Processed artifacts are written under each
dataset's `processed/` directory.

---

# NF-ToN-IoT: current state

This branch carries the ToN-IoT work. Two graph constructions are supported and
both are runnable from the same CLI. Everything below was produced on this
branch's code; result JSONs are committed under `results/`.

## Why there are two graph constructions

**Aggregated (`--dataset ton_iot`)** — the original design. Flows are grouped by
`(src_ip, dst_ip, attack_type)`. On ToN-IoT this collapses **1,379,274 raw flows
into 2,127 edges** and inverts the class balance:

| | benign | dos | ransomware |
|---|---:|---:|---:|
| raw flows | 270,279 | 17,717 | 142 |
| aggregated edges | 1,746 (82%) | **4** | **3** |

Two classes fall below the 5-fold minimum and must be dropped from the metric, so
macro-F1 is over **8 classes**. The GNN trains on 2,127 examples and overfits
(train 0.61 / OOF 0.40). This caps every model in the ladder.

**Per-flow + capping (`--dataset ton_iot_capped`)** — replaces the 3-column
grouping with a finer one, in two steps:

1. **Signature grouping** — one edge per distinct *model-visible behaviour*, i.e.
   per unique `(src, dst, attack, IN_BYTES, FLOW_DURATION_MILLISECONDS, PROTOCOL,
   L4_DST_PORT)`. Flows identical on all seven are the same input vector with the
   same label; left as separate rows a random split scatters copies across train
   and test, and the score measures memorisation. The repetition is preserved as a
   `flow_count` feature rather than as duplicate rows.
2. **Per-class capping** (default 20,000) — thins majority classes only. Classes
   below the cap are kept in full, so rare classes are never touched.

```
1,379,274 flows -> 727,789 distinct behaviours (47.2% collapsed) -> capped -> 115,010 edges
```

All ten classes now clear the 5-fold minimum, so macro-F1 is over **10 classes**.

`_build_flow_edges` emits the same `EDGE_ATTR_COLUMNS` schema as the aggregated
path, so Step 2 onward runs unmodified — the switch is one flag, not a rewrite.

### Leakage is verified, not assumed

`src/pipeline/step1/verify_flow_graph.py` runs as a pipeline stage and fails loudly
if duplicate signatures survive:

```
edges=115010  distinct signatures=115010
per-fold share of test edges also present in train:
  fold 0: 0.00%   fold 1: 0.00%   fold 2: 0.00%   fold 3: 0.00%   fold 4: 0.00%
PASS: no duplicate signatures, no fold leakage, all stages aligned.
```

An earlier per-flow experiment (2026-08-05, 156,401 edges) scored GNN macro-F1
0.7558 but predates signature grouping and was never leakage-checked — in it,
backdoor held 17,247 edges against only 1,204 distinct behaviours. **That number is
superseded and should not be quoted.**

## Results

Pooled five-fold out-of-fold macro-F1. Both use the whitened-prototype semantic
consultant (`trained_llm_head: false`).

| rung | aggregated (8 classes) | per-flow + capped (10 classes) |
|---|---:|---:|
| GNN | 0.3271 | **0.6722** |
| LLM | 0.2785 | **0.6200** |
| AGAF | 0.3233 | **0.5744** |
| Feedback loop | 0.3654 | **0.6691** |

Every rung roughly doubles, in the harder 10-class setting. Overfitting is gone:
train macro-F1 0.6549 against OOF 0.6722 — the out-of-fold score is now *higher*
than the training score, versus a 0.21 train/OOF gap on the aggregated graph.

**The ladder order does not hold on either graph** (`ladder_order_holds: false`).
On the capped graph:

- AGAF is the *worst* rung, 0.098 below the GNN (CI [-0.105, -0.091], P=0). Its
  modality gate never left 50/50, so it averages a strong GNN with a weaker
  semantic branch and lands below both.
- The loop beats AGAF (+0.095, P=1.0) and LLM (+0.049, P=1.0) but not the GNN
  (-0.003, CI excludes zero).
- **Semantic feedback is not distinguishable from its own control:**

| feedback mode | macro-F1 | vs real |
|---|---:|---|
| real | 0.6691 | — |
| head_only | 0.6687 | +0.0004, CI [-0.0024, +0.0027], P=0.63 |
| random | 0.6584 | +0.0107, P=1.00 |

Real feedback beats *random* but ties *head_only*, so the loop's performance is
explained by its GNN head, not by the semantic consultant.

### Per-class, capped GNN

| class | support | precision | recall | F1 |
|---|---:|---:|---:|---:|
| Benign | 20,000 | 0.997 | 0.979 | 0.988 |
| backdoor | 1,204 | 0.865 | 0.917 | 0.890 |
| xss | 20,000 | 0.748 | 0.916 | 0.824 |
| mitm | 1,009 | 0.859 | 0.773 | 0.814 |
| dos | 814 | 0.490 | 0.994 | 0.656 |
| ddos | 20,000 | 0.673 | 0.618 | 0.645 |
| scanning | 11,953 | 0.439 | 0.901 | 0.590 |
| password | 20,000 | 0.613 | 0.501 | 0.552 |
| injection | 20,000 | 0.892 | **0.338** | 0.490 |
| ransomware | 30 | **0.159** | 0.967 | 0.274 |

Two structural findings worth continuing from:

- **Injection is the largest genuine error source.** Its misses go to xss (25.5%),
  password (17.2%), ddos (11.7%) and scanning (11.4%) — not randomly. Injection
  spans only **61 distinct IP pairs**, and 20–34% of those also carry the classes
  it is confused with. These are the same attacker hitting the same web server, so
  `(src, dst)` is identical and the graph carries no signal to separate them. This
  is a ceiling on any purely structural model.
- **The class-weighted focal loss over-predicts rare classes.** dos reaches recall
  0.994 at precision 0.490, ransomware 0.967 at 0.159. Those predictions come out
  of the large ambiguous classes, and injection pays most.

## GAT architecture note

`VARIANT_NAMES` in `src/models/gnn_classifier.py` controls how many GAT branches
the encoder runs. The original design followed FedGATSage with three branches
(`temporal`, `content`, `behavioral`), each a full GATv2 encoder differing only by
a fixed rescaling of the same five edge attributes.

Those rescalings are absorbable into the learned linear layer that immediately
follows them (`W·diag(w)` is just another reachable `W`), so the branches span an
identical function class — they act as an ensemble, not as complementary views.
Measured on the aggregated graph over three seeds:

| configuration | macro-F1 | time | params |
|---|---:|---:|---:|
| 3 variants | 0.3634 ± 0.0073 | 36.9 s | 150,312 |
| 1 variant, plain GATv2 | 0.3657 ± 0.0104 | 14.4 s | 50,580 |

No measurable accuracy cost, 2.6x faster, one third the parameters. **This branch
therefore defaults to a single plain GATv2 branch.** To restore the FedGATSage
ensemble, set `VARIANT_NAMES = ("temporal", "content", "behavioral")` and the three
`EDGE_FOCUS_WEIGHTS` entries back.

## Reproducing

Per-flow + capping (~3h 12m on CPU: GNN 12 min, encode 11 min, AGAF 16 min,
feedback 95 min):

```bash
python -m src.pipeline.common.run_pipeline --dataset ton_iot_capped --cap 20000 \
    --skip top_k_sweep baselines validate
```

Aggregated baseline:

```bash
python -m src.pipeline.common.run_pipeline --dataset ton_iot
```

`top_k_sweep` is skipped above because it trains 21 candidate configurations — on
the capped graph that is roughly 8 hours on its own. As a result the capped run
used the **default `top_k_percent = 16.0`, which was never tuned for this graph**
(`"source": "default"` in `results/ton_iot_capped_ladder.json`).

Committed results:

| file | contents |
|---|---|
| `results/ton_iot_capped_ladder.json` | the 10-class ladder + bootstrap CIs |
| `results/ton_iot_capped_ablations.json` | real / random / head_only |
| `results/ton_iot_capped_gnn.json` | per-class GNN breakdown |
| `results/ton_iot_capped_agaf.json` | AGAF fusion detail |
| `results/ton_iot_aggregated_ladder.json` | the 8-class aggregated baseline |

## Open items

1. **Trained LLM head.** Both ladders above use the whitened-prototype scorer. On
   the aggregated graph the prototype scorer collapses (0.28) while a trained MLP
   head on the same CySecBERT embeddings reaches 0.50. Re-running the capped
   feedback with `--use-llm-head` is the most likely route to recovering the ladder.
2. **AGAF's gate.** It sits at 50/50 and drags the fusion below both inputs.
   Confidence-gated head fusion is the candidate fix.
3. **Tune `top_k_percent`** for the capped graph.
4. **`MAX_EPOCHS = 300` is binding, not early stopping.** Folds peaked at epochs
   285 and 279, and validation F1 was still rising at 300 — the model is
   under-trained rather than over-trained.

## Caveats to preserve

- Edge grouping includes the attack label in the key on **both** paths, so neither
  is a deployment-valid estimate.
- The 8-class and 10-class macro-F1 figures are **not** directly comparable.
- Ransomware has 30 edges (18/6/6) in the capped graph. With six test samples it
  moves the 10-class macro materially; report the 9-class macro alongside it.

---

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
