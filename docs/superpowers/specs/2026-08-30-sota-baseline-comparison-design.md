# SOTA Baseline Comparison — Design

**Date:** 2026-08-30
**Status:** approved (design); implementation plan pending
**Branch context:** `experiment/edge-feature-encoding`

## 1. Purpose

The report needs an answer to "did you compare against the field?". This spec defines a
controlled comparison in which two published intrusion-detection architectures are
re-implemented from their papers and trained on **our aggregated flow-graph
representation**, under **our evaluation protocol**, alongside our four ladder rungs.

### Claim boundary

We do **not** claim to beat the baselines' published numbers. Those were obtained on
per-flow NetFlow graphs of ~10^6 edges; ours are aggregated graphs of 2,127 (NF-ToN-IoT)
and 656 (NF-UNSW-NB15) edges. The claim is narrower and defensible:

> Given the same aggregated flow-graph representation, the same splits, and the same
> metric, our GNN+LLM fusion and feedback loop outperform / do not outperform published
> graph-based NIDS architectures.

Every table, caption and sentence in the report must use the phrasing
**"re-trained on our aggregated representation"**, never "outperforms <paper>".

### Reference points from the source papers

| Model | Published task | Dataset | Reported |
|---|---|---|---|
| E-GraphSAGE | binary | NF-ToN-IoT | F1 1.00 |
| E-GraphSAGE | multi-class | NF-ToN-IoT | **weighted-F1 0.63** |
| E-GraphSAGE | multi-class | NF-BoT-IoT | weighted-F1 0.81 |
| TE-G-SAGE | multi-class | NF-UNSW-NB15-v3 | Accuracy / Macro-F1 / FAR |

Multi-class is the hard task even at full scale. Note that E-GraphSAGE reports
**weighted**-F1, which is more forgiving under imbalance than the **macro**-F1 we report;
the two are not interchangeable and must not be compared directly.

## 2. Baselines selected

**E-GraphSAGE** — Lo, Layeghy, Sarhan, Gallagher, Portmann, NOMS 2022
(arXiv:2103.16329). Supervised, edge-level, multi-class, published on NF-ToN-IoT and
NF-UNSW-NB15. The canonical graph NIDS.

**TE-G-SAGE** — MDPI, Dec 2025 (`mdpi.com/2673-3951/6/4/165`), reference code
`github.com/Ricco555/TE-G-SAGE-XAI`. Supervised, edge-aware GraphSAGE + SHAP,
multi-class, published on NF-UNSW-NB15-v3.

### Rejected candidates and why

- **Anomal-E** (KBS 2022) — binary anomaly detection with CBLOF/HBOS/IF/PCA heads.
  Cannot emit a multi-class macro-F1 as published.
- **XG-NID** (arXiv:2408.16021) — fuses flow-level *and packet-level* modalities in a
  heterogeneous graph. We have no packet data; running it flow-only deletes half its
  architecture and would be a genuinely handicapped baseline.
- **GTCN-G** — code availability and exact datasets unverified.

## 3. Architectures, as specified in the papers

### 3.1 E-GraphSAGE

- Node features initialised `x_v = {1, ..., 1}`, dimensionality = number of edge features.
  **`data.x` is ignored** — our 10 centrality measures are unused by this model. This is
  their architecture as published, not a handicap, and it is a substantive difference to
  report: E-GraphSAGE derives node state solely from aggregated edge features.
- K = 2 convolutional layers, hidden dimension 128, ReLU, dropout 0.2 between layers.
- Neighbourhood aggregation over **edge** features:
  `h_N(v)^k = AGG_k({e_uv^(k-1) : u in N(v), uv in E})`, mean aggregator
  `h_N(v)^k = sum(e_uv^(k-1)) / |N(v)|_e`.
- Node update by concatenation: `h_v^k = sigma(W^k · CONCAT(h_v^(k-1), h_N(v)^k))`.
- Edge embedding `z_uv^K = CONCAT(h_u^K, h_v^K)` (256-dim) → classifier.
- Adam, lr 1e-3, plain (unweighted) cross-entropy, full neighbourhood sampling.
- Epoch count is not stated in the paper; see §6 deviation D3.

### 3.2 TE-G-SAGE

- 2 SAGEConv layers, hidden dimension 128, mean aggregator, dropout 0.3.
- Per-layer fanout [25, 15].
- Multi-class edge/flow classification.
- Reported metrics: Accuracy, Macro-F1, FAR, under a 60/30/10 chronological split.

## 4. Data contract — what "same data" means

All three model families receive **identical rows, identical labels, and identical
fold assignments**, taken from the existing artifacts:

- `data/{ton_iot,unsw_nb15}/processed/step1/aggregated_edges.csv`
- `data/{ton_iot,unsw_nb15}/processed/splits/folds.pt`

They differ only in how each paper's own preprocessing converts the two **categorical**
edge columns (`most_common_protocol`, `most_common_port`) into numbers:

| Model | Categorical handling | Continuous handling |
|---|---|---|
| Ours (v2) | learned embeddings (protocol 4-dim, port 32-dim) | log1p + Z-score, cols 0-2 |
| E-GraphSAGE | standard-scaled numeric | standard-scaled |
| TE-G-SAGE | one-hot, `rare_min_freq=50` | log1p + StandardScaler + correlation-prune @ 0.995 |

**Rationale (decision, approved).** Forcing our v2 tensor on the baselines would hand
them raw vocabulary indices as numeric magnitudes — the model would read port 443 as
"5x larger than" port 80. That is precisely the v1 encoding bug this project already
found and fixed, and CLAUDE.md forbids reintroducing it. Doing so would depress baseline
scores for a preprocessing reason rather than an architectural one, which tests nothing.
Each paper's documented recipe is therefore used. The *data* is identical; only the
featurization follows its own paper.

## 5. Configurations reported

Each baseline is reported in **two columns**:

- **`as_published`** — hyperparameters exactly as stated in the source paper
  (§3), nothing tuned, nothing repaired.
- **`refit`** — capacity and schedule re-selected on **our validation folds only**, over
  a fixed grid. No architectural change, no bug fixes, no added components. The grid,
  fixed here so it cannot be widened after seeing results:

  | Knob | Values |
  |---|---|
  | hidden dim | 32, 64, 128 |
  | layers (K) | 1, 2 |
  | learning rate | 1e-3, 5e-4 |
  | dropout | as published (0.2 E-GraphSAGE / 0.3 TE-G-SAGE) |
  | TE-G-SAGE `rare_min_freq` | 50 (published), 2 |

  **Loss weighting is not on the grid.** An earlier revision put it there, on the premise
  that E-GraphSAGE specifies plain unweighted cross-entropy while our rungs use
  inverse-frequency weights. That premise came from the paper text and was wrong: the
  authors' released notebook uses `nn.CrossEntropyLoss(weight=class_weights)` with
  sklearn inverse-frequency weights — the same scheme we use. Weighted cross-entropy is
  therefore the *faithful* setting and belongs in `as_published`. No asymmetry exists to
  correct.

  Epochs are not a grid axis in either column: the source papers do not state an epoch
  count (see D3), so both columns use our standard early-stopping schedule. `refit`
  therefore differs from `as_published` only in the knobs tabulated above.

- **`plus_node_features`** (both baselines) — identical to `refit`, except the
  featureless node initialisation (E-GraphSAGE's `x_v = {1,...,1}`; TE-G-SAGE's learned
  constant embedding) is replaced by our 10 centrality measures (`data.x`).

  This is **not E-GraphSAGE** and must never be tabulated as such. It is labelled
  "E-GraphSAGE + our node features" everywhere it appears. Its purpose is a single
  ablation: E-GraphSAGE learns its node representation from the edges, while we hand our
  model precomputed centralities derived from those same edges. The gap between `refit`
  and `plus_node_features` isolates how much of our advantage comes from that feature
  engineering rather than from GNN+LLM fusion — the question a reviewer will ask, and one
  the two-column design cannot answer.

  Note that this column withholds nothing from `refit`: centralities are derived from the
  same edge list both columns already receive. It changes the *representation*, not the
  data.

  **Confirmed: TE-G-SAGE also ignores node features.** Its released
  `edge_graphsage.py` uses `nn.Embedding(1, hidden)` — one learned constant broadcast to
  every node — when `in_node == 0`, which is the shipped configuration. The
  `plus_node_features` variant therefore applies to **both** baselines, and both carry
  `is_faithful_to_paper: false`.

Reporting both is deliberate. The published defaults were chosen for graphs ~300x larger
than ours; running them unchanged at our scale places them outside their designed
operating regime, and we chose that regime by choosing aggregation. The `as_published`
column is the honest "their model, untouched" number the comparison was asked for; the
`refit` column shows how much of any gap is scale mismatch rather than architecture, so
the result cannot be attributed to an unequal tuning budget.

## 6. Evaluation protocol

Identical to the canonical protocol for our own rungs:

- The same `folds.pt` — stratified 5-fold, seed 42.
- Pooled out-of-fold evaluation; primary metric **macro-F1**; accuracy and weighted-F1
  also recorded.
- >= 3 seeds (42, 1, 2). Hyperparameter selection on **validation folds only**, never test.
- `OMP_NUM_THREADS=1` exported for every run.
- NF-ToN-IoT: classes 3 (`dos`, 4 edges) and 7 (`ransomware`, 3 edges) stay in-graph but
  are excluded from the metric — the 8-class macro-F1, matching `ton_iot_current.json`.

### Forced deviations (must be recorded in the output JSON)

- **D1 — TE-G-SAGE chronological split.** Its published 60/30/10 chronological split is
  impossible: aggregation collapsed timestamps. Our stratified folds are used instead.
- **D2 — TE-G-SAGE fanout.** [25, 15] exceeds the degree available in a 656-edge graph;
  full-neighbourhood aggregation is used.
- **D3 — E-GraphSAGE epoch count.** Not stated in the paper. Our standard early-stopping
  schedule (patience 25, max 200 epochs, best val macro-F1 restored) is applied, and the
  fact is recorded.
- **D4 — TE-G-SAGE `rare_min_freq=50`.** At our scale this filter removes nearly every
  port category. Applied as published in `as_published`; recorded, and available as a
  swept knob in `refit`.

Each deviation is emitted as a `{code, reason, resolution}` entry under `deviations` in
the results JSON, so the report can cite them rather than bury them.

## 7. Deliverables

| Path | Contents |
|---|---|
| `src/models/baselines/__init__.py` | package init |
| `src/models/baselines/e_graphsage.py` | E-GraphSAGE encoder + edge classifier (PyG) |
| `src/models/baselines/te_g_sage.py` | TE-G-SAGE encoder + edge classifier (PyG) |
| `src/pipeline/baselines/__init__.py` | package init |
| `src/pipeline/baselines/preprocess.py` | per-paper featurization of `aggregated_edges.csv` |
| `src/pipeline/baselines/run_baselines.py` | driver: folds, seeds, both configs, both datasets |
| `results/{unsw_nb15,ton_iot}_sota_baselines.json` | results contract |
| `tests/test_sota_baselines.py` | contract tests (§9) |

Report artifact: one table of 2 baselines x 2 configs (plus the labelled E-GraphSAGE
 node-feature variant) x 2 datasets against our 4 rungs
(GNN / LLM / AGAF / Loop).

## 8. Results contract

`results/{dataset}_sota_baselines.json`, mirroring the `schema_version`/`dataset_key`/
`metric_protocol` conventions of `*_current.json`, with:

- `baselines.{e_graphsage,te_g_sage}.{as_published,refit}` each carrying
  `macro_f1`, `accuracy`, `weighted_f1`, `per_class_f1`, `seeds`, `hyperparameters`.
- `baselines.e_graphsage.plus_node_features` — same fields, plus
  `"variant_label": "E-GraphSAGE + our node features"` and
  `"is_faithful_to_paper": false`, so no downstream table can render it as E-GraphSAGE.
- `deviations` — the D1-D4 list.
- `data_provenance` — path and sha of `aggregated_edges.csv` and `folds.pt`, plus
  `excluded_classes`.
- `preprocessing.{model}` — the categorical/continuous recipe actually applied.

## 9. Tests

- Baselines consume the **same** `folds.pt` file as `*_current.json` (assert identical
  fold hash).
- The same classes are excluded from the ToN metric in both files.
- Every declared deviation has a non-empty `reason` and `resolution`.
- E-GraphSAGE ignores `data.x`: assert its forward pass under `as_published` and `refit`
  is invariant to perturbing `data.x`, confirming the ones-initialisation is faithfully
  implemented — and that the same forward pass under `plus_node_features` is *not*
  invariant, confirming the variant actually consumes the centralities.
- Any results entry with `is_faithful_to_paper: false` carries a `variant_label`.
- Seed determinism: two runs at seed 42 with `OMP_NUM_THREADS=1` agree to 1e-6.

## 10. Interpretation, fixed in advance

Fixed before any number is produced, so the conclusion is not selected after the fact:

- If `refit` E-GraphSAGE lands near our GNN rung (0.7219 UNSW / 0.4290 ToN), the fusion
  and loop deltas are attributable to our contribution.
- If either baseline lands **above** our Loop rung (0.7728 UNSW / 0.4478 ToN), we report
  that outcome plainly, and the report's contribution claim narrows accordingly.
- If `plus_node_features` closes most of the gap to our rungs, our advantage is largely
  the centrality feature engineering rather than the fusion, and the report must say so.
- On NF-ToN-IoT the ladder already does not hold (AGAF sits significantly below the GNN,
  P=0.0005; Loop is not separated from the GNN, P=0.862). The baseline comparison does
  not change that and must not be presented as if it does.
- Confidence intervals are required on every baseline-vs-rung difference, on the same
  bootstrap protocol used in `docs/RESULTS_ARCHIVE.md` §1.1. Use
  **"highest point estimate"**, never "top rung".

## 11. Out of scope

- Fidelity reproduction of published numbers on full raw NetFlow data.
- XG-NID, packet-level modality, Anomal-E's binary track.
- Any repair, redesign, or architectural augmentation of a baseline.
- Federated evaluation (FedGATSage).
