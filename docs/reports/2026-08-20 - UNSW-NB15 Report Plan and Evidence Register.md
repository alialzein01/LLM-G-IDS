# UNSW-NB15 Research Report — Plan and Evidence Register

Working document, 2026-08-20. Supersedes nothing; feeds the M2 research report.

## 0. Locked scope decisions

| Decision | Value |
|---|---|
| Dataset in scope | UNSW-NB15 only. ToN-IoT joins when the `experiment/ton-iot-parity` work lands. |
| Report shape | Instructor's five-part guide (`docs/reports/Report_Guideline.pdf`) — general shape, not a template. No fixed page count. |
| Headline result | Canonical prototype ladder (`results/unsw_nb15_current.json`). |
| Negative results | Reported in Results as their own subsections, plus Limitations. Not buried. |

The guide's four principles govern everything below:

1. Every claim needs a number attached.
2. State what was tried and did not work, alongside what did.
3. Compare against something.
4. Say what remains open.

Principle 2 is the reason the negative results get subsections rather than footnotes.

---

## 1. Evidence register

Every number below is verified against a file in this repo. Do not re-derive; cite from here.

### 1.1 The graph (source: `data/unsw_nb15/processed/step1/pyg_data.pt`)

| Property | Value |
|---|---|
| Nodes (unique IPv4) | 49 |
| Edges (aggregated `(src,dst,attack)`) | 656 |
| Node features | 10 centrality measures |
| Edge features | 5 (`flow_count`, `total_bytes`, `avg_duration`, `most_common_protocol`, `most_common_port`) |
| Classes | 10 (all evaluated; no dropped classes, unlike ToN) |

Class distribution: Normal 311 (47.4%); Analysis 25; every other class exactly 40.

Provenance of the small size: the UNSW-NB15 testbed replays traffic across a fixed,
small set of hosts. Aggregating ~2.5M raw flows by `(src_ip, dst_ip, attack_cat)`
collapses to 656 edges. There is **no** sampling cap in `src/pipeline/unsw_nb15/preprocess.py`
— verified by inspection. The only `head()` call writes a 200-row preview sidecar.

### 1.2 Structural degeneracy of the UNSW topology (new analysis, 2026-08-20)

The single most important explanatory result for this report.

| Measure | Value |
|---|---|
| Mean pairwise Jaccard overlap of `(src,dst)` pair sets between attack classes | **0.9167** (min 0.625, max 1.000) |
| `(src,dst)` pairs carrying all 10 labels | 25 |
| `(src,dst)` pairs carrying 9 labels | 15 |
| Edges sitting on a pair with >1 label | **385 / 656 = 58.7%** |
| Distinct pairs overall | 311 |

Worked example — `175.45.176.0 -> 149.171.126.10` carries nine labels
(Backdoors, DoS, Exploits, Fuzzers, Generic, Normal, Reconnaissance, Shellcode, Worms).

Mechanism, and why it matters: node features are per-IP centralities, so for a repeated
pair the GNN's `h_u` and `h_v` are **bit-identical across all nine labels**. The edge
classifier input is `[h_u ‖ h_v ‖ edge_attr]`. Therefore on 58.7% of edges the only
varying signal is the 5 edge attributes, and the structural branch degenerates to an
MLP over five numbers.

This is the mechanistic explanation for GNN macro-F1 = 0.5496. GATv2 is not
underperforming; UNSW-NB15's topology is near-uninformative for class discrimination
**by construction**. Reproduce with the snippet in §4.

### 1.3 Canonical ladder (source: `results/unsw_nb15_current.json`)

Consultant: whitened prototype on every rung. `top_k = 16.0%`, semantic-confidence
fraction 0.50, effective feedback 8.0%, seed 42, deterministic CPU.
Protocol: pooled five-fold out-of-fold.

| Rung | Accuracy | Macro-F1 |
|---|---|---|
| GNN | 0.6890 | 0.5496 |
| LLM | 0.7790 | 0.7353 |
| AGAF | 0.7637 | 0.7459 |
| Feedback loop | 0.8186 | **0.7764** |

`ladder_order_holds: true`.

### 1.4 Statistical comparisons (same file)

| Comparison | Δ macro-F1 | 95% CI | P(>0) |
|---|---|---|---|
| Feedback − AGAF | +0.0301 | [−0.0025, +0.0631] | 0.965 |
| AGAF − LLM | +0.0109 | [−0.0168, +0.0391] | 0.790 |

**Neither top-of-ladder gap is significant.** Both CIs cross zero. The ladder is an
observed point-estimate ordering, not proof.

What *is* significant (source: `results/unsw_nb15_trained_head_experiment.json`):

| Comparison | Δ macro-F1 | 95% CI | P(>0) |
|---|---|---|---|
| AGAF − GNN | +0.2837 | [+0.2421, +0.3236] | 1.000 |
| Loop − GNN | +0.2770 | [+0.2371, +0.3165] | 1.000 |

### 1.5 Feedback ablations (source: `results/unsw_nb15_current.json`)

| Mode | Macro-F1 |
|---|---|
| Real (entropy-selected) | 0.7764 |
| Random selection | 0.5155 |
| Head-only (GNN logits pass through) | 0.5578 |

| Comparison | Δ | 95% CI |
|---|---|---|
| Real − random | +0.2612 | [+0.2243, +0.2989] |
| Real − head-only | +0.2179 | [+0.1838, +0.2515] |

Entropy-based selection is doing real work — it is not noise. This is a genuine
positive result and should be stated as such.

Caveat carried from the contract: the real-vs-head-only ablation combines attention
feedback *and* final semantic fusion; it does not isolate attention re-entry alone.
§1.7 is what isolates it.

### 1.6 Negative result A — the trained LLM head beats the full system

Source: `results/unsw_nb15_head_baseline.json`, `results/unsw_nb15_trained_head_experiment.json`.
Marked `must_be_reported` in the contract itself.

A per-fold MLP on the same CySecBERT KG embeddings, consultant on every rung,
`top_k` retuned to 20% on validation only:

| Rung | Macro-F1 (head consultant) |
|---|---|
| GNN | 0.5496 |
| LLM head alone | **0.8321** |
| AGAF | 0.8331 |
| Feedback loop | 0.8269 |

`ladder_order_holds: false`.

Three facts to report:

- The LLM head alone (0.8321) **exceeds the canonical full system** (0.7764).
- AGAF − LLM head = +0.0010, CI [0.0000, 0.0033]. AGAF adds essentially nothing over
  its own consultant — degenerate, not a result.
- Retuning `top_k` from 16 to 20 recovered +0.0059 (0.8210 → 0.8269) and narrowed
  loop−AGAF from −0.0126 to −0.0067, but **never flipped the sign**. Validation macro-F1
  is nearly flat across n (0.8265–0.8312, spread 0.0047), so `top_k` is not the binding
  constraint. No n in 15–35 reaches the head on validation-selected terms.

### 1.7 Negative result B — the loop's gain is output fusion, not attention re-entry

Source: `results/unsw_nb15_llm_simplification_sweep.json`. This is the decisive experiment.

Two axes: `loop_full_llm_access` (attention bias on flagged ~8% **plus** output fusion of
the LLM branch on **all** edges) vs `loop_8pct_llm_access` (`--no-output-fusion`; attention
bias on the flagged ~8% only). Four text levels, GNN rung constant at 0.5496 throughout.

| Text level | LLM | AGAF | Loop (full) | Loop (8% only) | Ladder holds |
|---|---|---|---|---|---|
| L0 rich (canonical) | 0.7353 | 0.7459 | **0.7764** | **0.5343** | yes |
| L1 levels_only | 0.6850 | 0.7609 | 0.6969 | 0.5622 | no |
| L2 bare_numbers | 0.4883 | 0.6599 | 0.6435 | 0.5419 | no |
| L3 structure_only | 0.3947 | 0.5640 | 0.5369 | 0.5367 | no |

Findings to report verbatim in substance:

- With output fusion disabled, the loop collapses to **0.5343 at L0 — below the GNN
  alone (0.5496)**, and stays at 0.534–0.562 at every text level. The iterative 8%
  semantic consultation, in isolation, performs no better than the bare GNN.
- Therefore the loop's advantage comes from all-edge output fusion of LLM embeddings,
  **not** from the bidirectional attention mechanism that is the project's novelty claim.
- At L3 the two axes converge (0.5369 vs 0.5367): once the text is weak enough, output
  fusion contributes nothing — consistent with output fusion being the sole source of gain.
- The ladder holds **only** at L0 rich text. Every simplification breaks it, and in the
  direction *opposite* to the instructor's 2026-08-18 hypothesis: AGAF ends up above the
  loop at all three simplified levels.
- AGAF is more robust to simplification than the loop: L0→L3, the LLM falls 0.341, the
  loop falls 0.240, AGAF falls only 0.182.
- L1 AGAF (0.7609) > L0 AGAF (0.7459): the rich magnitude prose is not optimal for
  fusion. Removing it helped AGAF while hurting the LLM and the loop.

Integrity notes recorded in the same file: canonical embeddings and prototypes restored
afterwards; L0 axis A reproduces the canonical ladder exactly; NL text asserted label-free
for every variant via `knowledge_graph.assert_label_free`.

### 1.8 Standing limitations (from `results/unsw_nb15_current.json`)

- Edges are aggregated with the attack label in the grouping key, so these are **not
  deployment-valid estimates**. Note: this is mandated by the instructor's own
  `project_requirements.pdf` Step 1 — inherited from the spec, not a pipeline defect.
- The real-vs-head-only ablation does not isolate attention re-entry alone.
- `top_k` was selected on rotating validation folds; nested CV or a held-out dataset is
  required for a fully unbiased final estimate.

---

## 2. Report outline, mapped to the five-part guide

### Part 1 — Introduction / Context

Content: IDS on IoT/network traffic; the half-blind framing (GNNs read structure but carry
no semantic knowledge; LLMs know cybersecurity but cannot process topology natively);
objective; contributions; report structure.

Contributions to claim — phrased to match what the evidence actually supports:

1. An edge-level GNN+LLM pipeline for NetFlow IDS with an internally constructed,
   label-free knowledge graph.
2. AGAF, a gated attention fusion module over structural and semantic edge embeddings.
3. A bidirectional feedback loop with entropy-based uncertainty selection.
4. **A controlled ablation establishing which components actually carry the gain** — and
   showing that on UNSW-NB15 the attention-re-entry mechanism does not.

Status: writable now.

### Part 2 — Background / State of the Art

Content: the six-paradigm map (GNN-only; LLM-only; sequential GNN→LLM; sequential
LLM→GNN; interactive LLM↔GNN; parallel frozen LLM), then the gap. The guide asks for
coherent ordering — go paradigm by paradigm, not paper by paper.

Sources already in repo: `docs/Full 25-Work Comparison Table (Parallel GNN+LLM Fusion).md`,
`docs/Step 3 Fusion Literature Survey.md`, `docs/feedback_loop_mechanisms.md`, plus the
corpus in the `gnn-llm-ids-research` skill (FedGATSage, KLAGE, LOGIN, LLMGraph, eX-NIDS,
DoLLM, T5-IDS, GL-Fusion, GLANCE, GraphAdapter, …).

Status: writable now. Needs a citation pass — no BibTeX file exists in the repo yet.

### Part 3 — Method

Longest, most technical part. The guide explicitly asks for architecture figures — you
already have `agaf_architecture.{png,svg,pdf}` and `fusion_flow.png` at repo root.

- 3.1 Graph construction — `(src,dst,attack)` aggregation, 10 centralities, 5 edge attrs,
  Z-score normalisation with stats persisted for inference.
- 3.2 Knowledge graph and label-free NL — per-attack-type tercile discretization
  (low/medium/high), relation verbs, the `assert_label_free` guard and *why* it exists
  (label leakage into the LLM path).
- 3.3 GNN branch — GATv2 over GAT (static vs dynamic attention), edge attributes entering
  the attention computation, residuals + BatchNorm, edge classifier head.
- 3.4 LLM branch — CySecBERT (`markusbayer/CySecBERT`), mean pooling, 768-d.
- 3.5 AGAF — projection to common dim, gate over `[h ‖ s]`, softmax attention, fusion.
  Mention the surveyed alternatives (`concat`, `fixed`, `scalar`, `selfattn`) as a
  swappable interface.
- 3.6 Feedback loop — `UncertaintySelector` (Shannon entropy, top-k% quantile threshold
  calibrated per fold on train predictions), semantic scorer, `SemanticAttentionBias` →
  `BiasedGATv2Layer`, outer loop with churn tolerance, max 3 iterations.
- 3.7 Experimental protocol — stratified 5-fold, boolean masks, pooled OOF, macro-F1,
  class weights, seed 42, `IDS_FORCE_CPU=1` for reproducibility, leakage-free
  `edge_embeddings_oof.pt`.

Status: writable now from source. Every module named above is verified present in `src/`.

### Part 4 — Results

- 4.1 Dataset characterisation — §1.1 **and §1.2**. Put the degeneracy analysis here, early.
  It sets up every result that follows.
- 4.2 Canonical ladder — §1.3, with the CIs from §1.4 stated in the same breath. Do not
  present the ordering without them.
- 4.3 Ablations — §1.5. Real vs random vs head-only. This is the strongest positive result.
- 4.4 Negative result A — §1.6, the trained head baseline.
- 4.5 Negative result B — §1.7, the simplification sweep and the output-fusion isolation.

Status: fully evidenced. Writable now.

### Part 5 — Discussion / Conclusion

The argument that ties it together:

1. On UNSW-NB15 the graph modality carries little class-discriminative information —
   quantified: 58.7% of edges share topology across labels, mean Jaccard 0.917.
2. Consequently the semantic branch dominates, and a strong semantic head alone reaches
   0.8321.
3. The feedback loop's measured gain is attributable to all-edge output fusion, not to
   the 8% attention re-entry, which alone underperforms the bare GNN.
4. So the architecture's *fusion* contribution is supported; its *bidirectional
   iterative* contribution is not supported on this dataset.
5. Open: whether ToN-IoT, with genuinely differentiated topology per attack class,
   changes (3). That is the natural next step and the bridge to the ToN chapter.

This is a coherent, defensible M2 result. It is a well-instrumented nuanced finding with
a mechanistic explanation, not a failure — and it is exactly what guide principles 2 and 4
ask for.

---

## 3. Gaps to close

| Gap | Impact | Action |
|---|---|---|
| No BibTeX / reference manager file in repo | Blocks Part 2 | Create `docs/references.bib`; ~25 works from the survey docs |
| `CLAUDE.md` cites `notebooks/unsw_nb15_instructor_pipeline.ipynb`; no `.ipynb` exists (added in `fa1c106`, gone now) | Reproducibility claim in Part 3.7 | Confirm whether it was intentionally dropped in `c004ff6` |
| Per-class P/R/F1 table not yet extracted | Part 4 needs it — guide principle 1 | Pull from `data/unsw_nb15/processed/*/metrics.json` |
| No figure for the degeneracy finding | Part 4.1 would land harder with one | Jaccard heatmap over the 10 classes |
| ToN-IoT chapter | Deferred | Blocked on `experiment/ton-iot-parity` |

## 4. Reproducing the degeneracy analysis

```python
import pandas as pd, itertools
df = pd.read_csv('data/unsw_nb15/processed/step1/aggregated_edges.csv')
df['pair'] = df['IPV4_SRC_ADDR'] + '->' + df['IPV4_DST_ADDR']
atk = {a: set(g['pair']) for a, g in df.groupby('Attack') if a != 'Normal'}
j = [len(atk[a] & atk[b]) / len(atk[a] | atk[b])
     for a, b in itertools.combinations(sorted(atk), 2)]
print('mean Jaccard', sum(j) / len(j))                      # 0.9167
n = df.groupby('pair')['Attack'].nunique()
print('edges on multi-label pairs',
      df['pair'].isin(set(n[n > 1].index)).sum(), '/', len(df))   # 385 / 656
```
