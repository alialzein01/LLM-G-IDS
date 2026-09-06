# Claim–Evidence Map — LLM-G-IDS M2 Report

**Purpose.** Every claim the report could make, mapped to the artifact that supports it and
marked with the strength of that support. Written before drafting so that no sentence in
Parts 1–5 outruns its evidence. Companion to `WRITING_PLAN.md` (context pack) and
`REPORT_WORKFLOW.md` (structure and terminology lock).

**Provenance of this file.** Built from `WRITING_PLAN.md`, `docs/RESULTS_ARCHIVE.md`
§0–§5.5, and direct inspection of `results/*.json` on branch
`experiment/edge-feature-encoding` @ `4521bdc`. Every row marked `[verified]` was read out
of the named JSON during this pass; rows marked `[plan]` are taken from `WRITING_PLAN.md`
without independent artifact confirmation.

> **Note on `EVIDENCE_STATE.md`.** The task brief names
> `docs/research_report/EVIDENCE_STATE.md` as prior verification work not to be repeated.
> That file does not exist in the working tree and has never existed in git history
> (`git log --all --diff-filter=A` returns nothing). This map was therefore built from the
> plan plus first-hand artifact reads. If `EVIDENCE_STATE.md` exists outside the repo,
> reconcile it against this file before drafting.

---

## Status legend

| Marker | Meaning |
|---|---|
| **SEP** | Statistically separated: 95% CI excludes zero. Always add *"separated at seed 42; rung-side seed variance not yet estimated."* |
| **NOT-SEP** | Measured, CI crosses or touches zero. Report as a point estimate only. |
| **SEP-ADVERSE** | Separated in the direction that goes against us. Must be reported. |
| **PT** | Point estimate exists; no interval was computed for this contrast. |
| **DERIV** | Arithmetic on artifact-backed point estimates. No interval exists and none can be quoted. |
| **DIAG** | Diagnostic only — label leakage or non-canonical arm. Never reportable as performance. |
| **RETRACTED** | Was claimed, has been withdrawn. Must not reappear. |
| **FORBIDDEN** | The paper must not make this claim in any form. |
| **OPEN** | Would be a legitimate claim but no artifact supports it. Do not write it. |

Every interval in this map that involves one of our four rungs inherits the contracts'
own caveat, recorded verbatim in both SOTA files and confirmed in this pass as
`rung_seeds: 1`, `baseline_seeds: 3`:

> Our four rungs were run at seed 42 only, so no rung-side training stochasticity enters
> these intervals. They are therefore NARROWER than a symmetric multi-seed comparison
> would give.

---

## A. Scope, protocol and validity

| ID | Claim | Evidence | Status |
|---|---|---|---|
| A1 | Classification is edge-level on a directed multigraph; nodes are IPs with 10 centralities, edges are aggregated flows with 5 attributes. | `src/pipeline/step1/graph_construction.py`; `PROJECT_NOTES.md` architecture section | **PT** — implementation fact, no metric attached |
| A2 | Evaluation is pooled five-fold out-of-fold macro-F1. | `metric_protocol` field, all three ladder contracts `[verified]` | **PT** |
| A3 | All four rungs are measured at 3 training seeds (42/1/2), fold partition fixed; baselines at 3 seeds too. | `configuration.seed = 42` in both `*_current.json`; `rung_seeds: 1` in both SOTA contracts `[verified]` | **PT** — governs every CI in this map |
| A4 | Edge aggregation keys on `(src_ip, dst_ip, attack_type)`, so the ground-truth label participates in defining edge identity; scores are **not deployment-valid**. | `graph_construction.py:215`; archive §4; already stated in `01_introduction.tex` ¶3 | **PT** — must appear in Parts 1, 3 and 5, never buried |
| A5 | Runs drift ~0.012 macro-F1 at fixed seed unless `OMP_NUM_THREADS=1`; that exceeds most effects measured here. | archive §4, §2.2 | **PT** — the drift magnitude is the reason for the §2.2 retraction |
| A6 | UNSW scores 10 evaluation classes, ToN 8 (`dos` 4 edges and `ransomware` 3 edges stay in-graph for message passing but are excluded from the metric). | `eval_classes` / `dropped_classes` / `n_eval_classes` in `ton_iot_current.json` `[verified]` | **PT** — a 10-class ToN macro-F1 is not a reportable number |
| A7 | Only ladder **shape** is comparable across datasets; absolute macro-F1 is not. | `cross_dataset_comparison.json` `notes[0]` `[verified]` | **PT** |
| A8 | Hyperparameters (`top_k_percent`, `injection_scale`) were selected on validation folds over 3 seeds, never on test. | `top_k_source` and `selection` blocks, both `*_current.json` `[verified]` | **PT** |
| A9 | The graph is small: 656 UNSW edges, 2,127 ToN edges. | archive §5.5; `known_ceilings` counts `[verified]` | **PT** — scale caveat for every comparison |

---

## B. The performance ladder (3 training seeds, fold partition fixed) — UPDATED 2026-09-06

Source: `results/unsw_nb15_current.json` (schema 4), `results/ton_iot_current.json`
(schema 6), aggregate `results/multiseed_ladder_v2_legacy.json`, all `[verified]`.
Mean ± std over seeds 42/1/2. The LLM rung is deterministic given the folds (std 0).

| ID | Claim | Value | Status |
|---|---|---|---|
| B1 | UNSW ladder, pooled OOF macro-F1 | GNN 0.7437±0.023 / LLM 0.7353±0 / AGAF **0.7793**±0.012 / Loop 0.7644±0.004 | **PT** |
| B2 | ToN ladder, pooled OOF macro-F1 | GNN 0.4336±0.005 / LLM 0.2785±0 / AGAF 0.4102±0.028 / Loop **0.4521**±0.011 | **PT** |
| B3 | UNSW order by mean is LLM < GNN < Loop < AGAF | `multi_seed.mean_order` | **PT** — order only; nothing separated (§C) |
| B4 | ToN order by mean is LLM < AGAF < GNN < Loop | `multi_seed.mean_order` | **PT** — order only; nothing separated (§C) |
| B5 | **AGAF is the highest mean on UNSW; the loop is the highest mean on ToN.** | B1, B2 | **PT** — say "highest mean", never "top rung"; never say the loop is the strongest rung on UNSW |
| B6 | UNSW selected config: `top_k_percent` 31.0, `injection_scale` 2.0; verified in every run | `configuration`, `injection_scale_verified_in_run` | **PT** |
| B7 | ToN selected config: `top_k_percent` 25.0, `injection_scale` 20.0; verified in every run | same | **PT** |
| B8 | The loop is the most seed-stable rung on UNSW (std 0.004 vs GNN 0.023, AGAF 0.012) | `multi_seed.rungs.*.macro_f1_std` | **PT** — a property, not a win |
| B9 | AGAF is the least stable rung on ToN (0.3975–0.4422, range 0.045) | `multi_seed.rungs.agaf.macro_f1_per_seed` | **PT** |
| B10 | Seed-42 single-run values (for `reproduce_ladder.py`) | `results` block; GNN/LLM reproduce schema-3/5 exactly, AGAF/loop do not | **PT** — report `multi_seed`, not `results` |

---

## C. Ladder significance — two-level bootstrap (edges AND training seed), `[verified]`

| ID | Contrast | UNSW | ToN |
|---|---|---|---|
| C1 | AGAF − GNN | +0.0362 CI[−0.0096, +0.0822] P=0.946, sign-stable → **NOT-SEP** | −0.0238 CI[−0.0921, +0.0607] P=0.250, NOT sign-stable → **NOT-SEP** |
| C3 | Loop − AGAF | −0.0156 CI[−0.0552, +0.0268] P=0.229, sign-stable (below at all 3 seeds) → **NOT-SEP** | +0.0428 CI[−0.0316, +0.1105] P=0.874, sign-stable → **NOT-SEP** |
| C4 | Loop − GNN | +0.0206 CI[−0.0203, +0.0625] P=0.797, sign-stable → **NOT-SEP** | +0.0191 CI[−0.0180, +0.0596] P=0.839, sign-stable → **NOT-SEP** |

Comparisons against the LLM rung keep the schema-3/5 single-seed intervals (that rung has
no seed variance); they are under `supersedes.schema_*_statistical_comparisons`.

**C6 — the only defensible ladder sentence, both datasets.** *No rung comparison is
statistically separated once training-seed variance is included.* Orderings by mean may
be stated as orderings. The schema-3 UNSW separations (Loop−GNN, Loop−LLM) and the
schema-5 ToN separations (AGAF below GNN P=0.0005; Loop above AGAF P=1.0) do **not**
survive and must not be cited as current. Status: **NOT-SEP** everywhere.

**C7 — what the seed std is and is not.** It is training-seed variance at a fixed fold
partition. Fold-partition variance is unmeasured, so ±std is a lower bound on total
uncertainty. Say so wherever a ± appears.

---

## D. Multi-seed inversion — RESOLVED 2026-09-06

| ID | Claim | Evidence | Status |
|---|---|---|---|
| D1 | The conversation-recorded 3-seed caveat (AGAF 0.7757 / loop 0.7545) is replaced by an artifact: AGAF 0.7793±0.012 / loop 0.7644±0.004 on UNSW. Direction confirmed, values differ. | `multi_seed` block; `supersedes.multi_seed_caveat` | **PT** |
| D2 | "The feedback loop is the strongest rung." | — | **FORBIDDEN** on UNSW (below AGAF at every seed); on ToN say "highest mean, not separated" |
| D3 | Wording that is true on both datasets | — | *"highest mean; no comparison is separated once training-seed variance is included"* |
| D4 | The 3-seed rung re-run is done. Fold-partition variance remains the open follow-up. | archive 2026-09-06 | **PT** — state as open work in Part 5 |
| D5 | The schema-3/5 AGAF and loop values do not reproduce under current code at seed 42 (GNN and LLM do, bit-exactly). Cause not established. | `supersedes.reason`, both contracts | **PT** — a reproducibility finding; do not guess the cause |
| D6 | The canonical loop's injected bias was ~0.02–0.08 on unit-variance features: the mechanism was effectively off, and the loop's edge over `head_only` is late fusion. Switching it on (selector-head loss + calibrated temperature) did not help at 3 seeds. | archive 2026-09-06 Findings 1–3; `results/multiseed_ladder_v2_{fixed,fixed_reselected}.json` | **PT** — the RQ3 answer is negative and verified |

---

## E. Comparison against re-trained published baselines (archive §5, both `*_sota_baselines.json` `[verified]`)

### E.0 Claim boundary — mandatory framing

| ID | Claim | Evidence | Status |
|---|---|---|---|
| E0 | Published architectures were **re-trained on our aggregated flow-graph representation**. This is **not** a comparison against their published numbers, which came from per-flow graphs ~300× larger. | `claim_boundary` field, verbatim in both contracts `[verified]` | **PT** — the only permitted phrasing |
| E0b | "We outperform E-GraphSAGE / TE-G-SAGE." | — | **FORBIDDEN** in every form |
| E0c | `plus_node_features` is a labelled, non-faithful ablation, not the published architecture. | `is_faithful_to_paper: false`, `variant_label` fields `[verified]` | **PT** — never tabulate under a paper's name |
| E0d | Three candidate baselines were rejected before running with stated reasons: Anomal-E (binary, cannot produce multiclass macro-F1 as published), XG-NID (needs packet modality we do not have), GTCN-G (no verified code or matching dataset). | archive §5 | **PT** — include, it defends the baseline selection |

### E.1 Baseline point estimates (mean over seeds 42, 1, 2)

| ID | Dataset / architecture | as published | refit | + our node features | Status |
|---|---|---:|---:|---:|---|
| E1 | UNSW / E-GraphSAGE | 0.1194 | 0.1368 | 0.1173 | **PT** `[verified]` |
| E2 | UNSW / TE-G-SAGE | 0.3841 | 0.5842 | **0.7196** | **PT** `[verified]` |
| E3 | ToN / E-GraphSAGE | 0.4141 | 0.4141 | **0.4358** | **PT** `[verified]` |
| E4 | ToN / TE-G-SAGE | 0.1931 | 0.2345 | 0.3523 | **PT** `[verified]` |

Per-cell `macro_f1_std` is present in the contract for every entry and should be carried
into the results table (e.g. UNSW TE-G-SAGE refit ±0.0224).

### E.2 Our rungs vs the strongest baseline

| ID | Contrast | Value | Status |
|---|---|---|---|
| E5 | **UNSW: Loop vs TE-G-SAGE + our node features** | **+0.0535 CI[+0.0102, +0.0990] P=0.991** | **SEP** — the report's strongest baseline claim |
| E6 | UNSW: GNN vs same | +0.0027 CI[−0.0372, +0.0452] P=0.532 | **NOT-SEP** |
| E7 | UNSW: LLM vs same | +0.0160 CI[−0.0313, +0.0616] P=0.768 | **NOT-SEP** |
| E8 | UNSW: AGAF vs same | +0.0400 CI[−0.0029, +0.0836] P=0.962 | **NOT-SEP** |
| E9 | **UNSW headline**: the loop is the *only* rung separated from the strongest re-trained baseline. | E5–E8 | **SEP** for E5, **NOT-SEP** for the rest |
| E10 | ToN: Loop vs E-GraphSAGE + node features | +0.0133 CI[−0.0597, +0.0853] P=0.616 | **NOT-SEP** |
| E11 | ToN: Loop vs E-GraphSAGE as published | +0.0340 CI[−0.0121, +0.0809] P=0.925 | **NOT-SEP** |
| E12 | ToN: GNN vs E-GraphSAGE as published | +0.0151 CI[−0.0276, +0.0564] P=0.761 | **NOT-SEP** |
| E13 | **ToN: LLM vs E-GraphSAGE as published** | **−0.1338 CI[−0.1756, −0.0933] P=0.000** | **SEP-ADVERSE** |
| E14 | **ToN: AGAF vs E-GraphSAGE as published** | **−0.0812 CI[−0.1362, −0.0227] P=0.003** | **SEP-ADVERSE** |
| E15 | ToN: LLM vs E-GraphSAGE + node features | −0.1545 CI[−0.2274, −0.0879] | **SEP-ADVERSE** |
| E16 | ToN: AGAF vs E-GraphSAGE + node features | −0.1019 CI[−0.1843, −0.0218] | **SEP-ADVERSE** |
| E17 | ToN: LLM vs TE-G-SAGE + node features | −0.0729 CI[−0.1214, −0.0259] | **SEP-ADVERSE** |
| E18 | **ToN headline**: nothing of ours is separated from E-GraphSAGE, and two of our rungs are significantly behind it. | E10–E16 | **SEP-ADVERSE** — lead with this, do not soften |
| E19 | The intervals used are the **primary two-level bootstrap** (resamples edges *and* draws one of three baseline seeds). Seed-matched intervals are retained separately. | `resampled: edges_and_baseline_seed`; `seed_matched_comparisons` block `[verified]` | **PT** |

### E.3 The ~+0.60 UNSW margins

| ID | Claim | Evidence | Status |
|---|---|---|---|
| E20 | UNSW margins against E-GraphSAGE of ≈+0.60 are **ceiling-limited by the endpoint-only edge representation**, not architectural evidence. | `known_ceilings.e_graphsage_endpoint_only.reporting_rule` `[verified]`; archive §5.1, §5.5 | **PT** — do not lead with these numbers |

### E.4 Forced deviations D1–D5

| ID | Claim | Evidence | Status |
|---|---|---|---|
| E21 | Five deviations from the source papers were forced by our representation: no chronological split (D1); fanout/batch exceed the graph (D2); fixed epoch schedules replaced by 300 epochs / patience 25 (D3); `rare_min_freq=50` erases port categories at our scale (D4); E-GraphSAGE Eq. 4 vs released `W_msg([h_u ‖ e_uv])` (D5). | `deviations` list in both contracts; archive §5.4 `[verified]` | **PT** — reproduce the full table in Part 3 |
| E22 | Baselines used the same rows, labels, folds and evaluation classes as our rungs. | `data_provenance` block with SHA-256 of `aggregated_edges.csv` and `folds.pt` `[verified]` | **PT** — the hashes are worth quoting |
| E23 | Each paper's own categorical featurisation was preserved. | `preprocessing` block `[verified]` | **PT** |

---

## F. Decomposition of the naive UNSW gap

| ID | Claim | Evidence | Status |
|---|---|---|---|
| F1 | The staircase 0.3841 → 0.5842 → 0.7196 → 0.7644 (loop, 3-seed mean; AGAF 0.7793) | E2 + B1, all `[verified]` | **PT** — every rung of the staircase is artifact-backed |
| F2 | Attribution: +0.200 to fair tuning, +0.135 to our feature engineering, +0.053 to our architecture | arithmetic on F1 | **DERIV** |
| F3 | ≈51% tuning budget / ≈35% feature engineering / ≈14% architecture | arithmetic on F2 (0.2001/0.1354/0.0532 over a 0.3887 total; 51.5 / 34.8 / 13.7%) | **DERIV** — **no interval exists for any share.** Present as a decomposition of point estimates, never with a significance claim |
| F4 | Only the final +0.053 step (Loop vs TE-G-SAGE + features) carries an interval, and it is separated. | E5 | **SEP** |
| F5 | The decomposition is a centrepiece result and an honesty device: most of the naive margin is not architectural. | F1–F4 | **DERIV** — frame as interpretation |

**Drafting note.** `decomposition` appears as no field in either SOTA contract; it is our
arithmetic. Say so, or a reviewer will ask.

---

## G. The E-GraphSAGE endpoint-only ceiling (best explanatory result)

| ID | Claim | Evidence | Status |
|---|---|---|---|
| G1 | E-GraphSAGE classifies an edge from `CONCAT(h_u, h_v)`; the edge's own features never reach the classifier (paper Eq. 5; authors' notebook `self.W(th.cat([h_u, h_v], 1))`). | `known_ceilings...description` `[verified]` | **PT** — mechanism, verified in the released code |
| G2 | Parallel edges between one IP pair are therefore mathematically indistinguishable to it. | G1 | **PT** — deductive |
| G3 | 385 of 656 UNSW edges (58.7%) and 167 of 2,127 ToN edges (7.9%) are indistinguishable by endpoints. | `edges_indistinguishable_by_endpoints` = {unsw 385, ton 167}, `fraction` = {0.587, 0.079} `[verified]` | **PT** |
| G4 | The prediction (58.7% → collapse on UNSW; 7.9% → survives on ToN) matches the outcome: 0.1194 UNSW vs 0.4141 ToN. | G3 + E1, E3 | **PT** — hypothesis → prediction → confirmation; this is the report's model mechanism paragraph |
| G5 | Not a training-schedule artifact: at the authors' full 4999 epochs, fold-0 validation macro-F1 plateaus near 0.11, best 0.1384 at epoch 1400. | `not_an_epoch_artifact` field `[verified]` | **PT** |
| G6 | This is a representation–architecture interaction, **not** evidence that E-GraphSAGE is a weak model. | `reporting_rule` field `[verified]` | **PT** — mandatory framing |

---

## H. Gate 0 and Gate 0.5 — the intellectual centre

Contracts: `results/{unsw_nb15,ton_iot}_oracle_ceiling_v2{,_trained_head}.json`, all
`[verified]`. Output fusion is **off** in every arm, so advice reaches the GNN only
through the mechanism under test. `n_pairs: 15` (3 seeds × 5 folds), thread-pinned.

| ID | Claim | UNSW | ToN | Status |
|---|---|---|---|---|
| H1 | A **perfect consultant** (true label at ±4 nats) delivers a large gain through the edge channel. | paired +0.0369 CI[+0.0097, +0.0618] P=0.995 (pooled +0.0345) | paired +0.1354 CI[+0.0919, +0.1797] P=1.0 (pooled +0.1090) | **SEP** both — but **DIAG**: deliberate label leakage, never reportable as performance |
| H2 | The canonical whitened-prototype consultant captures essentially none of that headroom. | paired −0.0046 CI[−0.0273, +0.0158] P=0.356; **−4.5%** of headroom | paired +0.0099 CI[−0.0111, +0.0286] P=0.836; **11.9%** of headroom | **NOT-SEP** both |
| H3 | A stronger, leakage-free per-fold **trained head** consultant does not close the gap either. | paired +0.0007 CI[−0.0183, +0.0201] P=0.525; **12.5%** captured | paired −0.0378 CI[−0.0588, −0.0167] P=0.0001; **−38.0%** captured | **NOT-SEP** UNSW / **SEP-ADVERSE** ToN |
| H4 | Trained-head arm absolute values | control 0.7398 → prototype 0.7382 → head 0.7441 → oracle 0.7743 | control 0.4195 → prototype 0.4325 → head 0.3781 → oracle 0.5286 | **PT** |
| H5 | On ToN the trained head *significantly harms* the mechanism despite scoring 0.5165 standalone against the prototype's 0.2785. | — | H3, H4 | **SEP-ADVERSE** |
| H6 | **Headline: the channel works; neither realistic consultant uses it.** The binding constraint is consultant quality/calibration, not mechanism capacity. | H1–H5 | | **SEP** for the capacity half, **NOT-SEP** for every realistic-consultant half — exactly the point |
| H7 | Standalone classifier accuracy does **not** translate into useful injected advice. | H3 vs H5 | | **SEP-ADVERSE** on ToN, which is what makes the claim more than a null |
| H8 | Implication for future work: calibration and trust of consultant logits, not a bigger consultant. | H6, H7; archive §0 "Consequences" | | **PT** — interpretation, flag as such |

---

## I. Negative results and retractions — report as prominently as the positives

| ID | Claim | Evidence | Status |
|---|---|---|---|
| I1 | **Attention injection is inert.** Churn exactly 0.0000 in every configuration tested, even handed the true label at ±4 nats. Confirmed three times. | archive §2.1; Gate 0 `oracle_attention` UNSW paired −0.0007 P=0.482, ToN +0.0114 P=0.870 `[verified]` | **NOT-SEP** — a clean, verified null |
| I2 | Structural cause on UNSW: 58.7% of edges (100% of consulted ones) share a `(src,dst)` pair, so node-derived terms are identical and attention never touches `edge_attr`. | archive §2.1 | **PT** — mechanism |
| I3 | Independent second cause on ToN: 62.8% of edges have destination in-degree 1, where softmax over incoming edges is 1.0 regardless of bias. | archive §2.1 | **PT** — number not independently re-verified in this pass `[plan]` |
| I4 | Edge injection, not attention injection, is the live mechanism. | I1–I3; `injection_mode: edge` in both contracts `[verified]` | **PT** |
| I5 | **`real > control` is RETRACTED.** The 2026-08-27 run that produced `real > control > shuffled > random` was not thread-pinned; documented drift (~0.012) exceeds the effect, and `real − control` flips sign on UNSW between runs. | archive §2.2, §3.2; `retraction` block in `unsw_nb15_edge_injection_v2.json` | **RETRACTED** |
| I6 | The ordering `real > control > shuffled > random` as proof the semantic branch is a working consultant. | — | **FORBIDDEN** |
| I7 | **What survives the retraction:** the mechanism is *sensitive to advice content*. `random` degrades far beyond drift on both datasets (−0.0518 UNSW, −0.0593 ToN); `shuffled` does on ToN (−0.0228). UNSW's `shuffled` (−0.0098) sits inside the drift band and is unresolved. | archive §2.2 `[plan]` | **PT** — corrupting advice hurts; supplying it does not measurably help versus none |
| I8 | **Two numbers were once fabricated in this project** (an oracle headroom of +0.0508 with control 0.7237 → 0.7745, back-derived from an unrelated `feedback_minus_gnn` delta) and were caught in external review. | archive §3.1 | **PT** — worth a methods/limitations sentence on artifact discipline; do not reintroduce the values |
| I9 | A previously published ToN ladder (GNN 0.3398, Loop 0.4538) **does not reproduce** under unchanged code, graph and seed. | archive §3.5 | **PT** — reproducibility finding |
| I10 | The loop reaches a fixed point after roughly one correction; iterations plateau at 3.07 actual, and `max_iterations` 5 → 0.7327, 8 → 0.7501 show no improvement. | archive §2.3; `results/unsw_nb15_edge_injection.json` sweep `[plan]` | **PT** — state as *empirical*, not "by construction" |
| I11 | `semantic_logits` is computed once, outside the iteration loop, so only the flagged set changes across iterations, never the advice content. | `src/models/feedback_classifier.py:1056` vs `:1066`, archive §2.3 | **PT** — implementation fact, verify the line numbers before citing them in Part 3 |

---

## J. Encoding, capacity and ablation findings

| ID | Claim | Evidence | Status |
|---|---|---|---|
| J1 | The v1 encoding left port (std 11,359) and protocol (std 62) raw — up to ~11,000× the scale of the Z-scored columns, despite being categorical. | archive §4; `PROJECT_NOTES.md` | **PT** — this was a bug |
| J2 | Fixing it moved the GNN rung 0.5496 → 0.7219 (≈ +0.17). | archive §4 `[plan]` | **PT** — no interval; do not call it significant |
| J3 | A flat MLP on the raw v1 tensor scored 0.5631, reproducing the entire v1 GNN — GATv2 message passing bought nothing over badly-scaled numbers. | archive §4 `[plan]` | **PT** — a strong, honest methods anecdote |
| J4 | v2 encoding: cols 0–2 log1p then Z-scored; cols 3–4 are vocabulary indices consumed by learned embeddings (protocol 4-dim, port 32-dim → 39-wide). | `edge_attr_encoding: v2_log_cont_cat_idx` in all contracts `[verified]`; `gnn_classifier.py::EdgeFeatureEncoder` | **PT** |
| J5 | AGAF is capacity-limited on UNSW, not overfitting-limited: 207,756 params against 393 train edges (529 per example); every lightweight variant costs ≥0.08 macro-F1 and drops AGAF below both single branches. | archive §4 `[plan]` | **PT** — predates v2 and ToN's regression; **label as not re-run under v2** |
| J6 | **ToN's AGAF regression under v2 is undiagnosed.** Candidates named but untested: the fusion-dim fix changed effective capacity; v2 port/protocol embeddings interact badly with ToN's sparser graph (1,501 nodes vs UNSW's 49). | archive §1.2 | **OPEN** — state as undiagnosed, do not assert a cause |
| J7 | Under v1, AGAF (0.4447) was ToN's top rung and the loop sat below it; under v2 both are reversed. | archive §1.2 | **PT** — do not repeat the v1 finding as current |
| J8 | GraphSMOTE-style oversampling is off by default and was verified inert/harmful on v1 ToN (AGAF −0.073, ~4× variance at ratio 0.05); combining it with inverse-frequency class weights double-corrects and collapsed AGAF to 0.0227. | archive §4; `results/ton_iot_balancing.json` `[plan]` | **PT** — v1 only, not re-verified on v2 |
| J9 | Fusion-mechanism comparisons on the ToN graph are not resolvable: rare classes carry 12–35 edges and seed variance (~0.11) exceeds every between-mechanism gap. | archive §6 `[plan]` | **PT** — a good limitations sentence |

---

## K. Feedback ablations at ladder level — do not conflate with Gate 0

Source: `feedback_ablations` in both `*_current.json`, `[verified]`. **These run with output
fusion ON**, unlike Gate 0. They are a different experiment and answer a different question.

| ID | Claim | UNSW | ToN | Status |
|---|---|---|---|---|
| K1 | The loop beats a `head_only` control | +0.0472 CI[+0.0147, +0.0800] | +0.0408 CI[+0.0021, +0.0796] | **SEP** both |
| K2 | The loop beats `random` advice | +0.1241 CI[+0.0837, +0.1648] | +0.0428 CI[**−0.0008**, +0.0862] | **SEP** UNSW / **NOT-SEP** ToN |
| K3 | Absolute ablation scores | random 0.6492, head_only 0.7257 | random 0.4046, head_only 0.4068 | **PT** |
| K4 | **The tension the report must state.** With output fusion on, the loop is separated from `head_only` on both datasets (K1). With output fusion **off**, the same contrast is not separated on either (H2). The measured ladder-level gain is therefore attributable to output fusion, not to the injection path. | K1 vs H2 | | **SEP** / **NOT-SEP** — this is the honest reading and it strengthens the Gate 0 story |

> **Discrepancy found in this pass.** Archive §1.3 describes the ToN real-vs-random
> ablation as *"+0.0428, P=0.972 — separated."* The contract's own 95% interval is
> `[−0.00077, +0.08622]`, which **touches/crosses zero**, and no `prob_positive` field
> exists in that block to source the 0.972. Under the convention this project applies
> elsewhere (UNSW `agaf_vs_gnn`, CI low −0.0004, correctly written "touches zero"), K2-ToN
> is **NOT-SEP**. **Do not write "separated" for the ToN random ablation.** Archive §1.3
> should be corrected.

---

## L. Comparisons that go against us — must appear, not be buried

| ID | Claim | Evidence | Status |
|---|---|---|---|
| L1 | **A trained-head LLM-only baseline beats the full system on both datasets**: UNSW 0.8321 vs loop 0.7644 (AGAF 0.7793); ToN 0.5165 vs loop 0.4521. An MLP on CySecBERT text embeddings outperforms the complete architecture. | `llm_only_trained_head_macro_f1` in `cross_dataset_comparison.json` `[verified]`; re-verified under Gate 0.5 provenance per archive §4 | **PT** — **no interval was computed for this contrast.** Report the point estimates and say no interval exists |
| L2 | Swapping the trained head in as consultant degenerates the loop into echoing it (loop 0.8258 < LLM-head 0.8321 ≈ AGAF-head 0.8331). | archive §4 `[plan]` | **PT** — pre-v2, fusion on; label as a prior, not a settled result |
| L3 | The canonical architecture mandates the prototype scorer partly for cross-dataset comparability, not because it is the better consultant. | `PROJECT_NOTES.md` canonical rules; L2 | **PT** — an honest design-decision sentence |
| L4 | On ToN, AGAF sits significantly below the bare GNN. | C1 | **SEP-ADVERSE** |
| L5 | On ToN, two of our rungs are significantly behind E-GraphSAGE as published. | E13, E14 | **SEP-ADVERSE** |
| L6 | On UNSW, no adjacent rung step is separated. | C6 | **NOT-SEP** |

L1 is the sharpest publication risk in the project and is not yet addressed anywhere in
the drafted introduction. Part 5 must confront it directly.

---

## M. Novelty claims — keep narrow

| ID | Claim | Evidence | Status |
|---|---|---|---|
| M1 | "First GNN↔LLM feedback system." | LOGIN (arXiv 2405.13902), DAS (arXiv 2512.21106), GLANCE (arXiv 2510.10849), RoGRAD (arXiv 2510.01910), all with IDs in `docs/feedback_loop_mechanisms.md` `[verified]` | **FORBIDDEN** |
| M2 | The dual-stream idea itself is not novel; parallel BERT + graph encoders with attention fusion exist outside security (e.g. misinformation detection). | already conceded in `01_introduction.tex`; `moorthy2025dualstream` | **PT** — keep the concession |
| M3 | Permitted novelty envelope: a shared NetFlow pipeline in which a graph encoder and a cybersecurity language encoder both contribute **directly to edge-level classification**, plus an explicit test of whether iterative semantic feedback adds anything beyond late fusion. | `01_introduction.tex` §1.1 closing ¶ | **PT** — confine to IDS / network-flow / edge-level |
| M4 | The closest prior work is identified rather than hidden (LOGIN as the closest analogue to this codebase; XG-NID uses the LLM post-classification for explanation, a different purpose). | `docs/feedback_loop_mechanisms.md`; `farrukh2024xgnid` | **PT** — Part 2 must name these |
| M5 | The contribution is an implementation and controlled evaluation with evidence about its limits, not a universal improvement. | `01_introduction.tex` closing | **PT** |
| M6 | Whether the *negative* results (inert attention injection, the oracle/consultant gap) are novel contributions in the literature. | — | **OPEN** — resolve in the literature step before claiming |

---

## N. Claims blocked pending work

| ID | Blocked claim | Why | Unblocked by |
|---|---|---|---|
| N1 | "The feedback loop is the strongest rung." | D1 resolved: below AGAF at every UNSW seed; nothing separated | — (resolved; wording is forbidden) |
| N2 | Any adjacent-rung improvement claim on UNSW. | C6 | 3-seed re-run, or narrower framing |
| N3 | A cause for ToN's AGAF regression. | J6 | Dedicated diagnostic; none run |
| N4 | "The semantic branch is a working consultant." | I5 retraction; H2 | A thread-pinned 4-arm mechanism re-run |
| N5 | Deployment-valid detection performance. | A4 — the label defines edge identity | Label-free graph construction; not attempted |
| N6 | Any significance statement about the 51/35/14 decomposition shares. | F3 — no interval exists | Would need a bootstrap over the decomposition |
| N7 | A 10-class ToN macro-F1. | A6 | Not obtainable; classes have 3–4 edges |
| N8 | An interval on trained-head-vs-loop (L1). | No such contrast was computed | Would need a bootstrap over those predictions |

---

## O. Terminology and phrasing lock (mechanical check before each part is declared done)

| Rule | Correct form |
|---|---|
| CySecBERT | "pretrained cybersecurity language encoder" on first use, "semantic branch" thereafter. Never "generative LLM" |
| Phase 2 | "uncertainty-guided semantic feedback" — never renamed |
| Rung ranking | "highest mean" — never "top rung"; on UNSW the highest mean is AGAF |
| Baselines | "re-trained on our aggregated representation" — never "we outperform <paper>" |
| Ablation naming | `plus_node_features` never appears under a paper's own name |
| Separation | No ladder separation claims exist. Baseline separations carry "two-level bootstrap over edges and baseline seed; rung side at 3 seeds" |
| Cross-dataset | Compare ladder *shape* only |

`grep` the new LaTeX for: `top rung`, `outperform`, `state-of-the-art`, `significantly`
(check each against §C/§E), `LLM` (must be the locked term outside the RQ names).

**Note.** `results/ton_iot_current.json` contains a field literally named
`loop_is_top_rung: true`. It is a contract field, not permission — the rendered text must
still say "highest point estimate."

---

## P. Coverage check against the report's own research questions

| RQ | Answerable from | Answer available |
|---|---|---|
| RQ1 — individual branch performance | B1, B2 | Yes |
| RQ2 — does AGAF beat both single branches | C1, C2 | Yes, and the answer is **no on UNSW** (not separated) and **no on ToN** (separated the wrong way) |
| RQ3 — does feedback add value beyond static fusion, and from where | C3, K1, K4, H1–H6 | Yes, and the answer separates the two sub-questions: measurable at ladder level with fusion on, not attributable to the injection path |
| RQ4 — effect of topology, repeated endpoint pairs, imbalance, consultant strength | G3, I2, I3, J8, J9, H3 | Yes — repeated endpoint pairs and consultant strength are the strongest evidence; imbalance is v1-only |
| RQ5 — vs published architectures re-trained on the same representation, and how much is representation/tuning vs architecture | E1–E19, F1–F5 | Yes — this is the best-evidenced RQ |

RQ1, RQ2, RQ4 and the report-structure paragraph in `01_introduction.tex` are written
single-dataset (NF-UNSW-NB15 only) and must be re-scoped to both datasets, per
`WRITING_PLAN.md` Task 0. `REPORT_WORKFLOW.md` "Current scope" still restricts the report
to NF-UNSW-NB15 and contradicts the accepted two-dataset scope.

---

## Q. Open items carried into the next steps

1. `EVIDENCE_STATE.md` is missing — reconcile if it exists elsewhere.
2. Archive §1.3 calls the ToN random ablation "separated" against its own contract's CI (see §K).
3. `REPORT_WORKFLOW.md` scope note contradicts the accepted two-dataset scope.
4. L1 (trained-head baseline beats the full system) has no interval and no place in the current draft.
5. `references.bib` holds 6 entries against roughly 30 needed.
6. `figures/` and `tables/` are empty; two figures are load-bearing (the ceiling diagram, the decomposition staircase) and must be requested from the user per the diagram protocol.
