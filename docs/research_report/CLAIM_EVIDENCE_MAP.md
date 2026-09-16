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

**Audited against the final manuscript, 2026-09-11 (W8).** Sections C, D and E were
rewritten in that pass: they carried the prototype-consultant ladder and the seed-42-rung
baseline intervals, both of which are superseded. `EVIDENCE_STATE.md` now exists and has
been reconciled against this file. Rows unchanged since the original pass are still marked
`[verified]` from that pass.

Both sides of every interval in this map are now measured at three training seeds. The
earlier caveat that our rungs ran at seed 42 only, which made the intervals narrower than a
symmetric comparison would give, no longer applies and its blocks are under `superseded` in
both SOTA contracts.

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
| B1 | UNSW ladder, pooled OOF macro-F1 (loop consults the trained head, 2026-09-07) | GNN 0.7437±0.023 / LLM 0.7353±0 / AGAF 0.7793±0.012 / Loop **0.8341**±0.004 / head alone **0.8374**±0.005 | **PT** — quote the loop and head-alone rows together |
| B2 | ToN ladder, pooled OOF macro-F1 (loop consults the trained head, 2026-09-07) | GNN 0.4336±0.005 / LLM 0.2785±0 / AGAF 0.4102±0.028 / Loop **0.5044**±0.008 / head alone **0.5081**±0.007 | **PT** — quote the loop and head-alone rows together |
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

**Regenerated 2026-09-16 with the seed drawn INDEPENDENTLY for each side.** Section 3.9 always
described an independent draw; the ladder code drew one seed and handed it to both rungs, which
is a paired draw and gives a narrower interval. Every value in this table moved and every one
of them widened. **No verdict changed.** The superseded shared-seed intervals are the ones
this table carried before that date and must not be quoted: they are narrower than the
procedure the manuscript describes.

The loop is measured under both consultants, and the two halves of this table are the
report's central result. Intervals are `multi_seed.comparisons[*].two_level` in
`multiseed_ladder_v2_head.json` and `multiseed_ladder_v2_legacy.json`.

| ID | Contrast | Consultant | UNSW | ToN |
|---|---|---|---|---|
| C1 | AGAF − GNN | either | +0.0370 [−0.0212, +0.0929] → **NOT-SEP** | −0.0235 [−0.0910, +0.0602], not sign-stable → **NOT-SEP** |
| C2 | Loop − AGAF | prototype | −0.0145 [−0.0556, +0.0276] → **NOT-SEP** | +0.0433 [−0.0376, +0.1199] → **NOT-SEP** |
| C3 | Loop − AGAF | trained head | +0.0557 [+0.0208, +0.0905] → **SEP** | +0.0936 [+0.0139, +0.1720] → **SEP** |
| C4 | Loop − GNN | prototype | +0.0214 [−0.0278, +0.0657] → **NOT-SEP** | +0.0193 [−0.0180, +0.0604] → **NOT-SEP** |
| C5 | Loop − GNN | trained head | +0.0916 [+0.0376, +0.1425] → **SEP** | +0.0697 [+0.0129, +0.1283] → **SEP** |
| C6 | Loop − LLM | trained head | +0.0989 [+0.0721, +0.1267] → **SEP** | +0.2231 [+0.1703, +0.2771] → **SEP** |
| C7 | Loop − LLM | prototype | — | — | **OPEN in the manuscript.** The regenerated prototype aggregate now DOES carry this comparison, because the per-seed head logits it needs were built after that file was last written. The `[GAP]` marker in the `tab_comparisons` caption is stale as a result. Adding the row is a content decision that has not been taken |

**C8 — the defensible ladder sentences.** *With the trained-head consultant every loop
comparison is separated on both datasets.* *With the prototype consultant nothing is
separated on either.* *AGAF is not separated from the GNN rung under either consultant.*
Status: **SEP** for C3, C5, C6; **NOT-SEP** for C1, C2, C4.

**C9 — the one procedure disagreement.** UNSW AGAF − GNN is separated seed-matched
(+0.0356 [+0.0038, +0.0678]) and not separated two-level. The pre-registered rule makes the
two-level result decide. Both are shown in the manuscript, because the disagreement is the
clearest illustration of what admitting seed variance costs. **NOT-SEP.**

**C10 — what the seed std is and is not.** It is training-seed variance at a fixed fold
partition. Fold-partition variance is unmeasured, so ±std is a lower bound on total
uncertainty. Say so wherever a ± appears.

**C11 — the consultant change itself.** +0.0697 UNSW and +0.0523 ToN, three-seed means. The
`[GAP]` was closed on 2026-09-13 by `results/consultant_change_interval.json` and the interval
regenerated on 2026-09-16 under the independent draw: **+0.0701 [+0.0353, +0.1057] SEP** on
UNSW, **+0.0511 [−0.0082, +0.1110] NOT-SEP** on ToN. The change stays confounded with the
re-selected entropy percentile (31→29, 25→16), so every statement of it must say so, and the
ToN side must never be called separated.

**C12 — accuracy and weighted F1 (added 2026-09-16).** Reported in `tab_metrics` and
`fig_metrics`, from `rungs.*.accuracy` and `rungs.*.weighted_f1` in the two ladder aggregates,
scored on the rows the metric scores. **PT** — no interval was computed for any of them, and
none may be described as separated.

| rung | UNSW acc / wF1 | ToN acc / wF1 |
|---|---|---|
| GNN | 0.8105 / 0.8207 | 0.8739 / 0.8949 |
| semantic | 0.7790 / 0.7914 | 0.4995 / 0.6071 |
| fusion | 0.8364 / 0.8398 | 0.8329 / 0.8550 |
| feedback, trained head | 0.8852 / 0.8887 | 0.9186 / 0.9223 |

**C13 — what accuracy may and may not be used to say.** *The feedback model is the highest rung
on all three metrics on both datasets.* **PT**, permitted. *Accuracy separates the rungs.*
**FORBIDDEN** — on NF-ToN-IoT the benign class holds 82% of the scored edges, every graph rung
clears 0.83, and no accuracy difference carries an interval. The reason macro-F1 is the
headline must appear wherever accuracy does.

---

## D. The consultant change — RESOLVED 2026-09-07, audited 2026-09-11

| ID | Claim | Evidence | Status |
|---|---|---|---|
| D1 | The loop's consultant is a design variable, not a rung. It is reported under both the whitened prototype scorer and the per-fold trained head. | `consultant_decision`, both `*_current.json` | **PT** |
| D2 | "The feedback loop is the strongest rung." | C3, C5, C6 | **PERMITTED with the trained head, on both datasets**; **FORBIDDEN** for the prototype consultant, where nothing is separated |
| D3 | Wording that is true on both datasets | — | *"with the trained-head consultant the loop is separated above every other rung; with the prototype consultant no comparison is separated"* |
| D4 | The 3-seed rung re-run is done. Fold-partition variance remains the open follow-up. | archive 2026-09-06 | **PT** — stated as a limitation in Part 5 |
| D5 | The schema-3/5 AGAF and loop values do not reproduce under current code at seed 42 (GNN and LLM do, bit-exactly). Cause not established. | `supersedes.reason`, both contracts | **PT** — a reproducibility finding; do not guess the cause |
| D6 | Under the prototype consultant the injected bias was ~0.02–0.08 on unit-variance features, so that mechanism was effectively off and the loop's edge over `head_only` was late fusion. Under the trained head it is ~1.33. | archive 2026-09-06 Finding 3; archive 2026-09-07 head-echo table | **PT** — Appendix A |
| D7 | Switching the two signals on (selector-head loss, calibrated temperature) made the loop worse at 3 seeds on both datasets. | `results/multiseed_ladder_v2_{fixed,fixed_reselected}.json` | **PT** — Appendix A, conditions A/B/C |
| D8 | The head standing alone is NOT reported as a comparative rung. The author's scope decision, 2026-09-11. | — | **SCOPE** — the honest qualifier is carried instead by the isolation experiment (§H), which shows the injection path contributes nothing separated with either consultant |

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

### E.2 Our rungs vs the baselines — three seeds on both sides, 2026-09-07

Superseded: the seed-42-rung intervals that used to fill this block (UNSW loop +0.0535,
ToN loop +0.0340 and the rest) are in `superseded.statistical_comparisons` and must not be
quoted. The loop is now given under both consultants.

| ID | Contrast | Value | Status |
|---|---|---|---|
| E5 | UNSW: loop (trained head) vs TE-G-SAGE + our node features | +0.1155 [+0.0673, +0.1627] | **SEP** |
| E6 | UNSW: loop (prototype) vs same | +0.0453 [+0.0033, +0.0847] | **SEP** |
| E7 | UNSW: GNN vs same | +0.0249 [−0.0270, +0.0755] | **NOT-SEP** |
| E8 | UNSW: LLM vs same | +0.0163 [−0.0291, +0.0607] | **NOT-SEP** |
| E9 | UNSW: AGAF vs same | +0.0600 [+0.0096, +0.1093] | **SEP** |
| E10 | **UNSW headline**: the loop is separated above every faithful baseline under either consultant; the GNN and LLM rungs are not separated from the strongest one. | E5–E9 | **SEP** / **NOT-SEP** as marked |
| E11 | ToN: loop (trained head) vs E-GraphSAGE as published | +0.0890 [+0.0312, +0.1480] | **SEP** |
| E12 | ToN: loop (prototype) vs same | +0.0378 [−0.0098, +0.0857] | **NOT-SEP** |
| E13 | ToN: loop (trained head) vs E-GraphSAGE + node features | +0.0667 [−0.0181, +0.1505] | **NOT-SEP** two-level, **SEP** seed-matched (+0.0679 [+0.0189, +0.1156]); non-faithful variant |
| E14 | ToN: GNN vs E-GraphSAGE as published | +0.0203 [−0.0230, +0.0633] | **NOT-SEP** |
| E15 | ToN: AGAF vs E-GraphSAGE as published | −0.0061 [−0.0797, +0.0777] | **NOT-SEP** |
| E16 | **ToN: LLM vs E-GraphSAGE as published** | **−0.1336 [−0.1773, −0.0939]** | **SEP-ADVERSE** |
| E17 | ToN: LLM vs TE-G-SAGE + node features | −0.0724 [−0.1233, −0.0251] | **SEP-ADVERSE** |
| E18 | **ToN headline**: the consultant change is what moves the loop across the line against E-GraphSAGE. Our structural rungs match that baseline rather than improving on it, and the prototype LLM rung is separated below it. | E11–E17 | must be reported, not softened |
| E19 | The intervals are the **two-level bootstrap with BOTH sides at three seeds** (rung seed and baseline seed drawn independently, edges resampled). Seed-matched intervals are retained alongside. | `statistical_comparisons_3seed` / `seed_matched_comparisons_3seed`, schema 2 `[verified]` | **PT** |
| E20 | The earlier claim "the loop is the *only* rung separated from the strongest UNSW baseline" is withdrawn: at three seeds AGAF is separated there too. | E9 | **RETRACTED** |

### E.3 The ~+0.60 UNSW margins

| ID | Claim | Evidence | Status |
|---|---|---|---|
| E21 | UNSW margins against E-GraphSAGE of ≈+0.60 are **ceiling-limited by the endpoint-only edge representation**, not architectural evidence. | `known_ceilings.e_graphsage_endpoint_only.reporting_rule` `[verified]`; archive §5.1, §5.5 | **PT** — do not lead with these numbers |

### E.4 Forced deviations D1–D5

| ID | Claim | Evidence | Status |
|---|---|---|---|
| E22 | Five deviations from the source papers were forced by our representation: no chronological split (D1); fanout/batch exceed the graph (D2); fixed epoch schedules replaced by 300 epochs / patience 25 (D3); `rare_min_freq=50` erases port categories at our scale (D4); E-GraphSAGE Eq. 4 vs released `W_msg([h_u ‖ e_uv])` (D5). | `deviations` list in both contracts; archive §5.4 `[verified]` | **PT** — reproduce the full table in Part 3 |
| E23 | Baselines used the same rows, labels, folds and evaluation classes as our rungs. | `data_provenance` block with SHA-256 of `aggregated_edges.csv` and `folds.pt` `[verified]` | **PT** — the hashes are worth quoting |
| E24 | Each paper's own categorical featurisation was preserved. | `preprocessing` block `[verified]` | **PT** |

### E.5 The NF-ToN-IoT scoring asymmetry (2026-09-16 audit, point 2)

| ID | Claim | Evidence | Status |
|---|---|---|---|
| E25 | On NF-ToN-IoT the fusion model and all six re-trained baselines were scored **without** the dropped-class mask that the GNN, semantic and feedback models use, so only they could spend a scored edge on `dos` or `ransomware`. | `results/ton_iot_dropped_class_bound.json`; `train_fusion.py`, `train_gnn.py`, `harness.py` now mask, pinned by `tests/test_dropped_class_masking.py` | **PT** — must be disclosed wherever the ToN baseline margins are stated |
| E26 | Scored edges lost that way, per seed 42/1/2: fusion 3/374/6; E-GraphSAGE as published and refit 58/63/47; E-GraphSAGE + features 66/28/60; TE-G-SAGE as published 83/161/206; TE-G-SAGE + features 51/116/73; TE-G-SAGE refit 81/32/29. | same artifact, `models.*.per_seed` | **PT** `[verified 2026-09-16]` |
| E27 | Crediting every such prediction with the true label bounds the repair. The bound stays **below** the feedback model at every seed for the fusion model and five of six baselines. | same artifact, `macro_f1_upper_bound` | **PT** — it is an upper bound, never an estimate |
| E28 | The one exception is E-GraphSAGE + features at seed 1, bound 0.5206 against the feedback model's 0.5012. That comparison is **already** reported as not separated, `+0.0667 [−0.0181, +0.1505]`. | `bounds_that_exceed_the_feedback_model`; `ton_iot_sota_baselines.json:statistical_comparisons_3seed.feedback_vs_e_graphsage_plus_node_features` | **NOT-SEP** — therefore no claim in the report depends on the asymmetry |
| E29 | "The asymmetry was corrected and the numbers re-measured." | — | **FORBIDDEN** — nothing was re-run; the code was fixed and the effect bounded |
| E30 | On NF-ToN-IoT the GNN model selected checkpoints on 10-class validation macro-F1 while the feedback model and baselines selected on 8. Fixed in `train_gnn.py`; published numbers predate the fix. | §3.9 disclosure; `train_gnn.EVAL_CLASSES` now set from the dataset config | **PT** — disclose, do not re-run |

---

## F. Decomposition of the naive UNSW gap

| ID | Claim | Evidence | Status |
|---|---|---|---|
| F1 | The staircase 0.3841 → 0.5842 → 0.7196 → 0.7644 (loop, 3-seed mean; AGAF 0.7793) | E2 + B1, all `[verified]` | **PT** — every rung of the staircase is artifact-backed |
| F2 | Attribution: +0.200 to fair tuning, +0.135 to our feature engineering, +0.053 to our architecture | arithmetic on F1 | **DERIV** |
| F3 | ≈51% tuning budget / ≈35% feature engineering / ≈14% architecture | arithmetic on F2 (0.2001/0.1354/0.0532 over a 0.3887 total; 51.5 / 34.8 / 13.7%) | **DERIV** — **no interval exists for any share.** Present as a decomposition of point estimates, never with a significance claim |
| F4 | Only the final step (Loop vs TE-G-SAGE + features) carries an interval, and it is separated: **+0.0453 [+0.0033,+0.0847]** two-level / +0.0446 [+0.0162,+0.0740] seed-matched at 3 seeds on both sides (was +0.0535 with the rung at seed 42 only). | E5, archive §5.3 | **SEP** |
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

## B'. The loop's consultant (2026-09-07)

| ID | Claim | Evidence | Status |
|---|---|---|---|
| B3 | The loop's semantic consultant is the per-fold trained head on CySecBERT embeddings; the LLM rung and AGAF keep the whitened prototype encoder. Parity rule: same encoder, graph, folds and seeds on every rung; the loop additionally trains a classification head. | `consultant_decision` and `architecture_parity_rule` in both contracts `[verified]` | **PT** |
| B4 | The loop is separated above AGAF, the GNN and the prototype LLM rung on BOTH datasets. | `multi_seed.separated_comparisons_two_level` in both contracts `[verified]` | **SEP** |
| B5 | The loop is NOT separated from the head alone it consults: −0.0030 [−0.024,+0.014] UNSW, −0.0028 [−0.039,+0.032] ToN, sign unstable across seeds; head alone has the higher mean on both. | `multi_seed.comparisons.feedback_vs_head_alone`, `loop_vs_head_alone_headline` `[verified]` | **NOT REPORTED** — author's scope decision 2026-09-11: the head is a component of the loop, not a comparative rung. The qualifier B4 needs is carried instead by the isolation experiment (§H), which is stronger because it holds for both consultants |
| B6 | Five mechanism treatments on the prototype consultant were tried and none was kept (selector-head supervision, consultant temperature, reliability weighting, advice format, cross-fitting). | archive 2026-09-06 and 2026-09-07 sections `[verified]` | **PT** — this is the justification for B3 |
| B7 | The head-consultant knobs are selected, not tuned: both curves flat within noise, ToN's injection_scale on the swept range's upper boundary. | `configuration.selection_curve_is_flat`, `scale_on_range_boundary` `[verified]` | **PT** — never write "tuned" |

---

## P. Coverage check against the report's own research questions

| RQ | Answerable from | Answer available |
|---|---|---|
| RQ1 — individual branch performance | B1, B2 | Yes |
| RQ2 — does AGAF beat both single branches | C1, C2 | Yes, and the answer is **no on UNSW** (not separated) and **no on ToN** (separated the wrong way) |
| RQ3 — does feedback add value beyond static fusion, and from where | C3, K1, K4, H1–H6 | Yes, and the answer separates the two sub-questions: measurable at ladder level with fusion on, not attributable to the injection path |
| RQ4 — effect of topology, repeated endpoint pairs, imbalance, consultant strength | G3, I2, I3, J8, J9, H3 | Yes — repeated endpoint pairs and consultant strength are the strongest evidence; imbalance is v1-only |
| RQ5 — vs published architectures re-trained on the same representation, and how much is representation/tuning vs architecture | E1–E19, F1–F5 | Yes — this is the best-evidenced RQ |

All five questions are answered in the manuscript, and the answers to RQ2 and RQ3 are
negative. RQ2 was reworded in W6 to drop "outperform" and "robust" for the locked
vocabulary. Both datasets are in scope throughout; the single-dataset wording that used to
sit in RQ1, RQ2, RQ4 and the structure paragraph is gone.

---

## Q. Open items — reviewed 2026-09-11 (W8)

Closed since the original pass:

1. `EVIDENCE_STATE.md` now exists and is reconciled against this file.
2. `references.bib` holds 26 entries and every cited key resolves. No citation is missing.
3. `tables/` is populated: nine fragments generated from the contracts by
   `tables/make_tables.py`, with `--check` wired into `build.sh`.
4. The two stale `limitations` entries in `results/{unsw_nb15,ton_iot}_current.json`
   ("No rung comparison is statistically separated…" and the Finding-3 attribution sentence)
   were corrected to the schema 5/7 facts, with the old text kept under `supersedes`.
5. `REPORT_WORKFLOW.md` scope note and section-state table were brought into line with the
   two-dataset report.

Still open:

6. `figures/` is empty. All nine figures are being drawn externally, one at a time, against
   `figures/FIGURE_BRIEFS.md`. The manuscript compiles without them and each is wired in as
   it lands.
7. Three `[GAP]` markers stand in the manuscript: no interval between the two loop
   configurations, no loop-versus-semantic comparison for the prototype consultant, and no
   recorded consultant-versus-GNN disagreement split except for NF-ToN-IoT with the trained
   head.
8. Archive §1.3 calls the ToN random ablation "separated" against its own contract's CI.
   That section is superseded and is not cited by the manuscript, but the archive text is
   still wrong.
9. `PROJECT_NOTES.md` still carries the rule "Report `head_alone` in every table", which the
   manuscript no longer follows after the 2026-09-11 scope decision. `PROJECT_NOTES.md` is out of
   git by design and was not edited; the divergence is recorded here and in `HANDOFF.md`.
10. Fold-partition variance is still unmeasured. It is stated as a limitation rather than
    resolved.
