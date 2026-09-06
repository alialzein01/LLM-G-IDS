# Results Archive

Full results history, findings, retractions and reproducibility record for LLM-G-IDS.

**Why this file exists:** `PROJECT_NOTES.md` is read at the start of every working session, so it must
stay short. This file holds the complete narrative — current results, superseded results,
what was tried, what failed, and what was corrected — and is read on demand. Nothing here
is a duplicate; `PROJECT_NOTES.md` carries the *rules and traps*, this carries the *record*.

**Authoritative numbers always live in `results/*.json`.** If this file and a contract
disagree, the contract wins and this file is stale.

---

## 0. Headline finding (2026-08-30)

> ### The feedback channel works. Neither realistic consultant uses it reliably.
>
> This is the central result of the Gate 0 investigation and it redirects the project.
>
> Gate 0 handed the feedback mechanism a **perfect consultant** (the true label at ±4 nats)
> with output fusion disabled, so the advice could only reach the GNN through the feedback
> path being measured. It delivered a large, statistically separated gain on **both** datasets:
> **+0.0345 on UNSW-NB15** (P=0.995) and **+0.1090 on NF-ToN-IoT** (P=1.0).
>
> The **canonical whitened-prototype consultant captures essentially none of it** —
> **−4.5% of that headroom on UNSW** and **11.9% on ToN**, neither separated from zero.
>
> Gate 0.5 then replaced the prototype with a much stronger, leakage-free per-fold trained
> head while keeping output fusion off. It captured only **12.5%** of pooled oracle headroom
> on UNSW (unseparated from control) and **−38.0%** on ToN (a statistically separated
> regression). Standalone classifier quality therefore does **not** translate into useful
> injected advice.
>
> **Refined conclusion:** mechanism capacity is not the limit, but prototype quality alone is
> not the full explanation. The unresolved failure is the compatibility/calibration of
> realistic consultant logits with the bias path: even a strong classifier cannot exploit the
> capacity that the oracle demonstrates.
>
> **Consequences for the work plan:**
> - Replacing the prototype with a stronger classifier is **not sufficient**. Do not treat
>   trained-head accuracy as an upper bound on realistic mechanism gain.
> - Of the queued options, calibration/trust work (P2′) is now more directly motivated than a
>   simple consultant swap. P1′/P3′ remain experiments, not fixes established by Gate 0.
> - This is independently corroborated by §2.2 — the mechanism-only `real > control` claim
>   also failed to replicate.
>
> See §2.4 for Gate 0, §2.5 for the trained-head diagnostic, and §2.2 for the retraction.

---

## 1. Current ladder (both datasets, v2 encoding)

**Regenerated 2026-08-27** directly from live pipeline artifacts
(`data/{dataset}/processed/step4_feedback/{ladder_summary,ablation_summary}.json`) — every
number below, the three `results/*.json` contract files, and `tests/test_current_results.py`
were cross-checked against each other and against the on-disk `.pt` graphs in that pass.

Pooled OOF macro-F1. Class counts differ (UNSW 10, ToN 8) — absolute levels are **not**
comparable across datasets, only ladder shape.

| Dataset | top-k | scale | GNN | LLM | AGAF | Loop | Ladder holds |
|---|---:|---:|---:|---:|---:|---:|:--:|
| UNSW-NB15 (10 cls) | 31.0 | 2.0 | 0.7219 | 0.7353 | 0.7595 | **0.7728** | yes, by order (loop−AGAF CI touches zero, P=0.778) |
| NF-ToN-IoT (8 cls) | 25.0 | 20.0 | 0.4290 | 0.2785 | 0.3334 | **0.4478** | no — AGAF < GNN (P=0.0005); loop is the highest point estimate |

Contracts: `results/unsw_nb15_current.json`, `results/ton_iot_current.json`,
`results/cross_dataset_comparison.json`.

UNSW, all 3 metrics: GNN acc 0.7957 / F1 0.7219 / wF1 0.8057; LLM 0.7790 / 0.7353 / 0.7914;
AGAF 0.7942 / 0.7595 / 0.8100; Loop 0.8308 / 0.7728 / 0.8393.

### 1.1 Significance — no adjacent rung step is separated on UNSW

| comparison | UNSW | ToN |
|---|---|---|
| AGAF − GNN | +0.0373 CI[−0.0004, +0.0764] P=0.974 — *touches zero* | −0.0962 CI[−0.1489, −0.0441] P=0.0005 — **separated, wrong direction** |
| AGAF − LLM | +0.0240 CI[−0.0048, +0.0543] P=0.949 — *crosses* | +0.0526 CI[+0.0134, +0.0951] P=0.996 |
| Loop − AGAF | +0.0135 CI[−0.0197, +0.0462] P=0.778 — *crosses* | +0.1151 CI[+0.0594, +0.1670] P=1.0 |
| Loop − GNN | +0.0507 CI[+0.0193, +0.0829] P=0.998 | +0.0189 CI[−0.0139, +0.0534] P=0.862 — *crosses* |
| Loop − LLM | +0.0374 CI[+0.0036, +0.0680] P=0.987 | +0.1677 CI[+0.1276, +0.2103] P=1.0 |

**On UNSW not a single adjacent rung step is separated.** Only the non-adjacent Loop−GNN and
Loop−LLM are clean. The defensible UNSW claim is *"the loop beats either single modality"* —
not *"each rung improves on the one below."* **On ToN the loop is not separated from the GNN**
(P=0.862), which is ToN's second-best rung. Say "highest point estimate", never "top rung".

### 1.2 The ToN reversal

Under v1 encoding, AGAF (0.4447) was the top ToN rung and the loop sat significantly below it.
Under v2 + the loop-fusion fairness fix, **AGAF regressed to 0.3334, significantly below the
bare GNN** (agaf_vs_gnn −0.0962, P=0.0005) — the GNN improved with everything else fixed,
AGAF got worse. The loop is unaffected and is now ToN's highest rung (0.4478), significantly
above AGAF (+0.1151, P=1.000).

**Do not describe AGAF as ToN's strongest rung, or the loop as underperforming AGAF on ToN —
both are reversed from the pre-v2 finding.** AGAF's regression is **undiagnosed**; do not guess
a cause without checking. Candidates: the fusion-dim fix changed AGAF's effective capacity, or
v2's port/protocol embeddings interact badly with ToN's much sparser graph (1501 nodes vs
UNSW's 49).

### 1.3 ToN loop reads the advice (ladder-level ablation)

Real feedback loop 0.4478 vs `random`-advice ablation 0.4046, mean_diff +0.0428, P=0.972 —
separated, unlike the v1-encoding run where real and random were statistically
indistinguishable. See `feedback_ablations` in `results/ton_iot_current.json`. Note this is the
*ladder-level* ablation (fusion on), which is a different experiment from the mechanism-only
test in §2.

---

## 2. Feedback mechanism — the causal test

### 2.1 Attention injection is inert

`--injection-mode attention` produces churn of exactly 0.0000 in every configuration tested,
**even handed the true label at ±4 nats.**

Cause: attention only changes node embeddings, but 58.7% of UNSW edges (100% of consulted ones)
share a `(src,dst)` pair, so node-derived terms are identical and only `edge_attr` distinguishes
them — attention never touches `edge_attr`. On ToN, 62.8% of edges have dst in-degree 1, where
softmax-over-incoming-edges is mathematically 1.0 regardless of bias — a second, independent
reason attention cannot work there. Structural finding, independent of edge encoding.

Confirmed a third time under v2 by Gate 0 (§2.4): `oracle_attention` − control = −0.0016
(P=0.482) UNSW, +0.0114 (P=0.870) ToN. Neither separated from zero.

### 2.2 Edge injection — `real > control` RETRACTED 2026-08-30

The 2026-08-27 mechanism-only run reported `real > control > shuffled > random` on both
datasets and was cited as the causal proof that "the LLM is a working consultant."
**That run was never thread-pinned** (`OMP_NUM_THREADS` unset; the runner only began pinning
on 2026-08-30). Documented drift at fixed seed without pinning is ~0.012 macro-F1.

Same nominal configuration, two runs, same `pooled_5_fold_oof_macro_f1` metric, same 3 seeds:

| dataset | run | control (`head_only`) | real | shuffled | random |
|---|---|---:|---:|---:|---:|
| UNSW | 2026-08-27, **not pinned** | 0.7280 | 0.7344 (**+0.0064**) | 0.7181 (−0.0098) | 0.6762 (−0.0518) |
| UNSW | 2026-08-30 Gate 0, **pinned** | 0.7398 | 0.7382 (**−0.0016**) | not run | not run |
| ToN | 2026-08-27, **not pinned** | 0.4359 | 0.4386 (**+0.0026**) | 0.4131 (−0.0228) | 0.3767 (−0.0593) |
| ToN | 2026-08-30 Gate 0, **pinned** | 0.4195 | 0.4325 (**+0.0130**) | not run | not run |

Absolute levels moved 0.012–0.016 — exactly the drift magnitude — and `real − control` **flips
sign on UNSW** (+0.0064 → −0.0016) while **growing on ToN** (+0.0026 → +0.0130). Not a
systematic bias in either direction: noise. Gate 0's paired per-(seed,fold) bootstrap agrees —
UNSW −0.0046 CI[−0.0273, +0.0158] P=0.356; ToN +0.0099 CI[−0.0111, +0.0286] P=0.836.

**What survives:** the mechanism is **sensitive to advice content**. `random` degrades well
beyond drift on both (−0.0518 UNSW, −0.0593 ToN) and `shuffled` does on ToN (−0.0228). UNSW's
`shuffled` (−0.0098) is inside the drift band and unresolved.

**What does NOT survive:** `real > control`. Corrupting the advice hurts; supplying it does not
measurably help versus no advice at all.

Contract `results/unsw_nb15_edge_injection_v2.json` carries a `retraction` block. A
thread-pinned 4-arm re-run (Gate 0 did not measure shuffled/random) is what would settle
UNSW's shuffled leg.

**Superseded pre-v2 numbers** (UNSW only, different mechanism variant): `edge_trained_head`
0.5822 vs `control_head_only` 0.5558 (+0.0264) — `results/unsw_nb15_edge_injection.json`, not
re-verified on v2.

### 2.3 The loop reaches a fixed point after one correction

`semantic_logits` is computed once at `src/models/feedback_classifier.py:1056`, **outside** the
iteration loop at `:1066`. Across iterations only the flagged set changes, never the advice
content. Observed iter-3 churn (from `feedback_trace_real.json`): UNSW 0.000 / 0.006 / 0.000 /
0.000; ToN 0.008 / 0.006 / 0.005 / 0.001.

This is **empirical, not mathematical** — the flagged set does change each cycle, so a loop
could in principle keep moving. Do not state it as "by construction."

**Do not sweep `max_iterations` or lower `churn_tol`.** Already measured:
`results/unsw_nb15_edge_injection.json` contains the sweep — `max_iterations_5` → 0.7327,
`max_iterations_8` → 0.7501, actual iterations plateauing at **3.07** in both, its own reading
"Keep 3. More iterations do not help."

### 2.4 Gate 0 — oracle ceiling under v2 (2026-08-30, thread-pinned, both datasets)

Question: if the semantic consultant were perfect, how much could the *mechanism* deliver?
Method: consultant class logits replaced by one-hot true label at ±4 nats through the same bias
path. Deliberate label leakage — **diagnostic only, never reportable as performance.** Output
fusion off in every arm, so even the oracle reaches the GNN only through the mechanism.

| arm − `control_head_only` | UNSW | ToN |
|---|---|---|
| **`oracle_edge`** | **+0.0345** (paired +0.0369, CI[+0.0097,+0.0618], P=0.995) | **+0.1090** (paired +0.1354, CI[+0.0919,+0.1797], P=1.0) |
| `real_prototype_edge` | −0.0016 (paired −0.0046, P=0.356) | +0.0130 (paired +0.0099, P=0.836) |
| `oracle_attention` | −0.0016 (P=0.482) | +0.0114 (P=0.870) |
| prototype's captured share of headroom | **−4.5%** | **11.9%** |

**The channel works; the consultant does not use it.** A perfect consultant gets a cleanly
separated gain through the edge channel on both datasets. The canonical whitened prototype
captures ≈0% of it, and neither prototype figure is separated from zero. **This is a
consultant-quality finding, not a mechanism-capacity one.**

Contracts: `results/unsw_nb15_oracle_ceiling_v2.json`, `results/ton_iot_oracle_ceiling_v2.json`.
Raw: `results/raw/oracle_ceiling_v2.{json,log}`. Runner:
`python -m src.pipeline.step4.mechanism_only_edge_injection`.

**Superseded pre-v2 oracle** (`results/unsw_nb15_oracle_ceiling.json`, top_k=16, scale=10, UNSW
only): headroom +0.0417 (control 0.5558 → oracle_edge 0.5975); prototype captured +0.0028 (7%),
trained head +0.0264 (63%); `oracle_attention` 0.5491, *below* control.

### 2.5 Gate 0.5 — trained-head consultant diagnostic (2026-08-30)

Question: does a strong but realistic trained-head consultant close a meaningful share of the
v2 oracle gap? The per-fold heads were regenerated from the current graph, embeddings, and
`folds.pt`; each fold's exact `[E,C]` slice was used so test rows remain OOF. Output fusion was
off, making direct head echoing structurally impossible. **Diagnostic only: this arm is not a
ladder rung and does not change the canonical prototype architecture.**

| arm / comparison | UNSW | ToN |
|---|---:|---:|
| control | 0.7398 | 0.4195 |
| prototype edge | 0.7382 | 0.4325 |
| **trained-head edge** | **0.7441** | **0.3781** |
| oracle edge | 0.7743 | 0.5286 |
| trained-head pooled gain | +0.0043 | −0.0414 |
| captured share of oracle headroom | **12.5%** | **−38.0%** |
| paired trained-head − control | +0.0007, CI[−0.0183,+0.0201], P=0.525 | −0.0378, CI[−0.0588,−0.0167], P=0.0001 |

**The trained head does not close a meaningful share of the gap.** On UNSW its small point gain
is unseparated from zero. On ToN it significantly harms the mechanism despite scoring 0.5165
standalone against the prototype's 0.2785. The oracle result therefore cannot be reached merely
by substituting a more accurate classifier. The open problem is how consultant logits are
calibrated, selected, and translated into edge bias—not whether the bias channel has capacity.

Contracts: `results/{unsw_nb15,ton_iot}_oracle_ceiling_v2_trained_head.json`. Raw:
`results/raw/oracle_ceiling_v2_trained_head.{json,log}`. Regeneration logs:
`results/raw/build_llm_heads_{unsw,ton}_gate05.log`.

---

## 3. Corrections and retractions

Kept deliberately — these are the record of what went wrong and how it was caught.

### 3.1 Fabricated oracle numbers (corrected 2026-08-28)

`PROJECT_NOTES.md` carried, for some time: *"Oracle mechanism ceiling on UNSW: +0.0508 (control 0.7237
→ oracle_edge 0.7745) … the prototype consultant captures +0.0173"*, citing
`results/unsw_nb15_edge_injection.json`.

**Those values appear in no artifact.** `+0.0508` is the `feedback_minus_gnn` delta from
`results/cross_dataset_comparison.json:63` — the loop minus the GNN rung — relabelled as oracle
headroom, with `0.7237`/`0.7745` back-derived to be consistent with it. The cited file's
`oracle_headroom` block is a *different* experiment entirely (fusion routing vs the trained
head, verdict "NO"). The real contract was `results/unsw_nb15_oracle_ceiling.json` all along:
+0.0417, prototype +0.0028.

Caught in external review. **Do not reintroduce those numbers.**

### 3.2 `real > control` retracted (2026-08-30)

See §2.2. Root cause: missing thread pinning, not a statistical-method difference.

### 3.3 "Top rung" overclaim (2026-08-30)

Neither top comparison is separated — UNSW Loop−AGAF crosses zero (P=0.778), ToN Loop−GNN
crosses zero (P=0.862). See §1.1.

### 3.4 Refuted mechanism premises (2026-08-28)

Two premises used to justify a proposed "widen the advice channel" experiment were wrong:

- **"The 10-class-logit channel is a bottleneck"** — refuted. The oracle pushed a *perfect*
  answer through that same 10-logit path and delivered +0.0417. Ten numbers are demonstrably
  sufficient. Widening to 768-d would also grow the projection ~430 → ~30,000 params against
  393 UNSW train edges.
- **"36 of 39 injection dims have no continuous semantics"** — overstated. Protocol/port
  embeddings are learned continuous latent vectors; adding a learned residual to them is not
  inherently meaningless. The open question is narrower: dedicated advice channel vs
  all-dimension injection.

### 3.5 Reproducibility failure (2026-08-20)

The previously published ToN prototype ladder (GNN 0.3398, Loop 0.4538, "holds") **does not
reproduce** — re-running unchanged code/graph/seed gives GNN 0.3271, Loop 0.3876. LLM and AGAF
rungs reproduce exactly; only the two rungs that made the ladder hold didn't. UNSW's contract
re-verified exact on all 4 rungs.

---

## 4. Other verified findings

- **The GNN rung's old weakness was the v1 encoding bug, not missing features.** Step1
  Z-scored only `edge_attr[:,:3]`, leaving port (std 11,359) and protocol (std 62) raw, up to
  11,000× the scale of everything else despite being categorical. Fixing it moved GNN
  0.5496 → 0.7219. A flat MLP on the raw v1 tensor scores 0.5631 — reproducing the entire GNN,
  i.e. GATv2/message-passing bought nothing over badly-scaled numbers. Don't propose new
  node/edge features (e.g. FedGATSage traffic statistics) as the fix for a weak GNN rung.
- **AGAF is capacity-limited on UNSW, not overfitting-limited** (207,756 params / 393 train
  edges = 529/example). Every "lightweight" variant (dim 64/32, scalar fusion) costs ≥0.08
  macro-F1 there and drops AGAF below GNN and LLM. **Do not shrink AGAF to make a ladder
  hold** — it inverts the ladder by destroying the middle rung. Predates v2 and ToN's AGAF
  regression; not re-run on either.
- **Trained-head LLM-only baseline beats the full system on both datasets** (UNSW 0.8321 vs
  loop 0.7728; ToN 0.5165 vs loop 0.4478) — an MLP on CySecBERT KG-text embeddings outperforms
  the complete architecture. The per-fold logits were regenerated against the current graph,
  embeddings, and folds for Gate 0.5 on 2026-08-30 and reproduced these exact pooled OOF scores.
  The older `results/{unsw_nb15,ton_iot}_head_baseline.json` files remain dated 2026-08-20, but
  the two headline head scores are now independently re-verified under current provenance.
- **Swapping the trained head in as consultant degenerates the loop.** With the head throughout,
  loop 0.8258 < LLM-head 0.8321 ≈ AGAF-head 0.8331 — the loop echoes its consultant rather than
  improving on it. This, plus cross-dataset comparability, is why the canonical rule mandates
  the prototype scorer. Pre-v2, fusion on — a real prior, not a settled result.
- **GraphSMOTE-style oversampling (`--oversample-ratio`) is off by default, verified
  inert/harmful on v1 ToN** — hurt AGAF (−0.073 macro-F1, ~4× variance at ratio 0.05). Not
  re-verified on v2, and AGAF's ToN behaviour has since changed substantially. **Never combine
  oversampling with the existing inverse-frequency class weights** — double-corrects (collapsed
  AGAF to 0.0227 in the v1 test); use `splits.class_weights_from_labels` recomputed on the
  balanced distribution. See `results/ton_iot_balancing.json`.
- `assemble_ladder` doesn't persist the LLM rung's OOF predictions — LLM accuracy/weighted-F1
  are only readable from `ladder_summary.json`, never recomputed.
- ToN classes 3 (`dos`, 4 edges) and 7 (`ransomware`, 3 edges) stay in-graph for message passing
  but are excluded from the metric (`DatasetConfig.eval_classes`). **A 10-class ToN macro-F1 is
  not a reportable number.**
- Edge aggregation is label-conditioned through `(src_ip, dst_ip, attack_type)` on both datasets.
- **Nondeterminism:** runs drift ~0.012 macro-F1 at fixed seed unless `OMP_NUM_THREADS=1`. That
  is larger than most effects measured in this project. Always export it.

---

## 5. SOTA baseline comparison (2026-09-02)

Two published graph-based intrusion-detection architectures were re-trained on the same
aggregated edge rows, fold assignments, evaluation classes, and pooled out-of-fold
macro-F1 protocol as the project ladder:

- **E-GraphSAGE** (Lo et al., NOMS 2022): supervised edge-level GraphSAGE whose edge
  classifier consumes `CONCAT(h_u, h_v)`.
- **TE-G-SAGE** (2025): supervised edge-aware GraphSAGE whose edge head also consumes
  the edge feature vector.

Three candidates were rejected before running: **Anomal-E** is a binary anomaly detector
and cannot produce the required multiclass macro-F1 as published; **XG-NID** requires both
flow and packet modalities, while this project has no packet data; **GTCN-G** had neither
verified code availability nor a verified matching dataset.

**Claim boundary (verbatim from both results contracts):**

> Published architectures re-trained on our aggregated flow-graph representation. NOT a
> comparison against their published numbers, which were obtained on per-flow graphs
> ~300x larger.

### 5.1 Headline findings

- On UNSW, the closest comparison is TE-G-SAGE + our node features. GNN `+0.0027`,
  LLM `+0.0160`, and AGAF `+0.0400` are not separated from zero. Only feedback
  `+0.0535`, CI `[+0.0102, +0.0990]`, is separated.
- On ToN, E-GraphSAGE as published is significantly ahead of the LLM (`-0.1338`) and
  AGAF (`-0.0812`) rungs. GNN and feedback are not separated from it.
- The approximately `+0.60` UNSW margins against E-GraphSAGE are ceiling-limited by the
  endpoint-only edge representation and are not evidence of architectural superiority.

**CI caveat (verbatim from both results contracts):**

> Our four rungs were run at seed 42 only, so no rung-side training stochasticity enters
> these intervals. They are therefore NARROWER than a symmetric multi-seed comparison
> would give. Re-running the ladder at seeds 1 and 2 is the highest-value follow-up.

### 5.2 Baseline point estimates

Values are mean pooled OOF macro-F1 over seeds 42, 1, and 2. `plus_node_features` is an
explicit non-faithful ablation, not the published architecture.

| Dataset | Architecture | as published | refit | + our node features |
|---|---|---:|---:|---:|
| UNSW | E-GraphSAGE | 0.1194 | 0.1368 | 0.1173 |
| UNSW | TE-G-SAGE | 0.3841 | 0.5842 | 0.7196 |
| ToN | E-GraphSAGE | 0.4141 | 0.4141 | 0.4358 |
| ToN | TE-G-SAGE | 0.1931 | 0.2345 | 0.3523 |

### 5.3 Full two-level bootstrap comparison

Sign convention: `delta = project rung - re-trained baseline`. Positive values favour the
project rung; negative values favour the baseline. Each cell is `delta [95% CI];
P(delta > 0)`. The primary bootstrap resamples edges and draws one of the three baseline
seeds, thereby propagating edge-sampling and baseline-seed variance.

**UNSW CI caveat (verbatim):**

> Our four rungs were run at seed 42 only, so no rung-side training stochasticity enters
> these intervals. They are therefore NARROWER than a symmetric multi-seed comparison
> would give. Re-running the ladder at seeds 1 and 2 is the highest-value follow-up.

| Re-trained baseline | GNN | LLM | AGAF | Feedback |
|---|---|---|---|---|
| E-GraphSAGE, as published | +0.6006 [+0.5581,+0.6419]; 1.0000 | +0.6139 [+0.5705,+0.6549]; 1.0000 | +0.6378 [+0.5958,+0.6772]; 1.0000 | +0.6513 [+0.6092,+0.6927]; 1.0000 |
| E-GraphSAGE, refit | +0.5837 [+0.5405,+0.6264]; 1.0000 | +0.5970 [+0.5523,+0.6410]; 1.0000 | +0.6210 [+0.5782,+0.6639]; 1.0000 | +0.6344 [+0.5913,+0.6783]; 1.0000 |
| E-GraphSAGE + our node features | +0.6037 [+0.5590,+0.6448]; 1.0000 | +0.6170 [+0.5738,+0.6583]; 1.0000 | +0.6410 [+0.5975,+0.6837]; 1.0000 | +0.6544 [+0.6100,+0.6985]; 1.0000 |
| TE-G-SAGE, as published | +0.3375 [+0.2829,+0.3881]; 1.0000 | +0.3508 [+0.2972,+0.4015]; 1.0000 | +0.3748 [+0.3206,+0.4280]; 1.0000 | +0.3882 [+0.3363,+0.4418]; 1.0000 |
| TE-G-SAGE, refit | +0.1374 [+0.0799,+0.1991]; 1.0000 | +0.1507 [+0.0942,+0.2157]; 1.0000 | +0.1747 [+0.1181,+0.2376]; 1.0000 | +0.1881 [+0.1307,+0.2521]; 1.0000 |
| TE-G-SAGE + our node features | +0.0027 [-0.0372,+0.0452]; 0.5320 | +0.0160 [-0.0313,+0.0616]; 0.7675 | +0.0400 [-0.0029,+0.0836]; 0.9620 | **+0.0535 [+0.0102,+0.0990]; 0.9910** |

**ToN CI caveat (verbatim):**

> Our four rungs were run at seed 42 only, so no rung-side training stochasticity enters
> these intervals. They are therefore NARROWER than a symmetric multi-seed comparison
> would give. Re-running the ladder at seeds 1 and 2 is the highest-value follow-up.

| Re-trained baseline | GNN | LLM | AGAF | Feedback |
|---|---|---|---|---|
| E-GraphSAGE, as published | +0.0151 [-0.0276,+0.0564]; 0.7605 | **-0.1338 [-0.1756,-0.0933]; 0.0000** | **-0.0812 [-0.1362,-0.0227]; 0.0030** | +0.0340 [-0.0121,+0.0809]; 0.9245 |
| E-GraphSAGE, refit | +0.0151 [-0.0276,+0.0564]; 0.7605 | **-0.1338 [-0.1756,-0.0933]; 0.0000** | **-0.0812 [-0.1362,-0.0227]; 0.0030** | +0.0340 [-0.0121,+0.0809]; 0.9245 |
| E-GraphSAGE + our node features | -0.0056 [-0.0767,+0.0671]; 0.4730 | **-0.1545 [-0.2274,-0.0879]; 0.0000** | **-0.1019 [-0.1843,-0.0218]; 0.0050** | +0.0133 [-0.0597,+0.0853]; 0.6160 |
| TE-G-SAGE, as published | +0.2339 [+0.1803,+0.2866]; 1.0000 | +0.0850 [+0.0431,+0.1246]; 1.0000 | +0.1376 [+0.0780,+0.1960]; 1.0000 | +0.2528 [+0.1974,+0.3076]; 1.0000 |
| TE-G-SAGE, refit | +0.1937 [+0.1419,+0.2395]; 1.0000 | +0.0448 [+0.0033,+0.0791]; 0.9835 | +0.0974 [+0.0410,+0.1500]; 0.9995 | +0.2126 [+0.1540,+0.2617]; 1.0000 |
| TE-G-SAGE + our node features | +0.0760 [+0.0215,+0.1326]; 0.9965 | **-0.0729 [-0.1214,-0.0259]; 0.0005** | -0.0202 [-0.0839,+0.0426]; 0.2715 | +0.0950 [+0.0403,+0.1519]; 1.0000 |

The secondary seed-matched intervals are retained in
`seed_matched_comparisons` in both contracts; the tables above use the required primary
two-level intervals.

### 5.4 Forced deviations from the papers

The same five documented deviations apply to both datasets:

| Code | Reason | Resolution |
|---|---|---|
| D1 | TE-G-SAGE publishes a 60/30/10 chronological split, but aggregation removed chronological order. | Use the project's stratified five-fold `folds.pt`. |
| D2 | TE-G-SAGE fanout `[25,15]` and batch size 4096 exceed the available degree and graph size. | Use full-neighbourhood, full-batch training. |
| D3 | The released implementations use fixed schedules: E-GraphSAGE 4999 epochs and TE-G-SAGE 20, with no validation split or early stopping. | Use max 300 epochs, patience 25 on validation macro-F1, and restore the best state, matching the project rungs. |
| D4 | TE-G-SAGE `rare_min_freq=50` removes almost every port category at this aggregated scale. | Keep 50 in `as_published`; sweep it in `refit`. |
| D5 | E-GraphSAGE paper Eq. 4 aggregates edge features alone, while the released notebook uses `W_msg([h_u || e_uv])`. | Follow the released implementation that produced the published results. |

### 5.5 E-GraphSAGE endpoint-only ceiling

E-GraphSAGE classifies an edge from `CONCAT(h_u, h_v)` without passing that edge's own
features to the classifier. Parallel edges sharing an endpoint pair are therefore
indistinguishable: 385 UNSW edges (58.7%) and 167 ToN edges (7.9%). This is a
representation-architecture interaction, not evidence that E-GraphSAGE is weak, and the
large UNSW margin must not be presented as a fusion win. A full 4999-epoch fold-0 run
confirmed that the ceiling is not caused by the shortened training schedule: validation
macro-F1 plateaued near 0.11 and peaked at 0.1384 at epoch 1400.

Contracts: `results/unsw_nb15_sota_baselines.json` and
`results/ton_iot_sota_baselines.json`. Prediction matrices:
`results/predictions/{unsw_nb15,ton_iot}_{e_graphsage,te_g_sage}_{as_published,refit,plus_node_features}.pt`.

---

## 6. Off-architecture code

Kept in the tree but unreachable from any canonical run: `enhanced12` feature profile,
`sweep_fusion_variants.py`, non-default `fusion_mode` branches, extra fusion knobs (InfoNCE
alignment, no-regret floor, gate-entropy, head output-gate).

Fusion-mechanism comparisons on the ToN graph are **not resolvable** anyway — rare classes have
12–35 edges and seed variance (~0.11) exceeds every between-mechanism gap.

---

## 2026-09-06 — Fixing the loop's two signals: a negative result, and a knob artefact

Three conditions, feedback stage only, seeds 42/1/2, both datasets, fold partition fixed.
The GNN/AGAF/LLM rungs are reused from the seed-matched captures in `results/multiseed/`
(neither fix touches them; the LLM rung is an argmax, invariant to the temperature).
Aggregates: `results/multiseed_ladder_v2_{legacy,fixed,fixed_reselected}.json`.
Runner: `scripts/run_multiseed_feedback.sh`. Selection: `scripts/reselect_knobs.sh`.

- **A `legacy`** — `--selector-head-loss-weight 0.0 --legacy-temperature`, i.e. both
  pre-fix defects, at the correct per-dataset scales (2.0 / 20.0).
- **B `fixed`** — Tasks 1+2, same scales. A vs B isolates the two signal fixes.
- **C `fixed_reselected`** — Tasks 1+2 at re-selected knobs. B vs C isolates the
  re-selection.

### Pooled OOF macro-F1, mean over 3 seeds

| dataset | rung | A legacy | B fixed | C reselected |
|---|---|---:|---:|---:|
| UNSW | loop | 0.7644 | 0.7610 | 0.7809 |
| UNSW | agaf | 0.7793 | 0.7793 | 0.7793 |
| UNSW | gnn  | 0.7437 | 0.7437 | 0.7437 |
| ToN  | loop | 0.4521 | 0.4147 | 0.4630 |
| ToN  | agaf | 0.4102 | 0.4102 | 0.4102 |
| ToN  | gnn  | 0.4336 | 0.4336 | 0.4336 |

`loop_vs_agaf`, two-level CI (edges + training seed):

| dataset | A legacy | B fixed | C reselected |
|---|---|---|---|
| UNSW | +0.0156 below, [−0.0552, +0.0268] | −0.0192, [−0.0594, +0.0197] | +0.0008, [−0.0464, +0.0447] |
| ToN  | +0.0428, [−0.0316, +0.1105] | +0.0060, [−0.0667, +0.0739] | +0.0527, [−0.0118, +0.1194] |

**Decision rule (fixed in advance): the loop is NOT SEPARATED from AGAF on either
dataset in any of the three conditions.** Every CI includes zero. C has the highest
point estimate on both datasets; that is not a separation.

### Finding 1 — fixing the two signals made the loop WORSE

Both defects were removed and verified in `bias_diagnostics`:

| dataset | selector-head macro-F1 A→B | mean_disagreement A→B | consultant max-prob A→B | T A→B |
|---|---|---|---|---|
| UNSW | 0.019–0.036 → 0.646–0.684 | 0.8975 → 0.524–0.548 | 0.102 → 0.487 | ~9.4 → 0.105 |
| ToN  | 0.029–0.065 → 0.293–0.333 | 0.8988 → 0.580–0.638 | 0.101 → 0.500 | ~9.5 → 0.030 |

And the loop went DOWN: UNSW 0.7644 → 0.7610 (−0.0034), ToN 0.4521 → 0.4147 (−0.0374).
A better-calibrated consultant and a genuinely uncertainty-ranked selector do not help
this mechanism. This is consistent with Gate 0 (§2.4): the binding constraint is
consultant quality, and sharpening a consultant that is wrong concentrates its error.

**Do not re-derive "fix the selector head / the temperature and the loop improves". It
was measured at 3 seeds and it is false.**

### Finding 2 — the recovery in C is a knob artefact, not a tuned improvement

C beats B on both datasets, but the validation curves it selected from are flat:

- UNSW `top_k` over 15..35 spans 0.7994–0.8110 (range 0.0116); selected 35 — **on the
  boundary of the swept range**, so the sweep cannot say whether the optimum lies outside.
- ToN `top_k` spans 0.4710–0.5072 (range 0.0362); selected 21.
- Both ranges are smaller than the rungs' own seed-to-seed spread, so the argmax is not
  distinguishable from its neighbours.

Selected: UNSW k 31→35, scale 2.0→5.0; ToN k 25→21, scale 20.0→10.0. Full curves are in
each dataset's `selected_feedback_config.json` under `selection_curves`, and in
`results/knob_selection_v2/`.

### Finding 3 — the canonical mechanism was effectively switched off; the fixes switched it on

`bias_diagnostics.flagged_bias_absmean` — the mean magnitude actually added to the
39-wide focused edge vector (Z-scored features, std ≈ 1) on flagged edges, mean over
folds, per seed:

| | UNSW | ToN |
|---|---|---|
| A legacy (canonical) | 0.077 / 0.060 / 0.061 | 0.023 / 0.021 / 0.023 |
| B fixed, same knobs | 1.228 / 1.171 / 1.218 (**~17×**) | 2.796 / 2.854 / 2.762 (**~125×**) |
| C fixed, re-selected | 1.103 / 1.100 / 1.158 | 2.845 / 2.893 / 2.992 |

Two consequences:

1. **In the canonical run the advice was ~0.02–0.08 on unit-variance features.** The
   mechanism was inert in effect, not just in the attention variant. This is the concrete
   form of Gate 0's "the prototype consultant captures ≈0% of the oracle's headroom": the
   loop's edge over `head_only` was late fusion, and the canonical ladder never actually
   tested feedback. Calibrating the temperature is what turned it on.
2. **`injection_scale` is nearly redundant with the learned `log_bias_strength` and
   projection.** C moved the scale 2.0→5.0 (UNSW) and 20.0→10.0 (ToN), yet the end-to-end
   magnitude stayed ~1.1 / ~2.9. The model sets its own effective magnitude. That is why
   the scale curves were flat and why re-selecting the scale is not a lever. Any future
   sweep should treat the scale as a fixed constant and, if magnitude matters, constrain
   `log_bias_strength` instead.

Read together with Findings 1–2: switching a weak consultant ON at the edges the GNN is
genuinely unsure about does not produce a separable gain. The remaining untried lever is
content-dependence — making the advice depend on the GNN's current top-2 (plan Task 5) so
the 10-way weak consultant is asked a 2-way question.


### Per-class F1, condition C vs AGAF (mean over 3 seeds)

The classes the loop was previously losing did NOT recover relative to AGAF:

| UNSW class | AGAF | loop B | loop C | C−AGAF |
|---|---:|---:|---:|---:|
| Analysis | 0.589 | 0.504 | 0.519 | −0.070 |
| Fuzzers | 0.733 | 0.652 | 0.676 | −0.057 |
| Shellcode | 0.943 | 0.883 | 0.898 | −0.045 |
| Generic | 0.541 | 0.630 | 0.682 | **+0.141** |

| ToN class | AGAF | loop B | loop C | C−AGAF |
|---|---:|---:|---:|---:|
| password | 0.295 | 0.192 | 0.209 | −0.086 |
| scanning | 0.278 | 0.171 | 0.229 | −0.049 |
| mitm | 0.708 | 0.833 | 0.893 | **+0.185** |
| xss | 0.084 | 0.232 | 0.254 | **+0.170** |

UNSW Fuzzers/Analysis and ToN password/scanning — the four classes the diagnosis
predicted the semantic branch should rescue — remain the loop's worst deficits against
AGAF in every condition. The loop's macro-F1 gains come from elsewhere (Generic, mitm,
xss). **The mechanism is not helping where the story says it should.**

### `head_only` is untouched by both fixes

Bit-identical OOF logits (`torch.equal`) and identical pooled macro-F1 between A and B
across all six (dataset, seed) pairs — the selector loss is skipped in `head_only`, and
the scorer is never consulted there, so neither the RNG stream nor the output moves.

### The injection_scale trap (see also the `validity` block in `results/multiseed_ladder.json`)

`train_feedback` resolved `top_k_percent` and `bias_confidence_fraction` from
`selected_feedback_config.json` but not `injection_scale`, which was an argparse default
of 10.0 the config could not override. The 3-seed sweep of 2026-09-06 omitted the flag,
so **every loop number in `results/multiseed_ladder.json` ran at 10.0 on both datasets**
instead of 2.0 / 20.0. GNN/LLM/AGAF are unaffected. At the correct scale the legacy loop
is ToN 0.4521 (not 0.4430) and its `loop_vs_agaf` sign is stable across seeds (v1 said
otherwise), so the withdrawn v1 numbers were wrong in value and in sign stability.
Closed at four points: `resolve_injection_scale` raises rather than defaulting;
`_build_model`/`_train_one_fold` require the value; `assemble_ladder` reports the knobs
that RAN from `benchmark_summary.json`; `aggregate_multiseed` pins per-capture knobs and
refuses to pool seeds that disagree.
