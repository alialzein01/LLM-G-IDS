# Results Archive

Full results history, findings, retractions and reproducibility record for LLM-G-IDS.

**Why this file exists:** `PROJECT_NOTES.md` is read at the start of every working session, so it must
stay short. This file holds the complete narrative — current results, superseded results,
what was tried, what failed, and what was corrected — and is read on demand. Nothing here
is a duplicate; `PROJECT_NOTES.md` carries the *rules and traps*, this carries the *record*.

**Authoritative numbers always live in `results/*.json`.** If this file and a contract
disagree, the contract wins and this file is stale.

### The trail: what was tried, what it cost, what it bought

If you are here to see whether the work was investigated rather than guessed, read these
six entries in order. Together they are the record of one mechanism being pushed until it
either worked or was shown not to.

| # | Entry | What was attempted | Outcome |
|---|---|---|---|
| 1 | §2.1 | Inject advice through the attention bias | Inert. Churn exactly 0.0000 in every configuration, even handed the true label. Structural, confirmed three times |
| 2 | §2.2 | Show the edge-injection channel beats no advice at all | **Retracted.** Sensitive to advice *content*, but real advice does not measurably beat none |
| 3 | §2.4 | Ask whether the channel itself has capacity, using an oracle consultant | It does, on both datasets, and both separated. So the limit is not the mechanism |
| 4 | 2026-09-06 | Fix the two broken signals the loop depended on: train the selector head, calibrate the consultant's temperature | Both defects genuinely removed, and the loop got **worse** on both datasets |
| 5 | 2026-09-07 | Five further treatments on the same consultant: selector supervision, temperature, reliability weighting, advice format, cross-fitting | All five negative, each from a different angle. The table below is the case for what came next |
| 6 | 2026-09-07 | Change the consultant itself, from the embedding-only prototype to a per-fold trained head | +0.0697 UNSW and +0.0523 ToN, and every ladder comparison becomes separated |
| 7 | 2026-09-13 | Put an interval on step 6, and complete the flagged-set disagreement split | Separated on UNSW, **not** on ToN. Two earlier figures found to be at the wrong knobs and not reproducible; both replaced. See `results/consultant_change_interval.json` and `results/consultant_complementarity.json` |

The improvement in step 6 is only meaningful because steps 1 to 5 are on the record. A
report that showed only step 6 would be claiming a lucky configuration; the archive is what
turns it into a diagnosis.

---

## 2026-09-07 — Canonical loop redefined: trained-head consultant

### The decision, verbatim

> The loop's semantic consultant is the per-fold trained head on CySecBERT embeddings (llm_head_logits.pt, in-fold on train/val, out-of-fold on test; the cross-fitted variant was measured worse, 2026-09-07).
> The LLM rung and AGAF keep the embedding-only whitened prototype: the semantic branch is compared as an encoder, the loop is the proposed system.
> The head alone is reported as the strongest LLM-only baseline in every table.
> Architecture parity rule: same encoder, same graph, same folds, same seeds on every rung; the loop additionally trains a classification head on the semantic embeddings.

### Why: five mechanism treatments, all negative

Before changing the consultant, five separate attempts were made to get the
whitened-prototype consultant's advice through the bias channel. Every one failed,
from a different angle. This table is the case for the change.

| # | Treatment | What it changed | Result | Where |
|---|---|---|---|---|
| 1 | Selector-head supervision | `SELECTOR_HEAD_LOSS_WEIGHT` 0.0 -> 1.0, so the head whose entropy ranks edges is trained as a classifier | 3-seed: worse. Reverted to 0.0 as the default | archive 2026-09-06, Findings 1-3 |
| 2 | Consultant temperature | per-fold calibration instead of the T=10 pin, turning a near-uniform softmax into a rankable one | 3-seed: worse, together with (1). Reverted | archive 2026-09-06, Finding 1 |
| 3 | Reliability-weighted advice (E1) | gate score, and injected magnitude, scaled by the consultant's measured per-class train precision | seed 42: `gate` +0.0038 UNSW / -0.0107 ToN; `gate+scale` +0.0057 / +0.0094, both under the 0.02 threshold and inside the seed band. Not kept | archive 2026-09-07 (four mechanism fixes) |
| 4 | Advice format | the oracle's own +/-4-nat one-hot, and its softmax at the same peak, instead of raw logits | seed 42: every arm BELOW baseline on both datasets. Saturating the input made the *injected* magnitude fall 4x, because the projection is learned. Not kept | archive 2026-09-07 (advice format) |
| 5 | Cross-fitted advice | train rows replaced by inner out-of-fold logits, so the loop trains on advice as reliable as the advice it is scored with | seed 42: loop worse on both (-0.0246 / -0.0338) and further below the head alone. `learned_bias_strength` moved by at most 0.0205 across four settings — the loop never learned a trust that in-fold optimism was fooling | archive 2026-09-07 (cross-fitted advice) |

Read with §2.4 (the channel has capacity; no realistic consultant exploits it), the
conclusion is that the binding constraint was never selection, weighting, format or
train/test reliability matching. It was the consultant. So the consultant changed.

### The new 3-seed ladder

Pooled OOF macro-F1, seeds 42/1/2, fold partition fixed at the seed-42 split.
Class counts differ (UNSW 10, ToN 8); only ladder SHAPE is comparable across datasets.

**UNSW-NB15 (10 classes)**

| rung | mean | std | seed 42 | seed 1 | seed 2 |
|---|---:|---:|---:|---:|---:|
| gnn | 0.7437 | 0.0228 | 0.7219 | 0.7674 | 0.7418 |
| llm | 0.7353 | 0.0000 | 0.7353 | 0.7353 | 0.7353 |
| agaf | 0.7793 | 0.0116 | 0.7680 | 0.7912 | 0.7788 |
| **loop** | 0.8341 | 0.0042 | 0.8332 | 0.8387 | 0.8304 |
| head alone | 0.8374 | 0.0048 | 0.8321 | 0.8386 | 0.8416 |

**NF-ToN-IoT (8 classes)**

| rung | mean | std | seed 42 | seed 1 | seed 2 |
|---|---:|---:|---:|---:|---:|
| gnn | 0.4336 | 0.0053 | 0.4290 | 0.4393 | 0.4326 |
| llm | 0.2785 | 0.0000 | 0.2785 | 0.2785 | 0.2785 |
| agaf | 0.4102 | 0.0279 | 0.3975 | 0.3910 | 0.4422 |
| **loop** | 0.5044 | 0.0080 | 0.4985 | 0.5012 | 0.5135 |
| head alone | 0.5081 | 0.0073 | 0.5165 | 0.5035 | 0.5043 |

### What is separated, and what is not

Two-level bootstrap (resampling edges AND the training seed), 2000 iterations.
The split is **identical on both datasets**, which it never was under the prototype.

| comparison | UNSW mean [CI] | sep | ToN mean [CI] | sep |
|---|---|---|---|---|
| `feedback_vs_agaf` | +0.0546 [+0.0228,+0.0880] | **yes** | +0.0932 [+0.0180,+0.1667] | **yes** |
| `feedback_vs_gnn` | +0.0908 [+0.0412,+0.1407] | **yes** | +0.0694 [+0.0129,+0.1288] | **yes** |
| `feedback_vs_llm` | +0.0989 [+0.0721,+0.1267] | **yes** | +0.2231 [+0.1703,+0.2771] | **yes** |
| `head_alone_vs_gnn` | +0.0938 [+0.0420,+0.1413] | **yes** | +0.0723 [+0.0125,+0.1322] | **yes** |
| `agaf_vs_gnn` | +0.0362 [-0.0096,+0.0822] | no | -0.0238 [-0.0921,+0.0607] | no |
| `feedback_vs_head_alone` | -0.0030 [-0.0242,+0.0136] | no | -0.0028 [-0.0388,+0.0318] | no |

**The loop is separated above AGAF, above the GNN and above the prototype LLM rung on
both datasets — the first separated ladder step this project has measured.**

**And the loop is NOT separated from the head alone that it consults.** On both
datasets the head alone carries the higher mean, the interval includes zero, and the
sign is not stable across seeds. The loop matches its consultant; it does not beat it.
Every statement of the loop's gain over AGAF or the GNN must carry this alongside.

The one place the ordering flips is UNSW seed 42, where the loop scores 0.8332 against
the head's 0.8321. That is what an unseparated difference looks like from the inside,
and it is why the 3-seed mean must not be quoted as if it settled the question.

### Per-class F1, 3-seed mean

**UNSW-NB15 (10 classes)**

| class | gnn | llm | agaf | loop | head alone |
|---|---:|---:|---:|---:|---:|
| Normal | 0.9129 | 0.8566 | 0.9152 | 0.9543 | 0.9617 |
| Analysis | 0.4327 | 0.4722 | 0.5886 | 0.6196 | 0.6407 |
| Backdoors | 0.5929 | 0.5823 | 0.6578 | 0.8019 | 0.7983 |
| DoS | 0.5831 | 0.6500 | 0.6446 | 0.7424 | 0.7311 |
| Exploits | 0.9309 | 0.9070 | 0.9583 | 0.9580 | 0.9500 |
| Fuzzers | 0.6207 | 0.6538 | 0.7330 | 0.7749 | 0.7939 |
| Generic | 0.6591 | 0.5957 | 0.5409 | 0.6899 | 0.6955 |
| Reconnaissance | 0.9110 | 0.8434 | 0.8916 | 0.8731 | 0.8865 |
| Shellcode | 0.8669 | 0.9512 | 0.9428 | 0.9959 | 1.0000 |
| Worms | 0.9272 | 0.8409 | 0.9205 | 0.9311 | 0.9165 |

**NF-ToN-IoT (8 classes)**

| class | gnn | llm | agaf | loop | head alone |
|---|---:|---:|---:|---:|---:|
| Benign | 0.9599 | 0.6359 | 0.9275 | 0.9832 | 0.9840 |
| backdoor | 0.3633 | 0.0707 | 0.2439 | 0.4467 | 0.5064 |
| ddos | 0.1848 | 0.0921 | 0.2419 | 0.2913 | 0.3145 |
| injection | 0.5436 | 0.4530 | 0.4845 | 0.6277 | 0.5888 |
| mitm | 0.8838 | 0.7349 | 0.7082 | 0.8566 | 0.8339 |
| password | 0.0401 | 0.1127 | 0.2950 | 0.2100 | 0.2193 |
| scanning | 0.0958 | 0.0936 | 0.2781 | 0.3116 | 0.3004 |
| xss | 0.3728 | 0.0326 | 0.0844 | 0.2612 | 0.2678 |

### Knob selection: flat curves, one boundary

Knobs were re-selected for the head consultant on validation folds only, 3 seeds
(`results/knob_selection_head/`): UNSW top_k 31 -> **29**, scale 2.0 -> **2.0**; ToN
top_k 25 -> **16**, scale 20.0 -> **20.0**.

Both curves lie inside their own noise band — UNSW's top_k spans 0.8241-0.8277 across
21 candidates and its scale 0.8252-0.8277 across six; ToN's span 0.5168-0.5232 and
0.5196-0.5232. **The argmax is picking noise. Neither knob is tuned, and neither
should be described as tuned.** ToN's scale selected on the UPPER BOUNDARY of the
swept range (20.0), which means the range was too narrow to contain the optimum if
one exists; the contract records `scale_on_range_boundary: true`.

### What is preserved

The schema-4 (UNSW) / schema-6 (ToN) prototype-consultant ladders move under
`supersedes` in each contract, intact, and their knobs stay in each dataset's
`selected_feedback_config.json` under `prototype_superseded` (with condition C nested
inside). That ladder is still reproducible: run `train_feedback` WITHOUT
`--use-llm-head` at those knobs. The single-seed and prototype-consultant baseline
interval blocks are likewise kept under `superseded` in the two
`*_sota_baselines.json` files.

Artifacts: `results/multiseed_ladder_v2_head.json`, `results/multiseed_v2/head/`,
`results/multiseed_head_per_class.json`, `results/knob_selection_head/`.

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

> **SUPERSEDED 2026-09-06.** Everything in §1.1–§1.3 is the single-seed (seed 42) picture
> from the schema-3/5 contracts. It is preserved for provenance. The current contracts are
> schema 4 (UNSW) / 6 (ToN), measured at 3 training seeds with the fold partition fixed; see
> the **2026-09-06** section below and `results/multiseed_ladder_v2_legacy.json`. At 3 seeds
> **nothing is separated on either dataset**; the UNSW loop is below AGAF at every seed and
> the ToN "AGAF significantly below GNN (P=0.0005)" does not survive. Do not quote §1.1's
> intervals or §1.2's reversal as current.

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

### 1.3 ToN loop reads the advice (ladder-level ablation) — SUPERSEDED, corrected 2026-09-13

**The numbers first written here (real 0.4478, random 0.4046, mean_diff +0.0428, "P=0.972")
were from a single pre-multiseed run and are not in any contract. Do not quote them.** Two
further problems with the original entry: it pointed at a key called `feedback_ablations`,
which does not exist, and it wrote a bootstrap `prob_positive` as "P", which reads as a
p-value and is not one.

The live figures are under `feedback_ablations_per_seed` in `results/ton_iot_current.json`,
three seeds, pooled OOF macro-F1:

| consultant | real (42 / 1 / 2) | random (42 / 1 / 2) |
|---|---|---|
| trained head (canonical) | 0.4985 / 0.5012 / 0.5135 | 0.4002 / 0.3790 / 0.4067 |
| prototype (superseded, under `supersedes`) | 0.4414 / 0.4509 / 0.4641 | 0.3966 / 0.3626 / 0.4065 |

The finding the heading claims survives the correction and is stronger than the original
entry stated: real advice beats random advice at every seed under both consultants. What the
entry must not be read as is evidence that the loop beats *no* advice through the injection
path -- that is §2.2, and it is retracted. This is the *ladder-level* ablation with output
fusion on, a different experiment from the mechanism-only test in §2.

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

### 5.3 Full rung-vs-baseline comparison — 3-seed rungs, 3-seed baselines, 2026-09-07

**Updated for the trained-head consultant.** The `feedback` row is the new canonical
loop and `head_alone` is a new row: the loop's own consultant standing alone. The
gnn / llm / agaf rows are unchanged by the consultant switch and reproduce their earlier
values. The prototype-consultant version of this table is preserved in each contract
under `superseded.whitened_prototype_scorer_3seed`.

Sign convention: `delta = project rung - re-trained baseline`. Each cell is
`delta [95% CI]` with **sep** if the interval excludes zero and *ns* if it does not.
`*` marks a non-faithful variant (our node features added), not a published architecture.

**UNSW-NB15 (10 classes) — two-level (rung seed and baseline seed drawn independently, edges resampled)**

| Re-trained baseline | GNN | LLM | AGAF | Loop | head alone |
|---|---|---|---|---|---|
| E-GraphSAGE, as published | +0.6224 [+0.5684,+0.6764] **sep** | +0.6137 [+0.5711,+0.6558] **sep** | +0.6575 [+0.6103,+0.7027] **sep** | +0.7130 [+0.6708,+0.7540] **sep** | +0.7161 [+0.6744,+0.7583] **sep** |
| E-GraphSAGE, refit | +0.6054 [+0.5508,+0.6598] **sep** | +0.5967 [+0.5520,+0.6392] **sep** | +0.6404 [+0.5914,+0.6867] **sep** | +0.6960 [+0.6526,+0.7373] **sep** | +0.6991 [+0.6545,+0.7403] **sep** |
| E-GraphSAGE + our node features * | +0.6251 [+0.5699,+0.6796] **sep** | +0.6164 [+0.5735,+0.6592] **sep** | +0.6601 [+0.6120,+0.7086] **sep** | +0.7157 [+0.6731,+0.7558] **sep** | +0.7188 [+0.6759,+0.7598] **sep** |
| TE-G-SAGE, as published | +0.3594 [+0.3003,+0.4221] **sep** | +0.3508 [+0.2981,+0.4019] **sep** | +0.3945 [+0.3348,+0.4520] **sep** | +0.4500 [+0.3924,+0.5042] **sep** | +0.4531 [+0.3976,+0.5052] **sep** |
| TE-G-SAGE, refit | +0.1613 [+0.0919,+0.2337] **sep** | +0.1527 [+0.0933,+0.2156] **sep** | +0.1964 [+0.1319,+0.2629] **sep** | +0.2519 [+0.1914,+0.3169] **sep** | +0.2550 [+0.1943,+0.3197] **sep** |
| TE-G-SAGE + our node features * | +0.0249 [-0.0270,+0.0755] *ns* | +0.0163 [-0.0291,+0.0607] *ns* | +0.0600 [+0.0096,+0.1093] **sep** | +0.1155 [+0.0673,+0.1627] **sep** | +0.1186 [+0.0691,+0.1659] **sep** |

**UNSW-NB15 (10 classes) — seed-matched (42↔42, 1↔1, 2↔2, edges resampled)**

| Re-trained baseline | GNN | LLM | AGAF | Loop | head alone |
|---|---|---|---|---|---|
| E-GraphSAGE, as published | +0.6224 [+0.5873,+0.6544] **sep** | +0.6141 [+0.5755,+0.6508] **sep** | +0.6581 [+0.6214,+0.6919] **sep** | +0.7132 [+0.6754,+0.7491] **sep** | +0.7164 [+0.6806,+0.7506] **sep** |
| E-GraphSAGE, refit | +0.6054 [+0.5709,+0.6378] **sep** | +0.5970 [+0.5579,+0.6351] **sep** | +0.6410 [+0.6041,+0.6757] **sep** | +0.6961 [+0.6577,+0.7324] **sep** | +0.6993 [+0.6629,+0.7343] **sep** |
| E-GraphSAGE + our node features * | +0.6248 [+0.5902,+0.6582] **sep** | +0.6165 [+0.5778,+0.6540] **sep** | +0.6604 [+0.6242,+0.6951] **sep** | +0.7156 [+0.6778,+0.7510] **sep** | +0.7187 [+0.6821,+0.7529] **sep** |
| TE-G-SAGE, as published | +0.3595 [+0.3247,+0.3921] **sep** | +0.3511 [+0.3109,+0.3901] **sep** | +0.3951 [+0.3535,+0.4327] **sep** | +0.4502 [+0.4090,+0.4901] **sep** | +0.4534 [+0.4130,+0.4937] **sep** |
| TE-G-SAGE, refit | +0.1599 [+0.1292,+0.1933] **sep** | +0.1516 [+0.1157,+0.1895] **sep** | +0.1956 [+0.1593,+0.2323] **sep** | +0.2507 [+0.2133,+0.2909] **sep** | +0.2539 [+0.2172,+0.2924] **sep** |
| TE-G-SAGE + our node features * | +0.0240 [-0.0013,+0.0504] *ns* | +0.0157 [-0.0227,+0.0545] *ns* | +0.0596 [+0.0244,+0.0952] **sep** | +0.1147 [+0.0748,+0.1540] **sep** | +0.1179 [+0.0768,+0.1588] **sep** |

**NF-ToN-IoT (8 classes) — two-level (rung seed and baseline seed drawn independently, edges resampled)**

| Re-trained baseline | GNN | LLM | AGAF | Loop | head alone |
|---|---|---|---|---|---|
| E-GraphSAGE, as published | +0.0203 [-0.0230,+0.0633] *ns* | -0.1336 [-0.1773,-0.0939] **sep** | -0.0061 [-0.0797,+0.0777] *ns* | +0.0890 [+0.0312,+0.1480] **sep** | +0.0925 [+0.0361,+0.1506] **sep** |
| E-GraphSAGE, refit | +0.0203 [-0.0230,+0.0633] *ns* | -0.1336 [-0.1773,-0.0939] **sep** | -0.0061 [-0.0797,+0.0777] *ns* | +0.0890 [+0.0312,+0.1480] **sep** | +0.0925 [+0.0361,+0.1506] **sep** |
| E-GraphSAGE + our node features * | -0.0020 [-0.0734,+0.0726] *ns* | -0.1559 [-0.2301,-0.0848] **sep** | -0.0284 [-0.1258,+0.0775] *ns* | +0.0667 [-0.0181,+0.1505] *ns* | +0.0702 [-0.0139,+0.1512] *ns* |
| TE-G-SAGE, as published | +0.2396 [+0.1864,+0.2938] **sep** | +0.0858 [+0.0436,+0.1241] **sep** | +0.2132 [+0.1339,+0.3005] **sep** | +0.3083 [+0.2345,+0.3793] **sep** | +0.3119 [+0.2361,+0.3834] **sep** |
| TE-G-SAGE, refit | +0.1985 [+0.1456,+0.2474] **sep** | +0.0447 [+0.0031,+0.0777] **sep** | +0.1721 [+0.1021,+0.2539] **sep** | +0.2672 [+0.2015,+0.3348] **sep** | +0.2708 [+0.2034,+0.3362] **sep** |
| TE-G-SAGE + our node features * | +0.0814 [+0.0256,+0.1371] **sep** | -0.0724 [-0.1233,-0.0251] **sep** | +0.0550 [-0.0232,+0.1453] *ns* | +0.1502 [+0.0749,+0.2268] **sep** | +0.1537 [+0.0796,+0.2246] **sep** |

**NF-ToN-IoT (8 classes) — seed-matched (42↔42, 1↔1, 2↔2, edges resampled)**

| Re-trained baseline | GNN | LLM | AGAF | Loop | head alone |
|---|---|---|---|---|---|
| E-GraphSAGE, as published | +0.0199 [-0.0097,+0.0500] *ns* | -0.1339 [-0.1672,-0.0998] **sep** | -0.0051 [-0.0490,+0.0417] *ns* | +0.0889 [+0.0391,+0.1393] **sep** | +0.0923 [+0.0447,+0.1401] **sep** |
| E-GraphSAGE, refit | +0.0199 [-0.0097,+0.0500] *ns* | -0.1339 [-0.1672,-0.0998] **sep** | -0.0051 [-0.0490,+0.0417] *ns* | +0.0889 [+0.0391,+0.1393] **sep** | +0.0923 [+0.0447,+0.1401] **sep** |
| E-GraphSAGE + our node features * | -0.0012 [-0.0336,+0.0314] *ns* | -0.1549 [-0.1903,-0.1162] **sep** | -0.0261 [-0.0735,+0.0238] *ns* | +0.0679 [+0.0189,+0.1156] **sep** | +0.0713 [+0.0208,+0.1205] **sep** |
| TE-G-SAGE, as published | +0.2393 [+0.2044,+0.2776] **sep** | +0.0855 [+0.0617,+0.1091] **sep** | +0.2144 [+0.1709,+0.2599] **sep** | +0.3083 [+0.2504,+0.3650] **sep** | +0.3118 [+0.2530,+0.3675] **sep** |
| TE-G-SAGE, refit | +0.1986 [+0.1608,+0.2354] **sep** | +0.0448 [+0.0187,+0.0696] **sep** | +0.1736 [+0.1290,+0.2171] **sep** | +0.2676 [+0.2103,+0.3238] **sep** | +0.2710 [+0.2160,+0.3273] **sep** |
| TE-G-SAGE + our node features * | +0.0809 [+0.0452,+0.1176] **sep** | -0.0729 [-0.1022,-0.0435] **sep** | +0.0559 [+0.0095,+0.1018] **sep** | +0.1499 [+0.0918,+0.2048] **sep** | +0.1534 [+0.0956,+0.2090] **sep** |

**The loop and the head alone clear every faithful baseline on both datasets**, in
both interval types — which the prototype-consultant loop did not (it was *ns* against
E-GraphSAGE on ToN). The remaining nulls are all gnn / llm / agaf against E-GraphSAGE
on ToN, and the LLM (prototype) rung is separated BELOW E-GraphSAGE there.

Note the two rungs track each other closely everywhere: the loop's margin over any
baseline is within ~0.004 of the head alone's. That is the same fact as
`feedback_vs_head_alone` being unseparated, seen from the baseline side.

#### 5.3a Superseded tables

Two earlier versions are kept in each contract and must not be quoted as current:
`superseded.statistical_comparisons` (schema 1, rung at seed 42 only) and
`superseded.whitened_prototype_scorer_3seed` (both sides at 3 seeds, loop consulting the
prototype). The prototype-consultant table as it was published on 2026-09-07 follows.

Kept for the record and preserved in each contract under `superseded`. Its own caveat is
the reason it was replaced:

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

---

## 2026-09-07 — Four mechanism fixes, seed 42

Four things were tried against the canonical condition-A loop: reliability-weighted
advice in two forms (E1), AGAF's fusion block transplanted into the loop (E3), and — from
the day before — the trained LLM head as the loop's consultant (the head-echo test).
**None is kept.** All numbers below are seed 42 only, pooled five-fold OOF, with
`OMP_NUM_THREADS=1 IDS_FORCE_CPU=1`. No bootstrap across seeds was run on these arms, so
nothing here is a separation claim — they are single-seed orderings against a baseline
that itself moves ±0.023 (UNSW) / ±0.005 (ToN) across training seeds.

Artifacts: `results/dev/e1e3_summary.json`, `results/dev/head_echo/`,
`results/dev/{e1,e3}/<dataset>/<arm>/mechanism_diagnostics.json`. Row-level log:
`docs/experiment_log.md`.

### UNSW-NB15 (10 classes)

| arm | real | head_only | random | real − head_only [95% CI] P | flagged acc GNN/proto/loop | churn all/flagged | flips +/− flagged | flagged_bias_absmean | fusion params |
|---|---:|---:|---:|---|---|---|---|---:|---:|
| A baseline | 0.7642 | 0.7543 | 0.6529 | +0.0098 [-0.018, +0.038] P=0.766 | 0.603 / 0.672 / 0.652 | 0.087 / 0.221 | +19 / −9 | 0.0767 | 125,266 |
| E1 gate | 0.7679 | 0.7543 | 0.6845 | +0.0134 [-0.019, +0.047] P=0.789 | 0.603 / 0.672 / 0.642 | 0.110 / 0.294 | +24 / −16 | 0.1114 | 125,266 |
| E1 gate+scale | 0.7699 | 0.7543 | 0.6635 | +0.0153 [-0.016, +0.047] P=0.823 | 0.603 / 0.672 / 0.647 | 0.102 / 0.265 | +22 / −13 | 0.1066 | 125,266 |
| E3 agaf | 0.6114 | 0.7543 | 0.3970 | -0.1438 [-0.193, -0.096] P=0.000 | 0.603 / 0.672 / 0.534 | 0.235 / 0.417 | +19 / −33 | 0.0591 | 199,242 |

### NF-ToN-IoT (8 classes)

| arm | real | head_only | random | real − head_only [95% CI] P | flagged acc GNN/proto/loop | churn all/flagged | flips +/− flagged | flagged_bias_absmean | fusion params |
|---|---:|---:|---:|---|---|---|---|---:|---:|
| A baseline | 0.4414 | 0.3809 | 0.3966 | +0.0603 [+0.026, +0.096] P=1.000 | 0.359 / 0.348 / 0.573 | 0.126 / 0.423 | +135 / −21 | 0.0234 | 125,266 |
| E1 gate | 0.4307 | 0.3809 | 0.3796 | +0.0497 [+0.014, +0.087] P=0.995 | 0.359 / 0.348 / 0.551 | 0.133 / 0.429 | +122 / −20 | 0.0356 | 125,266 |
| E1 gate+scale | 0.4507 | 0.3809 | 0.3853 | +0.0689 [+0.034, +0.105] P=1.000 | 0.359 / 0.348 / 0.592 | 0.134 / 0.442 | +142 / −18 | 0.0370 | 125,266 |
| E3 agaf | 0.3312 | 0.3809 | 0.2613 | -0.0493 [-0.086, -0.010] P=0.006 | 0.359 / 0.348 / 0.449 | 0.196 / 0.560 | +107 / −59 | 0.0216 | 199,242 |

`head_only` is bit-identical across every arm on both datasets
(0.7542650444992878 / 0.38090190800088425). It is the same plain-GNN-in-a-loop run in all
of them, so every row is read against the same baseline.

### E1 — reliability-weighted advice (`--advice-reliability {gate,gate+scale}`)

`w[c]` is the precision of the prototype consultant's argmax for class `c`, measured on the
fold's TRAIN edges only (0 where it never predicts `c` there). `gate` multiplies the
confidence gate's ranking score by `w[argmax]`; `gate+scale` additionally scales the advice
handed to the bias module, so injected magnitude is proportional to how often that verdict
is right. Mean `w` over the five folds (both arms compute it identically):

| Dataset | class 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| UNSW-NB15 | 1.000 | 0.852 | 0.976 | 0.893 | 0.961 | 0.662 | 0.895 | 0.984 | 1.000 | 0.960 |
| NF-ToN-IoT | 1.000 | 0.071 | 0.179 | 0.021 | 0.570 | 0.740 | 0.226 | 0.020 | 0.095 | 0.089 |

The two datasets' `w` vectors are the finding, not the macro-F1 deltas. On UNSW the
consultant is already reliable almost everywhere — six of ten classes at ≥0.9, minimum
0.662 — so reweighting has little to reorder. On ToN six of ten classes sit at or below
0.226 (`dos` 0.021, `ransomware` 0.020, `backdoor` 0.071, `xss` 0.089, `scanning` 0.095,
`ddos` 0.179), which is the same weakness §2.4 measures as the binding constraint, now
quantified per class from train labels alone.

**Verdicts.** `gate` is reverted: +0.0038 UNSW, −0.0107 ToN. `gate+scale` is not kept:
+0.0057 UNSW / +0.0094 ToN, both below the pre-set 0.02 threshold and inside the training-
seed band. Both flags stay in the tree at their `off` default with tests, so the arms are
reproducible; neither becomes canonical.

### E3 — the loop's output fusion with AGAF's gate (`--fusion-block agaf`)

Hypothesis: AGAF's UNSW lead is its fusion block, not its inputs. The loop's 5-parameter
scalar gate plus concat→MLP was replaced by AGAF's feature-wise gate over
`[h, s, |h−s|, h·s]` followed by its feature attention — through the shared functions
`agaf_feature_gate_fuse` / `agaf_feature_attention` extracted from
`AGAFFusionEdgeClassifier`, which now calls them itself, so there is one implementation
rather than a copy that can drift. Output-fusion capacity rose from 125,266 to 199,242
parameters (projections included; only the block in use is allocated).

**Verdict: revert.** −0.1528 UNSW, −0.1101 ToN. This **refutes the premise** that AGAF's
lead is its gate: given the loop's own inputs, AGAF's fusion block is much worse than the
loop's own, at 1.6× the parameters. Whatever AGAF's advantage is, it is not the block.

#### The RNG-stream finding (a defect, caught by the `head_only` invariant)

The first E3 implementation moved `head_only` — 100% of logit elements, up to 0.064 — in a
mode that never reaches `_output_fusion` at all. The cause was not the fusion arithmetic:
the `agaf` block's differently shaped layers consume a different number of draws from the
**global RNG stream** during `__init__`, which shifted every dropout mask drawn afterwards
in training. Backbone and projection weights were verified bit-identical; only the RNG
position after construction differed. Fixed by always drawing the canonical `loop` block
from the main stream (discarded when unused) and building `agaf` under
`torch.random.fork_rng`, so the stream ends in the same place either way.

**General lesson for this repo:** any change to *what modules a model allocates* — not just
to what it computes — can move every other arm of an ablation through the shared RNG
stream. An unrelated-mode invariant (`head_only` unchanged) is what caught it; a macro-F1
comparison alone would not have. Pinned by
`tests/test_fusion_block.py::test_the_fusion_block_does_not_shift_the_global_rng_stream`.

### Trained head as consultant, v2, seed 42: echo confirmed

Re-measurement of the v1-era head-echo result (loop 0.8258 < head-alone 0.8321 ≈ AGAF-head
0.8331 on UNSW) under the v2 encoding. `llm_head_logits.pt` was verified current rather
than rebuilt: shapes `[5, 656, 10]` / `[5, 2127, 10]`, built 2026-08-30 from inputs that
all predate it, and re-scored to the same pooled OOF macro-F1 its build log reports. AGAF
consumed `edge_embeddings_oof.pt`, passed explicitly.

| Dataset | rung | macro-F1 | accuracy |
|---|---|---:|---:|
| UNSW-NB15 | head alone | 0.8321 | 0.8796 |
| UNSW-NB15 | AGAF-head | 0.8331 | 0.8811 |
| UNSW-NB15 | loop-head (real) | 0.8351 | 0.8872 |
| UNSW-NB15 | loop-head (head_only) | 0.7543 | 0.8171 |
| UNSW-NB15 | loop-head (random) | 0.6565 | 0.7607 |
| NF-ToN-IoT | head alone | 0.5165 | 0.9149 |
| NF-ToN-IoT | AGAF-head | 0.5141 | 0.9140 |
| NF-ToN-IoT | loop-head (real) | 0.5064 | 0.9093 |
| NF-ToN-IoT | loop-head (head_only) | 0.3809 | 0.8049 |
| NF-ToN-IoT | loop-head (random) | 0.3756 | 0.8373 |

ToN's AGAF figure is the 8-eval-class macro-F1 recomputed from `metrics.json["predictions"]`;
that file's own `overall_macro_f1` (0.4532) is a 10-class macro and is not a reportable ToN
number.

Mechanism detail on the two loop-head runs:

| Dataset | flagged | gated | flagged acc GNN / head / loop | churn all / flagged | flips +/− on flagged | flagged_bias_absmean |
|---|---|---|---|---|---|---:|
| UNSW-NB15 | 204/656 | 102 | 0.603 / 0.770 / 0.750 | 0.149 / 0.338 | +43 / −13 | 1.3430 |
| NF-ToN-IoT | 532/2127 | 266 | 0.359 / 0.759 / 0.724 | 0.189 / 0.609 | +225 / −31 | 1.3291 |

On both datasets the loop given the trained head lands within ~0.005 of head-alone and
AGAF-head (UNSW 0.8351 vs 0.8321 / 0.8331; ToN 0.5064 vs 0.5165 / 0.5141): it tracks its
consultant rather than improving on it. The v1 ordering (loop below both) does not survive
on UNSW under v2 — the loop is now the highest of the three by 0.0020 — but the three
remain within a span of 0.0030, and it does survive on ToN, where the loop is the lowest of
the three. **The echo characterisation stands on both datasets.** The trained head remains
a separate, stronger baseline and never the ladder's LLM rung (PROJECT_NOTES.md, §5).

---

## 2026-09-07 — Advice format: the bias path does not want saturated advice


> **Correction, 2026-09-13.** The 245/32 and 145/4 counts in the question below, and the
> flagged-set accuracies in the mechanism table above (0.603 / 0.770 / 0.750 and
> 0.359 / 0.759 / 0.724), come from trained-head runs at the PRE-RESELECTION knobs,
> `top_k` 31 and 25, and were computed inside training from live per-fold probabilities. That
> intermediate was never saved and no reconstruction from committed artifacts reproduces them.
> They are superseded by `results/consultant_complementarity.json`, which recomputes the same
> quantities at the canonical `top_k` 29 and 16 from the committed pooled OOF tensors, for both
> datasets and both consultants. The flagged/gated COUNTS in the table above (204/102 and
> 532/266) are correct for k = 31 and k = 25; at the canonical knobs they are 190/95 and
> 341/170. The question's reasoning is unaffected: the trained head's advice is still right far
> more often than it is wrong on the flagged disagreements, and Gate 0.5 still measures a loss.

**Question.** On ToN the trained head's advice on the flagged edges is right on 245
disagreements and wrong on 32 (145 vs 4 inside the confidence-gated half), yet Gate 0.5
measured a macro-F1 *loss* through the bias path while the oracle — one-hot true label at
±4 nats through the SAME projection — gains. Hypothesis: the projection cannot use raw
consultant logits but can use saturated one-hot advice, i.e. the problem is the advice
FORMAT, not selection.

**Answer: no.** Giving a realistic consultant the oracle's exact format does not recover
the oracle's behaviour on either dataset, in either setting. Seed 42 only.

`--advice-format {logits,onehot,softmax}`, default `logits` (canonical). `onehot` is the
oracle's own constructor applied to the consultant's argmax rather than the true label —
`ADVICE_SATURATION_MAGNITUDE = 4.0` is now one shared constant that
`mechanism_only_edge_injection` imports, so the two cannot drift. Selection and the
confidence gate keep ranking the raw logits; only the tensor handed to `bias_module`
changes. Full tables: `docs/experiment_log.md`.

### Diagnostic (trained head, output fusion OFF — Gate 0.5's setting)

| Dataset | control | logits | onehot | softmax | oracle_edge (seed 42) |
|---|---:|---:|---:|---:|---:|
| UNSW-NB15 | 0.7543 | 0.7312 | 0.7190 | 0.7355 | 0.7522 |
| NF-ToN-IoT | 0.3809 | 0.3644 | 0.3639 | 0.3839 | 0.5948 |

Every trained-head arm is below control on UNSW and two of three are below it on ToN. On
ToN the captured share of oracle headroom is −7.7% (logits), −7.9% (onehot), +1.4%
(softmax) — the saturated formats do not move it off zero.

**A caveat that matters for anyone quoting the UNSW headroom.** At seed 42 the UNSW oracle
arm scores 0.7522, *below* the 0.7543 control — headroom −0.0021. A "captured share of
headroom" is therefore not computable on UNSW at this seed; the +0.0345 in §0/§2.4 is a
3-seed mean. Report the raw difference from control instead, and never a UNSW share from a
single seed.

### The loop (canonical prototype, fusion ON)

| Dataset | format | real | head_only | real − head_only [CI] | Δ vs A |
|---|---|---:|---:|---|---:|
| UNSW-NB15 | logits (A) | 0.7642 | 0.7543 | +0.0098 [−0.018,+0.038] | — |
| UNSW-NB15 | onehot | 0.7452 | 0.7543 | −0.0093 [−0.045,+0.027] | −0.0191 |
| UNSW-NB15 | softmax | 0.7512 | 0.7543 | −0.0029 [−0.040,+0.034] | −0.0126 |
| NF-ToN-IoT | logits (A) | 0.4414 | 0.3809 | +0.0603 [+0.026,+0.096] | — |
| NF-ToN-IoT | onehot | 0.3810 | 0.3809 | +0.0007 [−0.042,+0.041] | −0.0596 |
| NF-ToN-IoT | softmax | 0.4194 | 0.3809 | +0.0388 [−0.012,+0.087] | −0.0215 |

Both formats are strictly worse than A on both datasets. The pre-set keep-threshold
(`real − head_only` must exceed A's by ≥0.02) is not approached from the right side by any
arm. Reverted; the flag stays at its `logits` default with tests.

### The finding worth keeping: saturating the input SHRINKS the injected advice

`flagged_bias_absmean`, the magnitude actually injected after the learned projection:

| Dataset | logits (A) | onehot | softmax |
|---|---:|---:|---:|
| UNSW-NB15 | 0.0767 | 0.0183 | 0.0205 |
| NF-ToN-IoT | 0.0234 | 0.0202 | 0.0194 |

The input magnitude went **up** (raw logits → ±4 nats) and the injected magnitude went
**down**, by 4x on UNSW. The projection is learned, so it adapts its weights to a larger,
lower-variance input and ends up injecting less. This is why "hand the realistic consultant
the oracle's format" does not reproduce the oracle: the oracle's advantage is not carried by
the format of the vector it presents. Combined with §2.4 (the channel has capacity, the
consultant does not use it) and the E1 result (per-class reliability reweighting does not
help), **three separate attempts to make the bias path usable by a realistic consultant have
now failed, from three different angles: selection, weighting, and format.**

Artifacts: `results/dev/advice_format/{task2_summary.json,task3_summary.json}` and the
per-run directories beneath.

---

## 2026-09-07 — Cross-fitted advice: the loop does not learn a trust it can be taught

**Premise, and it is real.** `build_llm_heads` stores, for outer fold f, the fold-f head
applied to every edge. On f's own train edges those logits are IN-FOLD — the head fitted
those labels — while on its test edges they are out-of-fold. On the entropy-flagged subset
the consultant is 0.85–0.98 accurate where the loop trains and 0.68–0.85 where it is scored
(per-fold table in `docs/experiment_log.md`). The loop has been training against advice more
reliable than the advice it is graded on. The oracle, the only consultant the channel
converts, is the only one whose reliability is identical on train and test.

**Hypothesis:** with cross-fitted advice the loop learns a realistic trust and the head's
test-time signal becomes usable. **Answer: no, and for an informative reason.**

`build_llm_heads --crossfit K` (default off, K=5) fills each outer fold's train rows with
inner out-of-fold logits and leaves val/test rows untouched, so the file's pooled OOF
macro-F1 is unchanged by construction and measured so (0.8321 / 0.5165). Train accuracy
moves toward test accuracy on every fold of both datasets. Leakage is asserted per fold:
permute fold f's test labels and row f returns bit-identical.

### The loop (fusion on, trained head as consultant, seed 42)

| Dataset | consultant | real | head_only | real − head_only | Δ vs in-fold | vs head alone |
|---|---|---:|---:|---|---:|---:|
| UNSW-NB15 | in-fold | 0.8351 | 0.7543 | +0.0811 [+0.043,+0.120] | — | +0.0030 |
| UNSW-NB15 | cross-fit | 0.8106 | 0.7543 | +0.0565 [+0.018,+0.095] | −0.0246 | −0.0215 |
| NF-ToN-IoT | in-fold | 0.5064 | 0.3809 | +0.1240 [+0.069,+0.181] | — | −0.0101 |
| NF-ToN-IoT | cross-fit | 0.4724 | 0.3809 | +0.0902 [+0.036,+0.147] | −0.0338 | −0.0441 |

Cross-fitting makes the loop **worse** on both datasets, and pushes it further below the
head alone. Mechanism-only (fusion off) the picture is the same on UNSW (−0.0230 in-fold →
−0.0654 cross-fit vs control) and unresolved on ToN (−0.0165 → +0.0070, CI [−0.030,+0.043],
which includes zero: +3.3% of oracle headroom).

### Why it is informative: the learned trust did not move

`learned_bias_strength` is the scalar the loop learns for how hard to push the advice. If
the in-fold optimism were what the loop was exploiting, making the advice honestly less
reliable should lower it. It does not:

| Dataset | setting | in-fold | cross-fit |
|---|---|---:|---:|
| UNSW-NB15 | fusion off | 1.5722 | 1.5664 |
| UNSW-NB15 | fusion on | 1.5227 | 1.5022 |
| NF-ToN-IoT | fusion off | 1.4768 | 1.4652 |
| NF-ToN-IoT | fusion on | 1.4842 | 1.4899 |

Four settings, largest change 0.0205, and on ToN with fusion on it moves the *wrong* way.
The loop is not calibrating its trust to the consultant's reliability at all — it is not a
trust parameter that in-fold advice was fooling. What *did* move is the injected magnitude
through the projection weights (`flagged_bias_absmean` 1.3430 → 1.1752 UNSW, 1.3291 →
1.0745 ToN, fusion on), i.e. the model absorbs the change in the projection rather than in
the scalar that is supposed to represent trust.

A second oddity worth recording: on ToN with fusion on, cross-fitting *raised* the loop's
accuracy on flagged edges (0.724 → 0.746) and raised its wrong→correct count (+225 → +240)
while pooled macro-F1 fell 0.5064 → 0.4724. The per-class table shows where: `xss` 0.3333 →
0.1667 and `backdoor` 0.4878 → 0.3793 — small rare classes losing more macro-F1 than the
flagged-edge gains recover.

**Not kept.** The keep-rule (cross-fit `real − head_only` exceeding in-fold's by ≥0.02 AND
the arm above the head alone) fails on both counts, both datasets. The flag stays in the
tree, default off, with tests; the canonical `llm_head_logits.pt` is untouched and the
canonical builder was verified to still reproduce it bit-for-bit after the refactor.

**Four failed angles now.** Selection (E1 reliability gate), weighting (E1 gate+scale),
format (one-hot / softmax at the oracle's magnitude) and now train/test reliability
matching. Combined with §2.4 — the channel has capacity, no realistic consultant uses it —
the constraint is not any of the things that have been varied.

Artifacts: `results/dev/crossfit/{task2_summary.json,task3_summary.json}` and the per-run
directories.

