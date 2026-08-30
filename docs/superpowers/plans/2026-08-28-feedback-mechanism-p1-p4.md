# Feedback Mechanism: Gate 0 + P1′–P4 Proposal and Priority Decision

> **Status:** decision document, not an execution plan. **A full bite-sized TDD plan will
> be written for whichever option is selected** — do not implement from this document.
>
> **Revision 2 (2026-08-28)** — supersedes revision 1 after Codex validation review.
> **All nine of that review's findings were independently verified and accepted**; two of
> revision 1's load-bearing premises were wrong and are retracted below. Revision 1's
> recommended sequence (`P3 → P1 → P2 → P4`) is **withdrawn**.

**Audience:** Codex (re-review and priority selection), then whoever implements.

**Spec this argues from:** `todo.md` Step 4 (Bidirectional Feedback Loop),
`docs/feedback_loop_mechanisms.md` (LOGIN / DAS / GLANCE / RoGRAD summaries).

---

## Corrections carried from the review

| Revision 1 claimed | Verified reality |
|---|---|
| Oracle headroom **+0.0508**, prototype captures **+0.0173**, cited to `unsw_nb15_edge_injection.json` | **Both values appear in no artifact.** `+0.0508` is the `feedback_minus_gnn` delta from `results/cross_dataset_comparison.json:63`, relabelled as oracle headroom. The authoritative contract is `results/unsw_nb15_oracle_ceiling.json`: control `0.5558` → oracle `0.5975` = **+0.0417**; prototype captures **+0.0028**. `CLAUDE.md:110` carried the same error and has been corrected. |
| P3's "36 of 39 injection dims have no continuous semantics" is a **known defect** | **Overstated.** Protocol/port embeddings are learned continuous latent vectors; adding a learned residual to them is not inherently meaningless. Refuted premise, not a defect. |
| The 10-class-logit channel is a **bottleneck** | **Refuted.** The oracle pushed a *perfect* answer through that same 10-logit path and delivered +0.0417. Ten numbers are demonstrably sufficient. Widening to 768-d would also grow the projection **~430 → ~30,000 params against 393 UNSW train edges**. |
| Frozen advice means the loop **cannot** compound "by construction" | **Empirical, not mathematical.** The flagged set does change each cycle, so a loop could in principle keep moving. Conclusion still stands on *better* evidence neither review cited: `results/unsw_nb15_edge_injection.json` already contains the sweep — `max_iterations_5` → 0.7327, `max_iterations_8` → 0.7501, actual iterations plateauing at **3.07** in both, its own reading "Keep 3. More iterations do not help." |
| The loop is the **top rung** on both datasets | **Highest point estimate**, not separated. UNSW Loop−AGAF CI crosses zero (P=0.778); ToN Loop−GNN `mean_diff +0.0189, CI [−0.0139, +0.0534], P=0.8615` — also crosses zero. |

---

## Why touch the feedback mechanism at all

Mechanism-only, fusion off, 3 seeds, pooled 5-fold OOF macro-F1. **Revision 3
(2026-08-30):** the `real − control` figures previously quoted here (+0.0064 UNSW,
+0.0026 ToN, from `results/unsw_nb15_edge_injection_v2.json`) came from a run that was
**not thread-pinned** and **do not replicate**. Gate 0 re-ran the same configuration
with pinning:

| dataset | run | control (`head_only`) | real | real − control |
|---|---|---:|---:|---:|
| UNSW | 2026-08-27, not pinned | 0.7280 | 0.7344 | +0.0064 |
| UNSW | **Gate 0, pinned** | 0.7398 | 0.7382 | **−0.0016** |
| ToN | 2026-08-27, not pinned | 0.4359 | 0.4386 | +0.0026 |
| ToN | **Gate 0, pinned** | 0.4195 | 0.4325 | **+0.0130** |

Levels moved 0.012–0.016 — the documented drift magnitude — and the sign flips on UNSW
while growing on ToN. Noise, not bias. Paired per-(seed,fold) bootstrap agrees: UNSW
−0.0046 P=0.356, ToN +0.0099 P=0.836. **`real > control` is retracted.** The mechanism is
still demonstrably *sensitive to advice content* (`random` degrades −0.0518/−0.0593,
`shuffled` −0.0228 on ToN) — but supplying real advice does not measurably beat supplying
none. That contract now carries a `retraction` block; `CLAUDE.md` is corrected.

**Gate 0 replaces the pre-v2 headroom table entirely** (the +0.0417 / 7% / 63% figures
below were pre-v2, `top_k=16, scale=10`, UNSW only):

| | UNSW | ToN |
|---|---:|---:|
| `oracle_edge` − control | **+0.0345** (paired +0.0369, CI[+0.0097,+0.0618], P=0.995) | **+0.1090** (paired +0.1354, CI[+0.0919,+0.1797], P=1.0) |
| `real_prototype_edge` − control | −0.0016 (P=0.356) | +0.0130 (P=0.836) |
| `oracle_attention` − control | −0.0016 (P=0.482) | +0.0114 (P=0.870) |
| prototype's captured share of headroom | **−4.5%** | **11.9%** |

**The channel works; the consultant does not use it.** A perfect consultant gets a cleanly
separated gain through the edge channel on both datasets. The canonical whitened prototype
captures ≈0% of it, and neither prototype figure is separated from zero. Attention remains
dead under v2 — third independent confirmation, now with CIs.

This is a **consultant-quality** finding, not a mechanism-capacity one, and it argues that
P1′/P3′ are surgery on a channel that already works. `CLAUDE.md`'s canonical rule mandates
the prototype consultant on every rung, precisely so cross-dataset comparisons stay
meaningful — so the live question is whether to *improve* the prototype scorer (it collapses
under class imbalance: ToN prototype 0.2785 vs a trained MLP on the same CySecBERT
embeddings at ~0.50) rather than to replace it or to reshape the channel.

## Global constraints

- **Score against `control_head_only`, never the bare GNN rung.**
- Harness: `python -m src.pipeline.step4.mechanism_only_edge_injection`, 3 seeds
  (42, 1, 2), canonical top_k/scale (UNSW 31.0/2.0, ToN 25.0/20.0). The runner now pins
  `OMP_NUM_THREADS=1`/`MKL_NUM_THREADS=1` and `torch.set_num_threads(1)` — without it
  runs drift ~0.012, larger than every effect here.
- Semantic consultant is the **whitened prototype scorer** on every rung. AGAF head
  fusion off.
- Any new hyperparameter: **≥3 seeds, validation folds only**, never re-picked on test.
  `injection_scale` re-selected whenever the injection path changes.
- **Every new shared knob must be added explicitly to the result contracts and to
  `tests/test_current_results.py::test_both_datasets_declare_the_same_architecture`** —
  that test compares declared fields and will not detect a knob nobody declared.

## Success criteria (replaces revision 1's churn-based criterion)

**Churn is not a success criterion.** It measures change, not improvement — oscillation
raises it too. Revision 1 was wrong to propose it. Every option below is judged on:

1. Paired per-seed/per-fold improvement over `control_head_only`, with CIs, **on both datasets**.
2. Iteration-2/3 predictions better than iteration-1 (not merely different).
3. Wrong→correct vs. correct→wrong transition counts, reported separately.
4. Change in the advice and in the selection set, as a *diagnostic* only.

---

## Gate 0 — Re-run the oracle under v2 (do this first)

> **Completed 2026-08-30. Decision: CONTINUE.** Pooled oracle-edge headroom is
> `+0.0345` on UNSW-NB15 and `+0.1090` on ToN-IoT, so the `<0.02 on both`
> stopping rule is not met. The canonical prototype captures `-4.5%` of pooled
> headroom on UNSW (prototype point estimate below control) and `11.9%` on ToN.
> Oracle attention remains unseparated from control on both datasets. See
> `results/unsw_nb15_oracle_ceiling_v2.json`,
> `results/ton_iot_oracle_ceiling_v2.json`, and the completed Gate 0 TDD plan.
>
> **Gate 0.5 refinement (2026-08-30):** a regenerated, leakage-free trained-head
> consultant captured only 12.5% of pooled headroom on UNSW (unseparated from
> control) and −38.0% on ToN (significant harm). A stronger standalone classifier
> is therefore not sufficient; the unresolved issue is realistic-advice
> calibration/translation through the bias path. See RESULTS_ARCHIVE.md §2.5.

**Why.** Every headroom argument in this document rests on a pre-v2 measurement taken at
non-canonical `top_k=16, scale=10`, on UNSW only. The v1→v2 encoding change moved the GNN
rung 0.5496 → 0.7219; there is no reason to assume the mechanism ceiling moved
proportionally, or at all. **If the v2 headroom is small, P1′–P4 are all competing for
nothing and the correct decision is to stop.** This is the cheapest experiment in the
document and it can invalidate the rest of it.

**Scope.** `control_head_only`, `real_prototype_edge`, `oracle_edge`, `oracle_attention`,
both datasets, 3 seeds, canonical top_k/scale, fusion off.

**Files.** Extend `src/pipeline/step4/mechanism_only_edge_injection.py` with an
`oracle_edge` arm (one-hot true label at ±4 nats through the same bias path); write
`results/unsw_nb15_oracle_ceiling_v2.json` + ToN equivalent, raw output to `results/raw/`.

**Cost.** Low — ~2 hours' work, one overnight run. **Risk.** Low.

**Decision rule.** If v2 headroom < ~0.02 on both datasets, **stop the program** and
report the mechanism as capacity-bound. If it is large and the prototype's share is again
small, the highest-value work is consultant quality, not mechanism surgery.

---

## P1′ — State-conditioned semantic adapter

*Renamed from revision 1's "state-conditioned re-consultation".* Reweighting cached
prototype scores by the GNN's posterior is **posterior conditioning, not re-consultation**:
no flow is re-serialized, CySecBERT is never re-run, the LLM receives no new query.
Revision 1's claim that this "makes todo.md's bidirectional claim true" was **wrong** and
is withdrawn. A real re-query is a separate, more expensive option — decide which one is
actually wanted before implementing either.

**Why.** Advice is computed once at `feedback_classifier.py:1056`, outside the loop at
`:1066`. Conditioning it on the GNN's current posterior makes each cycle a different
question. This is a legitimate mechanism change; it is not the thesis claim.

**Two variants — Codex should pick one:**

| | What it does | Cost | Delivers todo.md:81? |
|---|---|---|---|
| **1a adapter** | Reweight cached prototype scores by the GNN's top-*m* classes | ~1 day | no |
| **1b re-query** | Re-serialize the flagged flow *with the GNN's current verdict* and re-encode via `LiveCySecBERTScorer` (cache already keyed `(fold, edge_row_id)`) | ~3 days | **yes** |

**Files.** `feedback_classifier.py:1002-1037` (`_semantic_logits` gains `gnn_probs=None`,
byte-identical when `None`), `:1056` (move inside loop), `:1066-1105`; constructor
`:600-628` (`consult_top_m: int = 0`, `0` disables); `train_feedback.py` flag; tests in
`tests/test_biased_gat.py`.

**Blocking design flaw to fix before implementing — control contamination.** If advice is
conditioned on the target edge's own GNN posterior and *then* shuffled, the `shuffled`
control still carries target-specific class information, and the four-arm test that
underwrites the whole "LLM is a working consultant" claim silently stops being a control.
**Shuffle the complete conditioned package** so no target-specific prior survives. The
plan must state the conditioning/shuffling order explicitly; `_shuffle_perm:996-1000` and
`_semantic_logits:1014-1021` both need to move together.

**Risk.** Medium-high. Oscillation (mitigate: hold the calibrated entropy threshold fixed
within a fold, already current behaviour at `:68-88`); top-*m* restriction amplifies a
confident-but-wrong GNN.

---

## P3′ — Dedicated advice channel vs. all-dimension injection

*Narrowed from revision 1.* The 768-d widening is **dropped** (refuted by the oracle;
~70× parameter growth on 393 train edges). The "meaningless dims" framing is **retracted**.

**Why.** What survives is a clean, cheap, genuinely open question: `_one_pass:842` adds
the bias across all 39 encoded dims. Whether a dedicated appended channel does better is
untested and unknown — worth one experiment, framed as a comparison, not a bugfix.

**Mechanism.** Add `injection_target: str = "all"` (`"all"` | `"channel"`). **Keep class
logits as the advice source.** `"all"` reproduces today's model exactly.

**Files.** `feedback_classifier.py:823-854` (`_one_pass`), `:600-628` (constructor),
`:678-689` (the `bias_dim == edge_attr_dim` guard — `"channel"` widens `edge_attr`, so
`EdgeFeatureEncoder.out_dim`, `bias_dim` and this guard move together; the guard exists to
catch exactly this and will fail loudly); `train_feedback.py` flag; `tests/test_biased_gat.py`.

**Cost.** Cheap, ~half a day, **plus** a mandatory `injection_scale` re-selection per
target (current 2.0/20.0 were calibrated to all-dimension injection). **Risk.** Low.

---

## P2′ — Learned calibration of advice

*Reframed from revision 1's "per-edge trust".* Two corrections: (a) consultant
**correctness ≠ usefulness** — "the LLM was right" is not "injecting its advice helped
this edge"; (b) current behaviour is **not** uniform, as revision 1 implied — semantic
logits already produce edge-specific projected biases, and `_gate_flagged:932-961` already
selects by LLM confidence. What is global is only the `log_bias_strength` scalar at `:321`.

**Why.** On ToN the consultant scores **0.2785 against the GNN's 0.4290** — the loop
consults something worse than itself. A learned calibration term is the natural response.
**It must be evaluated against the existing confidence gate**, not against no gate, or the
comparison credits it with what the gate already does.

**Files.** `feedback_classifier.py:276-341` (`SemanticAttentionBias.forward(..., trust=None)`,
`None` reproduces current behaviour), `:932-994` (expose gate features); new
`TrustHead(nn.Module)`: `forward(features: [k,3]) -> [k,1]` in (0,1), **< 50 params**;
`train_feedback.py` (`--trust-loss-weight`).

**Cost.** ~1 day. **Risk.** Medium-high — ~61 UNSW / ~159 ToN train edges carry gradient
per pass. ToN is the primary evidence dataset.

---

## P4 — Advantage-based router (deferred)

**Why deferred.** Revision 1 underspecified the advantage label, and the cost estimate was
wrong. Edge interventions are **graph-coupled**: comparing one globally-biased pass against
one unbiased pass cannot attribute the loss change to an individual consulted edge. A
defensible per-edge counterfactual needs leave-one-out interventions or an explicitly
justified approximation — well beyond the "3+ days" claimed. With ~61/~159 gradient-carrying
edges, a bandit router is very likely under-determined regardless. **P2′ captures most of
the signal at a fraction of the cost.**

---

## Recommended sequence

**Gate 0 → (P1′ or P3′) → P2′ → P4 deferred.** This adopts Codex's recommended sequence.

1. **Gate 0 is unconditional.** It is cheap, and it can invalidate everything after it.
   Nothing else should start first.
2. **P3′ before P1′ if the goal is a clean cheap result** — half a day, low risk, a real
   comparison. **P1′ before P3′ if the goal is the thesis** — but then choose **variant 1b**,
   since 1a does not deliver the todo.md claim and should not be described as if it does.
3. **P2′ only if Gate 0 / P1′ / P3′ show usable headroom**, evaluated against the existing
   confidence gate.
4. **P4 stays deferred** pending a defensible counterfactual design.

**Open question for Codex, not resolved here:** if Gate 0 confirms the prototype captures
only ~7% of headroom while the trained head captures ~63%, is mechanism surgery the right
target at all, or is the canonical prototype-consultant rule the binding constraint? That
rule exists for cross-dataset comparability and is a thesis-level decision — but it should
be made deliberately rather than by default.

**Explicitly out of scope:** sweeping `max_iterations` / `churn_tol`. Already measured —
`max_iterations` 5 and 8 both plateau at 3.07 actual iterations, no improvement.

---

## Next step

Once Codex picks, the selected option gets a full TDD plan in this directory with
per-task failing tests, exact code, and commit boundaries, per `superpowers:writing-plans`.
Every task ends with a 3-seed mechanism-only run scored against `control_head_only`,
judged on the success criteria above — not on churn.
