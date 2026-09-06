# Feedback Loop: Fix the Two Signals the Loop Runs On

**Goal:** Make "uncertainty-guided semantic feedback" actually uncertainty-guided. Today
neither of the two signals the loop depends on carries information. Fix both, re-select
the two per-dataset knobs, re-measure at 3 seeds, and report honestly whether the loop
now beats AGAF. A negative answer is acceptable; a fabricated positive one is not.

**Evidence base:** `results/multiseed_ladder.json` (3-seed, both datasets),
`data/*/processed/step4_feedback/feedback_trace_real.json`, and the code reads below.
**Prior plans:** `2026-08-28-feedback-mechanism-p1-p4.md`, `2026-08-29-feedback-gate0-oracle-v2-tdd.md`.

## The diagnosis (read fully before touching code)

**Correction (2026-09-06, after Task 2):** the 3-seed sweep behind
`results/multiseed_ladder.json` ran `train_feedback` WITHOUT `--injection-scale`, so the
loop rung ran at the argparse default 10.0 on both datasets instead of the selected
2.0 (UNSW) / 20.0 (ToN). The canonical Aug-27 run used the correct scales
(`benchmark_summary.json` in `backups/pre_multiseed/`). Therefore in that file the
**loop numbers and every loop comparison are invalid**; the GNN, LLM and AGAF numbers do
not depend on the scale and stand. "AGAF above the loop at all three seeds" is NOT
established and must be re-measured at the correct scales (Task 4 below, revised).

What is established at 3 seeds: AGAF 0.7793 +/- 0.0116 on UNSW; AGAF spans 0.3975-0.4422
on ToN; no comparison is separated once seed variance is included. Per-class F1 shows the loop losing on
exactly the classes where the semantic branch should help (UNSW Fuzzers −0.10, Analysis
−0.06; ToN password −0.21, scanning −0.16). The mechanism is not doing what its name says,
for two concrete reasons, both visible in artifacts that already exist.

**Defect 1 — the selector reads a collapsed head.** Uncertainty selection uses
`probs = fusion_out(edge_emb).softmax()` inside `FeedbackLoopClassifier.forward`. In the
canonical prototype path, `_output_fusion` returns `logits = correction` — `fusion_out`'s
output never enters the loss except as two scalar gate features (`h_gnn`, `conf_gnn`). It
is trained to be a gate feature, not a classifier. Result, from
`feedback_trace_real.json` iteration-1 `per_class_f1` (the raw head, before the final
trace entry is overwritten with the fused logits at `feedback_classifier.py:1166`):

| dataset | fold0 | fold1 | fold2 | fold3 | fold4 |
|---|---|---|---|---|---|
| UNSW nonzero-F1 classes /10 | 2 | 1 | 2 | 1 | 1 |
| ToN  nonzero-F1 classes /10 | 2 | 2 | 1 | 1 | 1 |

The head predicts one class. Its entropy is not "how unsure the GNN is about this edge".
The top-31% / top-25% "most uncertain" set is therefore arbitrary, and so is where the
advice lands. This also means `top_k_percent` was *selected* on this arbitrary signal.

**Defect 2 — the consultant's confidence is flat.** `WhitenedPrototypeScorer` returns
`cos / T` with `T` initialised to `10.0` (`initial_log_temperature = log(10)`). Cosines on
whitened CySecBERT embeddings span roughly [−0.2, 0.6]; divided by 10 the logits span
~0.08, so the softmax is uniform to three decimals. Measured directly from
`prototypes.pt`: max class-probability mean 0.125 (UNSW) / 0.110 (ToN) even at T=1;
uniform is 0.100. Consequences:
- The confidence gate (`bias_confidence_frac=0.5`, keep the most confident half) ranks
  on the third decimal. It is selecting noise.
- `mean_disagreement` = `1 − p_consultant[GNN's class]` sits at **exactly 0.900** in
  every fold, both datasets, at evaluation time (`trace_summary`) — i.e. 1 − 1/10. The
  learned per-class temperature did not sharpen it.
- The injected bias is `projection(cos/T) · strength`, so the advice vector's magnitude is
  tiny and its direction is a near-flat 10-vector.

**What the loop's gain over `head_only` actually is.** `head_only` disables output fusion
and returns `fusion_out` directly, so it is a plain GNN. `real` beats it by +0.01–0.03
(UNSW) via `_output_fusion` = `concat[(1−g)·proj(edge_emb), g·proj(llm_emb)]` on **all**
edges — that is late fusion, the same thing AGAF does, with a weaker gate (a 4→1 sigmoid
on four near-constant scalars vs AGAF's learned 256→2 softmax with BatchNorm projectors).
The feedback injection itself contributes approximately nothing, which is what Gate 0
already measured (prototype captures ≈0% of the oracle's headroom). **The loop is a worse
AGAF with an inert mechanism attached.** That is why AGAF wins.

## Global Constraints

- `export OMP_NUM_THREADS=1` for every run. Fold partition fixed (seed-42 `folds.pt`);
  never rebuild splits.
- Score every change against `head_only` AND against AGAF, at seeds 42, 1, 2, with
  `python -m src.pipeline.step4.aggregate_multiseed`. Single-seed numbers are not results.
- Do NOT swap the trained LLM head in as consultant (loop echoes it — archive §4). Do NOT
  sweep `max_iterations`/`churn_tol`. Do NOT add node/edge features. Do NOT oversample.
- Re-select `top_k_percent` and `injection_scale` on validation folds, 3 seeds, after the
  fix — CLAUDE.md mandates this whenever the injection path changes, and Defect 1 means
  the current values were chosen on a meaningless signal. Never re-pick on test.
- Every task starts with a failing test that would have caught the defect. Tests live in
  `tests/`; run `OMP_NUM_THREADS=1 python -m pytest tests/ -q` before each commit.
- If a task's result is negative, record it in `docs/RESULTS_ARCHIVE.md` and move on. Do
  not tune until it turns positive.

---

### Task 1: Make the selector head a classifier

- [ ] **1.1** Red test `tests/test_feedback_selector_head.py`: build the canonical model,
  train one UNSW fold in `real` mode for a short schedule, then assert the iteration-1
  trace head has nonzero F1 on ≥ 6 of 10 classes and macro-F1 within 0.15 of `head_only`
  on the same fold. Must fail on current code (currently 1–2 classes).
- [ ] **1.2** In `FeedbackLoopClassifier._output_fusion`, prototype path: return
  `logits = correction + (1 − g)·gnn_logits + g·llm_logits_proj` (the residual form the
  `head_logits` path already uses at `:929`), where `llm_logits_proj` is a learned
  `Linear(num_classes → num_classes)` over the scorer's logits. Alternative (simpler, try
  first): add `criterion(gnn_logits[train_mask])` to the training loss in
  `train_feedback._train_one_fold` with weight 1.0. Pick whichever makes 1.1 pass with
  the smaller diff; record the other as not taken.
- [ ] **1.3** Add `selector_head_macro_f1` (iteration-1 head, test edges) to
  `bias_diagnostics` in the trace so this can never regress silently.
- [ ] **1.4** Commit.

### Task 2: Give the consultant a usable confidence

- [ ] **2.1** Red test: load `prototypes.pt` fold 0, run `WhitenedPrototypeScorer` at its
  init, assert `mean_disagreement` over a random 30% of edges is < 0.80 and mean max-prob
  > 0.25. Must fail (currently 0.90 / ~0.10).
- [ ] **2.2** Change `initial_log_temperature` so the init softmax is informative. Do not
  guess: compute on train-fold cosines the T for which mean max-prob ≈ 0.5 (expect
  ~0.05–0.1) and use that. Keep the per-class temperature learnable.
- [ ] **2.3** Add `consultant_mean_max_prob` and `mean_disagreement` to
  `bias_diagnostics`. Assert in the test that post-training `mean_disagreement` < 0.80.
- [ ] **2.4** Commit.

### Task 3: Re-select the two per-dataset knobs on the fixed signals

- [ ] **3.1** Run `sweep_top_k` for both datasets, range 15–35, 3 seeds, validation only.
  Then sweep `injection_scale` over {0.5, 1, 2, 5, 10, 20} at the selected k. Write both
  into `selected_feedback_config.json` with `source` updated and the old values kept under
  `superseded`.
- [ ] **3.2** Contract test: both datasets' configs carry the new `source` string and the
  `superseded` block.
- [ ] **3.3** Commit.

### Task 4: Measure, three seeds, both datasets

- [ ] **4.1** Re-run `oof_emb → fusion → oof → feedback → ladder` at seeds 1, 2, 42 (42
  last) using the sweep script pattern in this session's scratch (`run_multiseed.sh`), then
  `aggregate_multiseed`. Write to `results/multiseed_ladder_v2.json`; leave the v1 file.
- [ ] **4.2** Report, per dataset: loop vs AGAF, loop vs head_only, loop vs GNN — two-level
  CI and sign stability. Also report the new `selector_head_macro_f1` and
  `mean_disagreement` so the reader can see the two defects are gone.
- [ ] **4.3** Decision rule, fixed in advance: the loop "beats AGAF" only if the two-level
  CI for `loop_vs_agaf` excludes zero on that dataset. Otherwise write "not separated". No
  other wording.
- [ ] **4.4** Update `docs/RESULTS_ARCHIVE.md` with a dated section; update the memory
  note `feedback-loop-selector-diagnosis` with the outcome.
- [ ] **4.5** Commit.

### Task 5 (only if Task 4 is not separated): content-bidirectionality

The consultant's verdict is state-independent, so the loop reaches a fixed point after one
correction (see memory `feedback-loop-bidirectionality-framing`). If Tasks 1–2 do not
separate the loop from AGAF, the remaining lever is making the advice depend on the GNN's
current belief.

- [ ] **5.1** Restrict the scorer to the GNN's current top-2 classes for flagged edges
  (mask other prototypes to −inf before softmax) so each iteration asks a different
  question. Red test: advice on a flagged edge must change between iteration 1 and 2 when
  the GNN's top-2 changes.
- [ ] **5.2** Measure exactly as Task 4. Same decision rule.

## Out of scope

Superseding `results/*_current.json` and the CLAUDE.md results table — a separate decision
the user has not yet made. Do not touch them in this plan.
