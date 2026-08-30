# Gate 0: v2 Oracle-Ceiling Re-run — TDD Execution Plan

> **Status:** **completed 2026-08-30**. Implements the unconditional first
> decision gate from `2026-08-28-feedback-mechanism-p1-p4.md`. The prespecified
> decision is **continue**: oracle edge headroom exceeds 0.02 on both datasets.

## Observed Gate 0 result

| dataset | control | prototype edge | oracle edge | oracle attention | oracle headroom |
|---|---:|---:|---:|---:|---:|
| UNSW-NB15 | 0.7398 | 0.7382 | 0.7743 | 0.7382 | **+0.0345** |
| ToN-IoT | 0.4195 | 0.4325 | 0.5286 | 0.4310 | **+0.1090** |

Paired seed/fold oracle-edge intervals exclude zero on both datasets: UNSW mean
difference `+0.0369`, 95% CI `[+0.0097, +0.0618]`; ToN `+0.1354`, 95% CI
`[+0.0919, +0.1797]`. The prototype captures `-4.5%` of pooled oracle headroom
on UNSW (its point estimate is below control) and `11.9%` on ToN. Oracle
attention remains unseparated from control on both datasets.

Artifacts: `results/raw/oracle_ceiling_v2.{json,log}` plus
`results/{unsw_nb15,ton_iot}_oracle_ceiling_v2.json`. Full suite: 58 passed.

**Interpretation:** Gate 0 does not stop P1′–P4 for lack of mechanism capacity.
It strengthens the consultant-quality concern: the canonical prototype leaves
most available edge-channel headroom unrealized. Choosing consultant work versus
P1′/P3′ remains the next thesis-level decision; no later option started here.

**Goal:** Measure the feedback mechanism's v2 ceiling on UNSW-NB15 and ToN-IoT
using the canonical per-dataset `top_k_percent` and `injection_scale`, with
output fusion disabled in every arm.

**Required arms:**

1. `control_head_only` — feedback structurally absent.
2. `real_prototype_edge` — canonical whitened-prototype consultant, edge injection.
3. `oracle_edge` — true-label logits at −4/+4 nats, edge injection.
4. `oracle_attention` — the same true-label logits, literal attention injection.

**Decision rule:** if `oracle_edge − control_head_only < 0.02` on both datasets,
stop P1′–P4 and report the mechanism as capacity-bound. Otherwise, report the
remaining headroom and the prototype's captured share before choosing later work.

**Non-negotiable experiment controls:** seeds `(42, 1, 2)`; five existing folds;
v2 graph artifacts; UNSW `top_k=31.0, scale=2.0`; ToN `top_k=25.0, scale=20.0`;
whitened prototypes for the real arm; `use_output_fusion=False`; single-threaded
BLAS; test folds used only for final scoring. The oracle is deliberately leaky and
must be labelled diagnostic-only in every artifact.

---

## Task 1 — Pin the arm configuration and oracle-logit contract

**Files:**

- Modify `src/pipeline/step4/mechanism_only_edge_injection.py`.
- Create `tests/test_mechanism_only_edge_injection.py`.

### 1.1 Write failing unit tests

Add focused tests that assert:

- `SEEDS == (42, 1, 2)` and both dataset setups retain their canonical top-k/scale.
- The arm table declares the four required arms with explicit `feedback_mode`,
  `injection_mode`, and advice source.
- `_oracle_logits(labels, num_classes, magnitude=4.0)` has shape `[E, C]`, uses
  `+4.0` at the true class and `-4.0` everywhere else, preserves device/dtype,
  and rejects invalid labels or non-positive magnitude.
- Oracle arms always set `use_output_fusion=False` and pass their logits only via
  `head_logits`; the real arm passes `head_logits=None` and therefore uses the
  loaded whitened prototype scorer.

Run:

```bash
.venv/bin/python -m pytest tests/test_mechanism_only_edge_injection.py -q
```

Expected: fail because the arm metadata and oracle helper do not yet exist.

### 1.2 Implement the smallest passing code

- Replace the string-only `ARMS` list with immutable arm specifications.
- Implement `_oracle_logits` with `torch.full(..., -magnitude)` plus `scatter_`.
- Keep the existing control mappings byte-for-byte equivalent where applicable.
- Add a small `_train_arm_fold(...)` adapter so tests can verify call arguments
  without loading graphs or running training.

Re-run the focused test and expect it to pass.

**Commit boundary:** `test(step4): pin v2 oracle arm contract`

---

## Task 2 — Preserve fold-level and iteration-level evidence

The proposal's success criteria cannot be reconstructed from the current runner,
which stores only a seed-level pooled score and average churn. Capture the evidence
during the run rather than attempting to infer it afterward.

**Files:**

- Modify `src/pipeline/step4/train_feedback.py`.
- Modify `src/models/feedback_classifier.py`.
- Modify `tests/test_mechanism_only_edge_injection.py`.
- Add focused tests to `tests/test_feedback_loop.py` if that existing suite owns
  trace behavior; otherwise keep them in the new runner test file.

### 2.1 Write failing trace tests

For a tiny deterministic synthetic trace, require:

- Each iteration row exposes `test_macro_f1`, `wrong_to_correct`, and
  `correct_to_wrong` relative to iteration 1 (iteration 1 counts are zero).
- Each trace row exposes the final gated selection count and its Jaccard overlap
  with the prior iteration's selected set; iteration 1 has no prior overlap.
- `head_only` has zero selected edges and no semantic-advice diagnostics.
- Fixed oracle/prototype advice is declared unchanged across iterations rather
  than being misrepresented as newly queried advice.

Run the focused tests and expect failure on the missing fields.

### 2.2 Add trace fields without changing predictions

- In `FeedbackLoopClassifier.forward`, compute the gated selection once per
  iteration and reuse it for both trace diagnostics and bias construction.
- Record only scalar diagnostics in the public trace: `selected_count`,
  `selection_jaccard_previous`, and `advice_recomputed=False`. Do not retain graph
  tensors in JSON-bound structures.
- In `_train_one_fold`, compare each iteration's test predictions with iteration
  1 and record wrong→correct and correct→wrong counts.
- Preserve early stopping, selector behavior, bias construction, and final logits.

Run:

```bash
.venv/bin/python -m pytest tests/test_mechanism_only_edge_injection.py tests/test_feedback_loop.py -q
```

**Commit boundary:** `feat(step4): retain oracle iteration transition evidence`

---

## Task 3 — Add paired statistics and a stable raw schema

**Files:**

- Modify `src/pipeline/step4/mechanism_only_edge_injection.py`.
- Modify `tests/test_mechanism_only_edge_injection.py`.

### 3.1 Write failing statistics/schema tests

Using small fabricated fold records, assert that:

- `_paired_fold_ci(candidate, control, seed=...)` pairs by `(seed, fold)` and
  rejects missing or duplicate pairs.
- Identical inputs produce zero mean difference and `[0, 0]` CI.
- A uniformly better candidate produces a positive mean difference and CI.
- The raw payload declares schema version, dataset configuration, encoding,
  seeds, folds, arm semantics, per-seed pooled metrics, per-fold metrics,
  per-iteration metrics, transition totals, paired CIs, and elapsed time.
- The payload explicitly declares `use_output_fusion=false`,
  `semantic_consultant=whitened_prototype` for the real arm,
  `oracle_logit_magnitude=4.0`, and `diagnostic_label_leakage=true` for oracle arms.

### 3.2 Implement deterministic paired inference

- Preserve unrounded scores for calculations; round only presentation copies.
- Compute paired bootstrap CIs by resampling the 15 matched `(seed, fold)` cells
  with a fixed RNG seed. Report mean difference, 2.5/97.5 percentiles, and the
  proportion positive.
- Store pooled OOF macro-F1 per seed as the primary metric, plus fold metrics for
  pairing. Do not average fold F1 and call it pooled F1.
- Aggregate iteration-2/3 improvement and wrong→correct/correct→wrong counts
  separately. Churn and selection changes remain diagnostics, never pass criteria.

Run the focused test and expect it to pass.

**Commit boundary:** `feat(step4): add paired Gate 0 evidence schema`

---

## Task 4 — Make result writing atomic and produce both authoritative contracts

**Files:**

- Modify `src/pipeline/step4/mechanism_only_edge_injection.py`.
- Modify `tests/test_mechanism_only_edge_injection.py`.
- Later generate `results/raw/oracle_ceiling_v2.json` and
  `results/raw/oracle_ceiling_v2.log`.
- Later generate `results/unsw_nb15_oracle_ceiling_v2.json` and
  `results/ton_iot_oracle_ceiling_v2.json`.

### 4.1 Write failing output tests

Patch the training adapter with deterministic fake fold results and assert that:

- `run()` writes one raw cross-dataset payload and two dataset-specific contracts.
- Parent directories are created.
- Files are written through a temporary sibling and replaced only after complete
  JSON serialization, so an interrupted overnight run cannot leave valid-looking
  partial JSON.
- `FB.SEED` is restored in a `finally` block even when one fold raises.
- The decision section calculates total headroom, prototype gain, captured share,
  and the exact stop/continue outcome from the unrounded means.

### 4.2 Implement output assembly

- Give `run()` explicit raw and contract output paths while preserving a useful
  repo-root default.
- Emit each dataset contract from the same in-memory raw records; never manually
  transcribe scores.
- Include a `reproduce` command and supersession link to the pre-v2
  `results/unsw_nb15_oracle_ceiling.json`.
- Keep console progress sufficient for the committed raw log.

Run the focused tests and expect them to pass.

**Commit boundary:** `feat(step4): write atomic v2 oracle ceiling contracts`

---

## Task 5 — Extend repository result-contract checks

**Files:**

- Modify `tests/test_current_results.py`.
- Modify `tests/test_mechanism_only_edge_injection.py` if shared schema helpers
  need direct coverage.

### 5.1 Add the failing authoritative-artifact tests

Require both v2 oracle files to declare the same shared architecture knobs:

- encoding, seeds, fold count, arm set, max iterations, churn tolerance,
  bias-confidence fraction, gate mode, consultant identity, output-fusion state,
  injection modes, oracle magnitude, and deterministic thread settings;
- allow only validation-selected `top_k_percent` and `injection_scale` to differ;
- require all four success-criterion evidence groups and a decision outcome;
- reject reportable-performance wording for the deliberately leaky oracle arms.

This test should initially fail because the overnight experiment artifacts do not
exist yet. Do not weaken it or insert fabricated scores.

### 5.2 Run all non-artifact tests

```bash
.venv/bin/python -m pytest \
  tests/test_mechanism_only_edge_injection.py \
  tests/test_feedback_loop.py \
  tests/test_biased_gat.py -q
```

Then run the full suite while excluding only the two not-yet-generated artifact
assertions, documenting the exact deselection expression in the raw log.

**Commit boundary:** `test(results): require both v2 oracle contracts`

---

## Task 6 — Execute Gate 0 and apply the decision rule

No hyperparameter is selected here. This task only runs the prespecified protocol.

### 6.1 Preflight

Confirm all six graph/fold/prototype artifacts exist, the graph encoding declares
v2, the worktree diff contains only intended changes, and focused tests pass.

### 6.2 Run the experiment

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m src.pipeline.step4.mechanism_only_edge_injection \
  2>&1 | tee results/raw/oracle_ceiling_v2.log
```

Do not run other CPU-heavy training concurrently.

### 6.3 Validate generated evidence

```bash
.venv/bin/python -m json.tool results/raw/oracle_ceiling_v2.json >/dev/null
.venv/bin/python -m json.tool results/unsw_nb15_oracle_ceiling_v2.json >/dev/null
.venv/bin/python -m json.tool results/ton_iot_oracle_ceiling_v2.json >/dev/null
.venv/bin/python -m pytest tests/test_current_results.py tests/test_mechanism_only_edge_injection.py -q
.venv/bin/python -m pytest -q
```

Manually reconcile console totals with JSON, verify all 15 fold cells per arm and
dataset are present, and confirm no NaN/Infinity appears in committed JSON.

### 6.4 Stop at the gate

Report, for each dataset:

- control, real-prototype, oracle-edge, and oracle-attention pooled means/stds;
- paired oracle−control and prototype−control CIs;
- iteration 2/3 changes and wrong→correct versus correct→wrong totals;
- selection-set change diagnostics and the fact that advice was fixed;
- total headroom and prototype share.

Then apply the prespecified `<0.02 on both datasets` rule. Do not begin P1′, P2′,
P3′, or P4 in the same step, regardless of the outcome.

**Commit boundary:** `results(step4): record v2 oracle ceiling on both datasets`

---

## Definition of done

Gate 0 is complete only when the implementation and full test suite pass, the raw
log/payload and both authoritative dataset contracts are generated from one run,
all required success-criterion evidence is present, and the stop/continue decision
is recorded without presenting oracle scores as deployable performance.
