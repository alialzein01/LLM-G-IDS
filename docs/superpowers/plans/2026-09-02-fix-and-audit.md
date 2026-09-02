# Fix, Audit and Review Plan

**Goal:** Close the one outstanding baseline task, correct four factual errors in the
report's Part 1, then audit everything finished so far and report what is next.

**Spec:** `docs/superpowers/specs/2026-08-30-sota-baseline-comparison-design.md`
**Prior plan:** `docs/superpowers/plans/2026-08-30-sota-baseline-comparison.md` (Tasks 1-9 complete)

## Global Constraints

- `export OMP_NUM_THREADS=1` for every run.
- Do not re-run training to "improve" any number. This plan fixes text, completes Task 10,
  and audits. It produces no new experimental results.
- Do not start the 3-seed rung re-run. It is deliberately postponed.
- Report scope is undecided (UNSW-only vs both datasets). Do NOT resolve it yourself and do
  NOT edit Part 1's dataset framing beyond the edits specified in Task B.
- Every claim must trace to an artifact. If you cannot point to the file a number came
  from, flag it rather than restating it.

---

### Task A: Complete Task 10 of the baseline plan

Task 10 was never run. Follow it exactly as written in
`docs/superpowers/plans/2026-08-30-sota-baseline-comparison.md`.

- [ ] **A1.** Step 0 — write the `known_ceilings` block into BOTH
  `results/unsw_nb15_sota_baselines.json` and `results/ton_iot_sota_baselines.json`.
  Literal content is in that plan.
- [ ] **A2.** Step 1 — write `tests/test_sota_baselines.py` contract tests (folds hash,
  eval classes, deviations D1-D5, the ceiling block). The file currently holds only the
  Task 9 test; add to it, do not overwrite it.
- [ ] **A3.** Step 2 — run the full suite: `OMP_NUM_THREADS=1 python -m pytest tests/ -v`.
  All must pass. Paste the output.
- [ ] **A4.** Step 3 — add the `docs/RESULTS_ARCHIVE.md` section. Record as headline
  findings the CI results, NOT the large margins:
  - UNSW vs TE-G-SAGE + our node features: GNN +0.0027, LLM +0.0160, AGAF +0.0400 are all
    NOT separated; only Loop +0.0535 [+0.0102, +0.0990] is separated.
  - ToN vs E-GraphSAGE as published: our LLM (-0.1338) and AGAF (-0.0812) are
    significantly BEHIND; GNN and Loop are not separated from it.
  - The +0.60 margins against E-GraphSAGE on UNSW are ceiling-limited and are not
    architectural evidence.
  - Quote the `comparison_caveat` wherever a CI appears.
- [ ] **A5.** Step 4 — add the Traps bullet to `CLAUDE.md`.
- [ ] **A6.** Commit.

---

### Task B: Correct four errors in report Part 1

File: `docs/research_report/sections/01_introduction.tex`.
Make ONLY these four edits. Do not restyle, reorder, or rewrite anything else.

- [ ] **B1 — the mechanism is described wrong (must-fix).**

The canonical configuration is `injection_mode: edge` on both datasets
(`results/*_current.json`), and CLAUDE.md records attention injection as inert (churn
exactly 0.0000 in every configuration). The introduction currently describes the inert
mechanism as though it were the one producing the results.

Replace:

> Phase~2 adds uncertainty-guided semantic feedback: edges with high-entropy GNN
> predictions are selected for semantic consultation, and the resulting signal is
> reintroduced as a bias in the next graph-attention update.

with:

> Phase~2 adds uncertainty-guided semantic feedback: edges with high-entropy GNN
> predictions are selected for semantic consultation, and the resulting signal is
> reintroduced by modifying the edge representations that the next graph update consumes.
> A second design, which biases graph attention directly, was also implemented and
> measured; it produced no change in prediction and is reported as a verified negative
> result.

- [ ] **B2 — contribution 3 repeats the same error.**

Replace `uncertainty-guided semantic feedback that injects semantic evidence into graph
attention for uncertain edges` with `uncertainty-guided semantic feedback that
reintroduces semantic evidence into the edge representations of uncertain edges`.

- [ ] **B3 — RQ3 presumes the attention path.**

In RQ3, replace `does the gain come from semantic re-entry into graph attention or from
final-output fusion?` with `does the gain come from semantic re-entry into the graph
representation or from final-output fusion?`

- [ ] **B4 — no research question covers the external baselines, and contribution 4
  omits them.**

Add RQ5 after RQ4, in the same `description` list:

> \item[RQ5.] How does the feedback loop compare against published graph-based intrusion
> detection architectures re-trained on the same aggregated representation under the same
> protocol, and how much of any observed difference is attributable to representation and
> tuning rather than to architecture?

In contribution 4, extend the list of evaluations to include published baselines. Replace
`trained-head baselines, and mechanism-isolation tests` with `trained-head baselines,
mechanism-isolation tests, and two published graph-based intrusion detection
architectures re-trained on the same representation`.

In the `Report structure` subsection, the sentence describing the results section
currently reads `The results section presents the NF-UNSW-NB15 performance ladder,
statistical comparisons, ablations, and verified negative results.` Insert
`comparisons against published baselines,` before `ablations,`.

- [ ] **B5.** Rebuild the PDF if a LaTeX toolchain is available; if not, say so rather
  than guessing. Commit.

---

### Task C: Audit what is finished

Produce a written audit. Do not fix what you find beyond Tasks A and B — report it.

- [ ] **C1 — deliverables.** Confirm every file the baseline plan promised exists:
  `src/models/baselines/{e_graphsage,te_g_sage}.py`,
  `src/pipeline/baselines/{preprocess,harness,run_baselines,compare_to_ladder}.py`,
  both `results/*_sota_baselines.json`, the 12 `[3, E]` prediction matrices,
  `tests/test_sota_baselines.py`. List anything missing.

- [ ] **C2 — provenance.** For both datasets, recompute the sha256 of
  `aggregated_edges.csv` and `folds.pt` and confirm they match `data_provenance` in the
  results file. A mismatch means the results no longer describe the data on disk.

- [ ] **C3 — internal consistency.** Verify for both datasets that:
  - every `baselines.<model>.<mode>.macro_f1` equals the mean of its per-seed values;
  - every prediction matrix has shape `[3, E]` with `E` equal to the graph's edge count;
  - every `plus_node_features` entry has `is_faithful_to_paper: false` and a
    `variant_label`, and no `as_published`/`refit` entry is marked non-faithful;
  - `statistical_comparisons` and `seed_matched_comparisons` each hold 24 keys.

- [ ] **C4 — stale claims.** Grep the repository for text that contradicts current
  artifacts. Known classes of error to look for, but do not limit yourself to these:
  - any claim that a paper does not state an epoch count (both do: 4999 and 20);
  - any claim that E-GraphSAGE uses plain unweighted cross-entropy (its released code
    uses class weights);
  - any description of the feedback loop as biasing graph attention;
  - any phrase asserting we outperform a published paper's own reported numbers;
  - any use of "top rung" where "highest point estimate" is required.
  Report file and line for each hit. Fix only the ones inside Task B's scope.

- [ ] **C5 — report state.** Record the current state of `docs/research_report/`:
  which of the five parts exist, the status table in `REPORT_WORKFLOW.md`, the count of
  entries in `references.bib`, and whether `figures/` and `tables/` are empty.

- [ ] **C6.** Write the audit to `docs/reports/2026-09-02 - Baseline Work Audit.md` and
  commit.

---

### Task D: Report and stop

- [ ] **D1.** Post a summary containing:
  - what Tasks A and B changed;
  - the C1-C5 audit findings, with anything broken listed first;
  - the full test suite result;
  - a short "what is next" list, in priority order, distinguishing work that is blocked
    on a user decision from work that is not.
- [ ] **D2.** Do not begin anything in that list. Stop and wait.

Known open items to include in D1's "what is next", plus anything you find:

1. **User decision — report scope.** `REPORT_WORKFLOW.md` restricts the report to
   NF-UNSW-NB15, but results exist for both datasets, and the ToN finding (a published
   baseline significantly beating two of our rungs) currently has nowhere to live.
2. **User decision — approve Part 1**, which the workflow requires before Part 2 is
   drafted.
3. **Postponed by the user — 3-seed rung re-run.** Our rungs ran at seed 42 only, so no
   CI includes rung-side seed variance. Needs a `--seed` flag on
   `src/pipeline/step3/build_oof_gnn_embeddings.py` (currently hardcoded `SEED = 42`) and
   per-seed output directories to avoid overwriting the seed-42 artifacts that
   `results/*_current.json` depends on. Do not start it.
