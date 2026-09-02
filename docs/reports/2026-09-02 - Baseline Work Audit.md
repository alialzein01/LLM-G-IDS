# Baseline Work Audit — 2026-09-02

Audit of the SOTA baseline comparison (Tasks 1–10) and the report's Part 1.
Scope: verification only. No experimental results were produced or re-run.

## Verdict

**No broken artifacts.** All deliverables present, all provenance hashes match, all
internal consistency checks pass, full suite green. One documentation violation was found
and fixed; three historical result artifacts carry retired phrasing and were deliberately
left alone.

## C1 — Deliverables

All present:

| File | |
|---|---|
| `src/models/baselines/{e_graphsage,te_g_sage}.py` | OK |
| `src/pipeline/baselines/{preprocess,harness,run_baselines,compare_to_ladder}.py` | OK |
| `results/{unsw_nb15,ton_iot}_sota_baselines.json` | OK |
| `tests/test_sota_baselines.py` | OK (5 tests) |
| Prediction matrices | 12 of 12 |

## C2 — Provenance

`aggregated_edges.csv` and `folds.pt` sha256 recomputed and matched against
`data_provenance` for both datasets. **All four match** — the results still describe the
data on disk.

## C3 — Internal consistency

For all 12 model × mode × dataset cells:

- every `macro_f1` equals the mean of its per-seed values (exact to 1e-9);
- every prediction matrix is `[3, E]` with E = 656 (UNSW) / 2127 (ToN);
- `plus_node_features` is the only mode marked `is_faithful_to_paper: false`, and each
  such entry carries a `variant_label`;
- `statistical_comparisons` and `seed_matched_comparisons` hold 24 keys each, per dataset.

## C4 — Stale claims

Searched all tracked `.md`, `.tex`, `.py`, `.json` outside `.venv` and the plans
directory.

| Class | Result |
|---|---|
| "neither paper states an epoch count" | none |
| E-GraphSAGE uses plain/unweighted cross-entropy | none |
| feedback described as biasing graph attention | none outside the corrected negative-result sentence |
| "outperforms \<paper\>" | none |
| "top rung" | **1 violation, fixed**; 3 historical, left |

**Fixed:** `docs/RESULTS_ARCHIVE.md:65` read "loop is the top rung", violating the
project's own rule stated eleven lines below it. Now "loop is the highest point estimate".

**Left deliberately:** `results/unsw_nb15_loop_mechanism.json`,
`results/unsw_nb15_edge_injection.json`, `results/ton_iot_current.json` each use "top
rung" in narrative fields. These are records of past runs; rewriting a result artifact to
match current phrasing would falsify the record. **Do not quote those narrative fields in
the report without rephrasing.**

## C5 — Report state

| Part | Status |
|---|---|
| 1 Introduction and Context | Review |
| 2 Background and State of the Art | Waiting |
| 3 Method | Waiting |
| 4 Results | Waiting |
| 5 Discussion and Conclusion | Waiting |
| Final | Waiting |

`sections/` holds only `01_introduction.tex`. `references.bib` has **6 entries**.
`figures/` and `tables/` are **empty**. No LaTeX toolchain on PATH, so the PDF could not
be rebuilt after the Part 1 edits.

## Part 1 corrections applied

1. **Mechanism (must-fix).** The introduction described feedback as "reintroduced as a
   bias in the next graph-attention update". The canonical configuration is
   `injection_mode: edge` on both datasets, and attention injection is recorded as inert
   (churn exactly 0.0000). Now describes edge-representation injection, with the
   attention variant named as a verified negative result.
2. **Contribution 3** repeated the same error; corrected.
3. **RQ3** presumed the attention path; now asks about the graph representation.
4. **RQ5 added**, covering the published-baseline comparison, which no research question
   previously asked for. Contribution 4 and the report-structure paragraph updated to
   match.
