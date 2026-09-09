# Report completion plan (for the implementing agent)

Owner of this plan: the user. Implementer: Opus. Checker at the end: Fable (separate session).
Read `CLAUDE.md`, `docs/research_report/REPORT_WORKFLOW.md`, `WRITING_PLAN.md` (writing
standard + traps), `EVIDENCE_STATE.md`, `CLAIM_EVIDENCE_MAP.md`, and
`docs/RESULTS_ARCHIVE.md` "2026-09-07 — Canonical loop redefined" before starting.

## Working rule (non-negotiable)
One workstream at a time, in the order below. After each workstream: build the PDF, run the
checks, commit, then **stop and report to the user** (what was done, what changed, what the
checks said, anything that did not match). The next workstream starts only when the user
says go. Figures are **proposed to the user first** (purpose, content, layout in words,
colours, size) and drawn only after approval.

## Frozen facts (schema 5 / 7 contracts; every sentence in the report must agree)

| | UNSW (10 cls) | ToN (8 cls) |
|---|---|---|
| head alone (strongest single-modality baseline) | 0.8374 ± 0.0048 | 0.5081 ± 0.0073 |
| **loop** (canonical: trained-head consultant; k=29 / scale 2.0; k=16 / 20.0) | 0.8341 ± 0.0042 | 0.5044 ± 0.0080 |
| AGAF (prototype semantic input) | 0.7793 ± 0.0116 | 0.4102 ± 0.0279 |
| GNN | 0.7437 ± 0.0228 | 0.4336 ± 0.0053 |
| semantic rung (prototype, deterministic) | 0.7353 | 0.2785 |

- Loop vs AGAF, GNN, semantic rung: **separated** on both datasets (two-level and seed-matched).
- Loop vs head alone: **not separated**, sign flips at one seed → the word is "indistinguishable".
- Loop and head alone vs every faithful re-trained baseline: separated on both datasets.
- Feedback on vs off (`real` − `head_only`): separated at every seed, both datasets. This is
  fusion + injection together; the mechanism-only (fusion off) path with a realistic
  consultant never separates from control. Say both.
- Mechanism-only: oracle headroom of the *current injection mechanism* is +0.0345 UNSW (3-seed
  mean; negative at seed 42) / +0.1090 ToN; five realistic-consultant treatments (selector-head
  supervision, temperature calibration, reliability-weighted advice, advice format,
  cross-fitted advice) all fail; the loop's learned trust scalar barely moves (≤0.02) under
  any of them.
- Knob curves are flat (UNSW 0.8241–0.8277 over 21 k values; ToN 0.5168–0.5232); ToN's scale
  sits on the upper boundary. Never call a knob "tuned".
- Vocabulary lock: "separated" / "not separated"; never "significant(ly)", "outperform",
  "beats", "top rung", "best", "state of the art", "robust", "superior" for our system.
  "Semantic branch / pretrained cybersecurity language encoder / trained semantic head";
  never "the LLM reasons". Baselines are "re-trained on our aggregated representation".
- Numbers: only from `results/*.json`, `docs/RESULTS_ARCHIVE.md`, or dataset statistics you
  can recompute. If a number is not there, write `[GAP: what]` and report it; never invent.

## Workstreams

### W5 — number-audit tooling FIRST (so every later draft is checked mechanically)
`scripts/check_report_numbers.py`: extract every numeric token (≥2 significant digits) from
`docs/research_report/sections/*.tex`, build the allowed set from all `results/*.json`
(recursively, every float, rendered at 2/3/4 decimals and as ±), the archive's tables, and
dataset statistics (edge/node/class counts, imbalance ratios, shares); print any number not
found with its line. Add a banned-word grep (list above) and the terminology lock. Wire as a
warning step in `docs/research_report/build.sh`. `scripts/prose_stats.py`: per section,
sentence-length mean/std/histogram, share of sentences <8 and >35 words, em-dash count,
semicolons per 1,000 words, flagged-term counts, paragraph-length distribution. Run both on
the three existing sections and report.

### W2 — Results section (`sections/04_results.tex`, 2,500–3,200 words + tables)
Describe; do not explain (Discussion does). Eight subsections, `\label{sec:res-…}`:
1. Protocol as run: 3 training seeds, fixed seed-42 partition, pooled OOF macro-F1 over
   10 / 8 classes, two-level bootstrap (edges AND seed, 2000 iters), seed-matched as
   secondary, the decision rule, ± = training-seed variance only, levels not comparable across
   datasets, semantic rung deterministic. Seed-42 reproducibility of GNN/semantic rungs.
2. The ladder (RQ1, RQ2): table rung × dataset mean ± std + per-seed; ordering by mean; which
   comparisons are separated (Table: loop−AGAF, loop−GNN, loop−semantic, head_alone−GNN,
   AGAF−GNN, loop−head_alone; mean, CI, P, sign-stable, verdict), both bootstrap types.
3. Feedback ablations (RQ3): per-seed real / head_only / random with CIs; plus the
   mechanism-only table (fusion off): control, prototype, head in-fold, head cross-fit, oracle
   edge, oracle attention; paired CIs; captured-share only where headroom is positive.
4. The prototype-consultant investigation (RQ3, RQ4), as one negative-results subsection:
   conditions A/B/C 3-seed table, the selector-head and temperature diagnostics, the five
   treatments table, the injected-bias magnitudes, the complementarity table (flagged edges:
   consultant right / GNN right on disagreements, both consultants, both datasets), the
   trust-scalar observation. Keep every number from the archive sections dated 2026-09-06/07.
5. Attention injection (RQ4): churn 0.0000, structural facts (58.7 % co-located, 62.8 %
   in-degree-1), oracle_attention not separated.
6. Baselines (RQ5): the 2×3×2 table with faithful flag; endpoint-only ceiling; the UNSW
   decomposition staircase (point estimates); loop-vs-baseline and head_alone-vs-baseline
   3-seed intervals (`statistical_comparisons_3seed`); which cells are not separated, if any.
7. Per-class F1 (RQ4): two tables, 3-seed mean, GNN / AGAF / loop / head alone, with class
   sizes; state where the loop is above/below AGAF and above/below the head.
8. Knob selection: the flat curves, the boundary, effective feedback rates.
Anchor figures/tables with the W4 filenames. Run W5 checks; zero unknown numbers before
reporting.

### W3 — Discussion and Conclusion (`05_discussion.tex` ~2,000 words, `06_conclusion.tex` ≤1 page)
Answer RQ1–RQ5 in order, each in one paragraph that names the evidence. Then: imbalance as
the organising axis (AGAF collapses on ToN; the loop and the head do not); why the loop equals
its head (the fusion floor; the inert trust scalar; only always-right advice converts; the
five treatments); what the trained-head configuration buys (a separated system) and what it
does not (a separated feedback gain); the instantiation-plus-measurement contribution stated
narrowly against DAS/LOGIN/GLANCE/RoGRAD; limitations as findings (label-conditioned
aggregation → not deployment-valid; fold-partition variance unmeasured; in-fold train
advice; flat knob curves, one on a boundary; single-seed head file inside the loop; Gate 0
(seed, fold) pairs correlated; E-GraphSAGE within noise of our structural rungs on ToN);
future work (learning trust; a deployment-valid graph; advice conditioned on graph state was
deliberately excluded per the design). Conclusion: what was built, what was measured, the two
headline sentences, one paragraph of outlook. No new numbers; reuse Results' numbers only.

### W4 — figures and tables from the contracts (propose each to the user first)
`docs/research_report/figures/make_figures.py` (matplotlib, colourblind-safe palette, PDF,
one function per figure, reads only `results/*.json`) and `tables/make_tables.py` (booktabs
`.tex` fragments, `\input` from the sections); both wired into `build.sh` before tectonic.
Figures (fixed filenames): `fig_ladder.pdf` (two panels, mean ± std bars, head-alone dashed
line); `fig_imbalance.pdf` (class sizes, log scale, ≤35-edge classes shaded);
`fig_perclass.pdf` (heatmap, 3-seed per-class F1, GNN/AGAF/loop/head); `fig_mechanism.pdf`
(mechanism-only arms vs control, both datasets); `fig_decomposition.pdf` (UNSW staircase
0.3841 → 0.5842 → 0.7196 → 0.7793 / 0.8341, "point estimates"); `fig_architecture.pdf` (TikZ,
system diagram with AGAF beside it); `fig_loop_flow.pdf` (TikZ flow: GNN pass → entropy →
top-k flagged → confidence gate → trained-head judgement → projection to edge-feature bias →
next pass → churn check → stop/repeat → output fusion; numbered steps, one example edge
traced, iteration cap and churn tolerance annotated); `fig_ceiling.pdf` (TikZ, E-GraphSAGE
endpoint-only collapse). Tables: ladder, comparisons, ablations, mechanism-only, baselines,
loop-vs-baseline intervals, per-class (appendix), knob curves (appendix), five treatments
(appendix). **Propose every figure to the user in one message, wait for approval, then draw.**

### W6 — writing-quality pass on every section (this is where "human" is earned)
Hand-edit each `.tex` (01–06) to the standard in `WRITING_PLAN.md` "Writing standard" and
the checklist below; measure with `prose_stats.py` before/after and put both in the commit
message. If `docs/research_report/VOICE_SAMPLE.md` exists, match its rhythm and connectives.
- Sentence length mean 15–22 words, std ≥ 7; at least one sentence under 8 words per few
  paragraphs; no three consecutive sentences within ±3 words of each other.
- Paragraph lengths vary 2–10 sentences; no padding to three items.
- Zero em dashes; ≤ 2 semicolons per 1,000 words; no consecutive colon-list openers.
- Remove: delve, leverage, robust, crucial, pivotal, landscape, comprehensive, nuanced,
  showcase, underscore, testament, holistic, "it is worth noting", "in order to", "notably",
  "interestingly", stacked furthermore/moreover.
- Active voice with "we" where the actor matters; recurring names stay fixed ("the loop",
  "the head", "the GNN", "AGAF"); no synonym cycling.
- Every paragraph answers "what does the reader now know"; add one true first-person
  observation per section (what we tried first, what surprised us) where the archive supports it.
- Numbers stay exactly as in the contracts. Re-run `check_report_numbers.py` after the pass.
No tool is claimed to guarantee any detector score; the target is prose an engineer would
write about their own work.

### W7 — abstract, front matter, appendices
`00_abstract.tex` (200–250 words, written last, two headline sentences included);
title/subtitle check in `main.tex`; appendices A (negative mechanism results, five treatments
+ A/B/C), B (per-class, knob curves, complementarity), C (reproducibility: commands, seeds,
`OMP_NUM_THREADS=1`, contract paths, test count); acknowledgements; a plain AI-use statement
(drafting/editing assistance; all experiments, numbers, decisions by the author).

### W8 — audits
1. `check_report_numbers.py` clean on all sections. 2. Walk `CLAIM_EVIDENCE_MAP.md` and
`EVIDENCE_STATE.md` row by row against the final text; update rows. 3. Terminology grep:
zero hits outside quoted names. 4. Fix cross-references, figure/table numbering, bib warnings.
5. Also fix the stale `limitations` entries in `results/{unsw_nb15,ton_iot}_current.json`
("No rung comparison is statistically separated…" and the Finding-3 sentence) to the schema
5/7 facts, keeping the old text under `supersedes`; keep tests green.

### W9 — hand-off
Final build, full test suite, push the branch, and write `docs/research_report/HANDOFF.md`:
what each section claims, the numbers it relies on, open `[GAP]`s if any, and the prose
statistics per section. Fable checks from there.

## Verification (every workstream)
`./docs/research_report/build.sh` compiles; `scripts/check_report_numbers.py` zero unknown;
`.venv/bin/python -m pytest tests/ -q` green; commit with a message naming the workstream.
