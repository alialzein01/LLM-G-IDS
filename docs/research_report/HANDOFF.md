# Handoff — M2 research report

*Written 2026-09-13, after the figures landed. This is the document to read before auditing
the report, and the document to hand to anyone who has to defend a sentence in it.*

The report is `docs/research_report/main.tex`, compiled to `build/main.pdf`. Sixty-five pages,
17,850 words of body prose, nine sections, eleven tables, eleven figures, 26 references.

---

## 1. How to rebuild and check it

```bash
cd docs/research_report && ./build.sh            # tables, number audit, PDF
cd docs/research_report && ./build.sh figures    # the above, plus redraw all eleven figures
OMP_NUM_THREADS=1 python -m pytest tests/ -q     # 174 tests, 359 subtests
python scripts/check_report_numbers.py           # exit code; build.sh only warns
python scripts/check_report_numbers.py --explain 0.8341   # where one number comes from
python scripts/prose_stats.py                    # writing statistics per section
```

Last full run (2026-09-17): build clean, **no overfull boxes**, audit **0 unknown numbers, 0
unverified provenance, 0 banned terms** across ten sections, **174 tests pass**. Four
terminology warnings remain and are all correct usage, exempted in-line.

`OMP_NUM_THREADS=1` is not optional. Without it pooled macro-F1 drifts by roughly 0.012 at a
fixed seed, which is larger than several differences the report measures, and one earlier
finding in this project was retracted because of it.

---

## 2. What each section claims, and what it rests on

### §1 Introduction
Frames five research questions and states the scope limit up front: edges are aggregated on
`(source IP, destination IP, attack class)`, so the label participates in defining edge
identity and no number here is a deployment estimate. Carries no results of its own.

### §2 Related work
Positions the work against graph-based IDS and against text-encoder approaches. No own
numbers. One flagged term, "robustness", is quoted from the title of Zhan et al.'s poster
(`zhan2025agentgnn`) and is exempted in-line. Earlier versions of this file attributed that
quotation to RoGRAD, which is a different paper; corrected 2026-09-18.

### §3 Method
Architecture and protocol. Load-bearing constants: 10 node centralities, 5 edge attributes,
39-wide encoded edge vector, 64-d structural embedding, 768-d semantic embedding, five folds,
three training seeds (42/1/2) at a fold partition fixed at the seed-42 split, `k = 29.0` and
`k = 16.0`, injection scale 2.0 and 20.0. Graph sizes 656 and 2,127 edges.
**Artifacts:** `docs/research_report/dataset_stats.json`, `results/*_current.json`
(`architecture`, `configuration`), `number_allowlist.txt` for the source-code constants.

### §4 Results
The section the report stands on. Eight subsections:

| Subsection | Claim | Artifact |
|---|---|---|
| 4.1 protocol | what was actually run | both `*_current.json` |
| 4.2 ladder | four rungs, loop drawn under both consultants | `multiseed_ladder_v2_head.json`, `multiseed_ladder_v2_legacy.json` |
| 4.2 metrics | accuracy and weighted F1 beside macro-F1 | same two files, `rungs.*.accuracy` / `.weighted_f1` |
| 4.2 comparisons | which steps are separated | `statistical_comparisons` in both contracts |
| 4.3 ablations | feedback on vs off vs random | `feedback_ablations_per_seed` |
| 4.3 isolation | the mechanism with output fusion disabled | `*_oracle_ceiling_v2_trained_head.json` |
| 4.4 five treatments | why the consultant changed | archive 2026-09-06 and 2026-09-07 |
| 4.5 attention | injection through attention is inert | archive §2.1 |
| 4.6 baselines | vs re-trained E-GraphSAGE and TE-G-SAGE | `*_sota_baselines.json` |
| 4.6 ToN asymmetry | who could predict a dropped class, and what it cost | `ton_iot_dropped_class_bound.json` |
| 4.7 per-class | levels, then where the loop gains and loses | `multiseed_head_per_class.json` |
| 4.8 knobs | both selection curves are flat | `results/knob_selection_head/` |

The two results the report is built around:

1. **Under the trained-head consultant every loop comparison is separated on both datasets;
   under the prototype consultant no comparison against a structural rung is separated.**
   That contrast, not the ladder itself, is the finding. Corrected 2026-09-18: the prototype
   loop IS separated above the LLM rung on ToN, at `+0.1727 [+0.1326, +0.2174]`, where that
   rung scores 0.2785 against the GNN's 0.4336. `loop_vs_llm` was absent from the superseded
   contract block until it was declared on 2026-09-18, which is why the earlier phrasing
   said "nothing".
2. **With output fusion disabled, an oracle converts the feedback channel into a separated
   gain on both datasets and no consultant we could build does.** The binding constraint is
   the consultant, not the mechanism.

### §5 Discussion
Answers RQ1–RQ5, argues that class imbalance (12.4:1 against 582:1) is the axis the two
datasets differ on, decomposes where the loop's gain comes from, and states eight limitations.
Introduces no numbers that §4 has not already established.

### §6 Conclusion
449 words. No new numbers.

### Appendices
A: the negative mechanism results in full — conditions A/B/C, injected magnitudes, per-class
consultant reliability, the five treatments. B: per-class F1, knob curves, complementarity.
C: reproducibility, including the artifact table and the environment pin.

---

## 3. Where an auditor should push, and what the answer is

These are the report's real weak points. Each is already stated in the text; this is the list
so nobody has to discover them.

| # | The objection | Where the report answers it |
|---|---|---|
| 1 | Aggregating on the attack label means edge identity depends on the answer | §3.2, and again in §5 limitations. Every result is framed as a controlled comparison, never a detection estimate |
| 2 | The consultant change is confounded with a knob change (top-k 31→29 and 25→16) | §4.2 says so explicitly and marks it `[GAP]` |
| 3 | The loop's advice is byte-identical across its three training seeds, because `train_feedback` loads the seed-42 head logits at every seed | §5 limitations. It means seed spread understates consultant variance |
| 4 | Both knob selection curves are flat, and ToN's injection scale selected on the range boundary | §4.8. The report says "selected", never "tuned" |
| 5 | AGAF is not separated from the graph encoder on either dataset | §4.2, reported as a negative result rather than omitted |
| 6 | The graphs are small: 656 and 2,127 edges | §3.2 and §5. Classes of 12 to 40 edges move substantially on one edge, and the report says so beside every per-class claim |
| 7 | The margin against E-GraphSAGE on UNSW looks like an architectural win | §4.6 and Figure 7. It is a representation effect, decomposed: ~45% tuning, ~30% node features, ~25% architecture |
| 8 | `head_alone` is reported in prose, not as a rung | Decision reversed 2026-09-21; §4.2 and §5.4. See §5 below |

---

## 4. Gaps, closed 2026-09-13

Both `[GAP]` markers are gone. **There are no open gaps in the report.** Closing them turned
up two errors in figures that were already printed, so the record matters.

**Gap A — the consultant change now carries an interval.** Both loop configurations had stored
their pooled out-of-fold predictions at all three seeds, so no retraining was needed.
`scripts/close_consultant_interval.py` reuses the report's own two bootstrap procedures and
writes `results/consultant_change_interval.json`. The point estimates reproduce exactly, which
is what validates the method.

| Dataset | point | two-level | seed-matched | separated |
|---|---:|---|---|---|
| NF-UNSW-NB15 | +0.0697 | +0.0701 [+0.0353, +0.1057] | +0.0701 [+0.0401, +0.0994] | yes |
| NF-ToN-IoT | +0.0523 | +0.0511 [−0.0082, +0.1110] | +0.0509 [+0.0022, +0.1013] | **no** |

*Two-level column updated 2026-09-19.* It carried the pre-regeneration intervals, which §4b
item 5 of this same file says were replaced when the ladder code began drawing the two seeds
independently. The values above are `results/consultant_change_interval.json` as committed, and
are what §4.2 of the report prints. No verdict changed.

This is not the flattering result. The report's largest single effect is separated on one
dataset and not on the other, and NF-ToN-IoT behaves exactly like AGAF against the graph
encoder: separated under the seed-matched procedure, not once training-seed variance is
admitted. §4.2 says so. The comparison also stays confounded with the entropy percentile,
and every statement of it repeats that.

**Gap B — the disagreement split is now complete, and the old figures were wrong.**
`scripts/close_complementarity_split.py` computes, for both datasets and both consultants at
the canonical knobs, the flagged and gated counts, the flagged-set accuracies and the
disagreement split, all from committed tensors. It writes
`results/consultant_complementarity.json`. Two corrections came out of it:

1. **Wrong knobs.** The counts printed as the realisation of the 14.5% and 8.0% consultation
   rates, "204 of 656 flagged and 102 gated" and "532 of 2,127 flagged and 266 gated", are the
   counts at k = 31 and k = 25, the percentiles in force *before* the knobs were re-selected.
   At the canonical k = 29 and k = 16 they are 190 of 656 with 95 gated, and 341 of 2,127 with
   170 gated. Corrected in §4.8.
2. **Not reproducible.** The flagged-set accuracies (0.603 / 0.770 / 0.750 and
   0.359 / 0.759 / 0.724) and the NF-ToN-IoT split (245 right against 32, and 145 against 4
   inside the gate) came from those same pre-reselection runs, computed inside training from
   live per-fold probabilities. That intermediate was never saved, and no reconstruction from
   committed artifacts reproduces them. They are replaced throughout by the recomputed values,
   which anyone can re-derive.

The replacement strengthens the report's argument rather than weakening it. On NF-ToN-IoT the
prototype consultant is right 45 times and wrong 178 where its answer would change something,
and the confidence gate does not rescue it: inside its own most confident half the split is 28
against 63. That is the binding-constraint claim, stated in counts.

## 4b. The 2026-09-16 audit, and where each of its seven points is answered

A consistency audit of the code against the manuscript found seven mismatches. Every published
number reproduced from saved predictions, so nothing was fabricated, but two of the seven could
have changed a separation verdict and one showed the report asking for a metric it never
reported. All seven are closed. No retraining was done, and no separated verdict changed.

| # | The mismatch | How it was answered |
|---|---|---|
| 1 | §3.6 said the fusion model "never" receives embeddings from a model that saw the test labels. Its training-edge embeddings come from folds that include the held-out fold. | §3.6 now states what holds: test-edge embeddings are clean, training-edge embeddings are one step removed, this is standard out-of-fold stacking, and a nested construction was not run |
| 2 | On NF-ToN-IoT the fusion model and the baselines could predict the two excluded classes; the other three rungs could not. | Code fixed in three files and pinned by `tests/test_dropped_class_masking.py`; the published runs are bounded in `results/ton_iot_dropped_class_bound.json` and disclosed in §3.9, §4.6 and the Discussion |
| 3 | `reproduce_ladder.py` ran the prototype consultant and compared against the trained-head contract. | Fixed; it now prints "All four rungs match". Note that `assemble_ladder` must NOT be given `--use-llm-head`: there that flag replaces the LLM rung with the head |
| 4 | `run_pipeline --seed` reseeded the fold partition, the inverse of the protocol. | `--seed` is now the training seed and reaches every stage that trains; `--split-seed` owns the partition. `train_gnn` gained the `--seed` it never had |
| 5 | §3.9 said the two-level bootstrap draws a seed independently per side; the ladder code drew one shared seed. | Ladder code now draws independently, every ladder interval regenerated. Every interval widened; **no verdict changed** |
| 6 | §3.7 said the entropy threshold is "calibrated once and reused". `calibrate()` is never called. | Text and all three docstrings on `UncertaintySelector` now say the quantile is recomputed per pass |
| 7 | §3.6 said the fusion model uses the prototype scorer; §3.5 and §3.7 said the trained head is "reported alone" and "in every table". | All three sentences corrected. `PROJECT_NOTES.md` carried the prototype error twice and is fixed on disk (that file is gitignored) |

Two defects were found while closing these, neither of them in the audit's list:

1. **The NF-ToN-IoT per-class values were scored over the wrong rows** (all 2,127 rather than
   the rows the metric scores), so the per-class table did not average to the ladder table:
   0.4305 against 0.4336 for the GNN model, and a matching gap at every rung. Fixed, and
   `scripts/build_per_class_artifact.py` now asserts the identity on every run. Nothing
   qualitative moved; every change was an increase between +0.0003 and +0.0272.
2. **The method's per-variant focus weights were passing the number audit by coincidence.** A
   bootstrap CI bound of 0.01247 rendered as the percentage string "1.25". When S1 moved that
   interval the coincidence vanished. They are source-code constants and are now in
   `number_allowlist.txt` against `src/models/gnn_classifier.py:22-24`. Worth knowing that the
   audit accepts numbers this loosely.

---

## 5. Scope decisions a reader might mistake for omissions

**`head_alone` is reported, in prose, and is still not a rung.** The trained head standing
alone scores 0.8374 on UNSW and 0.5081 on ToN, above the loop's 0.8341 and 0.5044 on both, and
the loop is *not* separated from it (−0.0032 and −0.0032, sign unstable across seeds). The
2026-09-11 decision to leave this out of the report was reversed on 2026-09-21: §4.2 now gives
the head's standalone scores, its separated margin over the GNN model, and the non-separation
against the loop, and §5.4 reads it as the isolation experiment seen from the other side. The
ladder is unchanged — graph encoder, semantic rung, AGAF, loop — because the head is a
component the loop contains, not a parallel stage. `PROJECT_NOTES.md`'s standing instruction to report
`head_alone` alongside the loop's gains is therefore now consistent with the report as well as
with internal analysis.

**Fold-partition variance is not measured anywhere.** One partition, generated at seed 42 and
held fixed, so the seed-matched bootstrap is a paired comparison. Every error bar in the
report is training-seed variance only, and every caption says so.

**The retracted `real > control` ordering is not cited.** Archive §2.2. It appears in the
report only as a retraction.

---

## 6. Writing statistics

Measured by `scripts/prose_stats.py`. No em dashes anywhere, by editorial rule.

| Section | Words | Sentences | Paragraphs | Em dashes | Semicolons |
|---|---:|---:|---:|---:|---:|
| abstract | 248 | 13 | 3 | 0 | 0 |
| cover | 52 | 14 | 14 | 0 | 0 |
| introduction | 2,052 | 106 | 29 | 0 | 1 |
| related work | 2,377 | 134 | 31 | 0 | 0 |
| method | 4,475 | 227 | 58 | 0 | 4 |
| results | 4,372 | 251 | 55 | 0 | 7 |
| discussion | 2,320 | 123 | 30 | 0 | 0 |
| conclusion | 448 | 23 | 6 | 0 | 0 |
| appendices | 1,297 | 76 | 30 | 0 | 2 |
| statements | 209 | 11 | 5 | 0 | 0 |
| **total** | **17,850** | **978** | **261** | **0** | **14** |

Run `scripts/prose_stats.py` for the full per-section distributions.

---

## 7. Figures

Eleven, all generated from sources in `docs/research_report/figures/src/`. Three are standalone
TikZ documents (architecture, loop flow, E-GraphSAGE ceiling); eight are matplotlib
(`make_plots.py`). `./build.sh figures` redraws all eleven; a plain build uses the committed
PDFs.

Most plotted values are transcribed from the specifications in `FIGURE_BRIEFS.md`, which were
checked against the contracts. A figure disagreeing with a table would not be caught by the
compiler. It is caught by reading the figure against its brief, and those briefs carry every
number their figure may contain.

**Two figures are different.** `fig_metrics` and `fig_perclass_levels` read their values from
the contracts at draw time. Between them that would have been about a hundred transcribed
numbers, and transcription at that volume is how `fig_perclass`'s NF-ToN-IoT row went stale
when the per-class artifact was corrected on 2026-09-16. Their briefs name the artifact and the
keys, and say what the figure should show, which is what you check them against instead.

---

## 8. Related documents

| File | What it holds |
|---|---|
| `CLAIM_EVIDENCE_MAP.md` | every claim, its evidence, and whether it may be stated. The FORBIDDEN and RETRACTED rows matter most |
| `EVIDENCE_STATE.md` | the current evidential position, rung by rung |
| `REPORT_WORKFLOW.md` | how the report was produced, workstream by workstream |
| `docs/RESULTS_ARCHIVE.md` | the full record, including the trail of attempts and every retraction |
| `number_allowlist.txt` | 25 source-code constants the audit accepts, each with a justification |
| `derived_numbers.json` | quantities computed from contracts rather than read from them |

**One trap in the capture directories.** `results/multiseed_v2/head/*/ladder_summary.json`
are stale. They were written during a prototype-consultant run at k = 31 and record a loop
of 0.765; the tensors in the same directories are the trained-head ones every contract
describes. They are deliberately not rewritten, because rewriting them would mean re-running
the captures. `aggregate_multiseed` reads the tensors, and takes only the seed-invariant
prototype LLM rung from the summary, which it re-derives and checks before use. Anything else
that reads those files is reading a configuration the report does not use.
