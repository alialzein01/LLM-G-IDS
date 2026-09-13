# Handoff — M2 research report

*Written 2026-09-13, after the figures landed. This is the document to read before auditing
the report, and the document to hand to anyone who has to defend a sentence in it.*

The report is `docs/research_report/main.tex`, compiled to `build/main.pdf`. Sixty pages,
16,371 words of body prose, nine sections, nine tables, nine figures, 26 references.

---

## 1. How to rebuild and check it

```bash
cd docs/research_report && ./build.sh            # tables, number audit, PDF
cd docs/research_report && ./build.sh figures    # the above, plus redraw all nine figures
OMP_NUM_THREADS=1 python -m pytest tests/ -q     # 167 tests, 359 subtests
python scripts/check_report_numbers.py           # exit code; build.sh only warns
python scripts/check_report_numbers.py --explain 0.8341   # where one number comes from
python scripts/prose_stats.py                    # writing statistics per section
```

Last full run: build clean, **no overfull boxes**, audit **0 unknown numbers, 0 unverified
provenance, 0 banned terms** across nine sections, **167 tests pass**. Four terminology
warnings remain and are all correct usage, exempted in-line.

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
numbers. One flagged term, RoGRAD's "robustness", is a quotation of another paper's claim and
is exempted in-line.

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
| 4.2 comparisons | which steps are separated | `statistical_comparisons` in both contracts |
| 4.3 ablations | feedback on vs off vs random | `feedback_ablations_per_seed` |
| 4.3 isolation | the mechanism with output fusion disabled | `*_oracle_ceiling_v2_trained_head.json` |
| 4.4 five treatments | why the consultant changed | archive 2026-09-06 and 2026-09-07 |
| 4.5 attention | injection through attention is inert | archive §2.1 |
| 4.6 baselines | vs re-trained E-GraphSAGE and TE-G-SAGE | `*_sota_baselines.json` |
| 4.7 per-class | where the loop gains and loses | `multiseed_head_per_class.json` |
| 4.8 knobs | both selection curves are flat | `results/knob_selection_head/` |

The two results the report is built around:

1. **Under the trained-head consultant every loop comparison is separated on both datasets;
   under the prototype consultant nothing is separated at all.** That contrast, not the
   ladder itself, is the finding.
2. **With output fusion disabled, an oracle converts the feedback channel into a separated
   gain on both datasets and no consultant we could build does.** The binding constraint is
   the consultant, not the mechanism.

### §5 Discussion
Answers RQ1–RQ5, argues that class imbalance (12.4:1 against 582:1) is the axis the two
datasets differ on, decomposes where the loop's gain comes from, and states seven limitations.
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
| 8 | `head_alone` is not reported as a rung | Deliberate scope decision, recorded in `CLAIM_EVIDENCE_MAP.md` §B′ row B5. See §5 below |

---

## 4. Open gaps

Two `[GAP]` markers, three occurrences. Both deferred by decision on 2026-09-13, to be
measured after the report is otherwise final.

**Gap A — the consultant change has no interval.** §4.2. The report's largest single effect,
+0.0697 on UNSW and +0.0523 on ToN, is a point estimate. Every other difference in the report
carries a bootstrap interval.
*Cost to close: low.* The pooled out-of-fold predictions for both loop configurations are on
disk at `results/multiseed_v2/{head,legacy}/<dataset>_seed{42,1,2}/feedback_oof_real.pt`,
shape `[E, C]`. No retraining is needed; it is a bootstrap over an existing tensor pair.
The confound in objection 2 above must be restated wherever the new interval appears.

**Gap B — one diagnostic table is missing.** §4.7 and Appendix B.3. The
consultant-versus-graph-encoder disagreement split on the flagged set was recorded only for
NF-ToN-IoT with the trained head (right 245, wrong 32; inside the confidence gate 145 to 4).
Never recorded for NF-UNSW-NB15, and never for the prototype consultant.
*Cost to close: higher.* The per-edge flags were not saved, so this needs the loop re-run with
extra instrumentation. Nothing in the report depends on it.

---

## 5. Scope decisions a reader might mistake for omissions

**`head_alone` is not a rung.** The trained head standing alone scores 0.8374 on UNSW and
0.5081 on ToN, above the loop's 0.8341 and 0.5044 on both, and the loop is *not* separated
from it (−0.0030 and −0.0028, sign unstable across seeds). The report's ladder is graph
encoder, semantic rung, AGAF, loop. It reports the mechanism's effect inside the loop rather
than the head as a comparative rung. This is a deliberate scope decision taken on 2026-09-11,
recorded in `CLAIM_EVIDENCE_MAP.md` §B′ row B5 as **NOT REPORTED**, not an oversight and not a
suppressed negative. Note that `CLAUDE.md` still carries the older instruction to report
`head_alone` in every table; that file is the project's working rule sheet, not the report.

**Fold-partition variance is not measured anywhere.** One partition, generated at seed 42 and
held fixed, so the seed-matched bootstrap is a paired comparison. Every error bar in the
report is training-seed variance only, and every caption says so.

**The retracted `real > control` ordering is not cited.** Archive §2.2. It appears in the
report only as a retraction.

---

## 6. Writing statistics

Measured by `scripts/prose_stats.py`. No em dashes anywhere, by editorial rule.

| Section | Words | Sentences | Paragraphs | Mean length | Std | Short <8 | Long >35 | Semicolons |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| abstract | 246 | 13 | 3 | 18.9 | 11.3 | 15.4% | 7.7% | 0 |
| introduction | 2,078 | 107 | 29 | 19.4 | 8.6 | 5.6% | 5.6% | 0 |
| related work | 2,393 | 132 | 31 | 18.1 | 9.6 | 10.6% | 5.3% | 0 |
| method | 4,160 | 214 | 56 | 19.4 | 10.1 | 11.2% | 6.5% | 3 |
| results | 3,515 | 199 | 47 | 17.7 | 9.3 | 13.1% | 4.0% | 5 |
| discussion | 2,220 | 115 | 29 | 19.3 | 8.8 | 6.1% | 7.0% | 0 |
| conclusion | 449 | 24 | 6 | 18.7 | 10.1 | 25.0% | 0.0% | 0 |
| appendices | 1,120 | 62 | 24 | 18.1 | 12.1 | 19.4% | 9.7% | 0 |
| statements | 190 | 10 | 5 | 19.0 | 10.3 | 20.0% | 0.0% | 0 |
| **total** | **16,371** | **876** | **230** | | | | | **8** |

---

## 7. Figures

Nine, all generated from sources in `docs/research_report/figures/src/`. Three are standalone
TikZ documents (architecture, loop flow, E-GraphSAGE ceiling); six are matplotlib
(`make_plots.py`). The plotted values are transcribed from the specifications in
`FIGURE_BRIEFS.md`, which were checked against the contracts. `./build.sh figures` redraws all
nine; a plain build uses the committed PDFs.

A figure disagreeing with a table would not be caught by the compiler. It is caught by reading
the figure against its brief, and the briefs carry every number a figure may contain.

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
