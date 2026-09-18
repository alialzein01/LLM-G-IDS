# M2 report review record

This is review metadata, not a manuscript or independent scientific authority.

## Current checkpoint

- Report root and manuscript: `docs/research_report/`, `main.tex`
- Updated: 2026-09-18
- Current section and state: Method — complete
- Next action: audit Results (`sections/04_results.tex`), then pause at its discussion
  checkpoint. Results is the longest section and the one every claim rests on.
- Reviewed state: git `3443016` plus uncommitted edits to sections 1-3. Hashes after revision:
  `01_introduction.tex` `c575f5bb4ae12bdaaabeacc67255e5462235425ee128cd4ce1429daa52ed1b66`,
  `02_related_work.tex` `1bc62a03bcf270461c6b190f2126a4010aa795bb4666d423f3d91beca7401d43`,
  `03_method.tex` `761c1235543d487c4f7fb6050bfe8336e6958ae778a22950e5e55de955f09cdd`.
  Commits `cf62461`, `376ca41`, `3443016` landed mid-review and are accounted for below.
  `HANDOFF.md` and `number_allowlist.txt` also carry small corrections from this review.

## Writing profile and scope

- Audience/voice: formal accessible M2 English; explain concepts and choices before relying on them.
  Manuscript voice already uses first person ("we", "our") in every section; keep it.
  Editorial rule already in force: no em dashes (`HANDOFF.md` §6).
- Agreed editorial scope: explain and reorganise; discuss major structural changes first.
- Evidence scope: targeted checks against `results/*.json` and `dataset_stats.json`; no experimental reruns.
- Current project scope decisions:
  - `head_alone` is deliberately NOT a comparative rung in the report
    (author's decision 2026-09-11; `CLAIM_EVIDENCE_MAP.md` §B′ row B5; `HANDOFF.md` §5).
    `PROJECT_NOTES.md` still carries the opposite rule for internal analysis; the report is the exception.
  - Report stage names: GNN model / semantic model / fusion model / feedback model.
    Artifact keys keep the older names (`agaf_vs_gnn`, `loop_vs_agaf`). Do not rename keys.
  - "selected", never "tuned", for `top_k_percent` and `injection_scale`.
- Reference material consulted (text only, no visual inspection yet): `PROJECT_NOTES.md`;
  `docs/research_report/HANDOFF.md`; `CLAIM_EVIDENCE_MAP.md` §A, §B, §B′, §E; `main.tex`;
  `sections/01_introduction.tex`, `00_abstract.tex`, and the relevant passages of
  `04_results.tex` (§4.2, §4.6) and `06_conclusion.tex`;
  `tables/tab_comparisons.tex`, `tables/tab_vsbaselines.tex`;
  `results/{unsw_nb15,ton_iot}_current.json`, `results/{unsw_nb15,ton_iot}_sota_baselines.json`,
  `results/multiseed_ladder_v2_legacy.json`, `docs/research_report/dataset_stats.json`;
  `src/pipeline/step2/knowledge_graph.py:210`.
- Missing inputs: none blocking the introduction. The bundled reference PDFs
  (`Report_Guideline.pdf`, exemplar thesis) were not re-read for this section; no layout
  judgment was made, and `build/main.pdf` was not inspected.

## Section ledger

| Section | State | Approved changes | Evidence/check results | Unresolved necessary issues | Next step |
|---|---|---|---|---|---|
| Introduction | complete | I1-I9, all approved 2026-09-18 | tectonic build clean, 0 overfull boxes, no undefined refs/citations; `check_report_numbers.py` exit 0, 0 unknown / 0 unverified / 0 banned; 2 pre-existing terminology warnings, both correct usage; 0 em dashes | none | done |
| Related work | complete | R1-R9, all approved 2026-09-18 | build clean, 0 overfull, no undefined refs; PDF text re-extracted to confirm the restored sentence; audit exit 0, warn=0 for this section; 22/22 citation keys resolve; 0 em dashes | none in §2; R10 is a Positioning/Method wording question deferred to the Method audit | done |
| Method | complete | M1-M8, all approved 2026-09-18 | build clean, 0 overfull, no undefined refs; audit exit 0; Table 2, GATv2 config, head shape and focal weighting checked against source; node-feature claims checked by loading both graphs | none; R10 resolved as M1 | done |
| Results | not started | none | not checked | not assessed | audit next |
| Discussion | not started | none | not checked | not assessed | after results |
| Conclusion | not started | none | not checked | not assessed | after discussion |
| Abstract/front matter | not started | none | not checked | none — the over-claim was fixed upstream in commit `3443016` | after narrative |
| Appendices/whole-report review | not started | none | not checked | not assessed | final checks |

## Decisions and open questions

### Verified during the introduction audit (2026-09-18)

- `656` edges, `47.4%` benign, `12.4:1`, eight attack classes at 40 edges and one at 25:
  all reproduce from `dataset_stats.json:unsw_nb15`.
- `2,127` edges, `82.1%` benign, `582:1`, seven attack classes at or below 35 edges:
  reproduce from `dataset_stats.json:ton_iot`.
- "factor of 47": `cross_dataset.imbalance_ratio_of_ratios = 46.7846`.
- Ten and eight scored classes: `num_eval_classes` 10 / 8; dropped `dos` (4 edges),
  `ransomware` (3 edges).
- "Only two quantities selected on validation data differ": `top_k_percent` and
  `injection_scale`, per `PROJECT_NOTES.md` and both `*_current.json` selection blocks.
- Label-free text guard: `assert_label_free`, `src/pipeline/step2/knowledge_graph.py:210`,
  called in `knowledge_graph.py:241` and `build_simplified_nl.py:122`.
- All eight §1 citation keys resolve in `references.bib`.
- Headline positive claim: feedback model separated above the GNN model and the fusion model on
  both datasets (`statistical_comparisons.feedback_vs_gnn`, `.feedback_vs_agaf`) and above every
  faithful re-trained baseline (`statistical_comparisons_3seed`, both SOTA contracts;
  `tab_vsbaselines`). Correct as written.

### Approved and applied, 2026-09-18

Student instruction: "yes do all nine". Applied to `sections/01_introduction.tex`:

| ID | What changed |
|---|---|
| I1 | Headline pair rescoped. The negative now names the GNN and fusion comparisons and the NF-ToN-IoT E-GraphSAGE margin, instead of "none of those comparisons". The positive list gained "from the semantic model", which is supported and matches the abstract and conclusion |
| I2 | "five sections" to "six sections, followed by three appendices"; Conclusion sentence added; Results and Discussion now use `\ref` like Related Work and Method |
| I3 | "trained-head baselines" replaced by "a comparison between the two consultants" (first written as "consultant-substitution comparisons", which introduced an overfull line and was reworded) |
| I4 | "at or below that threshold" replaced by "at or below 35 edges" |
| I5 | RQ2 now glosses *separated* as "their intervals exclude zero" |
| I6 | The model ladder is glossed at first use, line 60 |
| I7 | One clause bridging *semantic branch* (component) and *semantic model* (rung) |
| I8 | "nothing we could distinguish from nothing" replaced by "indistinguishable from adding nothing" |
| I9 | *edge-level* defined at first use in paragraph 3 |

### Related Work, approved and applied 2026-09-18

Student instruction: "do all of them".

| ID | What changed |
|---|---|
| R1 | **A mid-sentence `%` on line 131 was deleting five words from the compiled PDF.** Page 14 read "reverses the order. descriptions of node interactions and flags...". The words are restored and the fix was verified by extracting the page text from `build/main.pdf`, not by reading the source |
| R2 | The `numcheck: allow` justification on that line credited the quoted word "robustness" to RoGRAD. It is from the title of Zhan et al.'s poster. Corrected in the section and in `HANDOFF.md` §2, which repeated the same misattribution |
| R3 | Positioning described the consultant as a frozen encoder "whose embeddings are scored against class prototypes", which is the superseded configuration. It now names the trained head as the consultant and the prototype scorer as the earlier configuration, matching §3.7 and §1 |
| R4 | §2.3's four fusion families now say where this report's fusion stage sits |
| R5 | Glosses added for SHAP, Majorization--Minimization, and MC-dropout |
| R6 | Zhan et al. is now identified as a poster |
| R7 | Positioning no longer restates the routing/content bidirectionality definition in full; it refers to Section 1 and keeps only what it adds |
| R8 | "Agent-based analysis for GNN robustness" read as a system name; it is now attributed to its authors as a description |
| R9 | §2.1 now says that E-GraphSAGE and TE-G-SAGE are the two architectures re-trained as baselines in §4.6 |

**One self-correction during R4.** The first version of the placement sentence said the fusion
stage interacts "through attention", taking the cue from the name Adaptive Gated Attention
Fusion. §3.6 describes a gate that emits two softmaxed weights and convexly combines the
projections, with no attention between the representations. The sentence was corrected to place
the stage in the gating family before the section was finalised.

### Open finding carried forward

- **R10 (§2.6 and §1, wording, needs a decision).** Both sections describe the final stage as
  "joint attention-based fusion", and that phrase sits inside the novelty sentence in both. By
  §2.3's own taxonomy the module is a gate, not attention between representations. The module's
  name contains "Attention" and a softmax gate is loosely called attention, so this is a
  defensible usage rather than an error, but the report defines the two families and then uses
  the other family's word for its own stage. Raise with the Method audit, where §3.6 is read in
  full. Changing it would touch a claim in two sections, so it was not changed unilaterally.

### Method, approved and applied 2026-09-18

Student instruction: "do all of them".

| ID | What changed |
|---|---|
| M1 | **R10 resolved.** §3.6 describes a gate: two projections, a gate emitting two logits, a softmax, a convex combination. No attention between the representations anywhere in the fusion path, and the feedback model's output fusion is another gate. The module keeps its code name (*Adaptive Gated Attention Fusion*), and the abstract/conclusion keep it as a name, but the **novelty claim** now reads "sample-adaptive gated fusion" in §1 and §2.6, and the descriptive sentence in Positioning reads "gated fusion" |
| M2 | §3.5's five quantile bands are fitted over all edges before the split. §3.3 flagged its own statistics as transductive; §3.5 now does the same |
| M3 | Table 2's nine focus weights are now stated to be fixed a priori, never swept, source-code constants. Verified identical to `EDGE_FOCUS_WEIGHTS`, `src/models/gnn_classifier.py:20-24` |
| M4 | `$C$` defined at first use as the number of classes, ten on both graphs |
| M5 | The consultant-identity paragraph moved out of §3.8 Baseline Protocol and merged into §3.7, where the consultant is described. The two overlapping sentences were dropped and only the new content kept: both configurations at the same seeds, partition and injection scale, differing only in entropy percentile (31 against 29, 25 against 16). §3.8 now runs from the five deviations to the two asymmetries |
| M6 | The trained head's description gained the batch normalisation inside `ModalityProjector`, folded into the M4 edit |
| M7 | §3.2 now states that `k_truss` takes the value 2.0 at every NF-UNSW-NB15 node, so seven of the ten channels vary there, and that it varies on NF-ToN-IoT. Placed with the duplicate-column disclosure rather than with the convergence sentence, since both are about the same loss of information |
| M8 | `number_allowlist.txt:44` pointed at a constant named `VARIANT_FOCUS`; the constant is `EDGE_FOCUS_WEIGHTS`. Line range was already right |

**M7 was found by computation, not by reading.** Loading both graphs and testing every centrality
column showed `k_truss` constant at 2.0 on NF-UNSW-NB15 and varying on NF-ToN-IoT. Whether that
is the documented convergence fallback firing or a genuine property of a 49-node graph was not
determined, and the added sentence states only what was observed. Note that the number audit
accepts "2.0" partly because it is also NF-UNSW-NB15's injection scale, so the audit is not
what verifies this.

### Verified against source during the Method audit

| Claim in §3 | Checked against | Result |
|---|---|---|
| Table 2 focus weights | `src/models/gnn_classifier.py:20-24` | exact match |
| GATv2, 8 heads of width 8 to 64, no self-loops | `gnn_classifier.py:92-111`, `train_gnn.py:41` (`HIDDEN_DIM = 64`, `HEADS = 8`) | match |
| trained head 768-128-128-C | `build_llm_heads.py:50-51`, `fusion_classifier.UnimodalEdgeClassifier` | match |
| focal loss, gamma 2.0, weights sum to class count, absent classes zero | `src/pipeline/common/splits.py:140-182` | match |
| `global_betweenness == betweenness`, `global_pagerank == pagerank` on UNSW; no duplicates on ToN | both `pyg_data.pt`, loaded and compared | match |
| 49/656 and 1,501/2,127 nodes and edges | `dataset_stats.json` | match |
| 58.7% and 7.9% parallel-edge coverage | `known_ceilings.edges_indistinguishable_by_endpoints` | match |
| k 29.0, scale 2.0, confidence 0.50, 14.5%, 3 iterations, churn 0.01 | `unsw_nb15_current.json:configuration` | match |
| edge representation dimension 295 | 4x64 + 39 | match |

### Evidence questions raised in §1, both closed upstream during the review

Commits `376ca41` and `3443016` landed while the introduction was being audited, by the author.

- **EQ1 — closed for the abstract, conclusion, §4.2 and the figure caption; §1 was missed.**
  Under the prototype consultant the feedback model is separated above all six NF-UNSW-NB15
  baselines and above both TE-G-SAGE variants on NF-ToN-IoT (`tab_vsbaselines`), so
  "nothing is separated at all" was wrong. `3443016` rescoped four statements to "no comparison
  against a structural rung is separated" but did not touch the introduction, which then
  disagreed with the abstract. Finding I1 closes it, and also covers the baselines, which the
  new house phrasing does not mention.
- **EQ2 — closed by `3443016`.** The `[GAP]` marker in `tab_comparisons` is filled. The
  prototype feedback-versus-semantic comparison is separated on NF-ToN-IoT,
  `+0.1727 [+0.1326, +0.2174]`, against a semantic rung scoring 0.2785, and not separated on
  NF-UNSW-NB15. No introduction claim depends on it.
- Citation keys `bayer2022cysecbert` and `farrukh2024xgnid` were renamed to
  `bayer2024cysecbert` and `farrukh2025xgnid` by `376ca41`; both resolve in `references.bib`.

## Final checks

Performed for the introduction, related work and method, 2026-09-18:

- `tectonic main.tex --outdir build`: builds, **0 overfull boxes**, no undefined references or
  citations. Pre-existing underfull warnings in `main.bbl` and `07_appendices` are untouched.
- `python scripts/check_report_numbers.py`: **exit 0**. 0 unknown numbers, 0 unverified
  provenance, 0 banned terms. The introduction's two terminology warnings (L85, L107) predate
  this review and are correct usage: both describe generative models other than CySecBERT.
- Editorial rule: 0 em dashes in the introduction.
- **Not performed: visual inspection of the rendered pages.** `pdftoppm` is not installed on
  this machine, so pages 4-9 of `build/main.pdf` were not looked at. Layout is clean by the
  compiler's own measure only. Install `poppler` if a visual pass is wanted.
- For §2 specifically: all 22 citation keys resolve in `references.bib`, and the bibliography
  entries for `zhan2025agentgnn`, `tegsage2025` and `wang2025rograd` were read. **The cited
  papers themselves were not read**, so the faithfulness of each description of prior work is
  not verified, only the keys and the internal consistency.
- Whole-report final checks are still outstanding and belong at the end of the review.
