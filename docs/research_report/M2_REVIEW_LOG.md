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
| Results | complete | S1-S5, all approved 2026-09-18 | build clean, 0 overfull, audit exit 0; roughly forty numbers reproduced from artifacts, code or the graphs; new passage checked in the rendered PDF | none in the report; two problems live in other files, see below | done |
| Discussion | complete | D1-D4, approved 2026-09-18 | build clean, 0 overfull, audit exit 0; every quoted value reproduced from the ladder, oracle and per-class artifacts | none | done |
| Conclusion | complete | C1-C2, approved 2026-09-18 | build clean, 0 overfull, audit exit 0; no new numbers introduced | none | done |
| Abstract/front matter | complete | A1 approved; AI statement left unchanged at the author's instruction | build clean, audit exit 0; cover, abstract and TOC inspected as rendered pages | none | done |
| Appendices/whole-report review | complete | P1-P3 and W1-W3, approved 2026-09-19 | build clean, 0 overfull, audit exit 0, 174 tests / 359 subtests; 38/38 bibliography entries reconciled; no undefined or multiply-defined labels; every figure-transcribed value matched to its contract | none | done |

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

### Results, approved and applied 2026-09-18

Student instruction: "do all of them".

| ID | What changed |
|---|---|
| S1 | The four captured-share percentages in §4.3 are computed against the headroom as a plain three-seed mean, +0.0345 and +0.1090, while the surrounding text quoted only the paired bootstrap means +0.0369 and +0.1354. The ToN denominator appeared nowhere in the report, so none of the four shares could be reproduced from the page. Both denominators are now stated where the shares are introduced, with one clause saying why they differ from the intervals |
| S2 | Table 6 prints a cell where random advice beats no advice (NF-ToN-IoT, seed 42: 0.4002 against \texttt{head\_only}'s 0.3809) and the text passed over it. Now stated, with the observation that it is one cell of six and that both claims, real against random and real against off, hold at every seed |
| S3 | `loop_vs_agaf`, a raw contract key, was left in the prose at §4.4 with no gloss anywhere in the report. Now reads as the feedback model against the fusion model |
| S4 | §4.2's two rung lists are rankings; they now say so |
| S5 | The decomposition caption said shares are of the gap to the feedback model, but the final row's share uses the smaller gap to the fusion model. Fixed in `tables/make_tables.py` and regenerated; only `tab_decomposition.tex` changed |

**One self-correction.** The S1 edit first gave the ToN headroom as +0.1091. The exact value is
0.109031, which rounds to +0.1090. The number audit passed the wrong value, which is a third
instance of the audit accepting a token that no artifact supports at that precision.

### Discussion, approved and applied 2026-09-18

| ID | What changed |
|---|---|
| D1 | **The fifth home of the prototype over-claim.** Section 5.4 still said "Under the prototype consultant none of those comparisons is separated on either dataset", pointing back at a list that includes the semantic model and every faithful baseline. Wrong twice: the prototype feedback model is separated above the semantic model on NF-ToN-IoT, and above all six NF-UNSW-NB15 baselines and both NF-ToN-IoT TE-G-SAGE variants. Commit `3443016` fixed the abstract, 4.2, the figure caption and the conclusion; the introduction was fixed earlier in this review; this was the last one. A grep over all sections, `make_tables.py` and `make_plots.py` now returns nothing |
| D2 | **RQ1 contradicted 4.2.** It said the experiment changed "only the classifier" and concluded the semantic rung's weakness "therefore traces to" the prototype scorer. Section 4.2 states that the entropy percentile moved with the consultant (31 to 29, 25 to 16) and explicitly refuses to attribute the lift to the consultant. RQ1 now restates the confound, cites 4.2, and gives the conclusion as consistent-with rather than established |
| D3 | "It stays above the GNN model on both datasets" is an ordering of means, and under the prototype consultant it is not separated. Now says so |
| D4 | Limitation 2's "the two comparisons this report calls not separated" now names them: the fusion model against the GNN model on each dataset |

**What D2 costs.** RQ1's answer is weaker after this edit, and deliberately so. The strong version
is recoverable only by an experiment this review does not run: the prototype configuration at
k = 29 and k = 16, or the trained head at k = 31 and k = 25, so that the consultant is the only
thing that moved.

### Conclusion, approved and applied 2026-09-18

| ID | What changed |
|---|---|
| C1 | The conclusion said the consultant changed "and nothing else about the architecture", which is literally true but left the conclusion more confident about attribution than 4.2 and the revised RQ1. It now says the entropy percentile was re-selected alongside the consultant, and that the lift is not attributed to either change alone |
| C2 | Graph scale was absent from what the conclusion leaves open. It now names the 656 and 2,127 edge counts and the 12-to-40 edge classes beside the deployment limitation |

**One self-correction.** The C1 edit first ended "The two changes are not separated from each
other", which misuses *separated*, a term this report defines as an interval excluding zero.
Replaced with "That lift is not attributed to either change alone" before the section was
finalised.

### Abstract, front matter and the first visual pass, 2026-09-19

`poppler` was installed on 2026-09-19, so rendered pages could be inspected for the first time.

| ID | What changed |
|---|---|
| A1 | The abstract was the only part of the report that never stated the scope limit. It now says edges are aggregated with the attack label in the grouping key and that no number is a deployment-valid estimate |
| V1 | **Found only by looking at the page.** `fig_comparisons` carries a note rendered inside the image. It still read "There is no feedback-versus-semantic-model comparison for the prototype consultant" while the same figure, after commit `3443016` added the two rows, plots exactly that comparison in both panels. The note now states the real verdict: separated on NF-ToN-IoT, not on NF-UNSW-NB15. Fixed in `make_plots.py` and regenerated with `python make_plots.py comparisons`, so the other ten figures were untouched |
| V2 | Figure 1 floated above the "3 Method" heading. Changed from `[tbp]` to `[!ht]`; it now sits under the paragraph that introduces it, still on page 17 |

**Why V1 matters beyond itself.** It was the sixth home of the prototype separation claim and the
only one that is not a `.tex` file. The compiler sees an image and `check_report_numbers.py`
reads `.tex` sources, so no existing check could reach it. Every other figure's baked-in note was
read for the same kind of staleness; `fig_ladder`'s even discloses the top-k confound correctly.

**Author's decision:** the AI-use statement in `99_statements.tex` is to stay exactly as written.
Raised because this review changed claims, not only prose; the author declined the change.

**Deliberate, verified, not a finding:** `\reportsubtitle` is defined in `main.tex` and never
printed. Commit `a4cd1d0` explains why: the faculty template has one title field and the
registered title belongs in it unaltered.

**Still open, cosmetic:** `\reportinstitutions` is defined and never used. The defence date
prints as dotted rules by design; the author should confirm the faculty accepts that.

**Pages inspected at this stage:** cover, abstract, table of contents, and the eleven figure
pages, including a print-resolution zoom on the architecture diagram's smallest annotations and
on the corrected note in Figure 6.

### Appendices, approved and applied 2026-09-19

| ID | What changed |
|---|---|
| P1 | **An arithmetic error.** A.3 said "on NF-ToN-IoT six of ten sit at or below 0.226". Seven values are at or below it; six are strictly below, and `password` sits exactly at 0.226. Changed to "below", and the six are now named. `RESULTS_ARCHIVE.md` carries the same slip and still says "at or below"; not corrected, it is a record |
| P2 | Table 12 listed class indices 0-9 only, so a reader could not tell which attack classes the consultant is weak at, and the two datasets do not share a mapping. Rebuilt with each dataset's class names beside its values, in its own label-mapping order (`LABEL_MAPPING`, `graph_construction.py:39` for NF-ToN-IoT; the per-class artifact's `class_names` for NF-UNSW-NB15). No value changed |
| P3 | Table 11 declared four columns and used three, leaving a stray `&` on every row |

**Test suite run, 2026-09-19:** `OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/ -q` gives
**174 passed, 359 subtests passed** in 133s. Appendix C states exactly those counts, so that
claim is verified rather than inherited.

**The audit's weakness, demonstrated again.** `check_report_numbers.py --explain 0.662` attributes
that value to `results.oracle_attention...mean_disagreement` in two oracle contracts. The real
source is the reliability table in `docs/RESULTS_ARCHIVE.md:1021`. A passing audit says a token
appears somewhere, not that it means what the sentence says.

### Whole-report pass, 2026-09-19

| ID | What changed |
|---|---|
| W1 | **The table of contents pointed at the wrong page for References**, 61 instead of 57. `\addcontentsline{toc}{section}{References}` sat after `\bibliography`, so it recorded the page the bibliography ended on. Moved above it, with a comment saying why. Every other entry, including all three appendices, was already correct |
| W2 | The final reference spilled two lines onto a page of its own. `\setlength{\bibsep}{0.9ex}` pulls it back; the bibliography now ends cleanly on page 60 and the report is 68 pages rather than 69 |
| W3 | Section 2.6 said the work "instantiates the loop". *Loop* is the old internal name for the feedback model. Now "that loop", which refers to the prior-work mechanism it is describing |

**Checks run across the whole document**

- Bibliography reconciled both ways: 38 entries defined, 38 cited, nothing orphaned or missing.
- No undefined references and no multiply-defined labels.
- RQ1 to RQ5 each appear in the introduction, the results and the discussion.
- Superseded names are gone: no `AGAF`, no "semantic rung". "Graph encoder" survives three
  times, all describing other people's systems, which is the documented rule.
- Exactly one sentence repeats across sections, the headline in the abstract and the
  conclusion. Deliberate.
- **Every hand-transcribed figure value checked against its contract**: all ladder means,
  standard deviations and per-seed values, and all 24 comparison intervals under both
  consultants. Zero mismatches.
- `OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/ -q`: 174 passed, 359 subtests passed.
- Audit exit 0. Zero em dashes. Build clean, 0 overfull boxes.
- Pages inspected as rendered images: cover, abstract, both table-of-contents pages, the eleven
  figure pages, a wide-table results page, the appendix tables, and every bibliography page.

### Citation verification, 2026-09-20

Six of roughly twenty-six substantive citations were read against the actual papers, chosen
because the report's argument leans on them. Three problems were found and fixed.

| ID | What changed |
|---|---|
| C1 | **Factual error.** §2.5 said LOGIN prompts the LLM with "node text and one-hop topology". The paper prompts with the node's own text and a description of its **two-hop** neighbourhood (`qiao2025login`, §5.4: "its original text $s_n$, the two-hop neighborhood $N_2(n)$ description, and the neighbor labels"). Corrected |
| C2 | **Two passages were verbatim from DAS and unquoted.** "the refined semantics are fed back to update the same graph learner" is twelve consecutive words from the DAS v1 abstract; "refinement loop monotonically decreases a task-adaptive surrogate objective" is eight words from its Appendix B. Both reworded in §2.5, and the introduction's shorter version of the same sentence with them |
| C3 | §2.5 and §1 described DAS's setting as "general text-rich graphs". Its abstract says it is evaluated "on both text-rich and text-free graphs", and its gains concentrate on structure-dominated ones. The structural-statistics-as-sentence device the report credits DAS with is what DAS uses for its **text-free** graphs. Both passages now say so |

**Similarity check.** An n-gram comparison between the report and the four papers downloaded in
full (DAS, E-GraphSAGE, LOGIN, Sentence-BERT) found two distinct 8-word overlaps before the fix,
both from DAS. After the fix, and after rewording one coincidental 7-word collocation that
appeared in §2.3 and §3.6 ("the balance between structure and semantics is"), **zero overlaps of
seven words or more remain** against those four sources.

**Verified correct, quoting the source**

| Claim | Source | Verdict |
|---|---|---|
| E-GraphSAGE classifies an edge from the concatenation of its two endpoint embeddings | Eq. 5: "the edge embeddings ... are calculated as the concatenation of the node embeddings of nodes u and v" | correct |
| Its published Eq. 4 aggregates edge features | "our newly proposed neighborhood aggregator function creates the aggregated embeddings of the sampled neighborhood edges" | correct |
| Node features initialised to ones | "We use the vector $x_v = \{1,...,1\}$ to initialise the node features" | correct, matches the contract's `node_init` |
| LOGIN uses MC-dropout variance to find uncertain nodes | §5.2, "Monte Carlo dropout variational inference", "the variance of the predictions" | correct |
| Right: explanation appended to node text and re-embedded | §5.5.1, "we append the explanation $e_n$ ... to its original text $s_n$" and Eq. 9 | correct |
| Wrong: edges pruned | §5.5.2, "we prune edges based on node similarity scores" | correct in substance; the report omits that pruning is by a similarity threshold rather than wholesale |
| DAS: closed feedback loop, fixed GNN, implicit supervision | v1 abstract, verbatim | correct |
| DAS: Majorization--Minimization, monotonic decrease | Appendix B and Theorem B.2 (Monotonic Descent) | correct |
| DAS: five structural statistics | §3, "degree, betweenness, closeness, clustering coefficient, and square clustering coefficient" | correct, exact list |
| DAS is named DAS in v1 and GES after the April-2026 revision | v1 abstract names DAS; the current arXiv version names GES | the bib note is correct |
| RoGRAD claims priority for iterative refinement | abstract: "the first iterative paradigm", "transforms LLM augmentation for graphs from static signal injection into dynamic refinement" | correct |
| Sentence-BERT reports MEAN as the stronger pooling | Table 6: MEAN 80.78 / 87.44, CLS 79.80 / 86.62, MAX 79.07 / 69.92; "The default configuration is MEAN" | correct in substance; the report says "the two common choices" where three were evaluated |

### Second citation pass, 2026-09-20

Twelve further papers read. One error, significant enough to move a paragraph.

| ID | What changed |
|---|---|
| C4 | **GMLM was the lead example of the wrong family, and the claim about it was backwards.** §2.3 said GMLM "uses separate graph and language towers, concatenates their representations, and passes the result to an MLP" and that "its own comparison explicitly distinguishes this design from cross-attention, gating, and learned modality weights". The paper's third stated contribution is "a bi-directional cross-attention fusion module that moves beyond simple concatenation", and its methods section repeats "we move beyond simple concatenation and employ a bi-directional cross-attention mechanism". It runs Graph-to-Text and Text-to-Graph attention and concatenates the two *attended* results. GMLM moved to the third family beside RAGFormer, CAST and BiGTex, described as its authors describe it; the first family now rests on BertGCN, which the text already called the cleaner example. The four-family structure and this report's own placement in the second family are unchanged |

**Verified correct in this pass, against the source**

| Paper | Claim | Verdict |
|---|---|---|
| GLANCE | lightweight router, advantage-based objective because LLM calls are non-differentiable, gains on heterophilous nodes | all three verbatim in the abstract |
| BertGCN | final prediction interpolates GCN and BERT with one global hyperparameter | exact: $Z = \lambda Z_{GCN} + (1-\lambda) Z_{BERT}$, "$\lambda$ controls the tradeoff" |
| GMU | multiplicative gates decide how modalities influence the activation | verbatim |
| RAGFormer | stacks the branch embeddings as a token sequence, self-attention with a residual | verbatim: "stack all node embeddings ... into a sequence", "a self-attention layer followed by a residual connection" |
| CAST | cross-attention, materials property prediction, graphs lose global structure that text restores | confirmed |
| BiGTex | GNN embeddings enter the LLM as soft prompts; text returns by cross-attention with the graph as query | all three confirmed, including "the graph embedding serves as the query" |
| DET | uses the terms "structural encoder" and "semantic encoder"; argues against late fusion | confirmed; it contrasts itself with methods that "concatenate the output embeddings of different encoding layers" |
| GL-Fusion | message passing folded into every transformer layer, no separate fusion module | confirmed |
| Dual-Stream | trainable scoring function emitting a scalar weight per modality per input | confirmed: "a trainable feedforward attention scoring function" |
| XG-NID | LLM used only after classification; "dual modality" means flow-level and packet-level | both confirmed |
| GATv2 | GAT's attention ranking is unconditioned on the query node | verbatim, "static attention" |
| TE-G-SAGE | edge-aware GraphSAGE with SHAP, chronological evaluation | confirmed |

**Coverage after two passes: 18 verified, 4 errors found and fixed.**

**Still unread, 19 citations, all lower risk:** BSTFNet, CPS-IDS, GCN-2-Former, TESSERACT,
Sommer, Arp et al., Scarfone, CySecBERT, the two dataset papers, NetFlow standardisation, and
the standard tooling citations (GraphSAGE, GAT, BERT, AdamW, PyTorch Geometric, scikit-learn,
focal loss, Efron).

**Two small imprecisions reported and left unfixed at the author's discretion:** LOGIN prunes
edges below a similarity threshold rather than wholesale, and Sentence-BERT compared three
pooling strategies rather than "the two common choices".

### Two problems found in Results that were NOT in the report — both fixed 2026-09-19

The report was correct in both cases. They were left alone during the section reviews because
contracts and `PROJECT_NOTES.md` sit outside an editorial review, then fixed when the author said to
continue fixing.

**Fixed:** `configuration.selection_curve_note` in both `*_current.json` now reads
0.8238--0.8277 and 0.5170--0.5232, which is what the twelve sweep files in
`results/knob_selection_head/` give. `PROJECT_NOTES.md` corrected to match, with a dated note.
`scripts/derive_report_numbers.py` had already recorded the disagreement in its `knob_curves`
note; that text now records the correction instead, and `derived_numbers.json` was regenerated,
with a diff confirming only the note string changed. No test reads this field, and
`refresh_contracts.py --check` still reports contracts matching the aggregates.

**Fixed:** `HANDOFF.md` §4's Gap A table now carries the committed intervals,
+0.0701 [+0.0353, +0.1057] and +0.0511 [-0.0082, +0.1110], with a dated line saying it had held
the pre-regeneration values that §4b item 5 of the same file describes as replaced. No verdict
changed.

The original description of both problems follows.

- **Knob-curve spans.** §4.8 gives 0.8238--0.8277 (UNSW top-k) and 0.5170--0.5232 (ToN top-k).
  The `configuration.selection_curve_note` string in both `*_current.json` says 0.8241--0.8277
  and 0.5168--0.5232, and `PROJECT_NOTES.md` repeats the contract. Recomputing the three-seed mean per
  candidate from all twelve files in `results/knob_selection_head/` gives the report's values
  exactly: UNSW top-k 0.8238--0.8277 argmax 29, scale 0.8252--0.8277 argmax 2.0; ToN top-k
  0.5170--0.5232 argmax 16, scale 0.5196--0.5232 argmax 20.0. The note strings are wrong.
- **`HANDOFF.md` §4 Gap A table is pre-regeneration.** It gives the consultant change as
  +0.0702 [+0.0359, +0.1035] on UNSW and +0.0504 [-0.0091, +0.1091] on ToN.
  `results/consultant_change_interval.json` and the report both give +0.0701 [+0.0353, +0.1057]
  and +0.0511 [-0.0082, +0.1110]. §4b item 5 of the same file says every interval was
  regenerated, so the handoff contradicts itself.

### Verified against artifacts during the Results audit

Reproduced exactly: both oracle gains and intervals, and the four captured shares, from the two
`*_oracle_ceiling_v2_trained_head.json`; the seed-42 caveat (0.7522 against 0.7543, headroom
-0.0021) from the same files' per-seed runs; both consultant-change intervals under both
procedures from `consultant_change_interval.json`; every §4.5 complementarity figure from
`consultant_complementarity.json`, which also records `seed = 42` as the text says; all thirteen
per-class values checked from `multiseed_head_per_class.json`; the decomposition arithmetic and
its four shares; the flagged and gated counts; the attention-injection intervals; the ablation
table's six intervals and the real-minus-random minimum of 0.098 behind "at least 0.09". The
62.8% in-degree-one figure was recomputed directly from the NF-ToN-IoT graph.

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
