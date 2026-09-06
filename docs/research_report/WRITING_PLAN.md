# Report Writing Plan — LLM-G-IDS M2 Research Report

## Context

The experimental work is finished. What remains is writing the report.

The project built an edge-level intrusion detection pipeline that fuses a GATv2 graph
encoder with a frozen CySecBERT semantic encoder, then adds an uncertainty-guided
feedback loop. Over the last week the work was benchmarked against two published
graph-based NIDS architectures (E-GraphSAGE, TE-G-SAGE) re-trained on the same
representation. Every number the report needs now exists and is committed.

Part 1 (Introduction) is drafted and provisionally approved; the user will re-read it
later. **Scope decision made: the report covers BOTH datasets** (NF-UNSW-NB15 and
NF-ToN-IoT), which contradicts the current `REPORT_WORKFLOW.md` scope note and means Part 1
needs revising — it is written single-dataset throughout.

The intended outcome is a five-part M2 report whose every quantitative claim traces to a
committed artifact, which reports negative results as prominently as positive ones, and
which reads like a person wrote it.

**Resolved 2026-09-06:** the four rungs are measured at 3 training seeds (fold partition
fixed) and the ladder CIs are two-level (edges AND training seed). **Nothing is separated
on either dataset.** Write no ladder separation claim; state orderings by mean as orderings.

---

## Which workflow governs (added 2026-09-05)

Academic Research Skills (**ARS v3.21.1**) is installed as a Claude Code plugin. It is the
**planning and quality-control layer**. It does not replace the deliverable's structure.

- **Structure** = the supervisor-accepted five-part M2 shape in `REPORT_WORKFLOW.md`.
- **Manuscript** = LaTeX: `main.tex` + `sections/*.tex`, compiled by `./build.sh`.
  ARS defaults to Markdown / APA 7.0 / DOCX. **Do not produce a parallel Markdown
  manuscript** — that forks the source of truth. ARS outputs are planning artifacts only.
- **Scientific truth** = the LLM-G-IDS artifacts named in this plan, never ARS's own
  summaries of them.

**Do not run `/ars-full`.** The sequence is in "Tasks" below.

Relevant ARS commands: `/ars-plan`, `/ars-outline`, `/ars-lit-review`, `/ars-reviewer`,
`/ars-citation-check`. Note ARS installs a `PreToolUse` write-scope guard on
`Write|Edit|Bash`; it is benign and runs on every such call.

---

## How to use this plan (token discipline)

This plan is the context pack. **Do not explore the repository.** Every number you need
to write Parts 2–5 is below.

Read exactly these, in this order:

1. This plan.
2. `docs/research_report/EVIDENCE_STATE.md` — the verification of the project's scientific
   state, completed 2026-09-03 against artifacts and source. **That work is done; do not
   repeat it.**
3. `docs/research_report/REPORT_WORKFLOW.md` — writing standards, terminology lock, section states.
4. `docs/research_report/sections/01_introduction.tex` — what Part 1 already says, so you don't repeat or contradict it.
5. `docs/RESULTS_ARCHIVE.md` — **only the sections a given task names**, not the whole file.
6. For Part 2 only: `docs/Full 25-Work Comparison Table (Parallel GNN+LLM Fusion).md` and `docs/Step 3 Fusion Literature Survey.md`.

Open a `results/*.json` **only** when you are about to cite a number not listed in this
plan. Do not read source code unless Part 3 requires a specific implementation detail.

---

## The system, in one paragraph

Network flows are aggregated into a directed multigraph keyed on
`(src_ip, dst_ip, attack_type)`. Nodes are IP addresses carrying 10 centrality measures;
edges are aggregated flows carrying 5 attributes (flow_count, total_bytes, avg_duration,
protocol, port). **Classification is edge-level.** A GATv2 encoder produces structural edge
embeddings; each edge is also serialised into controlled, label-free natural language and
encoded by CySecBERT into a 768-d semantic embedding. Phase 1 (AGAF) fuses the two with
gated attention. Phase 2 selects high-entropy edges, consults a whitened prototype scorer,
and reintroduces the result **by modifying edge representations** (`injection_mode: edge`).
Evaluation is pooled 5-fold out-of-fold macro-F1, seed 42.

---

## Every number you need

### Ladder (pooled OOF macro-F1, v2 encoding, 3 training seeds, mean ± std)

| Dataset | top-k | scale | GNN | LLM | AGAF | Loop |
|---|---:|---:|---:|---:|---:|---:|
| NF-UNSW-NB15 (10 cls) | 31.0 | 2.0 | 0.7437 ± 0.023 | 0.7353 ± 0 | **0.7793 ± 0.012** | 0.7644 ± 0.004 |
| NF-ToN-IoT (8 cls) | 25.0 | 20.0 | 0.4336 ± 0.005 | 0.2785 ± 0 | 0.4102 ± 0.028 | **0.4521 ± 0.011** |

Source: `results/{unsw_nb15,ton_iot}_current.json` (schema 4 / 6),
`results/multiseed_ladder_v2_legacy.json`. Bold = highest mean. **No comparison is
separated** (two-level CI includes zero everywhere). The ± is training-seed variance only;
fold-partition variance is unmeasured — say so wherever a ± appears.

### Ladder significance (archive §1.1)

| comparison | UNSW | ToN |
|---|---|---|
| AGAF − GNN | +0.0373 [−0.0004, +0.0764] P=0.974 — *touches zero* | −0.0962 [−0.1489, −0.0441] P=0.0005 — **separated, wrong direction** |
| AGAF − LLM | +0.0240 [−0.0048, +0.0543] P=0.949 — *crosses* | +0.0526 [+0.0134, +0.0951] P=0.996 |
| Loop − AGAF | +0.0135 [−0.0197, +0.0462] P=0.778 — *crosses* | +0.1151 [+0.0594, +0.1670] P=1.0 |
| Loop − GNN | +0.0507 [+0.0193, +0.0829] P=0.998 | +0.0189 [−0.0139, +0.0534] P=0.862 — *crosses* |
| Loop − LLM | +0.0374 [+0.0036, +0.0680] P=0.987 | +0.1677 [+0.1276, +0.2103] P=1.0 |

### SOTA baselines (archive §5)

| Dataset | E-GraphSAGE pub → refit → +feats | TE-G-SAGE pub → refit → +feats |
|---|---|---|
| UNSW | 0.1194 → 0.1368 → 0.1173 | 0.3841 → 0.5842 → **0.7196** |
| ToN | 0.4141 → 0.4141 → **0.4358** | 0.1931 → 0.2345 → 0.3523 |

**Loop vs the strongest baseline (two-level bootstrap, positive = our rung ahead):**

- UNSW vs TE-G-SAGE+features: **+0.0535 [+0.0102, +0.0990] — separated.**
- ToN vs E-GraphSAGE+features: +0.0133 [−0.0597, +0.0853] — **not separated.**
- ToN vs E-GraphSAGE as published: +0.0340 [−0.0121, +0.0809] — **not separated.**

**Other rungs vs the strongest baseline:**

- UNSW vs TE-G-SAGE+features: GNN +0.0027, LLM +0.0160, AGAF +0.0400 — **none separated.**
- ToN vs E-GraphSAGE as published: LLM −0.1338, AGAF −0.0812 — **separated, baseline ahead.**

### Decomposition of the naive UNSW gap (a centrepiece result)

| Step | macro-F1 | attributable to |
|---|---:|---|
| best baseline as published | 0.3841 | — |
| + fair tuning (`refit`) | 0.5842 | +0.200, our choice of graph scale |
| + our 10 centralities | 0.7196 | +0.135, our feature engineering |
| our AGAF rung (highest mean, 3 seeds) | 0.7793 | +0.060, our architecture |
| our Loop rung (3 seeds) | 0.7644 | +0.045 |

≈51% tuning budget, ≈34% feature engineering, **≈15% architecture** (using AGAF). Note the
baseline rows are 3-seed means too; the decomposition is of point estimates and carries no
interval — do not call any step significant.

### E-GraphSAGE endpoint-only ceiling (archive §5.5)

E-GraphSAGE classifies edges from `CONCAT(h_u, h_v)` alone — the edge's own features never
reach the classifier (paper Eq. 5; confirmed in the authors' notebook as
`self.W(th.cat([h_u, h_v], 1))`). Parallel edges between one IP pair are therefore
mathematically indistinguishable.

| Dataset | edges | indistinguishable by endpoints |
|---|---:|---:|
| NF-UNSW-NB15 | 656 | **385 (58.7%)** |
| NF-ToN-IoT | 2127 | 167 (7.9%) |

Not an epoch artifact: trained to the authors' full 4999 epochs, fold-0 validation
macro-F1 still plateaus at ~0.11 (best 0.1384 @ epoch 1400).

### Gate 0 — the project's central finding (archive §0, §2.4, §2.5)

A **perfect consultant** (true label at ±4 nats, output fusion disabled) delivers
**+0.0345 on UNSW** (P=0.995) and **+0.1090 on ToN** (P=1.0), both separated. The canonical
prototype consultant captures **−4.5% (UNSW) / 11.9% (ToN)** of that headroom, neither
separated. A stronger trained-head consultant captured **12.5% (UNSW, unseparated)** and
**−38.0% (ToN, a separated regression)**.

**Conclusion: the channel has real capacity; neither realistic consultant exploits it.**
The binding constraint is consultant quality/calibration, not mechanism capacity.

### Forced deviations from the source papers (D1–D5)

D1 no chronological split (aggregation dropped timestamps) → our stratified folds.
D2 fanout [25,15] and batch 4096 exceed the whole graph → full-batch, full-neighbourhood.
D3 both papers use a fixed epoch count with no validation (4999 / 20) → our 300-epoch,
patience-25 schedule, identical to our own rungs.
D4 `rare_min_freq=50` erases most port categories at our scale → as published in
`as_published`, swept in `refit`.
D5 E-GraphSAGE's paper Eq. 4 aggregates edge features alone, but the released code builds
messages as `W_msg([h_u ‖ e_uv])` → the code's form is used.

---

## Traps — violating any of these is a factual error

0. **MULTI-SEED — resolved 2026-09-06; read `EVIDENCE_STATE.md` §4 before any headline.**
   At 3 training seeds AGAF (0.7793) is above the loop (0.7644) at every seed on UNSW, and
   the loop (0.4521) is above AGAF (0.4102) at every seed on ToN. **Neither is separated**
   (two-level CIs [−0.055, +0.027] and [−0.032, +0.111]). *"The feedback loop is the
   strongest rung"* is **false on UNSW** and unsupported as a separation on ToN. The
   canonical loop's feedback was effectively off (bias ~0.02–0.08); switching it on did not
   help (archive 2026-09-06). RQ3's answer is negative and verified.

0b. **NOVELTY — two claims at different levels. Hold BOTH; do not collapse them.**

   **CLAIM A — the configuration (PERMITTED).** The *conjunction* of
   (1) bidirectional iterative GNN-semantic coupling, (2) joint classification by
   attention-fusion of structural and semantic embeddings in one classifier, and
   (3) network intrusion detection on an IP/flow graph, is not occupied by prior work.
   The four verified anchors each break at least one conjunct:

   | Work | (1) iterative | (2) joint fusion head | (3) IDS |
   |---|---|---|---|
   | **DAS** (2512.21106) | yes | **no** — refined semantics feed a fixed GNN that classifies alone | **no** |
   | **GLANCE** (2510.10849) | **no** — single-round routing | **no** — selective delegation, not two-branch fusion | **no** |
   | **RoGRAD** (2510.01910) | yes | **no** | **no** |
   | **LOGIN** (2405.13902) | training-only | **no** — GNN classifies alone | **no** |

   Supporting analysis: `docs/Full 25-Work Comparison Table (Parallel GNN+LLM Fusion).md`.

   **Conditions on Claim A.** Write it as *"we are not aware of prior work combining …"*,
   never "confirmed unique", "first", or "no prior art exists" — an absence-of-evidence
   claim cannot be verified and this project has retracted overclaims before. It is a
   **positioning** claim for Related Work, not a result.

   **CLAIM B — the components (FORBIDDEN).** No individual mechanism here is novel. Each of
   these is refuted by a specific verified paper:

   | Do NOT claim | Refuted by |
   |---|---|
   | a closed GNN-LLM feedback loop | **DAS** — its abstract describes exactly this |
   | iterative semantic refinement | **DAS** (with an MM convergence argument); **RoGRAD** claims "the first iterative paradigm" |
   | structural statistics serialised to natural language | **DAS** — node text from degree, betweenness, closeness, clustering |
   | selective / uncertainty-guided consultation | **GLANCE** — a per-node router deciding when to query the LLM |
   | LLM-consulted GNN training | **LOGIN** |

   **DAS is the closest prior work — closer than LOGIN.** See `EVIDENCE_STATE.md` §7b.

   **CLAIM C — performance (BOUNDED BY EVIDENCE).** Architectural novelty is not a result.
   A reviewer will grant Claim A and immediately ask whether it helps. The honest answer:
   AGAF is **not separated** from the GNN on UNSW and is **significantly below** it on ToN;
   the loop is not separated from AGAF; and the multi-seed field (trap 0) may invert the
   ordering. Never let Claim A imply Claim C.

   **How the three fit together:**

   > The configuration is, as far as we are aware, unoccupied (A). Its individual
   > mechanisms are established prior art (B). What this work contributes is the
   > edge-level intrusion-detection instantiation and the measured finding that a
   > realistic consultant does not exploit a channel with demonstrated capacity (C).

   Related Work carries A. Method carries the honest description of B. Results and
   Discussion carry C.

0c. **DEPLOYMENT VALIDITY.** `src/pipeline/step1/graph_construction.py:215` groups by
   `[SRC_COL, DST_COL, ATTACK_COL]` — the attack label defines edge identity. These are not
   deployment-valid estimates. State it plainly wherever results are presented.

1. **Absolute levels are NOT comparable across datasets.** UNSW scores 10 classes, ToN 8
   (classes 3 `dos` and 7 `ransomware` have 4 and 3 edges; they stay in the graph but are
   excluded from the metric). **Compare ladder *shape* across datasets, never absolute
   macro-F1.** A 10-class ToN macro-F1 is not a reportable number.
2. **Say "highest point estimate", never "top rung".**
3. **`real > control` is RETRACTED** (archive §2.2). Never cite
   `real > control > shuffled > random` as an established ordering.
4. **Attention injection is inert** — churn exactly 0.0000 in every configuration,
   confirmed 3×. The live mechanism is edge injection. Part 1 already states this
   correctly; keep it consistent.
5. **Never claim we outperform a published paper's own reported numbers.** The phrasing is
   *"re-trained on our aggregated representation."*
6. **`plus_node_features` is NOT E-GraphSAGE or TE-G-SAGE.** It is a labelled,
   non-faithful ablation. Never tabulate it under the paper's name.
7. **E-GraphSAGE's large UNSW margins are ceiling-limited, not architectural evidence.**
   Do not lead with +0.60 numbers.
8. **ToN's AGAF "regression" is withdrawn.** At 3 seeds AGAF spans 0.3975–0.4422 and
   AGAF−GNN is not sign-stable; the P=0.0005 claim was a seed-42 artifact. Do not describe
   a ToN AGAF regression as current, and do not guess a cause for one.
9. **On ToN the loop has the highest mean; on UNSW AGAF does.** Neither is separated. Say
   "highest mean", never "strongest rung".
10. **Ladder CIs are two-level (edges AND training seed) and none excludes zero.** Baseline
    CIs also resample the baseline seed. The ± is training-seed variance at a fixed fold
    partition; fold-partition variance is unmeasured — say so wherever a ± appears.
11. **Two numbers were once fabricated in this project and caught in review.** If you
    cannot point to the artifact a number came from, do not state it.

---

## Terminology lock (from `REPORT_WORKFLOW.md` — do not deviate)

- CySecBERT: **"pretrained cybersecurity language encoder"** on first use, **"semantic
  branch"** thereafter. Never "generative LLM". It is a frozen 110M-parameter BERT-class
  encoder used as a sentence encoder — no generation, no instruction-following.
- Phase 2 is **"uncertainty-guided semantic feedback"**. Do not rename it.
- Keep edge-level classification terminology consistent throughout.
- At final QA only, replace the cover wording with title **"Graph-Based Intrusion
  Detection with a Cybersecurity Language Encoder"** and subtitle **"Structural–Semantic
  Fusion with Uncertainty-Guided Feedback"**.

---

## Writing standard — professional and human

The guideline (`docs/reports/Report_Guideline.pdf`) states three principles: *every claim
needs a number attached*; *state what was tried and did not work*; *compare against
something*. This project is unusually well placed to satisfy all three — use that.

**Lead with the honesty.** Most M2 reports oversell. This one can say "our fusion stage is
statistically indistinguishable from a published baseline given the same features" and read
as more competent for it. Frame limitations as findings, not apologies.

**Write mechanisms as short narratives.** A mechanism paragraph should carry a hypothesis,
a prediction, and a test. The E-GraphSAGE ceiling is the model: endpoint-only edge
representation → should collapse where parallel edges are common → UNSW 58.7% gives 0.12,
ToN 7.9% gives 0.41. That is memorable. "E-GraphSAGE performed poorly" is not.

**Name the surprises in plain sentences.** *"E-GraphSAGE's published configuration won its
own grid."* *"The channel works; neither consultant uses it."* One clear sentence beats
three hedged paragraphs.

**Concrete prose habits:**

- Vary sentence length deliberately. A six-word sentence after a thirty-word one resets the
  reader. Uniform 25-word sentences are what makes technical writing feel machine-made.
- Cut: "it is worth noting", "it should be mentioned", "in order to", "utilise",
  "leverage" (as a verb), "delve", "furthermore" stacked on "moreover".
- Prefer active voice where the actor matters: *"we selected on validation folds"*, not
  *"selection was performed on validation folds"*.
- Never use a promotional adjective for your own result. Let the number carry it.
- Prefer a table to a paragraph of numbers; prefer a sentence to a table of two rows.
- Every paragraph should be answerable to "what does the reader now know that they
  didn't?" If nothing, cut it.

**Always distinguish** an observed point estimate from a statistically supported
difference. This is the single most common failure mode in this project's history.

---

## Tasks

**Preparation tasks P1-P3 come first. Drafting does not begin until the user approves the
outline.** After every task: commit, then **stop and report to the user**. After any task
that changes a `.tex` file, run `./docs/research_report/build.sh` first and confirm it
compiles, and update the status table in `REPORT_WORKFLOW.md`.

---

### Task P1 — Claim-evidence map

Write `docs/research_report/CLAIM_EVIDENCE_MAP.md`. One row per claim the paper could make.

| column | content |
|---|---|
| claim | stated as it would appear in the paper |
| evidence | the exact artifact + field, or `NONE` |
| status | `separated` / `not separated` / `unsupported` / `retracted` |
| notes | the caveat that must travel with it |

Source the rows from this plan's numbers and `EVIDENCE_STATE.md` §§4-6. Every claim whose
status is not `separated` must carry the wording it will be written with. A claim with
`NONE` in the evidence column may not enter the manuscript at all.

Include, explicitly: the loop-vs-AGAF ordering (§4 inversion), the three declared
limitations, Gate 0, the SOTA comparison, and the E-GraphSAGE ceiling.

### Task P2 — Novelty and related-literature verification

**The four core anchors are already verified** (LOGIN, DAS, GLANCE, RoGRAD — DOIs, titles,
authors, no editorial notices; `EVIDENCE_STATE.md` §7b), and **the novelty framing is
already settled** (trap 0b). Do not redo either.

P2's remaining job is narrower:

1. Use `/ars-lit-review` to check for work **published since 2026-09-05**, and for any
   **edge-level or IDS-specific** GNN-semantic feedback work the four anchors do not cover.
   A hit there would break Claim A in trap 0b and must be reported immediately.
1b. **Re-run the conjunction check against DAS, GLANCE and RoGRAD.** The existing
   gap analysis (`docs/Full 25-Work Comparison Table (Parallel GNN+LLM Fusion).md`, and the
   slide derived from it) predates all three — DAS is December 2025, GLANCE and RoGRAD
   October 2025. Trap 0b's table records my assessment that none of them breaks the
   conjunction; confirm it against the papers themselves before Claim A is written.
1c. **Fix a known error in that gap analysis.** Its LOGIN entry is headed as satisfying
   the IDS conjunct while its body correctly states "Not IDS, not joint classification."
   The body is right; the header label is wrong. Correct it wherever it is reused.
2. Assemble the rest of `references.bib` — it holds 6 entries and needs roughly 30.
   Sources for the broader map: `docs/Full 25-Work Comparison Table (Parallel GNN+LLM
   Fusion).md` and `docs/Step 3 Fusion Literature Survey.md`.
3. Verify every DOI resolves before it enters `references.bib`. **Scite is reconnected**
   (verified working 2026-09-05). Search by **DOI or exact title, not keywords** — Scite's
   free-text search failed to surface LOGIN, while the DOI lookup found it immediately.

**Do not cite anything you have not confirmed exists.** This project has previously
retracted a claim and caught two fabricated numbers.

### Task P3 — Missing experiments and publication risks

From `EVIDENCE_STATE.md` §6 plus anything P1 and P2 surface, write a short risk register:
what a reviewer will attack, what evidence would answer it, and whether that evidence is
obtainable before submission. Flag which risks are accepted-and-disclosed versus which
need work.

### Task P4 — `/ars-plan`, then `/ars-outline`

Run `/ars-plan` for framing and positioning. Report, stop.

Then `/ars-outline`. **The outline must not fix a headline claim that depends on the loop
outranking AGAF** (trap 0). Report, stop, and wait for explicit user approval.

**Drafting begins only after the user approves the outline.**

---

## Drafting tasks (blocked until the outline is approved)

### Task 0 — Re-scope Part 1 and the workflow to both datasets

`REPORT_WORKFLOW.md` "Current scope" restricts everything to NF-UNSW-NB15. The user has
decided on both datasets. Update that section, then revise
`sections/01_introduction.tex`, which says "NF-UNSW-NB15" in the framing paragraph, RQ1,
RQ2, RQ4, contribution 4, and the report-structure paragraph.

Add NF-ToN-IoT to the dataset paragraph, and add the cross-dataset contrast as a stated
contribution — the ladder holds on UNSW and inverts on ToN, which is a finding, not a
blemish. Carry trap #1 into the text explicitly: only ladder *shape* is comparable across
datasets.

Do not otherwise rewrite Part 1; the user approved its substance.

### Task 1 — Part 2: Background and State of the Art

Sources: `docs/Full 25-Work Comparison Table (Parallel GNN+LLM Fusion).md`,
`docs/Step 3 Fusion Literature Survey.md`, and `references.bib`.

The guideline warns: *"be coherent here, careful what SOTA to mention before the other."*
Order the section as a narrowing funnel — graph-based NIDS → language models in security →
GNN+LLM integration patterns → the specific gap this work occupies. Cover the six-paradigm
map (GNN-only, LLM-only, sequential GNN→LLM, sequential LLM→GNN, interactive, parallel
frozen), and be explicit that the closest prior work is identified rather than hidden.

Introduce E-GraphSAGE and TE-G-SAGE here, since Part 4 compares against them.

**The related-work section must position DAS, LOGIN, GLANCE and RoGRAD honestly and
early** — they are the mechanism's prior art, and DAS is the closest. **This is where
Claim A belongs** (trap 0b): present the conjunction table, and state the gap as
"we are not aware of prior work combining …". Do not state Claim A anywhere else, and do
not let it imply a performance result.

`references.bib` currently holds **6 entries** and will need roughly 30. Add entries as you
cite; do not cite anything you have not confirmed exists.

### Task 2 — Part 3: Method

The guideline calls this the longest, most technical section, written so someone else could
reproduce it. Cover: graph construction and aggregation; the v2 encoding (cols 0–2 log1p +
Z-scored, cols 3–4 vocabulary indices consumed by learned embeddings — and why raw
categorical port/protocol was a bug worth ~0.22 macro-F1); node and edge features; the
GATv2 encoder; label-free text serialisation and the validation guard; the semantic branch;
AGAF; uncertainty-guided semantic feedback via edge injection; and the evaluation protocol
(stratified 5-fold, pooled OOF macro-F1, `OMP_NUM_THREADS=1`, ≥3-seed hyperparameter
selection on validation folds only).

State the label-in-edge-identity limitation again here, concretely, where a reader
evaluating reproducibility will meet it.

Also describe the baseline protocol: same rows, labels and folds; each paper's own
categorical featurisation; the three columns (`as_published`, `refit`,
`plus_node_features`); and deviations D1–D5.

### Task 3 — Part 4: Results

Structure: ladder on both datasets → significance → SOTA baseline comparison → the
decomposition table → ablations and negative results.

Required content, all listed above: the ladder table; the §1.1 significance table; the
baseline table across three columns; the Loop-vs-baseline intervals; the decomposition
(51/35/14); the E-GraphSAGE ceiling with its UNSW/ToN inversion as confirming evidence;
Gate 0 and Gate 0.5; the attention-injection null result; the `real > control` retraction.

`figures/` and `tables/` are **empty**. Two figures carry the most weight:

1. **The ceiling diagram** — two parallel edges between one IP pair collapsing to an
   identical `concat(h_u, h_v)` representation. This is the report's best explanatory
   visual.
2. **The decomposition chart** — the 0.3841 → 0.5842 → 0.7196 → 0.7793 staircase (AGAF,
   3-seed mean; loop 0.7644 shown alongside), coloured by what each step is attributable to.
   No step carries an interval; label it a point-estimate decomposition.

Per `REPORT_WORKFLOW.md`, diagrams are supplied by the user through Claude: **pause and
request each figure** with purpose, full labels, hierarchy, colours, aspect ratio, format
and filename, rather than generating one unasked.

### Task 4 — Part 5: Discussion and Conclusion

What the results mean, honestly. The defensible claims are narrow: on UNSW the loop is the
only rung separated from the strongest baseline, by +0.0535; on ToN nothing is separated
from E-GraphSAGE and two of our rungs are significantly behind it. The decomposition shows
~14% of the naive gap is architectural.

Gate 0 is the intellectual centre of the discussion: the feedback channel demonstrably has
capacity, and no realistic consultant exploits it. This is also where the contribution is
stated in its narrow form (trap 0b) — DAS reports consistent gains from a closed GNN-LLM
refinement loop; this work shows that result does not transfer to edge-level network-flow
IDS, and isolates why. Say what that implies for future work —
calibration and trust of consultant logits, not a bigger consultant.

Open work: deployment-valid graph construction where edge identity does not depend on the
label; fold-partition variance (only training-seed variance is measured); a stronger
semantic consultant within the `todo.md` design (text-only judgment) — the oracle shows the
channel has headroom that no realistic consultant has used.

### Task 5 — Final QA

Abstract; complete `references.bib`; appendices; the cover title/subtitle swap; a full pass
checking every quantitative claim against its artifact; confirm no trap above is violated;
final compile.

---

## Verification

```bash
# compile (first run ~2 min, then ~1 s)
./docs/research_report/build.sh

# nothing in the repo broke
cd /Users/ali/Desktop/ids-framework && OMP_NUM_THREADS=1 python -m pytest tests/ -q
```

Expected: `build/main.pdf` written; **84 passed**.

Before declaring any part done:

1. Every number in the new text appears either in this plan or in a `results/*.json` you
   opened and can name.
2. No sentence claims a separation that the CI tables above do not support.
3. Terminology lock respected — grep the new section for "LLM", "top rung", and
   "outperform".
4. The PDF compiles and the new section appears in the table of contents.

## Do not

- Re-run the 3-seed sweep or any of conditions A/B/C. They are done and committed.
- Re-run any experiment or "improve" any number. Writing only.
- Generate figures without first requesting them per the diagram protocol.
- Read the whole repository. This plan and `EVIDENCE_STATE.md` are the context pack.
- Run `/ars-full`. The staged sequence in Tasks P1-P4 is deliberate.
- Produce a Markdown manuscript alongside the LaTeX one. ARS outputs are planning artifacts.
- Begin drafting before the user approves the `/ars-outline` output.
- Cite `multi_seed_caveat`'s agaf_mean / loop_mean as results — they are not artifact-backed.
