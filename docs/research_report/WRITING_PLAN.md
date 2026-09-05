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

**One deferred item:** the four ladder rungs were only ever run at seed 42, so no
confidence interval includes rung-side seed variance. The user postponed the 3-seed
re-run. Write every separation claim so it survives either outcome (wording below).

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

### Ladder (pooled OOF macro-F1, v2 encoding)

| Dataset | top-k | scale | GNN | LLM | AGAF | Loop |
|---|---:|---:|---:|---:|---:|---:|
| NF-UNSW-NB15 (10 cls) | 31.0 | 2.0 | 0.7219 | 0.7353 | 0.7595 | **0.7728** |
| NF-ToN-IoT (8 cls) | 25.0 | 20.0 | 0.4290 | 0.2785 | 0.3334 | **0.4478** |

Source: `results/{unsw_nb15,ton_iot}_current.json`.

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
| our Loop rung | 0.7728 | +0.053, our architecture |

≈51% tuning budget, ≈35% feature engineering, **≈14% architecture.**

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

0. **MULTI-SEED INVERSION — read this before writing any headline.**
   `unsw_nb15_current.json` → `multi_seed_caveat` records **agaf_mean 0.7757 vs loop_mean
   0.7545** over three seeds — the *reverse* of the single-seed headline (AGAF 0.7595, Loop
   0.7728). `loop_vs_agaf` already crosses zero. That field is explicitly
   *"not re-verified against a saved artifact"*, so **those two numbers may never be cited
   as results** — but they mean the claim *"the feedback loop is the strongest rung"* is
   **not established**. The 3-seed re-run is postponed. See `EVIDENCE_STATE.md` §4.

0b. **NARROW NOVELTY.** LOGIN (arXiv 2405.13902), DAS (2512.21106), GLANCE (2510.10849)
   and RoGRAD (2510.01910) are documented in `docs/feedback_loop_mechanisms.md`. **Never
   claim the first GNN-LLM feedback system.** Confine novelty to IDS / network-flow /
   edge-level feedback.

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
8. **ToN's AGAF regression is undiagnosed.** Do not guess a cause.
9. **AGAF is not ToN's strongest rung** and the loop does not underperform AGAF there —
   both are reversed from the pre-v2 finding.
10. **Every CI omits rung-side seed variance** (rungs ran at seed 42 only). Wherever a
    separation is claimed, write *"separated at seed 42; rung-side seed variance not yet
    estimated"* — this survives the pending re-run either way.
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

Use `/ars-lit-review`. Confirm against current literature that the narrow novelty claim
holds: interactive GNN-semantic feedback applied to **edge-level network-flow intrusion
detection**. LOGIN, DAS, GLANCE and RoGRAD are prior art for the mechanism in general —
the contribution is the IDS instantiation and its measured limits, not the mechanism.

Add every cited work to `references.bib`. **Do not cite anything you have not confirmed
exists.** Scite was disconnected on 2026-09-03; use `/ars-citation-check` plus
`WebSearch`/`WebFetch`, or ask the user to reconnect Scite.

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
2. **The decomposition chart** — the 0.3841 → 0.5842 → 0.7196 → 0.7728 staircase, coloured
   by what each step is attributable to.

Per `REPORT_WORKFLOW.md`, diagrams are supplied by the user through Claude: **pause and
request each figure** with purpose, full labels, hierarchy, colours, aspect ratio, format
and filename, rather than generating one unasked.

### Task 4 — Part 5: Discussion and Conclusion

What the results mean, honestly. The defensible claims are narrow: on UNSW the loop is the
only rung separated from the strongest baseline, by +0.0535; on ToN nothing is separated
from E-GraphSAGE and two of our rungs are significantly behind it. The decomposition shows
~14% of the naive gap is architectural.

Gate 0 is the intellectual centre of the discussion: the feedback channel demonstrably has
capacity, and no realistic consultant exploits it. Say what that implies for future work —
calibration and trust of consultant logits, not a bigger consultant.

Open work: deployment-valid graph construction where edge identity does not depend on the
label; the 3-seed rung re-run; ToN's undiagnosed AGAF regression.

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

- Start the 3-seed rung re-run. The user postponed it.
- Re-run any experiment or "improve" any number. Writing only.
- Generate figures without first requesting them per the diagram protocol.
- Read the whole repository. This plan and `EVIDENCE_STATE.md` are the context pack.
- Run `/ars-full`. The staged sequence in Tasks P1-P4 is deliberate.
- Produce a Markdown manuscript alongside the LaTeX one. ARS outputs are planning artifacts.
- Begin drafting before the user approves the `/ars-outline` output.
- Cite `multi_seed_caveat`'s agaf_mean / loop_mean as results — they are not artifact-backed.
