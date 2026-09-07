# Evidence State — verification completed 2026-09-03

Independent verification of the project's scientific state, performed against artifacts and
source code rather than recollection. **This work is done. Do not repeat it.**

Repository state at verification: branch `experiment/edge-feature-encoding`, commit
`4521bdc`, 71 commits spanning 2026-05-06 → 2026-09-03.

Every finding below was checked against a named file, line, or artifact field. Where a
claim could not be traced to an artifact, it is marked as such and must not be cited.

---

## 1. Canonical sources of experimental truth

| Artifact | Status |
|---|---|
| `results/unsw_nb15_current.json` | **schema 5**, generated 2026-09-07, `v2_log_cont_cat_idx`, pooled five-fold OOF, 3 seeds; loop consults the trained LLM head |
| `results/ton_iot_current.json` | **schema 7**, generated 2026-09-07, 8-class metric, 3 seeds; loop consults the trained LLM head |
| `results/{unsw_nb15,ton_iot}_sota_baselines.json` | **schema 2**, 3-seed rung-vs-baseline intervals for the new loop and head-alone rows (2026-09-07) |
| `results/cross_dataset_comparison.json` | **schema 6**, 2026-09-07 |
| `docs/RESULTS_ARCHIVE.md` | narrative record, including retractions (§3) |
| `reproduce_ladder.py` | reproduction entry point |

Canonical ladder (pooled OOF macro-F1):

| Dataset | GNN | semantic | AGAF | feedback |
|---|---:|---:|---:|---:|
| NF-UNSW-NB15 (10 cls) | 0.7437 ± 0.023 | 0.7353 ± 0 | **0.7793 ± 0.012** | 0.7644 ± 0.004 |
| NF-ToN-IoT (8 cls) | 0.4336 ± 0.005 | 0.2785 ± 0 | 0.4102 ± 0.028 | **0.4521 ± 0.011** |

**UPDATED 2026-09-06.** Values are means ± std over 3 training seeds (42/1/2), fold
partition fixed at the seed-42 split; contracts are schema 4 (UNSW) / 6 (ToN), aggregate
`results/multiseed_ladder_v2_legacy.json`. Bold = highest mean. **No rung comparison is
separated on either dataset** (every two-level CI includes zero). The single-seed
schema-3/5 values that used to sit here (UNSW loop 0.7728; ToN AGAF 0.3334) are withdrawn
and preserved under each contract's `supersedes`.

**Out-of-fold discipline is enforced in code, not merely asserted.** `reproduce_ladder.py`
documents two rules it exists to enforce, both "violated by earlier runs": every rung uses
the same whitened-prototype scorer, and AGAF consumes per-fold OOF embeddings and never
`step3_gnn/edge_embeddings.pt` (which is trained on all edges and would leak test labels
through features).

---

## 2. Components verified

- **Edge-level classification.** Flows are edges; a flow between two IPs is the atomic unit.
- **Structural path.** GATv2 encoder (`src/models/gnn_classifier.py`).
- **Semantic path.** CySecBERT (`markusbayer/CySecBERT`), frozen, mean-pooled to 768-d
  (`src/pipeline/step3/encode_kg.py:13`). A BERT-class encoder, not a generative model.
- **Label-free guarantee is enforced.** `assert_label_free()`
  (`src/pipeline/step2/knowledge_graph.py:210`) raises if any attack term appears in the
  serialised text. Called at `build_simplified_nl.py:122`. This is a hard guard, not a
  convention.
- **AGAF** gated fusion (`src/models/fusion_classifier.py`).
- **Feedback** via **edge-representation injection** — `injection_mode: edge` in both
  canonical contracts. The attention-bias variant is measured **inert** (churn exactly
  0.0000, confirmed 3×; archive §2.1).

---

## 3. Stale and superseded evidence

**Schema 2 is formally superseded.** `supersedes` in `unsw_nb15_current.json`: schema 2 was
produced on v1 encoding (raw port/protocol magnitudes) and before the loop-fusion fairness
fix, both of which understated AGAF and the loop.

**Git history carries explicit corrections** — verified commit subjects:

- `b3a1915` Retract "real advice beats no advice" — the source run was not thread-pinned
- `9766049` Correct the oracle-ceiling numbers and revise the feedback-mechanism plan
- `15b54be` Reconcile all result contracts with live pipeline output (v2 encoding + fusion fix)
- `f05c98f` Give the loop's fusion the same footing as the AGAF rung
- `c318fd9` Encode protocol and port as categories, not magnitudes (the v1 bug)
- `be67723` Gate 0.5: a stronger consultant does not exploit the feedback channel

**Stale artifacts that must not be used as evidence:** `AGAF_Step3_Fusion.pptx`,
`agaf_architecture.pdf` (both pre-v2). Three historical result JSONs
(`unsw_nb15_loop_mechanism`, `unsw_nb15_edge_injection`, `ton_iot_current`) use the retired
phrase "top rung" in narrative fields; do not quote those fields verbatim.

**ToN reversal:** under v1, AGAF was ToN's strongest rung. Under v2 + the fairness fix,
AGAF appeared to regress to 0.3334, "significantly below the bare GNN" (−0.0962, P=0.0005).
**Withdrawn 2026-09-06:** across 3 seeds AGAF spans 0.3975–0.4422 and AGAF−GNN is
−0.024 with CI [−0.092, +0.061], not even sign-stable. The single-seed "regression" was
within seed noise. The v1→v2 AGAF change on ToN is therefore not established either way.

---

## 4. Multi-seed inversion — RESOLVED 2026-09-06

The conversation-recorded caveat (AGAF 0.7757 / loop 0.7545) is replaced by an artifact.
Condition A of `results/multiseed_ladder_v2_legacy.json` (canonical architecture, selected
knobs verified in every run, 3 training seeds, fold partition fixed):

| dataset | loop − AGAF, two-level CI | per-seed sign | verdict |
|---|---|---|---|
| UNSW | −0.0156 [−0.0552, +0.0268] P=0.229 | negative at 42, 1, 2 | **not separated; AGAF is the higher mean** |
| ToN  | +0.0428 [−0.0316, +0.1105] P=0.874 | positive at 42, 1, 2 | **not separated; loop is the higher mean** |

Binding consequences for the report:

1. **"The feedback loop is the strongest rung on UNSW" is false** and may not be written.
   On ToN the loop has the highest mean; write "highest mean, not separated".
2. **Write no ladder separation anywhere.** Orderings by mean may be stated as orderings.
3. **The ± is training-seed variance only.** Fold-partition variance is unmeasured; say so
   wherever a ± appears.
4. **The canonical loop's feedback was effectively off** — injected bias ~0.02–0.08 on
   unit-variance features (archive 2026-09-06 Finding 3). Fixing the two signals that made
   it so (selector head collapsed to one class; consultant softmax uniform at T=10) and
   re-selecting the knobs did **not** produce a separable gain at 3 seeds (Findings 1–2).
   This is the verified, negative answer to RQ3. Per `todo.md`, the semantic judgment
   depends on flow text only; GNN-state-conditioned advice was ruled out as out of spec.
5. **Reproducibility finding:** the schema-3/5 AGAF and loop values do not reproduce at
   seed 42 under current code; GNN and LLM reproduce bit-exactly. Cause not established.

## 5. Statistical reality (from the canonical artifact)

| comparison | mean diff | 95% CI | verdict |
|---|---:|---|---|
| `loop_vs_gnn` | +0.0507 | [+0.0193, +0.0829] | **separated** |
| `loop_vs_llm` | +0.0374 | [+0.0036, +0.0680] | **separated** |
| `agaf_vs_gnn` | +0.0373 | [−0.0004, +0.0764] | not separated (touches zero) |
| `agaf_vs_llm` | +0.0240 | [−0.0048, +0.0543] | not separated |
| `loop_vs_agaf` | +0.0135 | [−0.0197, +0.0462] | not separated |

**Only the two non-adjacent comparisons are separated.** The defensible UNSW claim is
*"the loop beats either single modality"* — never *"each rung improves on the one below."*

**What is solid:** the correct ablation baseline. `real_vs_head_only` +0.0472
[+0.0147, +0.0800] — separated. Score feedback against `head_only`, never the bare GNN rung.

---

## 6. Publication risks

1. **Deployment validity — confirmed still present in code.**
   `src/pipeline/step1/graph_construction.py:215` groups by `[SRC_COL, DST_COL, ATTACK_COL]`.
   The attack label participates in defining edge identity. The canonical artifact declares
   this as limitation 1. This is the most serious construct-validity threat and a reviewer
   will find it immediately. State it plainly; do not bury it.
2. **Most of the ladder is not statistically separated** (§5).
3. **Multi-seed inversion risk** (§4).
4. **No isolated attention-feedback ablation** — declared limitation 2: the
   real-versus-head-only ablation combines attention feedback and final semantic fusion.
   Compounded by attention injection being independently measured as inert.
5. **Novelty must be narrow.** LOGIN (arXiv 2405.13902), DAS (2512.21106), GLANCE
   (2510.10849) and RoGRAD (2510.01910) are all documented in
   `docs/feedback_loop_mechanisms.md`. **Never claim the first GNN↔LLM feedback system.**
   Confine the claim to IDS / network-flow / edge-level feedback.
6. **Gate 0 is arguably the real contribution, and it is a negative result.** A perfect
   consultant gains +0.0345 UNSW (P=0.995) / +0.1090 ToN (P=1.0), both separated; the
   canonical prototype captures −4.5% / 11.9% of that headroom, neither separated; a
   stronger trained head captured 12.5% (unseparated) / −38.0% (a separated regression).
   The channel has capacity; no realistic consultant exploits it.
7. **Against published baselines** (updated 2026-09-07 to 3-seed rungs vs 3-seed baselines;
   the +0.0535 figure previously quoted here came from the seed-42-only rung and is
   superseded). On UNSW every rung is separated above all six baseline configurations; the
   only *ns* cells are GNN and LLM against **TE-G-SAGE + our node features**, which is a
   non-faithful variant, not a published architecture. On ToN nothing separates from
   E-GraphSAGE except the LLM rung, which is separated **below** it; the loop separates
   above E-GraphSAGE `as_published`/`refit` under the paired seed-matched interval only.
   Full table: archive §5.3.
8. **Scale** — 656 (UNSW) and 2,127 (ToN) aggregated edges, versus ~10⁶-edge per-flow
   graphs in the source papers.
9. **Selection bias** — declared limitation 3: `top_k_percent` was selected on rotating
   validation folds; nested CV or an untouched set is required for a fully unbiased final
   estimate.

---

## 7. Corrections to prior understanding

- `requirements/todo.pdf` **does not exist**. The equivalents are `todo.md` (repo root) and
  `docs/reports/project_requirements.pdf`.
- `README.md` does **not** overclaim novelty — a grep for first/novel/SOTA/outperform found
  only one hit, and it concerns a reproduction check.

---

## 7b. Verified citations (Scite, 2026-09-05)

Confirmed to exist, with matching titles and authors, and **no editorial notices**
(no retractions or concerns) at time of check:

| DOI | Work | Authors | Year |
|---|---|---|---|
| `10.48550/arXiv.2405.13902` | LOGIN: A Large Language Model Consulted Graph Neural Network Training Framework | Qiao, Ao, Liu | 2024 |
| `10.48550/arXiv.2510.10849` | Glance for Context: Learning When to Leverage LLMs for Node-Aware GNN-LLM Fusion | Loveland, Yang, Koutra | 2025 |
| `10.48550/arXiv.2510.01910` | Are LLMs Better GNN Helpers? Rethinking Robust Graph Learning under Deficiencies with Iterative Refinement (RoGRAD) | Wang, Gao, Kharel | 2025 |

| `10.48550/arXiv.2512.21106` | Semantic Refinement with LLMs for Graph Representations (DAS) | Thapaliya, Wang, Li | 2025 |

All four anchors are verified. DAS is open access under CC-BY.

### DAS is the CLOSEST prior work — closer than LOGIN

Verified 2026-09-05 from the paper's own abstract and full text. DAS:

- "couples a fixed graph neural network (GNN) and a large language model (LLM) in a
  **closed feedback loop**. The GNN provides implicit supervisory signals to guide the
  semantic refinement of LLM, and the refined semantics are **fed back to update the same
  graph learner**" — structurally our Phase 2;
- builds node descriptions by expressing **structural statistics in natural language**,
  explicitly "degree, betweenness, closeness, clustering coefficient, and square clustering
  coefficient" — the same idea as our 10 centrality measures plus label-free serialisation;
- gives a Majorization-Minimization argument that its iterative refinement loop
  monotonically decreases a task-adaptive surrogate objective.

**What still separates our work — state the contribution in exactly these terms:**

| | DAS | This work |
|---|---|---|
| granularity | node-level | **edge-level** (a flow is the unit) |
| domain | general text-rich / text-free graphs | **network-flow intrusion detection** |
| injection target | refines LLM semantics | **edge representations** consumed by the next graph update |
| headline result | consistent improvements | **the consultant does not exploit a channel that demonstrably has capacity** (Gate 0) |

The mechanism is **not** the contribution. The IDS instantiation and the negative
diagnostic result are. Any sentence implying we invented GNN-LLM feedback, iterative
semantic refinement, or structural-statistics-as-text is refuted by DAS.

### Two further competing-novelty findings — read before drafting any novelty claim

1. **RoGRAD's abstract explicitly claims "RoGRAD is the first iterative paradigm"** that
   moves LLM augmentation for graphs "from static signal injection into dynamic
   refinement." Our loop is also iterative refinement. Do not phrase our contribution in a
   way that collides with this claim.
2. **GLANCE trains a lightweight router that decides, per node, whether to query the LLM**,
   using inexpensive per-node signals. This is conceptually very close to our
   uncertainty-guided selection of high-entropy edges. Our selection policy is therefore
   **not** novel in itself; what is untried is the edge-level network-flow IDS setting and
   the measured finding that the consultant fails to exploit the channel.

**Practical note on Scite:** free-text search did **not** surface LOGIN; the DOI lookup
did. Search by DOI or exact title, not keywords.

---

## 7c. Canonical loop redefined (2026-09-07) — supersedes §4 and §5 for the loop rung

The loop's semantic consultant is now the per-fold trained LLM head, not the whitened
prototype; the LLM rung and AGAF keep the prototype encoder, and the head alone is
carried as its own rung. Five mechanism treatments on the prototype consultant had all
failed first (archive 2026-09-07). Consequences for anything written from this file:

- **The ladder moved.** Loop 0.8341 ± 0.004 (UNSW) / 0.5044 ± 0.008 (ToN), separated
  above AGAF, the GNN and the prototype LLM rung on **both** datasets. §5's "nothing is
  separated" no longer describes the loop rung.
- **What is still not separated:** AGAF vs GNN, and **loop vs head_alone**. The head
  alone carries the higher 3-seed mean on both datasets (0.8374 / 0.5081) with the
  interval straddling zero and the sign unstable across seeds. §6 risk 6's framing
  survives in a new form: the system now clears every baseline except the one it is
  built on top of.
- **Publication risk, restated.** The proposed system is statistically indistinguishable
  from its own semantic branch used alone. That must appear wherever the AGAF/GNN gains
  appear. It is recorded in both contracts as `loop_vs_head_alone_headline`.
- **The knobs are selected, not tuned** — both curves flat within noise, ToN's scale on
  the swept range's upper boundary.

## 8. Readiness assessment

**Ready:** the evidence base is well governed. Limitations are self-declared in the
artifact, provenance is traceable, leakage guards are enforced in code, and the project has
a documented history of retracting its own claims.

**RQ5 intervals: done (2026-09-07).** Both sides of every rung-vs-baseline comparison are
now three training seeds. `results/{unsw_nb15,ton_iot}_sota_baselines.json` are at schema 2
with `statistical_comparisons_3seed` and `seed_matched_comparisons_3seed`; the seed-42-only
blocks are preserved under `superseded` and must not be quoted. The asymmetry caveat that
stood on the old blocks ("our four rungs were run at seed 42 only ... re-running the ladder
at seeds 1 and 2 is the highest-value follow-up") is discharged.

**Ready for a locked outline (2026-09-06).** §4 is resolved: the headline is "nothing
separated; AGAF highest mean on UNSW, loop on ToN; the mechanism's feedback was effectively
off and switching it on did not help". Results and Discussion can be drafted from the
schema-4/6 contracts and `docs/RESULTS_ARCHIVE.md` 2026-09-06.
