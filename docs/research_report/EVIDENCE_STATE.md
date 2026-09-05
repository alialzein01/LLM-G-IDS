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
| `results/unsw_nb15_current.json` | **schema 3**, generated 2026-08-27, `v2_log_cont_cat_idx`, pooled five-fold out-of-fold |
| `results/ton_iot_current.json` | **schema 5**, generated 2026-08-27, 8-class metric |
| `results/{unsw_nb15,ton_iot}_sota_baselines.json` | 2026-09-02, published-baseline comparison |
| `results/cross_dataset_comparison.json` | cross-dataset contract |
| `docs/RESULTS_ARCHIVE.md` | narrative record, including retractions (§3) |
| `reproduce_ladder.py` | reproduction entry point |

Canonical ladder (pooled OOF macro-F1):

| Dataset | GNN | semantic | AGAF | feedback |
|---|---:|---:|---:|---:|
| NF-UNSW-NB15 (10 cls) | 0.7219 | 0.7353 | 0.7595 | 0.7728 |
| NF-ToN-IoT (8 cls) | 0.4290 | 0.2785 | 0.3334 | 0.4478 |

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
AGAF regressed to 0.3334, **significantly below the bare GNN** (−0.0962, P=0.0005). The
cause is **undiagnosed**. Do not guess it.

---

## 4. THE MOST SERIOUS UNRESOLVED ISSUE — multi-seed inversion

`results/unsw_nb15_current.json` → `multi_seed_caveat`:

```
agaf_mean 0.7757  (std 0.0152)
loop_mean 0.7545  (std 0.0223)
source: "conversation-recorded, not re-verified against a saved artifact in this pass"
```

**Over three seeds, AGAF exceeds the feedback loop — the reverse of the single-seed
headline** (AGAF 0.7595, Loop 0.7728). The `loop_vs_agaf` interval already crosses zero.

Two consequences, both binding:

1. Those two numbers are **not artifact-backed** and must never be cited as results.
2. The claim "the feedback loop is the strongest rung on UNSW" is **not established**. Write
   every separation as *"separated at seed 42; rung-side seed variance not yet estimated."*

The 3-seed re-run is postponed by the user. Until it runs, no headline may depend on the
loop outranking AGAF.

---

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
7. **Against published baselines**, only the loop separates on UNSW (+0.0535 vs TE-G-SAGE +
   our features); on ToN nothing separates from E-GraphSAGE and two rungs are significantly
   behind it.
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

**NOT yet verified:** DAS (`arXiv 2512.21106`, cited in `docs/feedback_loop_mechanisms.md`).
Task P2 must confirm it before it enters `references.bib`.

### Two competing-novelty findings — read before drafting any novelty claim

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

## 8. Readiness assessment

**Ready:** the evidence base is well governed. Limitations are self-declared in the
artifact, provenance is traceable, leakage guards are enforced in code, and the project has
a documented history of retracting its own claims.

**Not ready for a locked outline:** risk 3 (§4) can invert the headline. `/ars-plan` may
proceed for framing and positioning. `/ars-outline` should not fix a headline claim that
depends on the loop outranking AGAF until the 3-seed re-run completes.
