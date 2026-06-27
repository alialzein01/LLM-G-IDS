# Phase 2 — Bidirectional Feedback Loop: Advisory & Implementation Plan

**LLM‑G‑IDS · CESI LINEACT · prepared for Alex**
Grounded in the actual `ids-framework/` source tree and the **current** Phase 1 artifacts (read on 2026‑06‑15). Every shape, module name, and file path below was verified against the repository, not the prompt text.

---

## 0. Executive summary — read this first

The Phase 2 prompt is built on one load‑bearing assumption: *"Is the AGAF gate collapsing toward LLM (gate ≈ 0, the GNN is already suppressed)?"* I checked it directly against `step3_fusion/metrics.json` for both datasets. **The premise is empirically false, and the truth points to a different — and more fixable — pathology.**

### 0.1 The gate is not collapsed; it is *pinned at indecision*

Support‑weighted, pooled out‑of‑fold gate means (`fusion_diagnostics_summary`):

| Dataset | global `gate_GNN` | global `gate_LLM` | per‑class `gate_GNN` range |
|---|---|---|---|
| NF‑ToN‑IoT | **0.494** | 0.506 | 0.435 (dos) – 0.516 (mitm) |
| NF‑UNSW‑NB15 | **0.465** | 0.535 | 0.410 (Shellcode) – 0.526 (Exploits) |

The gate is not near 0. It sits within ±0.07 of **0.50 for every class on both datasets**. AGAF is not suppressing the GNN — it is averaging the two modalities almost uniformly and barely conditioning on the input.

**Root cause is in the code, not the data.** `train_fusion.py` minimises

```python
loss = cls_loss - gate_entropy_lambda * gate_entropy          # line 123-127
# gate_entropy = -[g·log g + (1-g)·log(1-g)].mean()   (maximised at g = 0.5)
```

With `GATE_ENTROPY_LAMBDA = 0.01`, the optimiser is *rewarded* for driving every gate value toward 0.5. The regulariser meant to "prevent gate collapse" has instead manufactured **gate indecision**: a near‑constant 50/50 blend regardless of edge or class. That is why the per‑class spread is so flat.

### 0.2 The consequence: AGAF does not beat the best unimodal MLP on *either* dataset

Pulling every current `benchmark_summary.json` (pooled out‑of‑fold macro‑F1, the same statistic `phase4_validate` treats as the headline):

| Model (current repo artifacts) | ToN‑IoT pooled‑F1 | ToN cv‑test mean ± std | UNSW pooled‑F1 | UNSW cv‑test mean ± std |
|---|---|---|---|---|
| GNN end‑to‑end (`step3_gnn`) | 0.2627 | 0.288 ± 0.020 | 0.5519 | 0.550 ± 0.042 |
| **GNN‑emb + MLP** (`baselines/gnn_embedding`) | 0.3261 | 0.331 ± 0.057 | **0.6342** | 0.633 ± 0.016 |
| **CySecBERT + MLP** (`baselines/llm_embedding`) | **0.3979** | 0.392 ± 0.023 | 0.5165 | 0.512 ± 0.065 |
| AGAF fusion (`step3_fusion`) | 0.3211 | 0.340 ± **0.102** | 0.6225 | 0.620 ± 0.038 |

Two things jump out. First, a 50/50 blend of a strong and a weak modality lands *near the average*, which is below the stronger modality's own MLP: on ToN the winner is CySecBERT (0.398) and AGAF (0.321) sits with the weaker GNN‑emb (0.326); on UNSW the winner is **GNN‑emb MLP (0.634)** and AGAF (0.623) is dragged *below* it by the LLM half. Second, AGAF's ToN cross‑fold **std is 0.102** — 4× CySecBERT's — so it is not only lower‑mean but unstable.

> The single most useful baseline that is missing from the prompt's comparison table is **GNN‑emb + MLP**. On UNSW it is currently the best model in the repository. Phase 2 must beat *it*, not just the end‑to‑end GNN at 0.5229.

### 0.3 What this reframes about Phase 2

The right one‑sentence motivation is **not** "rescue a suppressed GNN." It is:

> *AGAF fuses globally and statically, so it collapses to a near‑uniform average that underperforms the better modality. Phase 2 replaces global static fusion with **selective, uncertainty‑targeted, iterative** semantic correction — semantic signal is injected **only where the GNN is genuinely uncertain**, and only **as much as the modalities disagree**, instead of being averaged into every edge.*

This is a sharper and more defensible thesis than the prompt's, and it is exactly what the rest of this document operationalises. It also tells us three design constraints up front: (1) drop or invert the entropy regulariser that caused the 0.5 pinning; (2) make fusion *conditional and sparse*, not global; (3) freeze the strong modality's path so fusion can never drag the result below it (a residual/skip from the better unimodal logits — see §4 and §5).

### 0.4 Two housekeeping actions before any Phase 2 number is reported

1. **The prompt's Phase 1 table is stale.** Only the CySecBERT numbers (0.3979, 0.5165) still match the repo exactly; the GNN and AGAF numbers in the prompt (ToN 0.3006/0.3865; UNSW 0.5229/0.5705) do **not** match current artifacts (the models were re‑run since). Freeze one convention — **pooled out‑of‑fold macro‑F1**, which `phase4_validate.py` already computes — re‑emit all four Phase 1 rows under it, and treat *that* table as the thing Phase 2 must beat. Comparing Phase 2 against moving Phase 1 numbers is the fastest way to lose a reviewer.
2. **Run the redundancy pre‑flight in §11.5** (script provided) before building anything. If the proposed feedback signal correlates > ~0.6 with the existing gate, you are rebuilding AGAF and should stop.

---

## 1. Research interpretation and novelty framing

### 1.1 What "bidirectional" can honestly mean with a frozen LLM

CySecBERT is a fixed `[E × 768]` tensor; it cannot be updated. So "bidirectional" cannot mean "both networks learn from each other." It must be defined at the level of **representations and the message‑passing computation**, where it is genuinely true:

* **GNN → LLM direction** = *selection / routing*. The GNN's own uncertainty (entropy of its `[E × 10]` logits) decides **which** edges are allowed to draw on semantic evidence and **how strongly**. The LLM signal is not applied uniformly; the GNN gates the LLM's influence. The LLM "is consulted" only where the structural model asks for help.
* **LLM → GNN direction** = *re‑entry into message passing*. The semantic feedback is encoded as an extra edge channel and the GNN encoder is **re‑run** with it (§4, §6). Because the channel enters `GATv2Conv`'s `edge_attr` and the encoder is finetuned to use it (§5, Stage 2), the LLM signal literally changes attention coefficients and therefore the structural embeddings of neighbouring edges. The updated structural embeddings change next iteration's uncertainty, which changes selection — closing the loop.

So the loop is: `GNN logits → uncertainty mask (GNN→LLM) → semantic feedback channel → re‑run GNN attention (LLM→GNN) → new logits …`. Frozen weights on the LLM are fine: the LLM is a *fixed knowledge source*, and bidirectionality lives in the iterative coupling of selection and message passing, not in mutual weight updates. This is the same sense in which LOGIN is "interactive" — there too the LLM is a fixed consultant; what changes is the GNN.

### 1.2 What the feedback loop adds over AGAF — stated to survive review

AGAF and the feedback loop differ on three axes a reviewer will probe:

| Axis | AGAF (Phase 1) | Feedback loop (Phase 2) |
|---|---|---|
| **Locus of fusion** | After the GNN, on frozen `[E×64]` embeddings. The GNN never sees the LLM. | Inside the GNN: semantic signal re‑enters `GATv2Conv` and alters message passing. |
| **Selectivity** | Global. Every edge is gated, and §0.1 shows the gate is ~0.5 everywhere → effectively unconditional averaging. | Sparse and conditional. Only high‑entropy edges receive feedback; the rest keep pure structural predictions. |
| **Iteration** | Single forward pass. | Multiple passes; an edge's corrected embedding propagates to its graph neighbours, so correction is *relational*, not per‑edge. |

The sharp claim: **AGAF conditions the *output representation* on both modalities once; the feedback loop conditions the *computation* on both modalities repeatedly and selectively.** Empirically, §0.1–0.2 show AGAF's "conditioning" degenerated to a constant — so the distinction is not merely architectural, it is the difference between a blend that underperforms the best modality and a targeted correction that need not.

### 1.3 Relation to LOGIN (Qiao et al., arXiv 2024) and the IDS‑domain contribution

LOGIN is the correct anchor — it is the only "LLMs‑as‑Consultants" / interactive‑during‑training paper in the corpus. The mechanics rhyme: pick uncertain nodes, consult the LLM, feed the response back to the GNN (LOGIN: correct response → feature update, wrong → edge prune). The differences that make Phase 2 a contribution rather than a re‑application:

1. **Task granularity.** LOGIN classifies **nodes** on citation graphs (Cora/PubMed). LLM‑G‑IDS classifies **edges** (flows) on an IP communication multigraph — `data.edge_label`, the edge‑level prediction head, and the whole split/embedding stack are edge‑centric. Edge‑level interactive refinement on a directed multigraph is not what LOGIN does.
2. **No live LLM.** LOGIN issues live LLM queries on selected nodes mid‑training. Phase 2's hard constraint is **zero LLM calls** — all semantic evidence is the precomputed, **label‑free** `[E×768]` tensor. The consultation is reformulated as a lookup into a frozen semantic space (prototypes / a tiny scorer), which is a genuinely different mechanism with different failure modes (no hallucination, but no novel reasoning either).
3. **Domain‑specific selection.** "Uncertain" in IDS is dominated by the Benign majority class and by classes with single‑digit support (ransomware = 3 edges total). LOGIN's uniform uncertainty selection would spend the LLM budget on Benign. Phase 2 needs **class‑stratified** uncertainty selection (§2, §11.3) — an IDS‑specific design forced by the label distribution.
4. **Three‑variant structural encoder.** LOGIN has a single GNN. Here the encoder is a temporal/content/behavioural triptych (`EDGE_FOCUS_WEIGHTS`), which opens a routing question LOGIN never faced (§2, §11.2): should semantic feedback target the variant aligned with the suspected attack family?

Framing for the paper: *"We transfer the LLMs‑as‑Consultants paradigm from node‑level citation graphs with live LLM access to edge‑level network‑traffic graphs under a frozen, label‑free semantic encoder, and adapt selection and injection to the class‑imbalanced, multi‑variant IDS setting."* That is defensible and specific.

---

## 2. Practical architecture (modules, exact tensors)

All shapes use the verified repo dimensions: `E` edges, `N` nodes, `x ∈ [N×10]`, `edge_attr ∈ [E×5]` (cols 0–2 Z‑scored, cols 3–4 raw categorical), GNN embeddings `[E×64]`, CySecBERT `[E×768]`, `NUM_CLASSES = 10`, `proj_dim = 128`.

New file: `src/models/feedback_classifier.py`. Each module below is a `nn.Module`.

**`UncertaintyDetector`**
`forward(gnn_logits: [E×10]) -> (entropy: [E], uncertain_mask: [E] bool)`
Entropy `H = -Σ softmax(logits)·log softmax(logits)`, normalised by `log 10` to `[0,1]`. The threshold is **not** a fixed constant and **not** a single global percentile — both fail in IDS (§11.3). Use **class‑stratified top‑p**: within each *predicted* class `c`, flag the top‑`p` fraction by entropy. Default `p = 0.30`, swept in Exp 6. This guarantees the loop spends capacity on uncertain *attack* edges, not only on the Benign blob. `p` is a hyperparameter, not learned (a learned threshold on 3‑edge classes will overfit).

**`SemanticFeedbackScorer`**
`forward(llm_emb_subset: [U×768]) -> feedback: [U×10]` for the `U` uncertain edges.
Recommended parameter‑free realisation (Option C, §3): a soft class distribution from distances to **class prototypes** precomputed on the *training fold only*. Holds no trainable weights → cannot memorise ransomware. A learned variant (Option B/D) is kept behind a flag for ablation.

**`AttentionBiasGenerator`**
`forward(feedback: [U×10], uncertain_mask: [E]) -> edge_feedback_channel: [E×k]`
Scatters the per‑edge feedback back to full edge length (zeros on confident edges) and projects `[E×10] → [E×k]` with a single `nn.Linear(10, k)` (default `k = 8`). Output is the **appended edge channel** consumed by the biased encoder (§4). Confident edges receive an exact zero vector, so they are untouched.

**`BiasedGATStructuralEncoder`** (wrapper around the existing `GATStructuralEncoder`)
`forward(x, edge_index, edge_attr, edge_feedback=None)`.
When `edge_feedback is None` it is **bit‑for‑bit the current encoder** (back‑compat preserved). When provided, it concatenates the channel: `edge_attr_aug = cat([edge_attr, edge_feedback], dim=1)` → `[E×(5+k)]`, and the two `GATv2Conv` layers are instantiated with `edge_dim = 5 + k`. Appending (not adding) is what keeps the Z‑scored and categorical columns untouched (§11.6). The three‑variant `EDGE_FOCUS_WEIGHTS` still multiply only the original 5 columns; the appended `k` channels pass through unweighted (or with a learned per‑variant scale — §11.2).

**`IterativeController`** (plain Python state object, not a layer)
Holds `iteration`, `prev_pred [E]`, `prev_entropy [E]`. Exposes `should_stop()` using the convergence metric in §6 (prediction‑churn fraction + entropy delta). Caps at `MAX_ITERATIONS = 3` (Exp 5 shows 1→3 is the useful range).

**`FeedbackFusionClassifier`** (top‑level model)
Owns: a `BiasedGATEdgeClassifier` (the three‑variant model whose encoders are `BiasedGATStructuralEncoder`), the `UncertaintyDetector`, `SemanticFeedbackScorer`, `AttentionBiasGenerator`, and a **late residual fusion head** that combines the iterated structural logits with a frozen "strong‑modality" logit path (§4.4, §5). Produces final `[E×10]`. It is the only class the trainer and `phase5_validate` import.

---

## 3. The feedback signal — decision

Evaluated against the four stated criteria. Scores are qualitative (✓✓ strong, ✓ ok, ✗ weak):

| Option | (a) bidirectional | (b) no LLM at runtime | (c) low ToN overfit | (d) interpretable |
|---|---|---|---|---|
| A — GNN↔LLM cosine agreement (projected) | ✓✓ | ✓✓ | ✓ (needs AGAF projectors) | ✓ (one scalar/edge) |
| B — learned MLP `768→10` confidence | ✓✓ | ✓✓ | ✗ (memorises 3‑edge classes) | ✓ |
| **C — class‑prototype distance** | ✓✓ | ✓✓ | **✓✓ (0 params)** | **✓✓ (soft class vote)** |
| D — learned MLP `768→128` bias into AGAF gate | ✓✓ | ✓✓ | ✗ | ✗ (128‑dim, opaque) |

**Recommendation: Option C as the production signal; Option B as a regularised ablation; Option A as a redundancy probe.**

Why C wins on this dataset. ToN‑IoT has ransomware = 3 and dos = 4 edges *in the entire graph*. Any module with free parameters on a 768‑dim input will memorise those points; that is the dominant risk here (§11.4). Option C has **zero trainable parameters** in the signal itself: prototypes `μ_c = mean of CySecBERT embeddings over training‑fold edges of class c` (cosine‑normalised), and the feedback is `softmax(−τ · d(s_e, μ_c))` over the 10 prototypes. It is computed *per fold from the train mask only*, so it never peeks at val/test and respects `folds.pt`. It is natively interpretable — "this uncertain edge's flow text sits closest to the `password` and `injection` prototypes" — which is precisely the kind of evidence a paper can show. It is bidirectional in the §1.1 sense because the resulting `[E×k]` channel re‑enters GATv2 attention.

The one caveat C must clear is the redundancy test (§11.5): if the prototype soft‑vote already correlates with the AGAF gate, it carries no new information. The flat‑gate finding (§0.1) makes redundancy *unlikely* — a constant‑0.5 gate carries almost no class signal, so a class‑discriminative prototype distance is almost certainly orthogonal to it — but verify before building.

Mandatory constraint check: prototypes are built from the **label‑free** CySecBERT embeddings. We use training‑fold *labels* only to *group* edges into prototypes — no label text ever enters an LLM input, so Constraint 1 holds. (This is the same way `get_class_weights` uses labels: for grouping/weighting, not as model input.)

---

## 4. Injection point — decision

Options, against (a) elegance, (b) gradient stability, (c) no encoder retrain, (d) three‑variant compatibility:

| Injection | a | b | c | d |
|---|---|---|---|---|
| Concatenate to `edge_attr` (`edge_dim 5→5+k`) | ✓✓ | ✓✓ | ✗ (needs finetune) | ✓✓ |
| Additive on `edge_attr` in place | ✗ | ✗ (entangles Z‑scored + categorical cols) | ✓✓ | ✓ |
| Post‑message‑passing additive bias on `h` | ✓ | ✓✓ | ✓ | ✓ |
| Logit bias on final flow repr `[E×261]` | ✓ | ✓✓ | ✓✓ | ✗ (bypasses message passing) |

### 4.1 Primary: concatenate to `edge_attr`, finetune encoder (two‑stage)

This is the only option that makes the LLM signal *enter attention*, which is what licenses the bidirectionality claim and the LOGIN lineage. The objection — "requires retraining the encoder" — is handled by the **two‑stage schedule** in §5: Stage 1 trains everything else on the frozen Phase 1 GNN embeddings; Stage 2 unfreezes the encoder and finetunes at `lr = 1e-4` (vs the original `1e-3`), only long enough to teach the new `k` channels. Appending rather than adding solves the normalisation hazard (§11.6) for free: the Z‑scored columns 0–2 and the raw categorical columns 3–4 are byte‑for‑byte unchanged; the `k` new columns live in their own bounded scale (a projected softmax, roughly `[−1,1]`).

### 4.2 Why not additive‑on‑edge_attr

Columns 3–4 are `most_common_protocol` and `most_common_port` as **raw integers**, and columns 0–2 are Z‑scores. Adding a learned bias on top mixes a correction into a categorical code (meaningless) and silently shifts the normalised attention statistics (`data.edge_attr_mean/std` no longer describe the tensor). Rejected.

### 4.3 Fallback: post‑MP additive bias on `h`

If Stage‑2 finetuning proves unstable on ToN (high‑variance graph, §0.2), fall back to adding `bias_node` to the node embedding **after gat1, before the residual add** (`gnn_classifier.py:71`). It still influences gat2's message passing (one hop of propagation) without touching the attention edge inputs, so it is gentler. It is a weaker "feedback into the GNN" claim but still defensible. Keep it as the `--injection post_mp` option.

### 4.4 The non‑negotiable: a frozen strong‑modality skip

Independent of injection point, the final head must include a **residual path from the better unimodal logits** (CySecBERT‑MLP on ToN, GNN‑emb‑MLP on UNSW), frozen. Concretely `final_logits = strong_logits + α · feedback_correction`, `α` small and learnable, initialised near 0. This is what structurally prevents Phase 2 from repeating AGAF's failure of landing *below* the best modality (§0.2). It also gives a clean ablation ("α = 0" recovers the strong baseline exactly) and a clean story ("feedback is a correction *on top of* the best single view").

---

## 5. Training strategy — decision

**Option C (two‑stage), with the strong‑modality skip from §4.4.**

* **Stage 1 — bias module on frozen embeddings.** Load `step3_gnn/edge_embeddings.pt [E×64]` and `step3_llm/edge_embeddings.pt [E×768]`. Freeze the GNN encoder. Build prototypes per fold (train mask only). Train `AttentionBiasGenerator`, the late‑fusion head, and `α` with `FocalLoss(get_class_weights(...), gamma=2.0)`. Fast, stable, and on its own answers "is the feedback signal worth anything?" — but note Stage 1 alone is *not yet a feedback loop* (the bias never re‑enters message passing); it is a smarter fusion. Report it as the **"feedback‑fusion (no re‑entry)"** ablation (Exp 8).
* **Stage 2 — unfreeze + finetune the encoder.** Swap to `BiasedGATEdgeClassifier`, load Stage‑1 weights, unfreeze the encoder, finetune at `lr = 1e-4`, `weight_decay = 5e-4`, grad‑clip 1.0, with the iterative loop active (§6). This stage is what makes it bidirectional. Early‑stop on val macro‑F1, patience 40 (matches `train_fusion`).

Why not the alternatives. **Option A (full end‑to‑end from scratch)** risks catastrophic forgetting of the structural patterns and, on ToN with single‑digit minority classes, will overfit — the cv‑std already hit 0.102 for static AGAF; an end‑to‑end loop would be worse. **Option B (frozen, bias module only)** is exactly Stage 1 and, as noted, is not honestly a "feedback loop" — keep it only as the ablation that quantifies how much of the gain comes from re‑entry.

**Loss and regularisers (answering the explicit sub‑questions):**

* **Keep `FocalLoss(gamma=2.0)` and `get_class_weights`** (Constraint 7). They are the right tool for the imbalance and are shared across all Phase 1 models, so comparisons stay clean.
* **Do not reuse `gate_entropy_lambda = 0.01` as‑is.** §0.1 proved it manufactures the 0.5‑pinned gate. For Phase 2 there is no scalar modality gate to keep honest; instead regularise the feedback to be **sparse**, not high‑entropy: add `λ_sparse · ||α · feedback_correction||₁` (default `λ_sparse = 1e-3`) so the loop leaves confident edges alone and only spends magnitude where it helps. This is the inverse philosophy of the AGAF regulariser and directly encodes the §0.3 thesis. If you retain any soft gate, set its entropy λ to **0** and let the data choose.

---

## 6. Inference loop — concrete pseudocode

```python
# Preconditions (all verified to exist):
#   data           = torch.load(cfg.graph_path)                 # x[N×10], edge_index[2×E], edge_attr[E×5]
#   llm_emb        = torch.load(cfg.llm_embedding_path)         # [E×768], frozen CySecBERT
#   model          = FeedbackFusionClassifier(...).eval()       # owns BiasedGATEdgeClassifier + sub-modules
#   prototypes     = build_prototypes(llm_emb, labels, train_mask)   # [10×768], train fold only
#   strong_logits  = frozen_strong_head(strong_emb)             # [E×10] from §4.4 skip, computed once
MAX_ITERATIONS, P_UNCERTAIN, EPS_ENTROPY, CHURN_TOL = 3, 0.30, 1e-3, 0.005

x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
edge_feedback = torch.zeros(E, K)                       # iteration 0: no feedback
prev_pred, prev_entropy = None, None

for it in range(MAX_ITERATIONS):
    # ---- LLM -> GNN: run (biased) message passing with current feedback channel ----
    gnn_logits, _ = model.encoder_forward(x, edge_index, edge_attr, edge_feedback)   # [E×10]

    # ---- final prediction = frozen strong view + sparse learned correction ----
    logits = strong_logits + model.alpha * model.correction_from(gnn_logits, edge_feedback)
    pred   = logits.argmax(1)                                                          # [E]
    entropy = -(gnn_logits.softmax(1) * gnn_logits.log_softmax(1)).sum(1) / math.log(10)  # [E] in [0,1]

    # ---- stopping criterion (discrete-prediction-stable) ----
    if prev_pred is not None:
        churn = (pred != prev_pred).float().mean()              # fraction of edges that flipped
        d_ent = (entropy - prev_entropy).abs().mean()           # mean |Δ entropy|
        if churn < CHURN_TOL and d_ent < EPS_ENTROPY:
            break                                               # converged
    prev_pred, prev_entropy = pred, entropy

    # ---- GNN -> LLM: class-stratified uncertainty selection ----
    uncertain = class_stratified_topp(entropy, pred, p=P_UNCERTAIN)     # [E] bool
    # ---- semantic lookup (no LLM call): prototype soft-vote on uncertain edges only ----
    s = F.normalize(llm_emb[uncertain], dim=1)                          # [U×768]
    dist = 1 - s @ F.normalize(prototypes, dim=1).T                     # [U×10] cosine distance
    feedback_u = F.softmax(-model.tau * dist, dim=1)                    # [U×10]
    # ---- build next-iteration edge channel (zeros on confident edges) ----
    feedback_full = scatter_rows(feedback_u, uncertain, size=(E,10))    # [E×10]
    edge_feedback = model.bias_generator(feedback_full)                 # [E×K]; confident rows stay 0

final_logits = logits
```

**Where it breaks.** Two AND‑ed conditions, both in tensor terms: prediction churn `(pred != prev_pred).float().mean() < CHURN_TOL` (the discrete classification has settled) **and** `(entropy - prev_entropy).abs().mean() < EPS_ENTROPY` (the confidence has settled). Using *both* a discrete churn term and a continuous entropy term is what prevents the two classic failures: entropy creeping while labels oscillate, or labels frozen while a few edges flip‑flop. Hard cap at `MAX_ITERATIONS = 3` as the backstop (Exp 5 establishes 1–3 is where the curve flattens). Oscillation guard in §11.6.

---

## 7. Experiments

All on **both** datasets, same `folds.pt`, headline metric = pooled out‑of‑fold macro‑F1.

### 7.1 Primary comparison (re‑emit Phase 1 rows under the frozen convention)

| Model | ToN‑IoT pooled‑F1 | UNSW pooled‑F1 |
|---|---|---|
| GNN end‑to‑end | _(re‑emit; repo: 0.2627)_ | _(0.5519)_ |
| GNN‑emb + MLP **(add this row)** | _(0.3261)_ | _(0.6342)_ |
| CySecBERT + MLP | _(0.3979)_ | _(0.5165)_ |
| AGAF fusion | _(0.3211)_ | _(0.6225)_ |
| **Phase 2 feedback loop** | **?** | **?** |

### 7.2 Ablations (each isolates one mechanism)

1. **No uncertainty filtering** — apply feedback to *all* edges (`p = 1.0`). Tests whether selectivity is the point. Expect a drop toward AGAF‑like averaging.
2. **Random edge selection** — select the *same count* as the entropy filter, but at random. Disentangles "selecting uncertain edges" from "injecting any feedback."
3. **Random feedback signal** — replace prototype soft‑vote with Gaussian noise of identical shape `[U×10]`. Isolates semantic content from the mere structural effect of perturbing edges.
4. **Feedback without the skip / additive only** — drop the §4.4 frozen strong path (or replace late fusion with plain addition). Tests how much of the result is the residual safety net.
5. **Iterations 1/2/3/5** — plot macro‑F1 vs iterations; show convergence and that >3 adds nothing (or destabilises).
6. **Uncertainty threshold sweep** — `p ∈ {0.10, 0.20, 0.30, 0.40, 0.50}`; plot macro‑F1 vs `p`; report the stable operating band.
7. **Variant‑specific vs uniform injection** — if §11.2 routing is implemented, ablate to uniform. Shows whether the temporal/content/behavioural structure earns its keep in Phase 2.
8. **Stage 1 only (no re‑entry)** — the frozen‑encoder "feedback‑fusion" model. Tests whether Phase 2 *subsumes* AGAF (a smarter static fusion) or whether the **re‑entry into message passing** (Stage 2) is what actually adds value. This is the single most important ablation for the novelty claim.

### 7.3 Diagnostic experiment (do this first, §11.5)

Correlate the prototype soft‑vote against `gate_weights.pt` on held‑out edges. If `|ρ| > 0.6`, the signal is redundant with AGAF and the design must change before any training.

---

## 8. Metrics and reporting

* **Primary:** macro‑F1, `f1_score(average='macro', labels=range(10), zero_division=0)` — exactly `classification_metrics` in `metrics.py`.
* **Secondary:** weighted‑F1, accuracy, per‑class P/R/F1/support — already emitted by `classification_metrics`.
* **Confusion matrices** for both datasets via the `phase4_validate` pattern (`confusion_matrix.json`).
* **Selection analysis (Phase‑2‑specific):** per iteration, report (i) #edges selected, (ii) entropy distribution of selected edges, (iii) the **predicted‑class histogram of selected edges**. The headline question is whether selection concentrates on minority attack classes or wastes itself on Benign (§11.3). Report this as a table per iteration.
* **Per‑class deltas vs AGAF** on UNSW — name which classes move (Shellcode/Worms/Analysis are the low‑support ones to watch).
* **Rare‑class honesty (Constraint 8):** ToN supports are tiny and several fall **below 10 even pooled across all five test folds**: ransomware = 3, dos = 4, xss = 12, scanning = 13, backdoor = 16, password = 25. Per‑fold these are mostly < 5. **Report exact support and do not claim improvement on any class with < 10 test samples in a fold.** Show support columns next to every per‑class F1, and prefer macro‑F1 computed with `zero_division=0` so empty folds don't inflate.

---

## 9. Success criteria and fallback claims

* **Strong success:** Phase 2 pooled‑F1 > AGAF on **both** datasets. Given §0.2, the more honest and harder bar is *> the best unimodal MLP on both* (ToN 0.3979 CySecBERT; UNSW 0.6342 GNN‑emb). Hit that and Phase 2 is unambiguously the best model.
* **Moderate success:** best model on both datasets even without beating AGAF on ToN — i.e. UNSW > 0.6342 and ToN > 0.3979.
* **Partial success (most likely a priori):** wins on UNSW, not on ToN. The correct academic framing, *backed by the gate diagnostic*: **feedback helps when structure is informative and hurts when it is not.** The evidence chain is clean — ToN's best modality is semantic (CySecBERT 0.398 ≫ GNN‑emb 0.326), UNSW's is structural (GNN‑emb 0.634 ≫ CySecBERT 0.517); a loop that injects *semantic* correction into a *structural* model naturally helps where structure leads (UNSW) and adds little where semantics already dominate (ToN). The §4.4 skip guarantees you at least *match* CySecBERT on ToN rather than fall below it. The flat‑gate finding (§0.1) supports the framing: AGAF's inability to commit the gate is the static‑fusion symptom Phase 2's selectivity is designed to cure.
* **If Phase 2 never beats CySecBERT on ToN:** the publishable structural claims, in order of strength: **(b) lower cross‑fold variance / more stable predictions** is the strongest and is *directly motivated by data you already have* — AGAF's ToN cv‑std is 0.102 vs CySecBERT's 0.023; if the §4.4 skip + sparse correction cut that std materially, "comparable mean F1 at a fraction of the variance" is a real, defensible contribution. Next, **(a) improved minority‑class recall** — but only reportable for classes with ≥ 10 fold support, so likely injection/mitm/Benign on ToN, not ransomware. **(c) calibration** (lower ECE on the corrected predictions) is a clean secondary story. **(d) "the mechanism adds value despite flat aggregate metrics"** is the weakest and should not be the headline. Lead with (b).

---

## 10. Implementation roadmap (ordered by dependency)

1. **`datasets.py` — add `DatasetConfig` step4 fields.** Frozen dataclass, so add fields with defaults: `feedback_output_dir`, `prototypes_path`, `feedback_embedding_path`, and a `strong_modality: str` per dataset (`"llm"` for ton_iot, `"gnn"` for unsw_nb15, derived from §0.2). Populate both registry entries. *Imported unchanged elsewhere.*
2. **`src/models/feedback_classifier.py`** — the six modules in §2 + `FeedbackFusionClassifier` + `BiasedGATEdgeClassifier`/`BiasedGATStructuralEncoder`. `BiasedGATStructuralEncoder` subclasses or wraps `GATStructuralEncoder` and **must keep `forward(x, edge_index, edge_attr)` working when `edge_feedback is None`** so existing code and `train_gnn` are untouched.
3. **`src/pipeline/step4/build_prototypes.py`** — per fold, from `folds.pt` train masks only, write `[10×768]` prototypes (+ per‑fold for CV, + all‑edge for the final model). Pure NumPy/torch; no training.
4. **`src/pipeline/step4/train_feedback.py`** — the two‑stage loop (§5). Mirror `train_fusion.py` structure exactly (5‑fold CV → pooled OOF → final model → `write_benchmark_summary(..., model_name="feedback_loop")`). Reuse `FocalLoss`, `get_class_weights`, `classification_metrics`, `write_benchmark_summary`, `create_edge_splits` **unchanged**. Save `feedback_weights.pt [E×k]`, per‑iteration selection stats, `model.pt`, `scalers.pt`.
5. **`src/pipeline/phase5_validate.py`** — clone `phase4_validate.py`'s skeleton (`check`/`finalize_report`/`report_dir`/`write_json` from `reports.py`). Checks: artifacts exist; `feedback_weights`/embeddings align to edge count and are finite; splits align; predictions cover every edge; **`feedback_pooled_macro_f1 > agaf_pooled_macro_f1`** *and* `> best_unimodal_pooled_macro_f1`; paired‑fold bootstrap delta (reuse `_bootstrap_delta`) vs AGAF and vs the strong unimodal is positive; a non‑approval note if it fails to beat the strong baseline (same gating philosophy as phase4 line 130‑133).
6. **Build order for incremental testing:** (1) datasets → smoke test config loads → (3) prototypes → sanity‑check soft‑votes separate classes → run **§11.5 redundancy test** → (2) models with a forward‑pass unit test on 100 fake edges → (4) Stage‑1 only (Exp 8) to confirm it ≥ AGAF → enable Stage‑2 loop → (5) phase5_validate → full ablation grid (§7).
7. **Imported unchanged:** `splits.py` (all), `metrics.py` (all), `reports.py` (all), `gnn_classifier.py` (as a dependency of the biased subclass — do not edit it). **Modified/extended:** only `datasets.py` (additive fields). **New:** the model file, three step4 scripts, `phase5_validate.py`. The existing encoder file stays read‑only, which keeps Phase 1 reproducible.

---

## 11. Risks — each with a concrete, repo‑grounded fix

**11.1 "Gate collapse on ToN."** *Resolved by measurement.* The gate is at ~0.5, not 0 (§0.1); the GNN is **not** suppressed. The real defect is the entropy regulariser pinning the gate. Fix: Phase 2 has no global modality gate; use the sparse‑correction regulariser (§5) and the frozen strong‑modality skip (§4.4). Do **not** port `gate_entropy_lambda` over.

**11.2 Three‑variant bias interference.** Injecting one bias into all three encoders can blur the temporal/content/behavioural specialisation. Fix: append the `k` feedback channels with a **per‑variant learnable scale** `γ_variant` (init 1.0), and optionally a fixed routing that up‑weights the variant matching the suspected attack family (injection→content, scanning/ddos→behavioral, dos→temporal — consistent with `EDGE_FOCUS_WEIGHTS`). **Default: uniform (`γ` learned, no hard routing).** Hard routing is Exp 7; adopt only if it wins.

**11.3 Uncertainty selecting mostly Benign.** Benign is the majority and the GNN is genuinely unsure about many Benign edges (its feature entropy is highest there — ToN Benign `featH = 4.08` vs minority 0.9–2.3 in the AGAF diagnostics). A global entropy threshold would burn the LLM budget on Benign. Fix: **class‑stratified top‑p** in `UncertaintyDetector` (§2) — top‑`p` *within each predicted class* — so attack classes always get represented. Detect the failure by logging the predicted‑class histogram of selected edges every iteration (§8); if Benign > ~60% of selections, lower its per‑class `p`.

**11.4 Overfitting `SemanticFeedbackScorer` on ToN.** ransomware = 3, dos = 4. A learned `768→…` head will memorise them. Fix: **use Option C (prototype distance, 0 parameters)** as the default (§3); if a learned head is used for ablation, add L2 + early stopping and **never** report per‑class gains on < 10‑support classes (§8, Constraint 8).

**11.5 Feedback redundant with the AGAF gate.** If the signal duplicates what AGAF already encodes, it adds nothing. Fix: run this **before building** (it needs only existing artifacts + torch in your real venv; I could not run it in the sandbox because torch would not install there):

```python
import torch, torch.nn.functional as F, numpy as np
from scipy.stats import pearsonr
cfg_dir = "data/ton_iot/processed"
llm  = torch.load(f"{cfg_dir}/step3_llm/edge_embeddings.pt")          # [E×768]
gate = torch.load(f"{cfg_dir}/step3_fusion/gate_weights.pt")          # [E×128]
labels = torch.load(f"{cfg_dir}/step1/pyg_data.pt", weights_only=False).edge_label
folds  = torch.load(f"{cfg_dir}/splits/folds.pt", weights_only=False)
tr = folds[0]["train_mask"]; te = folds[0]["test_mask"]
proto = torch.stack([F.normalize(llm[tr & (labels==c)],dim=1).mean(0) for c in range(10)])  # [10×768]
sv = F.softmax(-(1 - F.normalize(llm[te],dim=1) @ F.normalize(proto,dim=1).T)*10, dim=1)     # [U×10]
signal = sv.max(1).values.numpy()           # prototype confidence per test edge
gate_scalar = gate[te].mean(1).numpy()       # AGAF per-edge GNN weight
print("pearson(prototype-confidence, AGAF gate) =", pearsonr(signal, gate_scalar)[0])
# |r| < ~0.3  -> orthogonal, signal carries new info (expected, given the flat gate)
# |r| > ~0.6  -> redundant, redesign the signal
```

**11.6 Iterative instability / oscillation.** Predictions may flip‑flop without converging. Fixes, layered: (i) the dual stopping criterion in §6 (churn AND entropy delta); (ii) **damping** — `edge_feedback ← β·new + (1−β)·old`, `β = 0.5`; (iii) **monotonic guard** — keep the iterate with the best *training* macro‑F1 seen so far (same "best‑state checkpoint" pattern as `train_fusion.py:215‑230`) and roll back if an iteration regresses; (iv) hard cap `MAX_ITERATIONS = 3`.

**11.7 `edge_attr` normalisation incompatibility.** `data.edge_attr` cols 0–2 are Z‑scored, cols 3–4 are raw categorical (`graph_construction.py:183‑185` stores `edge_attr_mean/std` only for the first three). Fix: **append, never add** (§4.1). The feedback channel is its own bounded scale (projected softmax), the original five columns are untouched, and `EDGE_FOCUS_WEIGHTS` continues to multiply only those five. No re‑normalisation needed, and `data.edge_attr_mean/std` stay valid.

---

## Appendix A — verified facts (read from the repo, 2026‑06‑15)

| Fact | Value | Source |
|---|---|---|
| GNN hidden dim / heads | 64 / 8 → emb `[E×64]` | `train_gnn.py:38,73-78` |
| Flow repr | `[src‖dst‖src*dst‖|src−dst|‖edge_attr]` = `4·64+5 = 261` | `gnn_classifier.py:148-166` |
| `encode_edges` output | `[E×64]` | `gnn_classifier.py:180-186` |
| LLM backbone | `markusbayer/CySecBERT`, frozen, `[E×768]` | `encode_kg.py:13,47,87` |
| `edge_attr` | 5 cols; 0–2 Z‑scored, 3–4 raw categorical | `graph_construction.py:31-36,183-185` |
| AGAF gate (ToN, pooled OOF) | global GNN 0.494; class range 0.435–0.516 | `step3_fusion/metrics.json` |
| AGAF gate (UNSW, pooled OOF) | global GNN 0.465; class range 0.410–0.526 | `step3_fusion/metrics.json` |
| Anti‑collapse regulariser | `loss = cls_loss − 0.01·gate_entropy` (maximised at g=0.5) | `train_fusion.py:45,123-127` |
| ToN supports (pooled) | ransomware 3, dos 4, xss 12, scanning 13, backdoor 16, password 25, ddos 35, injection 116, mitm 157, Benign 1746 | `step3_fusion/benchmark_summary.json` |

## Appendix B — current pooled out‑of‑fold macro‑F1 (the convention to standardise on)

| Model | ToN‑IoT | UNSW‑NB15 |
|---|---|---|
| GNN end‑to‑end | 0.2627 | 0.5519 |
| GNN‑emb + MLP | 0.3261 | **0.6342** |
| CySecBERT + MLP | **0.3979** | 0.5165 |
| AGAF fusion | 0.3211 | 0.6225 |

Best per dataset in **bold**. AGAF is the best model on *neither*.
