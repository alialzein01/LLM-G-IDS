# GNN↔LLM Feedback Loop: Mechanism Summary and Implementation Spec for Phase 4.4 / 4.5

Audience: Claude Code, implementing `src/models/feedback_classifier.py` Phase 4.4
(`BiasedGATv2Layer`, `SemanticAttentionBias`) and Phase 4.5 (`FeedbackLoopClassifier`).

Phases 4.1–4.3 already exist and are assumed complete:
- `UncertaintySelector` — entropy-based top-k% edge flagging, calibrated threshold buffer.
- `LiveCySecBERTScorer` — cached CySecBERT re-encoding of flagged edges' NL text.
- `WhitenedPrototypeScorer` — whitened cosine similarity → `[k, C]` semantic logits per
  flagged edge.

This doc summarizes the four papers that motivated this design, extracts the reusable
mechanism from each, and specifies exactly what Phase 4.4 / 4.5 need to implement so the
loop composes cleanly with what's already built.

---

## 1. The four mechanisms, summarized

### LOGIN (Qiao et al., arXiv 2405.13902) — closest analogue to this codebase

- **Uncertainty signal**: 5 forward passes with dropout p=0.3 (MC-dropout), variance of
  predictions per node → top-k uncertain nodes selected. (This codebase uses entropy of
  a single softmax instead — cheaper, same role.)
- **What's sent out**: a crafted prompt per uncertain node containing its text attributes
  and one-hop neighborhood topology description.
- **What comes back**: an LLM-predicted label + free-text explanation.
- **Incorporation rule (the important part)**: *conditional on correctness*.
  - If the LLM's predicted label matches the ground-truth label for that node → append
    the explanation text to the node's original text attribute, then re-embed (i.e.
    reinforce/enrich the semantic signal).
  - If the LLM's prediction is wrong → prune edges in the node's one-hop neighborhood
    (the assumption being wrong predictions correlate with noisy/heterophilous local
    structure, so cut the structural signal rather than trust it).
- **Loop shape**: single pass per training round, repeated across training epochs — not
  an inference-time iterative refinement. GNN is retrained after each correction batch.
- **Non-differentiability**: LLM step is entirely outside the autograd graph — it only
  edits the input (text or edge_index) between epochs.

### DAS / "Semantic Refinement with LLMs for Graph Representations" (arXiv 2512.21106)

- **Framing**: explicitly a "closed feedback loop" between a *frozen* GNN and an LLM.
- **Direction of supervision is inverted relative to LOGIN**: the GNN's own predictions
  act as an implicit supervisory signal that steers *how the LLM refines node semantics*
  (e.g. which semantic aspects to emphasize per node), and the refined semantic
  embeddings are fed back to re-run the same GNN.
- **Key idea to borrow**: treats the split between "structure-dominated" and
  "semantics-dominated" data as *learned per-instance*, not fixed. Relevant to this
  project's DDoS-vs-SQLi framing in the CLAUDE.md skill notes — the fusion should not
  assume a fixed weighting.
- **Loop shape**: iterative, alternating GNN-forward → LLM-refine → GNN-forward, until
  representations stabilize (no fixed cycle count given in the abstract; treated as a
  data-centric adaptation loop, not a fixed-depth unrolled computation graph).

### GLANCE (arXiv 2510.10849) — answers "when should you even bother calling the LLM"

- **Uncertainty/gating signal**: a lightweight *router* network, not raw entropy —
  learns per-node whether querying the LLM is worth it, from inexpensive per-node
  features.
- **Training the router**: advantage-based objective — compares the utility of
  "query LLM and use its answer" vs. "just trust the GNN," and only calls the LLM when
  the router predicts positive advantage. This is necessary because the LLM call is
  non-differentiable, so the router can't be trained by backprop through the LLM step;
  it's trained like a bandit/RL policy on the *outcome* of querying vs. not.
- **Result pattern**: biggest gains concentrated on heterophilous / structurally
  ambiguous nodes — i.e. exactly the nodes flagged by an uncertainty-based selector.
  This validates that `UncertaintySelector`'s entropy-based flagging is a reasonable
  (cheaper, non-learned) stand-in for GLANCE's router.

### RoGRAD / "Are LLMs Better GNN Helpers?" (arXiv 2510.01910)

- **Core claim**: static, one-shot LLM signal injection underperforms *iterative*
  refinement — i.e. running the GNN→LLM→GNN cycle more than once compounds benefit,
  particularly under graph deficiencies (noise, sparsity, label scarcity — all of which
  apply to rare attack classes like mitm/ransomware in this project's data).
- **Takeaway for design**: Phase 4.5's `FeedbackLoopClassifier` should support **more
  than one cycle** (a `num_cycles` hyperparameter), not hard-code a single GNN→LLM→GNN
  pass, and should expose a way to check whether predictions have stabilized between
  cycles (a convergence criterion, even a simple one like "the flagged-edge set didn't
  change" or "max logit shift < ε").

---

## 2. Common structure across all four (what to actually implement)

Every mechanism factors into the same four stages, which map directly onto this
project's phase split:

| Stage | Question answered | This codebase |
|---|---|---|
| 1. Flag | Which edges is the GNN unsure about? | `UncertaintySelector` (done) |
| 2. Consult | What does the semantic/LLM branch say about those edges? | `LiveCySecBERTScorer` + `WhitenedPrototypeScorer` (done) |
| 3. Incorporate | How does that semantic signal change the GNN's next pass? | **Phase 4.4 — not yet built** |
| 4. Loop | Repeat until stable, then classify | **Phase 4.5 — not yet built** |

None of the four papers do stage 3 the same way this project has committed to (additive
attention bias inside GATv2 — closest to how DAS "feeds refined semantics back to the
graph learner," but architecturally more explicit). That's fine — it's a legitimate,
more mechanistic version of the same idea, and it's cheaper than LOGIN's
"edit input, retrain" cycle since it happens within a single forward pass at
inference time as well as training time.

---

## 3. Phase 4.4 spec — `SemanticAttentionBias` and `BiasedGATv2Layer`

**`SemanticAttentionBias`**
- Input: `[k, C]` semantic logits from `WhitenedPrototypeScorer` for the `k` flagged
  edges (`C` = 10 = `DEFAULT_NUM_CLASSES`).
- Output: a per-edge, per-head scalar bias `[k, heads]` (or `[k, 1]` broadcast across
  heads — start with the simpler shared-across-heads version first, matching
  `GATStructuralEncoder`'s `heads=8` first layer).
- Mechanism: a small linear layer `Linear(C, heads)` (or `Linear(C, 1)`) projecting
  semantic logits into bias space. Initialize near-zero (e.g. `nn.init.zeros_` on the
  final layer) so that at the start of training the bias is a no-op and the GNN
  degrades gracefully to its unbiased behavior — this mirrors LOGIN's
  "correct→reinforce, wrong→prune" idea in spirit: a badly-calibrated semantic
  signal should not be able to swamp the structural signal early in training.
- For unflagged edges (the `E - k` edges not selected by `UncertaintySelector`), bias
  is `0` — semantic branch never touches edges the GNN wasn't unsure about.

**`BiasedGATv2Layer`**
- Same shape as `GATStructuralEncoder`'s `GATv2Conv` layers, but the attention
  logit computation gets an additive bias term before the softmax:
  `e(i,j) = a^T · LeakyReLU(W·[h_i ‖ h_j]) + bias(i,j)`
  where `bias(i,j)` is `0` for non-flagged edges and the `SemanticAttentionBias`
  output for flagged ones.
- `torch_geometric`'s `GATv2Conv` doesn't expose a raw per-edge additive bias hook
  directly, so the practical implementation path is one of:
  1. Subclass `GATv2Conv` and override `message()`/`edge_updater()` to add the bias
     before the internal softmax — most faithful, more surgery.
  2. Simpler and recommended for a first working version: don't touch attention logits
     inside GATv2Conv at all. Instead, treat the semantic bias as an **edge_attr
     perturbation** — concatenate/add the projected semantic bias into `edge_attr`
     for flagged edges only, before calling the existing `GATv2Conv(..., edge_dim=...)`.
     Since `edge_dim` already feeds into GATv2's attention computation (confirmed in
     `GATStructuralEncoder`, `edge_attr_dim=5`), this achieves the same causal effect
     (semantic signal biases attention) without subclassing PyG internals, and is
     far less likely to break silently.
- Recommend implementing option 2 first, validate on a dashboard/unit test that
  semantic bias measurably shifts attention weights for flagged edges, and only
  move to option 1 if the effect proves too weak (option 2 dilutes the bias signal
  through a fixed 5-dim edge_attr rather than a dedicated channel).

---

## 4. Phase 4.5 spec — `FeedbackLoopClassifier`

Outer loop, composing everything above:

```
forward(x, edge_index, edge_attr, node_texts, fold_index, num_cycles=2):
    logits, edge_emb = GATEdgeClassifier(x, edge_index, edge_attr)   # cycle 0, unbiased
    for cycle in range(num_cycles):
        probs = softmax(logits)
        flagged_mask = UncertaintySelector(probs)          # [E] bool
        flagged_ids  = nonzero(flagged_mask)
        sentences    = [node_texts[i] for i in flagged_ids]  # flow NL serialization
        cysec_emb    = LiveCySecBERTScorer.encode(flagged_ids, sentences, fold_index)
        sem_logits   = WhitenedPrototypeScorer(cysec_emb)     # [k, C]
        bias         = SemanticAttentionBias(sem_logits)      # [k, heads] or [k,1]
        biased_edge_attr = edge_attr.clone()
        biased_edge_attr[flagged_ids] = inject(bias, edge_attr[flagged_ids])
        logits, edge_emb = BiasedGATv2Layer(x, edge_index, biased_edge_attr)
        if converged(prev_flagged_mask, flagged_mask):  # RoGRAD-style early stop
            break
        prev_flagged_mask = flagged_mask
    return logits, edge_emb, flagged_mask   # flagged_mask returned for explainability/diagnostics
```

Design decisions to make explicit in code (flag these as TODOs/config, don't silently
pick one):

1. **`num_cycles` default.** RoGRAD's finding argues for >1. Start with 2 (one
   correction cycle) as default, expose as a constructor arg, and log/plot
   val-macro-F1 vs. num_cycles as an ablation (this project already has the
   infrastructure for per-fold metrics in `training_history.json`).
2. **Convergence check.** Cheapest option: stop when `flagged_mask` is identical to
   the previous cycle's (no new edges need semantic help). Slightly richer option:
   stop when max absolute logit change on flagged edges < ε. Implement the cheap one
   first.
3. **Gradient flow through the loop.** `LiveCySecBERTScorer`'s encoder is already
   frozen (`requires_grad_(False)` — confirmed in the existing code), so CySecBERT
   itself never needs backprop. But decide whether earlier cycles' `BiasedGATv2Layer`
   calls stay in the autograd graph (full backprop through all cycles — expensive,
   more "correct") or whether only the final cycle backprops and earlier cycles run
   under `torch.no_grad()` (cheaper, closer to how LOGIN/GLANCE treat the LLM step as
   effectively outside the training loop). Recommend starting with **only the last
   cycle differentiable**, matching how every one of the four papers avoids
   backpropagating through the actual LLM call.
4. **Where `flagged_mask` recalculation happens relative to calibration.**
   `UncertaintySelector.calibrate()` sets a threshold once per fold on train-fold
   predictions (existing code). Decide whether that threshold is recalibrated between
   cycles within a forward pass, or held fixed for all cycles in that fold. Recommend
   holding fixed within a fold's training — recalibrating mid-loop risks the flagged
   set oscillating and never satisfying the convergence check in point 2.
5. **Caching interacts with cycles.** `LiveCySecBERTScorer`'s cache key is
   `(fold_index, edge_row_id)` — since NL text per edge doesn't change between cycles
   (only which edges get flagged changes), the cache already gives cycle 2+ near-free
   re-encoding for edges flagged again. No changes needed here, just note it as a
   reason cycling is cheap after the first pass.

---

## 5. Acceptance test shape (matching this project's existing per-phase dashboard convention)

Per the file's own docstring convention ("do not compose downstream phases until the
current phase's dashboard has passed its acceptance test"), Phase 4.4's test should
verify in isolation: (a) bias is exactly zero for unflagged edges, (b) bias is non-zero
and gradient-connected to `WhitenedPrototypeScorer.log_temperature` for flagged edges,
(c) `BiasedGATv2Layer` output differs from `GATStructuralEncoder` output only on
flagged edges' neighborhoods (spot-check a toy 4-node graph). Phase 4.5's test should
verify: (a) `num_cycles=1` reduces to the existing unbiased `GATEdgeClassifier` forward
pass exactly (regression guard), (b) the convergence check actually halts before
`num_cycles` on a synthetic graph with well-separated classes (should stabilize fast),
(c) val-macro-F1 with the loop enabled is ≥ the frozen GATEdgeClassifier baseline on at
least one fold (sanity floor, not a strong claim).

---

## Source papers

- LOGIN: https://arxiv.org/abs/2405.13902
- DAS / Semantic Refinement with LLMs for Graph Representations: https://arxiv.org/abs/2512.21106
- GLANCE: https://arxiv.org/abs/2510.10849
- RoGRAD ("Are LLMs Better GNN Helpers?"): https://arxiv.org/html/2510.01910v1
