# Plan — Audit and Re-implement Step 4 (Phase 2 Feedback Loop) for UNSW-NB15

## Context

Step 4 of the LLM-G-IDS framework is the project's central research contribution: a bidirectional, uncertainty-gated feedback loop in which a frozen LLM signal is supposed to re-enter the GNN's message passing to correct uncertain edge classifications. The framework currently has Step 4 implemented end to end and verified, but two problems make the current state unfit to ship as a research contribution:

1. **The implementation diverges from its own spec in `todo.md` Step 4** at two concrete points:
   - todo.md says "Serialize uncertain flows to natural language and **route them to the LLM**." The code uses a precomputed lookup against frozen CySecBERT class prototypes instead of a runtime LLM call.
   - todo.md says "This score is injected back into the GAT as **an attention bias** on the relevant edges." The code concatenates a learned 8-dim feedback channel to `edge_attr` and lets `GATv2Conv`'s message MLP consume it — functionally related but not an additive bias on the attention coefficient softmax.

2. **The current scores do not justify the architecture.** With the end-to-end GATv2 as strong baseline (the configuration the user requested), pooled out-of-fold macro-F1 on UNSW is **0.5453**, which is **below GNN-only (0.5519)** and well **below AGAF (0.6225)**. The verifier's six failed checks include "OOF real prototype feedback beats random feedback" — `random_feedback` scores 0.5571, +0.0118 above prototype feedback. The semantic content of the feedback is empirically not the lever moving the result.

The goal of this plan is to fix both problems in the right order: rewrite the architecture so it can clear AGAF (≥ 0.625 pooled F1) first, then bring the implementation into literal alignment with todo.md once the F1 target is reached. Scope is **UNSW-NB15 only**; ToN-IoT keeps its existing `strong_modality="llm"` path untouched.

## Diagnosis (the three coupled root causes)

A Plan-agent audit identified three coupled defects in [feedback_classifier.py](src/models/feedback_classifier.py) that explain why the current loop fails:

1. **The "bias" is not a bias.** [`AttentionBiasGenerator.forward`](src/models/feedback_classifier.py#L271-L278) and [`BiasedGATStructuralEncoder._augment_edge_attr`](src/models/feedback_classifier.py#L377-L384) concatenate to `edge_attr`. PyG's `GATv2Conv` consumes `edge_attr` through its message MLP — the 8 channels compete as features against the original 5, not as a steering term on attention.
2. **The residual head leaks the GNN path.** [`ResidualCorrectionHead.forward`](src/models/feedback_classifier.py#L580-L590) takes `gnn_logits` directly in its input concatenation. The correction head can correct via the GNN path even when feedback is identically zero — which is why `head_only` already matches `feedback_loop` in the ablation.
3. **Prototype scoring is rank-degenerate.** [`SemanticFeedbackScorer.build_prototypes`](src/models/feedback_classifier.py#L154-L170) L2-normalises → means → re-normalises, collapsing per-class spread. With `temperature=10.0` ([line 27](src/models/feedback_classifier.py#L27)) the cosines saturate. This is why a uniform random softmax (`random_feedback` arm) carries more class-discriminating energy than the prototype vote.

## User decisions (locked)

- **Priority**: F1 first, faithfulness second.
- **Bias rewrite**: subclass `GATv2Conv` for a true additive attention bias.
- **Live LLM**: cached CySecBERT re-encoding (functionally identical to precomputed; satisfies the spec call graph).
- **Scope**: UNSW only; ToN-IoT untouched.

## Staged implementation

Each stage is independently runnable end to end: edit code, re-run `train_feedback` for UNSW, re-run `verify_feedback`, read pooled F1 from `benchmark_summary.json` and arm deltas from `ablation_summary.json`. Stop when pooled F1 ≥ 0.625 (beating AGAF); then ship Stage 4 for faithfulness.

### Stage 1 — True additive attention bias (biggest F1 lever)

**Target**: pooled F1 0.5453 → ~0.58; `random_feedback` falls below `prototype_feedback`.

**Files**: [`src/models/feedback_classifier.py`](src/models/feedback_classifier.py).

**Changes**:
- Add new module `BiasedGATv2Layer` that re-implements GATv2's message step:
  - Standard `(x_i, x_j) → attn_logit` MLP from `GATv2Conv`.
  - Accept a new kwarg `edge_attn_bias: [E, heads]` and add it to the unnormalised attention logits *before* the edge-level softmax (the `softmax(..., index=edge_index[1])` step inside PyG's GAT internals).
  - `edge_attr` itself stays at the original 5 dims (no concat).
- Add new module `SemanticAttentionBias` to replace `AttentionBiasGenerator`. Input: `feedback_subset [k, C]`, `uncertain_mask [E]`. Output: `bias [E, heads]`. Implementation: small Linear(C → heads), gated by `uncertain_mask` (exact zero on confident edges), scaled by a learnable `log_bias_strength` initialised so the effective bias is ~+1.5 nats on the predicted class.
- Replace the two `GATv2Conv` layers inside `BiasedGATStructuralEncoder` ([feedback_classifier.py:356-374](src/models/feedback_classifier.py#L356-L374)) with `BiasedGATv2Layer`. Thread `edge_attn_bias` (not `edge_feedback`) through `encode_edges_by_variant`, `forward_with_aux`, and `forward` ([feedback_classifier.py:478-547](src/models/feedback_classifier.py#L478-L547)).
- Keep `FeedbackNodeMessageAdapter` removed in this stage — feedback enters only via the attention coefficient.
- Add a unit-equivalence assertion in tests: with `edge_attn_bias = 0`, the new layer must match a stock `GATv2Conv` to 1e-5 on a synthetic graph (write a small standalone script under `tests/`).

**Verification**:
- Re-run `train_feedback` + `verify_feedback` on UNSW.
- Expected: `head_only` now diverges from `feedback_loop`; `random_feedback < feedback_loop`; α rises from ~0.10 → ~0.25; pooled F1 ≥ 0.58.

### Stage 2 — Decouple the residual head from the GNN path

**Target**: pooled F1 +0.000 to +0.005, but the `head_only` vs `feedback_loop` gap opens to > 0.01 — the ablation becomes interpretable.

**Files**: [`src/models/feedback_classifier.py`](src/models/feedback_classifier.py).

**Changes**:
- Replace `ResidualCorrectionHead` ([feedback_classifier.py:556-590](src/models/feedback_classifier.py#L556-L590)) with `SemanticCorrectionHead`: input is `cat([edge_emb, edge_feedback_summary])` only — drop `gnn_logits` from the concat.
- `edge_emb` must come from the **biased** GAT aggregation (i.e. with feedback already in the attention), so the residual is genuinely the semantic-only contribution.
- In `FeedbackFusionClassifier.correction_from` ([feedback_classifier.py:805-817](src/models/feedback_classifier.py#L805-L817)) and `forward` ([feedback_classifier.py:996-1034](src/models/feedback_classifier.py#L996-L1034)), pass only `edge_emb` + `edge_feedback`.
- Keep the existing `correction_gate = (edge_feedback.abs().sum > 0)` masking ([feedback_classifier.py:944-947](src/models/feedback_classifier.py#L944-L947)).

**Verification**:
- `head_only` should drop *below* `zero_feedback` (no feedback path = nothing for the head to do).
- `feedback_loop − head_only` delta should exceed +0.01 with bootstrap CI strictly > 0.

### Stage 3 — Repair the prototype signal

**Target**: with Stages 1+2 already in, +0.01 to +0.02 pooled F1. Without Stages 1+2, gain is marginal — only do this after the architecture is fixed.

**Files**: [`src/pipeline/step4/build_prototypes.py`](src/pipeline/step4/build_prototypes.py), [`src/models/feedback_classifier.py`](src/models/feedback_classifier.py).

**Changes**:
- In `build_prototypes.py`: fit a per-fold whitener (ZCA on train-fold CySecBERT embeddings; LDA fallback if rank-deficient). Save the projection matrix alongside `prototypes.pt`.
- In `SemanticFeedbackScorer.forward` ([feedback_classifier.py:218-234](src/models/feedback_classifier.py#L218-L234)): apply the whitener before cosine.
- Replace the scalar `self.temperature` with a learnable `self.log_temperature: nn.Parameter([num_classes])`, initialised to `log(10.0)`. Apply per-class before `_aggregate_to_classes` ([feedback_classifier.py:198-216](src/models/feedback_classifier.py#L198-L216)).

**Verification**:
- New diagnostic in training history: per-class mean cosine separation should exceed 0.3 after whitening (currently ~0.1, per `prototype_diagnostics.json`).

### Stage 4 — Live CySecBERT re-encoding with cache (spec faithfulness)

**Target**: zero F1 change; this stage exists to satisfy todo.md's literal "route to LLM" requirement once the F1 target is met.

**Files**: [`src/models/feedback_classifier.py`](src/models/feedback_classifier.py), [`src/pipeline/step4/train_feedback.py`](src/pipeline/step4/train_feedback.py).

**Changes**:
- New module `LiveCySecBERTScorer`: holds `AutoTokenizer` + `AutoModel.from_pretrained("markusbayer/CySecBERT")` (use the constants from [`src/pipeline/step3/encode_kg.py`](src/pipeline/step3/encode_kg.py) — the model name, mean-pooling, eval mode). Adds a deterministic LRU cache keyed by `edge_row_id`.
- In `train_feedback.py`: load `data/unsw_nb15/processed/step2/kg_triples_nl.txt` once per fold (row-aligned to edges, verified by Explore agent). When the loop selects uncertain edges, slice the NL list by the selected indices, run `LiveCySecBERTScorer`, then pass through the whitened `SemanticFeedbackScorer`.
- Cache strategy: key by `(fold_index, edge_row_id)`. Text is invariant so cache hit rate hits 100% after every edge has been selected once. Net runtime cost ≈ one pass over the full edge set per fold (≈ same as the precomputed encoding).
- Sanity assertion: on a random sample, the live re-encoded embedding must match the precomputed `step3_llm/edge_embeddings.pt` to 1e-4.

**Verification**:
- Pooled F1 should be within ±0.001 of Stage 3's result (within seed noise).
- Cache hit rate > 99% by epoch 2 (log this in training history).
- A new verifier check "Semantic feedback uses live LLM re-encoding" PASSes by source-text match on `AutoModel.from_pretrained` inside `LiveCySecBERTScorer`.

### Stage 5 — Hyperparameter sweep on a fixed architecture

**Target**: +0.005 to +0.015 pooled F1. Only run if pooled F1 after Stages 1–3 is still < 0.625.

**Files**: [`src/pipeline/step4/train_feedback.py`](src/pipeline/step4/train_feedback.py).

**Changes**:
- Expose `feedback_strength`, `max_iterations`, `alpha_warmup_epochs`, `alpha_warmup_value` as CLI flags (they already exist as constants).
- Add an α-warmup schedule: linear ramp from 0.0 → learnable target over the first 5 epochs (currently α is held at the warmup value then released).
- Sweep grid: `feedback_strength ∈ {1.0, 2.0, 4.0}`, `max_iterations ∈ {2, 3, 4}`. Selection criterion: 5-fold cross-validated mean val macro-F1, not single-fold best.

**Verification**:
- Churn fraction should drop monotonically across loop iterations (currently sometimes oscillates).
- α trajectory should converge by epoch 20.

## Stop condition

After each stage, read the headline pooled F1 from `data/unsw_nb15/processed/step4_feedback/benchmark_summary.json` and the random-feedback delta from `ablation_summary.json`. If pooled F1 ≥ 0.625 **and** `feedback_loop` > `random_feedback` with bootstrap CI > 0, stop building and ship Stage 4 for spec compliance. If still short after Stage 5, the architecture is genuinely unable to clear AGAF on UNSW with the current LLM/data; that is a publishable result on its own and should be reported as a Phase 2 limitation.

## Critical files

- [`src/models/feedback_classifier.py`](src/models/feedback_classifier.py) — all of Stages 1, 2, 3, 4 (new modules and routing).
- [`src/pipeline/step4/build_prototypes.py`](src/pipeline/step4/build_prototypes.py) — Stage 3 (per-fold whitener save).
- [`src/pipeline/step4/train_feedback.py`](src/pipeline/step4/train_feedback.py) — Stage 4 (NL loader + live scorer plumbing), Stage 5 (sweep flags).
- [`src/pipeline/step4/verify_feedback.py`](src/pipeline/step4/verify_feedback.py) — add Stage 4 source-check (Live LLM), Stage 1 source-check (additive attention bias).
- [`src/pipeline/step3/encode_kg.py`](src/pipeline/step3/encode_kg.py) — referenced (not modified) for the CySecBERT model name, mean-pooling, eval mode used by `LiveCySecBERTScorer`.
- [`src/pipeline/step2/knowledge_graph.py`](src/pipeline/step2/knowledge_graph.py) — referenced for the NL sentence template (rows of `kg_triples_nl.txt`).
- [`src/models/gnn_classifier.py`](src/models/gnn_classifier.py) — not modified; consulted for `GATEdgeClassifier` and the FedGATSage three-variant pattern.

## Reusable utilities to preserve

- `GATEdgeClassifier` ([gnn_classifier.py:82](src/models/gnn_classifier.py#L82)) — the strong-baseline model. The new `BiasedGATv2Layer` should plug into the same three-variant `nn.ModuleDict` pattern this class uses.
- `FocalLoss` and `get_class_weights` ([common/splits.py](src/pipeline/common/splits.py)) — keep as the criterion across all stages.
- `_train_strong_router` ([train_feedback.py](src/pipeline/step4/train_feedback.py)) — already correct; leave untouched.
- `create_edge_splits` and the per-fold splits at `data/unsw_nb15/processed/splits/folds.pt` — reused without changes.

## End-to-end verification

For each completed stage, run from the repo root:

```bash
source .venv/bin/activate
python -m src.pipeline.step4.build_prototypes --dataset unsw_nb15
python -m src.pipeline.step4.train_feedback --dataset unsw_nb15
python -m src.pipeline.step4.verify_feedback --dataset unsw_nb15
```

Then read three artifacts to score the stage:

1. `data/unsw_nb15/processed/step4_feedback/benchmark_summary.json` → `pooled_cv_macro_f1`. Compare to the previous stage's number.
2. `data/unsw_nb15/processed/step4_feedback/ablation_summary.json` → `arms.feedback_loop.overall_macro_f1` vs `arms.random_feedback.overall_macro_f1` and the four `primary_delta_vs_*` fields. The integrity test is `feedback_loop > random_feedback` with the bootstrap CI in `verify_feedback.py` strictly > 0.
3. `data/unsw_nb15/processed/reports/phase2_feedback_verification/verification_report.md` → confirm the architectural checks still PASS and the previously-failing outcome checks (`OOF B reentry-only beats A head-only`, `OOF real prototype feedback beats random feedback`, `Feedback benchmark beats Step 3 end-to-end GNN path`, `Feedback benchmark beats Step 3 AGAF fusion`) flip to PASS.

When all of the above are green and `pooled_cv_macro_f1 ≥ 0.625`, run Stage 4 for spec faithfulness, re-run verification to confirm a new PASS for "Semantic feedback uses live LLM re-encoding", commit the change, and update `PHASE2_FEEDBACK_LOOP_PLAN.md` Section 0.2 with the new headline number.
