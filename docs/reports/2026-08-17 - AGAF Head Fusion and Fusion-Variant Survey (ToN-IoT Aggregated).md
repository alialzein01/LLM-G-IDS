# 2026-08-17 — AGAF Head Fusion and Fusion-Variant Survey

**Scope:** NF-ToN-IoT, aggregated graph (2127 edges), 8 eval classes.
**Outcome:** AGAF improved from 0.4447 to 0.5141 macro-F1. Ladder still does not hold, for a new reason.

---

## 1. What we set out to do

Improve Step 3 (AGAF fusion) on the ToN-IoT aggregated graph, using mechanisms taken
from a literature survey rather than guesswork.

Two source documents drove the work:

- `docs/Step 3 Fusion Literature Survey.md`
- `docs/Full 25-Work Comparison Table (Parallel GNN+LLM Fusion).md`

Survey conclusion: parallel GNN+LM fusion with per-sample weights exists (Moorthy et al.,
*Scientific Reports* 2025, fake-news domain), but no intrusion-detection work performs
parallel embedding-level GNN+LLM fusion. Our mechanism is not novel; our application is.

---

## 2. Fusion-variant survey (negative result)

Five mechanisms were implemented, each from a published paper, and run on identical
folds and embeddings. Only the fusion step changed.

| Variant | Paper | Mechanism | pooled macro-F1 (10-class, seed 42) |
|---|---|---|---|
| selfattn | RAGFormer | two modality tokens + self-attention | 0.4387 |
| concat | GMLM | concatenate, one linear layer, no gate | 0.3932 |
| feature_gate | GMU | **existing AGAF** — per-feature sigmoid gate | 0.3911 |
| align λ=0.1 | DCVD | + InfoNCE cross-modal alignment, weak | 0.3470 |
| scalar | Moorthy | one weight per branch per edge | 0.3310 |
| fixed λ=0.5 | BertGCN | constant 50/50 blend | 0.3181 |
| align λ=0.5 | DCVD | + InfoNCE alignment, strong | 0.2981 |

### Three findings

**(a) The gate earns nothing.** concat 0.3932 vs feature_gate 0.3911 — a gap of 0.002.
Removing the gating mechanism entirely does not change the score.

**(b) The DCVD alignment loss hurt.** Both settings scored below baseline. The
hypothesis that the two branches needed geometric alignment did not transfer to this
data. Recorded as a failed prediction.

**(c) The differences are seed noise.** Repeating the top two variants across three
seeds:

| Seed | selfattn | feature_gate |
|---|---|---|
| 42 | 0.4387 | 0.3911 |
| 7 | 0.3247 | 0.3645 |
| 123 | 0.3764 | 0.3350 |

selfattn swings 0.114 across seeds; its mean advantage is 0.016. The noise is seven
times the effect.

**Cause:** rare eval classes hold 12–35 edges (ddos 35, password 25, backdoor 16,
scanning 13, xss 12). A handful of predictions moves macro-F1 by whole points.

**Conclusion:** fusion-mechanism ranking is not resolvable on the aggregated graph.
No variant was adopted. Artifacts in `data/ton_iot/processed/step3_fusion_variants/`.

---

## 3. The change that worked: head fusion

The variant survey changed *how* the two branches are mixed. The actual win came from
changing *what the semantic branch contributes*.

### Before

```
GNN embedding (64-d)   ──┐
                         ├── gate ── classify
LLM embedding (768-d)  ──┘
```

The semantic input was raw text meaning. AGAF had to learn to interpret it from scratch,
and had to beat a strong semantic model while doing so. It failed.

### After (`--use-head-logits`)

```
GNN embedding (64-d)      ──┐
                            ├── gate ── classify ──┐
LLM head logits (10-d)    ──┘                      ├── output gate ── final
                                                   │
LLM head logits (10-d)  ───────────────────────────┘
```

Two changes:

1. **Semantic input is now a verdict, not a description.** AGAF receives the trained
   head's 10 class scores instead of 768 embedding dimensions.
2. **The head's verdict is re-injected at the output** through a second learned
   per-edge gate, initialised at ~0.95 trust in the head.

The head becomes AGAF's floor. The GNN branch only has to add lift.

**Result: AGAF 0.4447 → 0.5141.**

---

## 4. Current authoritative ladder

`results/ton_iot_current.json` — trained head, head-fused AGAF.

| Rung | macro-F1 (8) | accuracy |
|---|---|---|
| GNN | 0.3398 | 0.8016 |
| LLM (trained head) | 0.5165 | 0.9149 |
| AGAF (head-fused) | 0.5141 | 0.9140 |
| Feedback loop | 0.4538 | 0.9050 |

**Ladder order holds: false.**

| Comparison | Δ | CI 95% | Reading |
|---|---|---|---|
| AGAF − GNN | +0.172 | [+0.110, +0.235] | real gain |
| AGAF − LLM | −0.002 | [−0.011, +0.002] | **tied** |
| Loop − AGAF | −0.060 | [−0.100, −0.019] | **loop is really worse** |

### Per-class F1

| Class | support | GNN | LLM | AGAF | Loop |
|---|---|---|---|---|---|
| Benign | 1746 | 0.922 | 0.983 | 0.983 | 0.979 |
| mitm | 157 | 0.762 | 0.840 | 0.840 | 0.836 |
| injection | 116 | 0.500 | 0.607 | 0.609 | 0.514 |
| ddos | 35 | 0.131 | 0.323 | 0.328 | 0.265 |
| password | 25 | 0.094 | 0.242 | 0.215 | 0.200 |
| backdoor | 16 | 0.122 | 0.474 | 0.500 | 0.324 |
| scanning | 13 | 0.118 | 0.312 | 0.312 | 0.333 |
| xss | 12 | 0.060 | 0.300 | 0.300 | 0.148 |
| dos* | 4 | 0.000 | 0.000 | 0.000 | 0.000 |
| ransomware* | 3 | 0.000 | 0.000 | 0.444 | 0.000 |

\* excluded from macro-F1

**Reading:** AGAF now matches the LLM almost class for class. It is no longer losing —
but it is not adding either. The GNN branch contributes nothing measurable on top of
the trained head.

---

## 5. Prototype-scorer comparison

`results/ton_iot_prototype_comparison.json` — kept for auditability, not authoritative.

| Rung | prototype scorer | trained head |
|---|---|---|
| GNN | 0.3398 | 0.3398 |
| LLM | 0.2785 | 0.5165 |
| AGAF | 0.4447 | 0.5141 |
| Loop | 0.4538 | 0.4538 |
| Ladder holds | **true** | false |

The prototype ladder holds only because its LLM rung is crippled. The same embeddings
with a trained head score 0.517 instead of 0.279. Claiming the ladder on the prototype
run while using the trained head elsewhere would be inconsistent, so the trained-head
run is authoritative.

**This is the single most important framing point for the supervisor.**

---

## 6. Corrections logged

Two errors found and fixed during this session:

1. **A stale contract file.** `results/ton_iot_current.json` was dated 6 August and
   reported AGAF (0.3233) below the GNN (0.3271). The prediction artifacts were from
   14 August and disagreed. Recomputing per-class F1 from the artifacts showed
   AGAF at 0.4447, above the GNN. The "fusion loses to its own input" conclusion was
   an artifact of a stale file, not a real result. Contract regenerated.

2. **`data/ton_iot/processed/step3_fusion_concat/` is an orphan.** No code in the tree
   produces it, it is not in git, and its `model_name` still reads `agaf_fusion`.
   Its per-fold-mean and pooled macro-F1 disagree about whether it beats AGAF.
   Do not cite it.

---

## 6b. Follow-up runs (same day)

Two items from the original "open" list were closed.

### The gate-entropy regulariser was not the cause

Re-running head-fused AGAF with `--gate-entropy-lambda 0` produced **bit-identical**
predictions (pooled macro-F1 `0.4532371043908496` in both runs) and an unchanged gate
spread (0.0155 → 0.0158).

The reason is visible in the trained weights:

```
head_gate bias:  init 3.0  ->  trained 2.9947
implied trust in the LLM head: 0.9523
```

The output gate never moved off its initialisation. **AGAF defers ~95% to the trained
LLM head**, so its own fused branch barely reaches the output — which is why a change to
the fused branch's regulariser cannot alter the predictions.

This reframes section 3. Head fusion did not make fusion work; it made AGAF stop hurting
by having it copy the stronger branch. The 0.4447 → 0.5141 gain is real as engineering,
but it is not evidence that structural and semantic evidence combine.

### The feedback loop, re-run against head-fused AGAF

The loop had never seen the head-fused AGAF benchmark. Re-running it:

| Run | top_k | Loop macro-F1 | Loop − AGAF | CI 95% |
|---|---|---|---|---|
| prototype path (original) | 15.0 | 0.4538 | −0.060 | [−0.100, −0.019] |
| head, fallback top_k | **16.0** | 0.4784 | −0.035 | [−0.067, −0.004] |
| head, ToN-selected top_k | **15.0** | **0.4986** | −0.015 | [−0.046, +0.016] |

**A configuration trap worth recording:** with no `selected_feedback_config.json` in the
output directory, `train_feedback` silently falls back to `top_k = 16.0` — UNSW's selected
value, not ToN's 15.0. The first re-run used the wrong value and understated the loop by
0.020. Always pass `--top-k-percent 15.0` for ToN.

At the correct setting the loop's CI crosses zero, so **the loop is now statistically tied
with AGAF**, not below it.

### The loop's selection mechanism does work

| Mode | macro-F1 |
|---|---|
| real | 0.4986 |
| head_only | 0.3641 |
| random | 0.3498 |

real − random = **+0.147**, CI [+0.093, +0.204], P=1.000
real − head_only = **+0.132**, CI [+0.080, +0.184], P=1.000

This is a genuine change of state. On the prototype path, real vs random was −0.0004 with
the CI crossing zero — choosing which flows to consult was no better than picking at
random. It is now decisively better. The selection logic is sound; what it feeds is the
limitation.

### Resulting ladder state

```
GNN 0.3398   <<   Loop 0.4986  ~  AGAF 0.5141  ~  LLM 0.5165
```

Only the GNN is clearly separated. Every pairwise CI among the top three crosses zero.
The strict order Loop > AGAF > {GNN, LLM} is not demonstrated.

---

## 7. Open items

**~~The gate-entropy regulariser~~ — closed in 6b.** Not the cause. Disabling it gave
bit-identical predictions, because the output gate defers 95% to the LLM head.

**~~The feedback loop regressed~~ — closed in 6b.** At the correct `top_k = 15.0` the
loop reaches 0.4986 and ties AGAF (CI crosses zero).

**Still open — AGAF does not actually fuse.** Its output gate sits at 0.95 trust in the
LLM head. Any claim that structural and semantic evidence combine is currently
unsupported on ToN. Testing this properly needs a GNN worth listening to.

**The GNN remains the bottleneck.** At 0.3398 it is far below FedGATSage's 0.6193 on
the same dataset. Their recipe includes per-node traffic-statistic features, which our
`structural10` profile lacks. Until the structural branch improves, fusion has nothing
to add on top of the semantic branch.

---

## 8. Reproduction

```bash
python -m src.pipeline.step3.train_fusion --dataset ton_iot --use-head-logits \
  --output-dir data/ton_iot/processed/step3_fusion_head

python -m src.pipeline.step4.assemble_ladder --dataset ton_iot --use-llm-head \
  --fusion-dir data/ton_iot/processed/step3_fusion_head
```

Prototype comparison:

```bash
python -m src.pipeline.step4.assemble_ladder --dataset ton_iot
```
