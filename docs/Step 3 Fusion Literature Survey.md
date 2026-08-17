# Literature Survey: Parallel Dual-Encoder Fusion (Step 3 / Phase 1)

**Question asked:** Has anyone built a similar "parallel dual-encoder" — one GNN branch + one LLM/transformer branch, running independently and fused via attention into a joint classifier — the way Step 3 of the LLM-Enhanced Graph-Based IDS proposes?

**Bottom line:** The exact *mechanism* (parallel GAT-family + LM, fused by a trainable scoring function that yields per-sample scalar weights) already exists — in fake-news detection, not intrusion detection. No network-IDS paper does true parallel embedding-level GNN+LLM fusion; existing IDS work is sequential (LLM before or after the GNN) or lacks a real language-model branch. That keeps your novelty claim intact but narrows it: the contribution is the *domain application plus per-attack-type interpretability validation*, not the fusion mechanism itself.

---

## The one work that matters most for your novelty claim

**Moorthy et al., "Dual-Stream Graph-Augmented Transformer Model Integrating BERT and GNNs for Context-Aware Fake News Detection," *Scientific Reports* 15:25436 (2025).** [Full text (PMC mirror)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12259997/)

- Runs **BERT in parallel** with a **GAT + Graph Transformer** stream over a heterogeneous article/user/source graph.
- Fuses the two with a **trainable feedforward attention-scoring function** that produces a **separate scalar weight per modality, per input sample** — verified on the fetched page: "the attention mechanism adaptively balances the textual and graph-based contributions... relying more on textual indicators for deceptive language and more on graph features for suspicious propagation patterns."
- Their own ablation table confirms attention fusion beats plain concatenation (AUC 0.97 vs 0.95) and the full model hits 99% accuracy / 0.99 AUC on FakeNewsNet.

This is mechanism-for-mechanism the closest thing to your Step 3 design anywhere in the literature I could confirm. It means "parallel GNN + LM with attention-weighted fusion" as a *general idea* is not novel — but it has never been applied to NetFlow/IoT intrusion detection, and this paper does not validate its weights per class (e.g. showing DDoS is structurally-dominant, injection is semantically-dominant) the way your Step 5 validation plan intends to.

## What existing intrusion-detection work actually does (none of it matches)

| Work | What it actually does | Why it's not your design |
|---|---|---|
| [XG-NID (Farrukh, Wali, Khan, Bastian, arXiv:2408.16021, 2024)](https://arxiv.org/html/2408.16021v1) | Heterogeneous GNN classifies traffic; LLM only generates a post-hoc human-readable explanation of the verdict | LLM is a report-writer, not a parallel encoder — confirmed on the page: no fusion, no attention between GNN and LLM outputs |
| [Zhan, Zhou, Haddadi (arXiv:2506.20806, 2025)](https://arxiv.org/pdf/2506.20806) | LLM agent scores/filters suspicious nodes *before* the GNN sees them | Sequential (LLM → GNN), opposite topology of "parallel" |
| [GCN-2-Former (*Scientific Reports*, 2025)](https://www.nature.com/articles/s41598-025-18401-3) | Two parallel towers (spatial GCN + temporal Transformer), concatenated | Parallel topology matches, but second tower reads time-series, not natural language — no semantic/KG branch, and fusion is plain concat, not attention |
| [FedLLM IDS (Research Square preprint)](https://www.researchsquare.com/article/rs-7738954/v1) | Claims to combine LLM semantic encoding + GNN embeddings via a "transformer fusion detector" | Nearest in spirit but the preprint states no fusion equations, no attention mechanism, no per-sample weights — underspecified, not a real precedent |
| [CPS-IDS](https://www.techscience.com/cmc/v87n3/66942) / [BSTFNet](https://www.techscience.com/cmc/v78n3/55933/html) | Gated fusion of a DeBERTa/ET-BERT semantic branch with a statistical/spatiotemporal branch | No graph tower at all |

## Closest architectural templates outside IDS (useful for building your fusion classifier)

| Work | Domain | Fusion mechanism | Why it's useful to you |
|---|---|---|---|
| [DCVD (Tang et al., arXiv:2605.11015, 2026)](https://arxiv.org/html/2605.11015v1) | Software vulnerability detection | Parallel GAT-structural + LLM-semantic branches → InfoNCE contrastive alignment → **bidirectional cross-attention** with explicit Q/K/V formulas → concat + projection | Best fully-specified template if you want richer fusion than a scalar weight: verified formulas `H_s = softmax(Q_sK_tᵀ/√d′)V_t`, `H_t = softmax(Q_tK_sᵀ/√d′)V_s` |
| [CAST (Lee et al., LG AI Research, arXiv:2502.06836)](https://arxiv.org/html/2502.06836v1) | Materials property prediction | Cross-attention fusing a graph encoder with a text encoder over generated structure descriptions | Cleanest published motivation for *why* a text branch adds value a graph alone can't capture — directly reusable framing for your KG-triple sentences |
| [RAGFormer (Li et al., arXiv:2402.17472, 2024)](https://arxiv.org/html/2402.17472v1) | Fraud detection | Stacks the two branch embeddings as a 2-token sequence, applies self-attention + residual, then MLP head | Simplest possible "attention over modalities" implementation if you want a minimal working version first |
| [GMLM (Sinha, arXiv:2503.05763)](https://arxiv.org/html/2503.05763v3) | Heterophilic node classification | Plain concatenation + MLP, **explicitly no attention/gating** | Use this as your "concat baseline" ablation arm — it's the naive alternative your attention fusion should beat |
| [BertGCN (Lin et al., ACL Findings 2021)](https://aclanthology.org/2021.findings-acl.126.pdf) | Text classification | Linear interpolation of two separate classifiers' predictions with a **global** (not per-sample) weight | Use as your other ablation baseline — the key contrast for your novelty argument is that your weight is *learned per sample*, theirs is a fixed hyperparameter |
| [Gated Multimodal Units (Arevalo et al., arXiv:1702.01992)](https://arxiv.org/abs/1702.01992) | General multimodal classification | Multiplicative, input-dependent gate deciding modality influence | Canonical citation for per-sample gated fusion if you want a gate instead of/alongside attention |

## Recommended framing for your writeup

1. **State the honest novelty position:** parallel GNN+LM fusion with per-sample attention weights is an established mechanism (Moorthy et al., fake news), but it has never been applied to NetFlow-based network intrusion detection, and no prior work validates the learned weights *per attack type* the way your Step 5 plan does.
2. **Cite XG-NID and the Zhan et al. poster as your key differentiating baselines** — both are labeled "GNN + LLM for IDS" in title but are sequential, not parallel-fused, which is exactly the gap you're filling.
3. **Consider an ablation ladder** mirroring the real baselines found: (1) GAT-only, (2) LLM-only, (3) concat+MLP (à la GMLM), (4) global-weight interpolation (à la BertGCN), (5) your per-sample attention fusion. This directly demonstrates that the *per-sample* nature of your weights — not just having two towers — is what earns the performance/interpretability gain, and maps cleanly onto your existing Step 5 validation plan.
4. **If you want a more expressive fusion than a scalar per-modality weight**, DCVD's bidirectional cross-attention formulas are the most rigorously documented template available and would upgrade your fusion layer without changing its "attention-based" character.

---

*This is a condensed synthesis. A fuller version covering 25 surveyed works (including molecular GNN+LLM composition patterns, phishing/bot-detection analogues, and gating-mechanism variants) is saved separately if you want to dig deeper into any thread.*
