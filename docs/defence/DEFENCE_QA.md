# Defence Q&A

Prepared 2026-09-21 from a simulated five-seat review panel over the M2 report.
**Resynced 2026-09-24 to report commit `4506e6e`** (second instructor revision: GLASS name,
RQ2 closed against the semantic stage). Table and section numbers refer to that PDF.
Each entry is a question a jury member is likely to ask, and the answer the report
can support. **Every number here traces to a committed artifact** — none is from
recollection.

Ordered by how likely the question is and how much it costs to get wrong.
Items 1–4 are the ones to rehearse out loud.

---

## 1. "Your system doesn't beat the head it consults."

**This is the most likely hostile question in the defence.** §3.5 raises it; §4.2 and
§5.4 now answer it.

> It doesn't beat it, and that is the finding. The feedback model sits at −0.0032 against
> the trained head alone on both datasets — [−0.0228, +0.0128] on NF-UNSW-NB15 and
> [−0.0342, +0.0299] on NF-ToN-IoT — not separated, with the sign unstable across the three
> seeds. That is the same result §4.3 reports from the other direction: with output fusion
> disabled, the injection path contributes nothing separated from zero. So the mechanism
> adds nothing over its own consultant, which is what we conclude.
>
> What we can also show is that the channel is not inert. An oracle consultant converts it
> into a separated gain on both datasets, +0.0369 and +0.1354. The limit is the quality of
> the consultant we could build, not the architecture around it. And the head alone is
> separated above the GNN model, at +0.0946 and +0.0725, so the honest summary is that on
> these graphs the semantic view is the strongest single signal and nothing we built on top
> of it adds a separated margin.

**Two oracle numbers, both correct.** Table 7 gives the oracle's paired difference against
control, +0.0369 and +0.1354, and below it the pooled headroom from the mean column, +0.0345
and +0.1090. The deck (slide 11) quotes the headroom. If a juror points at the mismatch:
"two estimators, same table; the caption explains it" (see Q10).

**Do not say** "it's nearly equal, so it's fine." That is the question, not the answer.
**Do not say** "the graph would help in deployment." Not measurable here.

---

## 2. "Is the feedback mechanism doing anything at all?"

> It moves predictions, yes. Between consecutive loop passes the injection path changes the
> predicted class of 0.1372 of edges on NF-UNSW-NB15 and 0.1246 on NF-ToN-IoT, three-seed
> means over every edge in the graph — roughly one edge in eight per pass. Per seed that runs
> from 0.0938 to 0.1675 and from 0.1070 to 0.1402.
>
> That measures change, not improvement. Isolate the path by disabling output fusion and the
> gain is +0.0007 and −0.0378, neither a separated improvement, and the second a separated
> loss. The mechanism is alive; it does not convert that movement into score.

**Do not say** the changes "run toward the correct class". The 1,001-vs-66 and 3,168-vs-195
flip counts were **removed from the report** because they were measured against the loop's
own first pass, not against an independent classifier.

**Keep distinct** — three different churn numbers:

| Quantity | Value | Meaning |
|---|---|---|
| Attention-bias variant | **0.0000** | nothing ever changes; genuinely inert |
| Edge injection vs previous pass | **0.0938–0.1675** (per seed, both datasets) | the live mechanism |

Source: `results/feedback_churn.json`; Appendix A. Its source traces live in the untracked
`tmp/multiseed_v2/`, which the report discloses.

---

## 3. "Only three seeds?"

> Three was our compute budget. The bootstrap's seed level therefore resamples three points,
> so we read the intervals as approximate where seed variance is concerned. The comparisons
> with the least room are two NF-ToN-IoT steps with the trained head, feedback over the GNN
> and over fusion, whose lower bounds are +0.0129 and +0.0139 (Table 5).
> Neither is load-bearing on its own: both are sign-stable at all three seeds and both
> replicate on the other dataset.

---

## 4. "Why is your best model's spread so much tighter than the GNN's?"

(±0.0042 against ±0.0228 on UNSW — a visible oddity in Table 3.)

> By design. The consultant is a single artifact built at seed 42 and used at every training
> seed, so its advice is identical across the three runs. The reported spread is graph-side
> training variance only; a fully re-seeded pipeline would show more. That is Limitation 5,
> and it is the reason we do not lean on the two narrowest NF-ToN-IoT comparisons alone.

---

## 5. "Did you check the gate needs to be adaptive?"

> Yes. We ran our gate against the four fusion families §2.3 names, on identical folds and
> embeddings. Feature-wise adaptive gating beats BertGCN-style fixed weighting by 0.119
> macro-F1 on NF-UNSW-NB15 and 0.069 on NF-ToN-IoT, and beats ungated concatenation by more.
> It is not a capacity effect: a quarter-sized version of our gate, 33,516 parameters against
> fixed weighting's 142,092, still beats it on both datasets.
>
**Not in the report.** This sweep was run after the report froze (commit `99338dc`), at one
seed, with no intervals. Say so before quoting it: "a post-report check at one seed".
Numbers: canonical gate 0.7680 / 0.3949, fixed weighting 0.6489 / 0.3256, quarter-size gate
0.7249 / 0.3458. Source: `results/{unsw_nb15,ton_iot}_fusion_variants.json`. Compare on
`eval_class_macro_f1`, never `all_class_macro_f1`.

The earlier per-class gate spread (0.29–0.41, 0.038) has been dropped from this sheet: no
committed artifact or report passage carries it.

---

## 6. "Would a random forest do just as well? Why a graph at all?"

> The research questions compare representations within a graph framing, so a tabular model
> answers a different question and we did not run one. We do have a no-graph reference point
> in the ladder: the semantic model uses no graph structure. On NF-ToN-IoT it scores 0.2785
> against the GNN model's 0.4336, so structure carries information there. On NF-UNSW-NB15 the
> two are close, 0.7353 against 0.7437, and we report that rather than hide it.
>
> But the direct question is fair and we concede it: no flat classifier was fitted to the same
> edge and centrality features, and no run disables message passing. That is Limitation 9, and
> it is the experiment we would run first.

---

## 7. "How many comparisons did you run?"

> Several dozen intervals across the ladder, baselines and diagnostics, with no multiplicity
> correction, so a few accidental separations are expected. The headline ones are not
> candidates: they hold on both datasets, under both bootstrap procedures, with the sign
> stable at every seed — the "stable" column of Table 5.

---

## 8. "Is CySecBERT an LLM? Your title says so and §3.5 says not."

> It is a pretrained language model — encoder-only, frozen, not prompted and not generative.
> The title uses the field's loose term; §3.5 gives the precise one.

---

## 9. "How does the fusion gate actually work?"

> Feature-wise. It reads both projections plus their elementwise difference and product, and
> emits 128 independent sigmoid weights per edge — one per dimension — then a feature-wise
> attention step reweights the fused vector's channels before the classifier. The objective
> also subtracts the gate's mean binary entropy with weight 0.01, which keeps the gate away
> from 0 and 1. Where we quote a gate value it is the mean over the 128 channels, read as the
> share given to the structural branch.

Plain version, as on slide 6: each view becomes 128 numbers, "channels". For every edge the
gate gives each channel its own weight *w*; that channel takes *w* from the structure and
1 − *w* from the sentence. Equation 3 in §3.6. **Not** "two weights per edge" — an earlier
version of the deck said that and was wrong.

---

## 10. "Your Table 7 percentages don't match the differences beside them."

> Two different estimators. The shares divide the mean column — headroom as oracle minus
> control, each share as that arm's mean minus control over the same denominator. The
> interval column is a paired bootstrap mean over resampled (seed, fold) pairs. The caption
> states this.

---

## 11. "Isn't fitting the scaler before the split leakage?"

> It is transductive, not label leakage: the scaler sees feature values, never targets, and
> the graph is fixed. Flagged explicitly in §3.3 and §3.5. We did not run a fold-local
> sensitivity check.

---

## 12. "Can you trust the per-class results on the small classes?"

> No, and we do not. No per-class value carries an interval, and on a 12-edge class a single
> edge moves F1 by several points. §4.7 says so. Nothing in the ladder rests on them.

---

## 13. "You claim imbalance is what differs between your datasets, but a lot differs."

> The datasets were selected for the imbalance contrast, and they also differ in the share of
> edges repeating an address pair, 58.7% against 7.9%, and in size and density. §1 and §4.5
> state both. With two datasets we cannot isolate imbalance from those differences. What the
> comparison establishes is that fusion is the stage whose ordering against the GNN flips
> between the two regimes — as an ordering of means; the difference is separated on neither
> dataset. Which difference between the datasets drives that is not settled here.

---

## 14. "Is NF-ToN-IoT really a graph? Most nodes have one connection."

Weakest-covered item. No prepared measurement.

> It is sparse — 1,501 nodes against 2,127 edges, and 62.8% of edges end at a destination of
> in-degree one. We use that fact in §4.5 to explain why attention injection cannot work
> there. We did not separately measure how much message passing contributes on that graph,
> and that is a fair limitation.

---

## 15. "Why do two of your baseline rows have identical numbers?"

> Because they are the same model. On NF-ToN-IoT the refit's validation grid re-selected
> exactly the published configuration — 128 hidden units, two layers, learning rate 0.001,
> dropout 0.2 — so both rows are one run reported twice. On NF-UNSW-NB15 the grid chose
> differently and the rows differ, 0.1194 against 0.1368.

Source: `results/ton_iot_sota_baselines.json`, `baselines.e_graphsage.{as_published,refit}.hyperparameters`
(validation macro-F1 of the refit selection 0.4653).

---

## 16. "What does it cost to run?"

No figures in the report. A frozen BERT forward pass per edge dominates. Get an order of
magnitude before the defence if asked-about topics include deployment.

---

## 17. "What does fusion actually buy you?" (RQ2)

Answered in the 2026-09-24 revision; Table 5, §4.2, §5.1.

> Two comparisons, two answers. Against the prototype semantic model, fusion is separated
> above it on both datasets: +0.0443 [+0.0067, +0.0795] on NF-UNSW-NB15 and +0.1299
> [+0.0704, +0.2054] on NF-ToN-IoT, both procedures, sign stable. Against the GNN model it is
> not separated on either, +0.0370 [−0.0212, +0.0929] and −0.0235 [−0.0910, +0.0602]. So
> fusion improves on the semantic side but is not shown to improve on both single-branch
> stages, which is what RQ2 asked. Under the rule fixed before the runs, RQ2's strong form is
> unsupported.

If asked how the new interval was produced: from the saved predictions, same two bootstrap
procedures, 2,000 iterations, bootstrap seed 42, seeds 42/1/2. No retraining, no change to
model selection; the reconstructed scores are asserted against the existing multi-seed
artifact. `scripts/close_rq2_interval.py` → `results/rq2_fusion_semantic_interval.json`.

**Do not say** fusion beats "the semantic branch" without "prototype". Against the trained
head it was never compared, and the head alone is far above it.

---

## 18. "What is GLASS, and what is new? DAS already does graph–language feedback."

> GLASS — Graph-Language Adaptive Semantic-Structural fusion — is the name of the whole
> pipeline: graph encoder, frozen CySecBERT over templated edge sentences, the per-channel
> gate, and the uncertainty-guided feedback stage. Neither fusion nor iterative feedback is
> new by itself, and the report says so: DAS already runs graph–language feedback, for node
> classification. What is ours is the edge-level intrusion-detection setting and the
> controlled evaluation that separates the full system from its consultant and from the
> injection path. That separation is what produced the main finding.

**Do not say** "the first bidirectional GNN–LLM loop". That sentence was removed from the report.

---

## 19. "Do you beat every baseline?"

> Every *faithful* re-trained baseline, on both datasets, with the trained-head consultant.
> Not every variant: on NF-ToN-IoT the feedback model is not separated from E-GraphSAGE with
> our node features added, +0.0667 [−0.0181, +0.1505] (Table 9). That variant is our ablation,
> not the published model, so it is not tabulated under the paper's name.

Slide 10 says "every faithfully re-trained baseline" for this reason — keep the qualifier.

---

## Things to volunteer rather than wait to be asked

1. **The most transferable result.** A controlled, label-free sentence describing a flow,
   read by a domain-adapted encoder, is a strong edge classifier on both datasets. That is
   useful to anyone building a NIDS, independent of graphs or feedback loops.
2. **The negative results are the contribution.** Five treatments tried and reverted, the
   injected magnitude of 0.077 that showed the mechanism was effectively switched off, the
   trust scalar that barely moves when reliability changes deliberately. This is the
   intellectual content.
3. **The label-in-key boundary.** Say it before anyone asks. It is the limitation that most
   restricts what the work can claim, and volunteering it is what makes the rest credible.

---

## Known weak spots, honestly

| Item | Status |
|---|---|
| ToN graph degeneracy (Q14) | no measurement; concede |
| Identical baseline rows (Q15) | resolved: refit re-selected the published config |
| Fusion-variant sweep (Q5) | one seed, not in the report; say so before quoting |
| Cost and latency (Q16) | absent from the report |
| Label-free key: how much would the edge set change? | never sized |
| False-positive / alert-volume framing | §1 opens on it, never returns |
