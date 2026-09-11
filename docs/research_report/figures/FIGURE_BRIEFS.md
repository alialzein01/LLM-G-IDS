# Figure briefs — M2 research report

Nine figures. Each entry has a **description** (what it is for and what it must show) and a
**prompt** that can be handed to a generating agent on its own. Every number needed is inside
the prompt, so no prompt requires access to this repository.

Status: **approved. All nine are drawn externally, one at a time.**

## How to use this file

Each figure is generated on its own. Hand the generating agent the **Prompt** block for one
figure plus the **Conventions** section below, and nothing else. The prompts are
self-contained: every number a figure needs is written into its prompt, so the agent does not
need to open a contract or any other file in this repository.

**Where the output goes.** Save the result into `docs/research_report/figures/` under exactly
the filename in that figure's heading, for example `fig_ladder.pdf`. Nothing else needs to
move, and the file does not need to be sent anywhere: it is read from that directory when the
figure is wired into the manuscript. A figure that is not yet drawn is simply absent, and the
report compiles without it.

**After a figure lands**, it gets checked against the contracts and given a caption, a label
and a position in the text. That is a separate step from drawing it, and it is where a wrong
number or a mislabelled axis is caught.

**Suggested order.** Figures 1, 2 and 5 are the explanatory diagrams and carry the most weight
per unit of effort, so draw those first. Then 3, 6 and 7, which are the results the report
turns on. Then 4, 8 and 9.

---

## Conventions that apply to every figure

- **Palette (Okabe–Ito, colourblind-safe), roles fixed across all figures:**
  graph encoder `#0072B2` (blue), semantic rung `#E69F00` (orange), AGAF `#009E73` (green),
  loop `#D55E00` (vermillion), oracle `#CC79A7` (purple), control / neutral `#999999` (grey).
- **Size:** report text width is 15.6 cm. Every figure is **6.1 in wide**, vector **PDF**,
  serif font at 9 pt.
- **The two datasets never share a y-axis.** Class counts differ (10 scored classes on
  NF-UNSW-NB15, 8 on NF-ToN-IoT), so absolute levels are not comparable. Only ladder shape is.
  Always two panels with independent axes.
- **Every error bar is training-seed variance at a fixed fold partition**, and captions say so.
  Fold-partition variance is not measured.
- No promotional language anywhere in a title, label or caption. A difference is "separated"
  only when its interval excludes zero.
- Rung names are fixed: "graph encoder", "semantic rung", "AGAF", "loop". Do not invent
  synonyms and do not use "LLM" for our own component.

### Layout problems to avoid

These are not hypothetical. A trial render of the six data figures hit all of them, and every
prompt below has been adjusted, but they are worth stating once because they recur.

- **Long category labels collide.** Where a bar chart has labels like "loop, trained head",
  rotate the tick labels about 25 degrees and right-align them rather than shrinking the font
  until it is unreadable.
- **Annotations drift outside the axes.** An arrow or a delta label placed between two bars
  needs its text anchored inside the axis limits, with a white background box so it does not
  sit on a gridline.
- **Legends land on the data.** Put the legend below the panels as a figure-level legend rather
  than inside a panel, unless there is obvious empty space.
- **Group labels on a stacked heatmap need real coordinates**, not an approximate offset, or
  they end up over the title.
- Leave headroom at the top of a bar panel for the error bar plus any annotation above it.

---

## 1. `fig_architecture.pdf` — the system

**Description.** The reader needs one picture that holds the whole pipeline, showing that the
four rungs are four exit points from one shared graph, not four different systems. It also has
to make the label-conditioned aggregation step visible, because that is the limitation that
bounds every claim in the report.

**Prompt.**
> Draw a left-to-right system diagram of a network intrusion detection pipeline, as a clean
> academic figure, 6.1 inches wide and about 3.6 inches tall, vector output, serif labels at
> 9 pt, white background, no drop shadows or 3-D effects.
>
> Stage 1, far left: a small stack of rows labelled "NetFlow records".
> Stage 2: a box "aggregate by (source IP, destination IP, attack class)". Put a small warning
> marker on this box with the side note "edge identity depends on the label; results are
> controlled comparisons, not deployment estimates".
> Stage 3: a small directed multigraph, 4 or 5 circular nodes labelled as IP addresses, with
> two visibly parallel edges between one pair. Annotate "nodes: 10 centrality measures;
> edges: 5 aggregated flow attributes".
>
> After the graph the path forks into two parallel branches drawn one above the other:
> upper branch in blue #0072B2: "GATv2 encoder" to "structural edge embedding (64-d)".
> lower branch in orange #E69F00: "label-free text serialisation" to "frozen cybersecurity
> language encoder" to "semantic edge embedding (768-d)".
>
> The two branches then meet twice, drawn as two separate destinations on the right:
> a green #009E73 box "AGAF: gated attention fusion", and a vermillion #D55E00 box
> "feedback loop (see the loop figure)".
>
> Mark five small read-out arrows leaving the diagram to the right edge, labelled
> "graph encoder", "semantic rung", "AGAF", "loop", each ending in a small tag
> "pooled 5-fold out-of-fold macro-F1". The point of these tags is that every rung is scored
> by the identical protocol on the identical folds.
>
> Keep colour meaning consistent: blue is structural, orange is semantic, green is fusion,
> vermillion is the loop. Use grey #999999 for the shared preprocessing stages.

---

## 2. `fig_loop_flow.pdf` — the feedback mechanism, step by step

**Description.** The report is about this mechanism, so its steps, its thresholds and its two
distinct paths must be unambiguous. The single most important thing this figure has to show is
that the consultant reaches the final classifier by **two** routes: the injection path being
tested, and output fusion. The isolation experiment works by cutting the second one, and a
reader who has not seen the two paths drawn separately will not follow that experiment.

**Prompt.**
> Draw a numbered process diagram of an iterative feedback loop in a graph neural network,
> academic style, 6.1 inches wide and about 4.0 inches tall, vector output, serif labels at
> 9 pt, white background.
>
> Draw steps 1 to 7 as a cycle, going clockwise, with the numbers shown in small circles:
> 1. "graph pass" — the GNN produces per-edge class logits.
> 2. "per-edge entropy" — Shannon entropy of each edge's prediction, in nats.
> 3. "flag the top k% by entropy" — annotate "k = 29% on NF-UNSW-NB15, k = 16% on NF-ToN-IoT,
>    both selected on validation folds".
> 4. "confidence gate keeps the top half" — annotate "effective consultation: 14.5% and 8.0%
>    of edges".
> 5. "consultant judgement" — a classification head trained on the semantic embeddings returns
>    class logits for the flagged edges.
> 6. "learned projection to an edge-feature bias" — annotate "added to the 39-wide encoded edge
>    vector".
> 7. "next graph pass, then churn check" — annotate "stop when churn < 0.01, at most 3
>    iterations".
> Arrow from 7 back to 1 to close the cycle.
>
> Separately, and this is the important part, draw a SECOND path as a distinct dashed arrow
> going straight from step 5 to a box on the right labelled "final classifier", bypassing the
> cycle entirely. Label that dashed arrow "output fusion". Label the solid path around the
> cycle "injection path". Put a small caption inside the figure: "the isolation experiment
> disables output fusion, so advice can only arrive through the injection path".
>
> Trace one example edge through the cycle in vermillion #D55E00, with a small annotation at
> each step showing its state: "entropy 1.42 nats, flagged", "gated", "consultant says
> Backdoors", "bias applied", "prediction changes". Use grey #999999 for the unflagged
> majority of edges so the reader sees that most edges never enter the loop.
>
> Keep everything else in muted greys and blues so the vermillion trace and the dashed fusion
> path are the two things the eye finds first.

---

## 3. `fig_ladder.pdf` — the four rungs on both datasets

**Description.** Shows the ladder and, in the same frame, what changing the loop's consultant
did to it. The loop was first built with an embedding-only prototype scorer as its consultant
and later with a classification head trained on the semantic embeddings. Both versions are
drawn, because the change between them is a result in its own right. The two panels must not
share a y-axis.

**Prompt.**
> Draw a two-panel grouped bar chart, academic style, 6.1 inches wide and about 2.8 inches
> tall, vector PDF, serif labels at 9 pt, white background, light horizontal gridlines only,
> no chart junk.
>
> Left panel titled "NF-UNSW-NB15 (10 classes)", right panel titled "NF-ToN-IoT (8 classes)".
> The two panels have INDEPENDENT y-axes and the figure must not invite comparison of their
> absolute heights. Y-axis label on both: "pooled out-of-fold macro-F1". Left panel y-range
> 0.20 to 0.90; right panel y-range 0.20 to 0.60.
>
> Five bars per panel in this order, with error bars showing plus or minus one standard
> deviation over three training seeds:
>
> NF-UNSW-NB15: graph encoder 0.7437 ± 0.0228 (blue #0072B2); semantic rung 0.7353 ± 0.0000
> (orange #E69F00); AGAF 0.7793 ± 0.0116 (green #009E73); loop, prototype consultant 0.7644
> (vermillion #D55E00 but drawn hollow with a hatched fill, no error bar); loop, trained-head
> consultant 0.8341 ± 0.0042 (solid vermillion #D55E00).
>
> NF-ToN-IoT: graph encoder 0.4336 ± 0.0053 (blue); semantic rung 0.2785 ± 0.0000 (orange);
> AGAF 0.4102 ± 0.0279 (green); loop, prototype consultant 0.4521 (hollow hatched vermillion,
> no error bar); loop, trained-head consultant 0.5044 ± 0.0080 (solid vermillion).
>
> Overlay the three individual seed values on each solid bar as small open circles:
> NF-UNSW-NB15 graph 0.7219 / 0.7674 / 0.7418, AGAF 0.7680 / 0.7912 / 0.7788, loop 0.8332 /
> 0.8387 / 0.8304. NF-ToN-IoT graph 0.4290 / 0.4393 / 0.4326, AGAF 0.3975 / 0.3910 / 0.4422,
> loop 0.4985 / 0.5012 / 0.5135. The semantic rung is deterministic, so it has no spread;
> annotate it "deterministic, std 0.0000" rather than drawing a zero-length error bar.
>
> Draw a thin arrow between the two loop bars in each panel labelled with the change:
> "+0.0697" on the left and "+0.0523" on the right, and a shared note under the figure:
> "consultant change, point estimates; the two loop versions also differ in their selected
> top-k (31 to 29, and 25 to 16)".
>
> Footnote inside the figure: "error bars are training-seed variance at a fixed fold
> partition; fold-partition variance is not measured".

---

## 4. `fig_imbalance.pdf` — the axis the two datasets differ on

**Description.** The Discussion argues that class imbalance is the axis on which the two
datasets differ and the axis on which the fusion stage fails. That argument rests on 12.4:1
against 582:1, which is hard to feel from a sentence and immediate from a log-scale chart.

**Prompt.**
> Draw a two-panel horizontal bar chart of class sizes, academic style, 6.1 inches wide and
> about 3.2 inches tall, vector PDF, serif labels at 9 pt, white background.
>
> Both panels use a LOGARITHMIC x-axis labelled "edges (log scale)", running from 1 to 2000.
> Bars sorted largest at the top. Use a single neutral blue #0072B2 for all bars.
>
> Left panel "NF-UNSW-NB15, 656 edges": Normal 311, Backdoors 40, DoS 40, Exploits 40,
> Fuzzers 40, Generic 40, Reconnaissance 40, Shellcode 40, Worms 40, Analysis 25.
>
> Right panel "NF-ToN-IoT, 2,127 edges": Benign 1746, mitm 157, injection 116, ddos 35,
> password 25, backdoor 16, scanning 13, xss 12, dos 4, ransomware 3.
>
> Shade a light red band behind every class at or below 35 edges, with one legend entry
> "35 edges or fewer". That is 1 class on the left and 7 on the right, and the contrast
> between those two counts is the point of the figure.
>
> Draw the two smallest NF-ToN-IoT bars, dos (4) and ransomware (3), hollow with a dashed
> outline, and label them "excluded from the metric, kept in the graph for message passing".
>
> Annotate each panel with its most-to-least-frequent class ratio in a corner box:
> "12.4 : 1" on the left, "582 : 1" on the right. Add a single caption line between the
> panels: "the two graphs differ by roughly a factor of 47 in imbalance severity".

---

## 5. `fig_ceiling.pdf` — why E-GraphSAGE collapses on one dataset and not the other

**Description.** The clearest explanatory visual available: a mechanism, a prediction derived
from it, and two datasets confirming the prediction in opposite directions. It also protects
the report from a misreading, since a very large margin against a published architecture looks
like an architectural win and is in fact a representation effect.

**Prompt.**
> Draw a two-part explanatory diagram, academic style, 6.1 inches wide and about 2.4 inches
> tall, vector PDF, serif labels at 9 pt, white background.
>
> Left part, the mechanism: two circular nodes labelled "u" and "v" with three curved parallel
> edges running between them. Label the three edges with visibly different attributes, for
> example "TCP:443, 120 flows", "UDP:53, 8 flows", "TCP:22, 3 flows", each in a different
> shade. Then draw all three collapsing through a single funnel arrow into one grey #999999
> box labelled "[ h_u ‖ h_v ]". Caption beneath: "E-GraphSAGE classifies an edge from its two
> endpoint embeddings only, so parallel edges between one address pair are identical to the
> classifier".
>
> Right part, the confirmation: a small two-bar chart, with each bar annotated by the share of
> affected edges. Bar 1, "NF-UNSW-NB15": macro-F1 0.1194, annotation "385 of 656 edges
> indistinguishable (58.7%)". Bar 2, "NF-ToN-IoT": macro-F1 0.4141, annotation "167 of 2,127
> edges indistinguishable (7.9%)". Use grey #999999 for both bars, since neither is one of our
> rungs. Y-axis "pooled out-of-fold macro-F1", range 0 to 0.5.
>
> Place a short sentence between the two parts: "the prediction and the inversion agree".
> Add a small footnote: "not a training-schedule artifact: at the authors' full 4,999 epochs
> on fold 0, validation macro-F1 plateaus near 0.11".

---

## 6. `fig_mechanism.pdf` — the channel has capacity; no realistic consultant uses it

**Description.** The central finding of the report, currently visible only as a table. A forest
plot is the right form because the claim is entirely about which intervals cross zero. Output
fusion is disabled in every arm, so this figure is measuring the injection path alone.

**Prompt.**
> Draw a two-panel forest plot (dot-and-interval plot), academic style, 6.1 inches wide and
> about 2.6 inches tall, vector PDF, serif labels at 9 pt, white background.
>
> Left panel "NF-UNSW-NB15", right panel "NF-ToN-IoT". X-axis on both: "paired difference in
> pooled out-of-fold macro-F1 against control". Draw a bold solid vertical line at zero in
> each panel. Independent x-ranges: left about −0.06 to +0.09, right about −0.09 to +0.20.
>
> Four rows per panel, top to bottom, each a point estimate with a 95% confidence interval
> drawn as a horizontal whisker with end caps:
>
> NF-UNSW-NB15: "oracle, edge" +0.0369 [+0.0097, +0.0618] in purple #CC79A7;
> "trained-head consultant, edge" +0.0007 [−0.0183, +0.0201] in vermillion #D55E00;
> "prototype consultant, edge" −0.0046 [−0.0273, +0.0158] in grey #999999;
> "oracle, attention" −0.0007 [−0.0211, +0.0180] in light purple, drawn hollow.
>
> NF-ToN-IoT: "oracle, edge" +0.1354 [+0.0919, +0.1797] in purple;
> "trained-head consultant, edge" −0.0378 [−0.0588, −0.0167] in vermillion;
> "prototype consultant, edge" +0.0099 [−0.0111, +0.0286] in grey;
> "oracle, attention" +0.0114 [−0.0083, +0.0307] in light purple, hollow.
>
> Fill a pale grey band across each panel spanning −0.012 to +0.012, labelled once in the
> legend as "run-to-run drift without thread pinning". Any interval sitting inside that band
> is smaller than the measurement noise this project has documented.
>
> Mark the two rows whose intervals exclude zero with a small filled marker and the others with
> an open marker, and add a one-line note: "only the oracle converts the channel; on
> NF-ToN-IoT the trained-head consultant is separated in the wrong direction".

---

## 7. `fig_decomposition.pdf` — where the NF-UNSW-NB15 margin actually comes from

**Description.** Stops the large headline margin against a published architecture from reading
as an architectural result. Roughly 45% of it is how we tuned, 30% is our node features, and
25% is the architecture.

**Prompt.**
> Draw a waterfall chart, academic style, 6.1 inches wide and about 3.0 inches tall, vector
> PDF, serif labels at 9 pt, white background, y-axis "pooled out-of-fold macro-F1" from 0.30
> to 0.90.
>
> Start with a full grey #999999 bar at 0.3841 labelled "TE-G-SAGE as published".
> Then three floating increment bars, each labelled with its size and its share of the total
> gap:
> increment 1, +0.2001 to 0.5842, labelled "fair tuning: graph scale and schedule, 44.5%",
> coloured light blue;
> increment 2, +0.1354 to 0.7196, labelled "our ten node centralities, 30.1%", coloured
> medium blue #0072B2;
> increment 3, +0.1145 to 0.8341, labelled "architecture, 25.4%", coloured vermillion #D55E00.
> End with a full vermillion bar at 0.8341 labelled "our loop rung".
>
> Draw a small horizontal tick across the final increment at 0.7793 labelled "AGAF rung", to
> show where the fusion stage lands on the same scale.
>
> Put this exact sentence in the figure as a footnote: "point estimates only; every level is a
> three-seed mean and no step carries an interval or a separation claim".

---

## 8. `fig_perclass.pdf` — per-class differences

**Description.** The per-class levels are already fully tabulated in the report, so a heatmap of
the same levels would only repeat the table. What the Discussion argues about is the
differences: where the loop gains over the fusion stage and over the graph encoder, and where
it loses. A diverging heatmap centred on zero shows both the direction and the concentration,
and the two negative cells on each dataset are exactly the ones the text discusses.

**Prompt.**
> Draw a diverging heatmap, academic style, 6.1 inches wide and about 4.2 inches tall, vector
> PDF, serif labels at 9 pt, white background.
>
> Two columns: "loop − AGAF" and "loop − graph encoder". Eighteen rows, grouped into two
> labelled blocks with a gap between them. Add a narrow left-hand margin column showing each
> class's edge count as plain text, not as colour.
>
> Diverging colour scale centred exactly at zero, symmetric limits −0.22 to +0.22, blue for
> negative and red for positive, white at zero. Print each cell's value inside it to four
> decimal places in a small font. Draw a thin black outline around every cell whose value is
> negative, so losses are findable at a glance.
>
> Block 1, "NF-UNSW-NB15" (class, edges, loop−AGAF, loop−graph):
> Normal 311 +0.0390 +0.0414; Analysis 25 +0.0309 +0.1869; Backdoors 40 +0.1442 +0.2090;
> DoS 40 +0.0978 +0.1594; Exploits 40 −0.0003 +0.0271; Fuzzers 40 +0.0419 +0.1542;
> Generic 40 +0.1490 +0.0308; Reconnaissance 40 −0.0185 −0.0379; Shellcode 40 +0.0531 +0.1290;
> Worms 40 +0.0107 +0.0040.
>
> Block 2, "NF-ToN-IoT": Benign 1746 +0.0558 +0.0234; backdoor 16 +0.2027 +0.0834;
> ddos 35 +0.0494 +0.1064; injection 116 +0.1433 +0.0842; mitm 157 +0.1484 −0.0273;
> password 25 −0.0850 +0.1699; scanning 13 +0.0335 +0.2159; xss 12 +0.1768 −0.1116.
>
> Footnote: "three-seed means; no per-class difference carries an interval, and on classes of
> 12 to 40 edges a single edge moves F1 substantially".

---

## 9. `fig_comparisons.pdf` — which ladder steps are separated

**Description.** Not in the original plan, proposed as an addition. The report's verdicts all
turn on whether an interval crosses zero, and that is a picture, not a table. It carries the
main result in one frame: under the trained-head consultant every loop comparison clears zero
on both datasets, and under the prototype consultant none of them does. It also shows the one
case where the two bootstrap procedures disagree, AGAF against the graph encoder on
NF-UNSW-NB15, which is the clearest illustration of what admitting training-seed variance
costs.

**Prompt.**
> Draw a two-panel forest plot with paired rows, academic style, 6.1 inches wide and about
> 3.4 inches tall, vector PDF, serif labels at 9 pt, white background.
>
> Left panel "NF-UNSW-NB15", right panel "NF-ToN-IoT". X-axis "difference in pooled
> out-of-fold macro-F1". Bold vertical line at zero in each panel. Independent x-ranges:
> left about −0.02 to +0.15, right about −0.11 to +0.30.
>
> Five comparisons per panel. Each comparison occupies one row slot containing TWO intervals
> drawn slightly offset vertically: the primary "two-level" interval drawn in a solid dark
> tone with round markers, and the "seed-matched" interval drawn in a lighter tone with square
> markers directly beneath it. Legend: "two-level (edges and training seed)" and
> "seed-matched (edges only)".
>
> Rows are grouped: first the loop comparisons under the TRAINED-HEAD consultant, then the same
> comparisons under the PROTOTYPE consultant, then AGAF against the graph encoder, which does
> not depend on the consultant. Label the two groups clearly, because the whole point of the
> figure is that the first group clears zero and the second does not.
>
> NF-UNSW-NB15, trained-head consultant:
> loop − semantic rung: two-level +0.0989 [+0.0721, +0.1267]; seed-matched +0.0991 [+0.0743, +0.1253].
> loop − graph encoder: +0.0908 [+0.0412, +0.1407]; +0.0907 [+0.0571, +0.1250].
> loop − AGAF: +0.0546 [+0.0228, +0.0880]; +0.0551 [+0.0331, +0.0779].
> NF-UNSW-NB15, prototype consultant:
> loop − graph encoder: +0.0206 [−0.0203, +0.0625]; +0.0207 [−0.0001, +0.0421].
> loop − AGAF: −0.0156 [−0.0552, +0.0268]; −0.0150 [−0.0425, +0.0126].
> NF-UNSW-NB15, consultant-independent:
> AGAF − graph encoder: +0.0362 [−0.0096, +0.0822]; +0.0356 [+0.0038, +0.0678].
>
> NF-ToN-IoT, trained-head consultant:
> loop − semantic rung: +0.2231 [+0.1703, +0.2771]; +0.2228 [+0.1732, +0.2713].
> loop − AGAF: +0.0932 [+0.0180, +0.1667]; +0.0940 [+0.0406, +0.1480].
> loop − graph encoder: +0.0694 [+0.0129, +0.1288]; +0.0691 [+0.0171, +0.1200].
> NF-ToN-IoT, prototype consultant:
> loop − AGAF: +0.0428 [−0.0316, +0.1105]; +0.0431 [−0.0059, +0.0936].
> loop − graph encoder: +0.0191 [−0.0180, +0.0596]; +0.0182 [−0.0050, +0.0418].
> NF-ToN-IoT, consultant-independent:
> AGAF − graph encoder: −0.0238 [−0.0921, +0.0607]; −0.0249 [−0.0697, +0.0226].
>
> There is no loop-versus-semantic-rung comparison for the prototype consultant; omit that row
> rather than inventing it.
>
> Draw a small callout on the NF-UNSW-NB15 "AGAF − graph encoder" row: "separated under the
> seed-matched procedure, not separated once training-seed variance is admitted; the two-level
> result decides".
>
> Colour every row by the rung the comparison is about, using vermillion #D55E00 where the loop
> is the subject and green #009E73 for the AGAF row. Draw the prototype-consultant group in a
> visibly lighter tint of the same colours, so the reader sees one group clearing zero and the
> other straddling it.
