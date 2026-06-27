# LLM-Enhanced Graph-Based Intrusion Detection

## Base Report for the PowerPoint Presentation

### 1. Presentation Goal

The goal of this presentation is to explain the research work completed so far in a coherent story. The presentation should not be a coding update. It should explain the context, the problem, why the problem matters, the research gap, the proposed solution, the papers that influenced the work, the implementation steps, the results, the problems we are currently facing, and the reasons behind those problems.

The target audience may include people who are not familiar with graph neural networks, language models, or intrusion detection research. Therefore, every technical concept should be explained briefly and clearly when it is introduced.

The intended presentation duration is around 20 minutes.

---

## 2. Context: Why Intrusion Detection Matters

Modern IoT networks contain many connected devices such as cameras, sensors, gateways, smart appliances, and industrial devices. These devices communicate continuously, often with limited security controls and limited computational resources. As a result, IoT networks are vulnerable to attacks such as denial of service, scanning, password attacks, injection attacks, man-in-the-middle attacks, ransomware, and backdoors.

An Intrusion Detection System, or IDS, is a system that monitors network behavior and tries to detect whether traffic is benign or malicious. In our project, the goal is not only to detect whether traffic is malicious, but also to classify the attack type.

The datasets used so far are:

- NF-ToN-IoT
- NF-UNSW-NB15

Both are NetFlow-style datasets. This means the data describes network flows rather than raw packets. A flow summarizes communication between two endpoints using features such as source IP, destination IP, protocol, port, bytes, duration, and attack label.

---

## 3. Problem Statement

Many intrusion detection approaches treat each network flow as an independent row in a table. This is useful, but it misses an important part of network behavior: communication structure.

For example, a DDoS attack may not be obvious from a single flow alone. It becomes clearer when many source devices communicate with the same target. Similarly, scanning behavior may appear as repeated communication attempts across many destinations or ports. These are structural patterns.

At the same time, pure graph-based models can capture structure but may not understand the semantic meaning of cybersecurity concepts. For example, a graph model can learn that a node has many outgoing edges, but it does not naturally understand that high-frequency short connections over certain ports may resemble scanning behavior.

This creates the central problem:

**Can we combine graph structure and cybersecurity semantics to improve edge-level intrusion detection?**

In this project, the classification target is the communication edge, not the IP node. Each edge represents an aggregated communication pattern between a source IP and a destination IP, and the model predicts the attack class of that edge.

---

## 4. Motivation

The motivation is that graph neural networks and language models provide complementary strengths.

A Graph Neural Network, or GNN, learns from graph topology. In our case, it learns from how IP addresses communicate with each other, using node centrality features and edge traffic attributes.

A cybersecurity language model, such as CySecBERT, learns semantic patterns from natural-language descriptions. In our case, it receives label-free descriptions of traffic behavior, such as connection frequency, byte volume, duration, protocol, and port type.

The motivation for combining them is simple:

- The GNN sees the network structure.
- CySecBERT sees a semantic description of traffic behavior.
- A fusion model tries to combine both signals before classification.

If both views contain complementary information, the fused model should perform better than either view alone.

---

## 5. Research Gap

The literature shows several related directions, but there is still a gap.

Some papers use GNNs for intrusion detection, but they often focus mainly on graph structure and do not use language-model semantics.

Some papers use knowledge graphs and language models for cybersecurity explanation, but the LLM is often used after classification to generate reports rather than as part of the classification process.

Some papers combine GNNs and LLMs, but many of them are designed for text-attributed graphs such as citation networks, not network intrusion detection. In those datasets, nodes already contain text, such as paper abstracts. In network traffic, the data is mostly numerical and categorical, such as bytes, ports, protocols, and durations. Therefore, we need to create a semantic text representation ourselves.

The gap we are targeting is:

**A graph-based IDS pipeline that performs edge-level attack classification by combining structural graph embeddings with semantic cybersecurity embeddings, then validating whether fusion improves over both standalone baselines.**

---

## 6. Main Idea of Our Solution

Our pipeline represents the same network traffic in two complementary ways.

First, we build a communication graph:

- Nodes are IP addresses.
- Directed edges are aggregated communications from source IP to destination IP.
- Edge labels are attack classes.
- The GNN performs edge classification.

Second, we build a semantic text representation:

- Each edge is converted into a label-free natural-language sentence.
- The sentence describes traffic behavior without revealing the attack label.
- CySecBERT encodes each sentence into a semantic embedding.

Third, we train a fusion model:

- The GNN embedding represents the structural view.
- The CySecBERT embedding represents the semantic view.
- Adaptive gated attention fusion learns how much to use each modality.

The success condition is strict: fusion must outperform both the GNN-only baseline and the CySecBERT-only baseline, especially using Macro-F1.

---

## 7. Step 1: Communication Graph Construction

In Step 1, raw network flow records were converted into a directed communication graph.

Each unique IP address became a node. Each aggregated flow became a directed edge from source IP to destination IP. Edges were grouped by:

```text
source IP, destination IP, attack label
```

This means that if the same source contacted the same destination many times under the same attack type, those flows were collapsed into one aggregated edge.

Each edge stores:

```text
flow_count
total_bytes
avg_duration
most_common_protocol
most_common_port
attack label
```

The first three numerical attributes were normalized using Z-score normalization. Z-score normalization means subtracting the mean and dividing by the standard deviation, so features with different scales become easier for the model to learn from. Protocol and port were kept as categorical integer-style features.

The task is edge classification. This means the model predicts the attack class of each communication edge, not the class of each IP address.

### Step 1 Validation

The main checks were:

- Node count equals the number of unique IP addresses.
- Edge count equals the number of aggregated `(source IP, destination IP, attack label)` groups.
- Edge labels are preserved correctly.
- Edge attributes are valid and finite.
- The graph can be loaded into PyTorch Geometric.
- Graph connectivity statistics are reported.

Step 1 passed for both datasets:

```text
ToN-IoT:    passed
UNSW-NB15:  passed
```

---

## 8. Step 2: Knowledge Graph and Semantic Text Construction

In Step 2, each graph edge became one knowledge-graph-style row. The goal was to create a semantic description of each edge that could be understood by CySecBERT.

The numerical attributes were discretized into:

```text
low / medium / high
```

This was done for:

```text
flow_count
avg_bytes
avg_duration
```

The discretization was performed separately inside each attack class. This is important because "high" traffic for one attack type may not mean the same thing as "high" traffic for another attack type. Per-class discretization preserves relative meaning inside each class and avoids letting dominant high-volume classes control the thresholds for all classes.

The structured KG file includes semantic relation names, such as:

```text
Benign -> communicated with
ddos -> launched ddos against
scanning -> initiated scan against
Reconnaissance -> initiated reconnaissance against
```

However, the natural-language sentences sent to CySecBERT are intentionally label-free.

This is a critical design decision. If we put attack names or attack-specific relation verbs inside the input sentence, the language model would see the answer during training. This is called label leakage. Label leakage means the model receives information that directly reveals the target label, making the result invalid.

Instead, CySecBERT receives sentences like:

```text
Observed traffic from source X to destination Y with high connection frequency,
large payloads, long duration, over TCP, to web service port 443.
```

It does not receive sentences like:

```text
This is DDoS traffic.
```

### Step 2 Validation

The main checks were:

- Every aggregated edge has one KG row.
- KG rows align with graph edges by row index.
- Discretization values are valid: low, medium, or high.
- Semantic relation mapping covers every attack label.
- Natural-language KG sentences are label-free.
- Sentence count equals edge count.

Step 2 passed for both datasets:

```text
ToN-IoT:    passed
UNSW-NB15:  passed
```

---

## 9. Step 3A: GNN Path

The GNN path uses the communication graph from Step 1.

The model is a GATv2-based edge classifier. GATv2 is a graph attention network. Attention means the model learns which neighboring nodes or edges are more important instead of treating all neighbors equally.

The GNN first learns node embeddings using graph structure and edge attributes. An embedding is a learned numerical representation. Then, for each edge, the model builds an edge representation by combining:

```text
source node embedding
destination node embedding
edge attributes
```

This edge representation is passed into an MLP classifier. An MLP, or multilayer perceptron, is a standard neural network classifier.

The GNN path learns from:

- IP-to-IP communication topology.
- Node centrality features.
- Edge traffic statistics.
- Direction of communication.
- Edge-level labels.

The embedding shape is:

```text
[number_of_edges, 64]
```

The model is evaluated using shared 5-fold splits and out-of-fold predictions. This means each edge is tested in a fold where it was not used for training.

### GNN Results

```text
NF-ToN-IoT:
GATv2 pooled Macro-F1 = 0.3006

NF-UNSW-NB15:
GATv2 pooled Macro-F1 = 0.5229
```

The GNN performs better on UNSW-NB15 than on ToN-IoT.

---

## 10. Step 3B: LLM Path

The LLM path uses the label-free natural-language sentences from Step 2.

Each sentence is encoded using:

```text
markusbayer/CySecBERT
```

CySecBERT is a BERT-style language model specialized for cybersecurity text. In our pipeline, it is used as a feature extractor. This means we do not rely on CySecBERT to directly output the attack class. Instead, CySecBERT converts each sentence into a semantic embedding, and a classifier is trained on top of those embeddings.

The embedding shape is:

```text
[number_of_edges, 768]
```

The LLM path learns from:

- Natural-language descriptions of traffic behavior.
- Discretized flow frequency.
- Discretized byte volume.
- Discretized duration.
- Protocol and port semantics.

The same train, validation, and test folds are used as in the GNN path, so the comparison is fair.

### LLM Results

```text
NF-ToN-IoT:
CySecBERT pooled Macro-F1 = 0.3979

NF-UNSW-NB15:
CySecBERT pooled Macro-F1 = 0.5165
```

On ToN-IoT, the LLM path is stronger than the GNN path.

On UNSW-NB15, the LLM path is close to the GNN path, but slightly weaker.

---

## 11. Step 3C: Fusion Path

The fusion path combines both modalities:

```text
GNN structural embedding:      64 dimensions
CySecBERT semantic embedding:  768 dimensions
```

Because the two embeddings have different dimensions, both are projected into a shared space. Projection means transforming both embeddings into the same size so they can be compared and combined.

The fusion model uses adaptive gated attention fusion.

A gate is a learned mechanism that decides how much information to take from each input. In our case, the model learns how much to rely on the GNN structural representation and how much to rely on the CySecBERT semantic representation.

Conceptually:

```text
fused representation =
    learned weight for GNN * GNN representation
  + learned weight for LLM * LLM representation
```

Feature-level attention is then applied before classification. Feature-level attention means the model can emphasize the most useful dimensions of the fused representation.

The fusion model is useful only if it improves over both standalone baselines. It is not enough to beat the weaker model. It must beat both the GNN-only model and the CySecBERT-only model.

### Fusion Validation Requirements

The main checks were:

- GNN and LLM embeddings are row-aligned.
- The same edge index refers to the same traffic record in the graph edge, KG row, KG sentence, GNN embedding, LLM embedding, and label.
- Fusion is evaluated on the same folds as the baselines.
- Fusion must beat both baselines using Macro-F1.
- Attention and gate weights are saved for explainability.
- Fold-level deltas are reported.

---

## 12. Results Summary

### NF-ToN-IoT

```text
GATv2 Only:      Macro-F1 = 0.3006
CySecBERT Only:  Macro-F1 = 0.3979
Fusion Model:    Macro-F1 = 0.3865
```

Fusion improves over the GNN:

```text
0.3865 > 0.3006
```

But fusion does not improve over CySecBERT:

```text
0.3865 < 0.3979
```

Therefore, Phase 4 fails for ToN-IoT.

### NF-UNSW-NB15

```text
GATv2 Only:      Macro-F1 = 0.5229
CySecBERT Only:  Macro-F1 = 0.5165
Fusion Model:    Macro-F1 = 0.5705
```

Fusion improves over both baselines:

```text
0.5705 > 0.5229
0.5705 > 0.5165
```

Therefore, Phase 4 passes for UNSW-NB15.

### Current Phase Status

```text
Phase 1 Graph Construction:
ToN-IoT    passed
UNSW-NB15  passed

Phase 2 Knowledge Graph:
ToN-IoT    passed
UNSW-NB15  passed

Phase 3 Baselines:
ToN-IoT    passed
UNSW-NB15  passed

Phase 4 Fusion:
ToN-IoT    failed
UNSW-NB15  passed
```

---

## 13. How to Explain the Metrics and Losses

### Macro-F1

Macro-F1 is the main metric because the datasets are imbalanced. It calculates the F1-score for each class separately, then averages them equally. This prevents the majority class from hiding poor performance on rare attack classes.

Short presentation sentence:

**We use Macro-F1 because every attack class should matter equally, even if some attacks appear much less often than others.**

### Cross-Entropy Loss

Cross-Entropy loss is the standard loss function for multiclass classification. It penalizes the model when it assigns low probability to the correct class.

Short presentation sentence:

**Cross-Entropy is the standard classification loss that teaches the model to give high probability to the correct attack class.**

### Focal Loss

Focal Loss is a modified classification loss that gives more attention to difficult or misclassified examples. It is often useful when data is imbalanced because easy majority-class examples can otherwise dominate training.

Short presentation sentence:

**We use Focal Loss because the datasets are imbalanced, and it forces the model to focus more on difficult and minority-class examples.**

### Class Weights

Class weights increase the penalty for mistakes on rare classes. This helps prevent the model from mainly learning the dominant classes.

Short presentation sentence:

**Class weights compensate for imbalance by making mistakes on rare attack classes more costly during training.**

### Attention

Attention is a mechanism that lets the model learn which parts of the input are more important.

Short presentation sentence:

**Attention helps the model decide which neighbors, features, or modalities deserve more importance for a specific prediction.**

### Fusion

Fusion means combining multiple sources of information. In our case, fusion combines graph structure from the GNN with semantic traffic descriptions from CySecBERT.

Short presentation sentence:

**Fusion combines the structural view and the semantic view so the final classifier can use both kinds of evidence.**

---

## 14. Papers Read and How They Influenced the Work

### FedGATSage

This paper is directly related to graph-based intrusion detection in IoT networks. It uses graph attention and GraphSAGE ideas in a federated learning setting. It supports the idea that graph structure is important for IoT intrusion detection because attacks often appear through communication patterns, not only individual flow features.

How it influenced our work:

- Reinforced the importance of graph-based IDS.
- Supported the use of attention-based GNNs.
- Highlighted that coordinated attacks require structural modeling.

### KLAGE: Knowledge Graphs and LLMs for Explainable Threat Detection

KLAGE uses knowledge graphs, Graph-BERT, explainability methods, and LLM-generated reports for network security. It is important because it connects knowledge graphs and LLMs to explainable cybersecurity.

How it influenced our work:

- Supported the idea of using KG-style representations for network security.
- Showed that LLMs are useful for explanation and semantic interpretation.
- Also showed a limitation: many systems use LLMs after classification, while our work uses semantic embeddings as part of the classification pipeline.

### KGNN

KGNN is not an IDS paper, but it shows how knowledge graphs and GNNs can be combined to learn multi-hop relationships. It is useful as a conceptual reference for graph-based reasoning.

How it influenced our work:

- Helped frame the value of knowledge graphs.
- Showed that graph neighborhoods can carry useful semantic and structural information.
- Highlighted scalability and noise issues when expanding too far in the graph.

### LOGIN

LOGIN proposes an LLM-consulted GNN training framework. The GNN identifies uncertain nodes and consults an LLM during training. This is important because it shows a way for GNNs and LLMs to interact, rather than simply running separately.

How it influenced our work:

- Motivated the idea that LLMs can help with uncertain graph predictions.
- Helped position future work around stronger GNN and LLM interaction.
- Showed that selective LLM use can reduce cost compared with asking the LLM about every sample.

### GL-Fusion

GL-Fusion studies how to combine GNNs and LLMs, and is directly relevant to the fusion part of our work. It supports the idea that simple combination is not always enough, and that the way embeddings are fused matters.

How it influenced our work:

- Reinforced the need to evaluate fusion carefully.
- Supported the idea that GNN and LLM embeddings can be complementary.
- Helped frame the current challenge: fusion must be stronger than both individual branches, not only one.

### Graph Meets LLM Survey

This survey organizes the broader research area of graphs and LLMs. It helps distinguish between different patterns: LLMs for graphs, graphs for LLMs, GNN and LLM fusion, graph retrieval, and graph reasoning.

How it influenced our work:

- Helped position our project inside the larger GNN and LLM research landscape.
- Clarified that IDS has different constraints than citation graphs or text-attributed graphs.
- Supported the argument that network flow data needs a custom semantic serialization strategy.

### EMT-IDNet

EMT-IDNet is an explainable multimodal temporal IDS framework for IoT environments. It combines multiple data modalities and uses attention for explainability.

How it influenced our work:

- Reinforced the value of multimodal IDS.
- Supported the use of attention-based fusion.
- Showed that explainability can be built into the model through attention weights.

### LLMs for Cybersecurity Review

This review summarizes LLM applications, challenges, and future directions in cybersecurity.

How it influenced our work:

- Highlighted the potential of LLMs for threat detection and analysis.
- Also emphasized risks such as hallucination, high computational cost, privacy issues, and vulnerability to prompt attacks.
- Supported our decision to use CySecBERT embeddings in a controlled classification pipeline rather than relying on unconstrained text generation.

---

## 15. Problems We Are Facing

The pipeline is implemented and validated, but the main current problem is that fusion does not yet pass on both datasets.

For UNSW-NB15, fusion improves over both baselines:

```text
Fusion = 0.5705
GNN    = 0.5229
LLM    = 0.5165
```

This shows that the structural and semantic views are complementary on UNSW-NB15.

For ToN-IoT, fusion improves over the GNN but does not beat CySecBERT:

```text
Fusion = 0.3865
GNN    = 0.3006
LLM    = 0.3979
```

This means the fusion model is learning useful information from the GNN, but it may be adding structural noise or failing to exploit the stronger semantic branch effectively.

The problem is not that the code does not run. The issue is methodological: fusion must be made strong and stable enough to outperform the best individual modality.

---

## 16. Possible Reasons Behind the Problems

### 1. Class Imbalance

Intrusion detection datasets often contain many examples of some classes and very few examples of others. This makes training difficult because the model can perform well on common classes while failing on rare classes.

This is why Macro-F1 is important and why Focal Loss and class weights are useful.

### 2. ToN-IoT May Have Weaker Structural Signal

The GNN result on ToN-IoT is much lower than the CySecBERT result:

```text
GNN = 0.3006
LLM = 0.3979
```

This suggests that, for ToN-IoT, the graph structure may be less informative than the semantic traffic description. If the GNN branch is weak, fusion can be pulled down unless the gate learns to rely more on the LLM branch.

### 3. Fusion May Not Be Adaptive Enough Yet

The fusion model should learn when to trust the GNN and when to trust CySecBERT. On ToN-IoT, it may not be selecting the stronger modality consistently enough.

### 4. Some Attack Classes May Be Hard to Separate

Some attacks may share similar flow patterns. For example, DoS and DDoS may both involve high-volume traffic, while injection and XSS may have more semantic similarity than structural separability. This can confuse both the GNN and fusion model.

### 5. Label-Free Text Is Necessary but Harder

Removing attack labels from the text is methodologically correct because it avoids label leakage. However, it also makes the task harder because CySecBERT must infer the class only from behavior descriptions, not from explicit attack words.

### 6. Edge Aggregation Can Hide Temporal Detail

Aggregating flows into edges simplifies the graph and makes training feasible, but it may remove temporal patterns. Some attacks are defined not only by who communicates with whom, but also by how behavior changes over time.

---

## 17. Current Takeaway

The project has reached a meaningful validation point.

The pipeline works end to end:

- Raw flows are converted into communication graphs.
- Graph edges are converted into label-free semantic sentences.
- GNN and CySecBERT baselines are trained using shared splits.
- Fusion is trained and evaluated fairly.
- Validation is strict and dataset-aware.

The key result is mixed but informative:

- On UNSW-NB15, fusion works and improves over both baselines.
- On ToN-IoT, fusion improves over the GNN but does not beat CySecBERT.

This means the research direction is valid, but the fusion strategy needs improvement for ToN-IoT.

---

## 18. Next Steps

The next research task is to improve fusion on ToN-IoT.

Possible directions include:

- Analyze per-class errors to identify which attack classes reduce Macro-F1.
- Inspect fusion gate weights to see whether the model relies too much on the weaker GNN branch.
- Improve the fusion architecture so it can fallback to the stronger modality when one branch is unreliable.
- Experiment with stronger class imbalance handling.
- Add temporal features or temporal aggregation if ToN-IoT requires time-based attack signals.
- Compare different fusion strategies, such as late fusion, confidence-aware fusion, or class-specific gates.
- Improve semantic text construction while keeping it label-free.

---

## 19. Suggested PowerPoint Storyline

### Slide 1: Title

LLM-Enhanced Graph-Based Intrusion Detection

### Slide 2: Context

IoT networks are vulnerable, and IDS is needed to detect attacks from network traffic.

### Slide 3: Problem

Traditional flow-based IDS treats flows independently and may miss communication structure.

### Slide 4: Motivation

GNNs understand structure. CySecBERT understands semantic descriptions. Combining them may improve detection.

### Slide 5: Research Gap

Existing work often uses either GNNs, KGs, or LLMs separately or sequentially. Our work tests structural-semantic fusion for edge-level IDS.

### Slide 6: Proposed Pipeline

Step 1 graph construction, Step 2 semantic text construction, Step 3 GNN/LLM/fusion training.

### Slide 7: Step 1

Raw NetFlow data becomes a directed IP communication graph.

### Slide 8: Step 2

Each edge becomes a label-free semantic sentence for CySecBERT.

### Slide 9: Step 3A and 3B

GNN structural path and CySecBERT semantic path.

### Slide 10: Fusion

Adaptive gated attention fusion combines the two embeddings.

### Slide 11: Papers Read

Summarize FedGATSage, KLAGE, LOGIN, GL-Fusion, Graph Meets LLM, EMT-IDNet, and the cybersecurity LLM review.

### Slide 12: Results

Show GNN, CySecBERT, and fusion Macro-F1 for both datasets.

### Slide 13: Current Problem

Fusion passes on UNSW-NB15 but fails on ToN-IoT because it does not beat CySecBERT.

### Slide 14: Reasons

Class imbalance, weak structural signal on ToN-IoT, difficult minority classes, possible fusion limitation.

### Slide 15: Next Steps

Per-class diagnosis, gate analysis, stronger fusion, imbalance handling, temporal features.

### Slide 16: Conclusion

The pipeline is working and validated. The research question now is how to make fusion robust across datasets.

---

## 20. Short Closing Statement

Our work shows that graph structure and cybersecurity semantics can be combined for intrusion detection. The pipeline is validated on two datasets, and fusion successfully improves over both baselines on UNSW-NB15. On ToN-IoT, fusion improves over the GNN but does not yet beat CySecBERT, which reveals the next research challenge: designing a more reliable fusion strategy that can adapt to the strength of each modality and handle class imbalance more effectively.
