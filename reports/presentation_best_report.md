# LLM-Enhanced Graph-Based Intrusion Detection

**Base report for the PowerPoint presentation**  
**Author:** Mohamad Salman Ali  
**Supervisor:** Fouad Al Tfaily, CESI LINEACT / Lebanese University  
**Date:** June 2026

---

## 1. Purpose of the Presentation

The purpose of this presentation is to explain the work completed so far as a coherent research story, not as a coding update. The presentation should answer:

- What is the context?
- What is the problem?
- Why does the problem matter?
- What gap are we trying to address?
- What solution did we implement?
- What papers influenced the work?
- What results did we obtain?
- What problems are we facing, and why?
- What are the next steps?

The audience may include people who are not specialized in machine learning, graph neural networks, or cybersecurity. Therefore, technical concepts such as GNN, CySecBERT, Cross-Entropy, Focal Loss, Macro-F1, attention, and fusion should be explained briefly when introduced.

The target presentation duration is around 20 minutes.

---

## 2. Context: Why Intrusion Detection Matters

IoT networks are made of many connected devices, such as sensors, cameras, gateways, smart appliances, and industrial controllers. These devices continuously exchange traffic and often have limited security resources. This makes IoT networks attractive targets for attacks such as denial of service, scanning, password attacks, injection attacks, man-in-the-middle attacks, ransomware, and backdoors.

An Intrusion Detection System, or IDS, monitors network traffic and tries to identify malicious activity. In our project, the goal is not only to decide whether traffic is benign or malicious, but also to classify the attack type.

We work with two NetFlow-style intrusion detection datasets:

- **NF-ToN-IoT**
- **NF-UNSW-NB15**

NetFlow data summarizes communication flows rather than storing raw packets. Each record describes communication between two endpoints using features such as source IP, destination IP, protocol, port, byte count, duration, and attack label.

---

## 3. From Traditional IDS to Graph-Based IDS

Traditional IDS approaches have evolved over time.

**Signature-based IDS** detects attacks by matching traffic against known attack patterns. It is fast for known attacks, but weak against new or modified attacks.

**Classical machine learning IDS** uses models such as Random Forests or Support Vector Machines to learn patterns from labeled traffic. These methods are stronger than signatures, but they often treat each flow as an independent row.

**Deep learning IDS** uses models such as CNNs, LSTMs, or Transformers to capture more complex patterns. However, many of these approaches still treat traffic as flat vectors or sequences.

**Graph-based IDS** models the network as a graph. IP addresses become nodes, and communications become edges. This is useful because many attacks are relational. A DDoS attack may appear as many sources targeting one destination. A scanning attack may appear as one source contacting many destinations or ports.

This motivates the graph-based part of our work.

---

## 4. Problem Statement

Many IDS methods treat each flow independently. This misses the structure of the network.

At the same time, graph neural networks can learn structure but do not naturally understand cybersecurity meaning. A GNN can learn that an edge has a high byte count or that a node has many neighbors, but it does not inherently know the meaning of protocol names, service ports, or attack behavior descriptions.

Language models have the opposite strength. A cybersecurity language model can encode semantic information from text, but it cannot naturally process graph topology.

This leads to our main research question:

**Can we combine graph structure and cybersecurity semantics to improve edge-level intrusion detection?**

The classification target in our project is the communication edge, not the IP node. Each edge represents an aggregated communication pattern between a source IP and a destination IP, and the model predicts the attack class of that edge.

---

## 5. Motivation

The project is motivated by the complementary strengths of GNNs and cybersecurity language models.

- The **GNN** learns from network structure: who communicates with whom, direction of communication, node centrality, and edge traffic attributes.
- **CySecBERT** learns from semantic descriptions: connection frequency, byte volume, duration, protocol, and port meaning expressed as text.
- The **fusion model** tries to combine both views into one final prediction.

The intuition is simple:

**The GNN sees the structure. CySecBERT sees the behavior description. Fusion tries to use both.**

If both views contain complementary information, fusion should perform better than either view alone.

---

## 6. Research Gap

The literature contains several related directions, but none directly solves our exact setting.

Some papers use GNNs for IDS, but they mainly focus on graph structure. Some papers use knowledge graphs and LLMs for cybersecurity, but the LLM is often used after classification to generate explanations or reports. Some papers combine GNNs and LLMs, but many are designed for text-attributed graphs, such as citation networks where every node already has text.

Network flow graphs are different. IP addresses do not naturally come with text descriptions. Network traffic is mostly numerical and categorical: bytes, duration, protocol, ports, and direction. Therefore, we need to create a semantic text representation ourselves.

Our current implemented contribution is:

**An edge-level IDS pipeline that builds both a structural graph view and a label-free semantic text view, then tests whether adaptive fusion improves over both standalone baselines.**

The broader research direction is to move from parallel fusion toward stronger GNN and LLM interaction, but the current validated system is the parallel/fusion version.

---

## 7. Main Solution Overview

The pipeline has three main technical stages.

**Step 1: Communication graph construction**

Raw NetFlow records are converted into a directed graph. IP addresses are nodes, and aggregated communications are edges.

**Step 2: Knowledge-graph-style semantic text construction**

Each edge is converted into a structured KG-style row and a label-free natural-language sentence. The sentence describes traffic behavior without revealing the attack label.

**Step 3: Model training**

Three paths are trained and compared:

- GNN structural path
- CySecBERT semantic path
- Fusion path combining both embeddings

The fusion model is only considered successful when it beats both standalone baselines.

---

## 8. Step 1: Communication Graph Construction

In Step 1, raw network flow records are converted into a directed communication graph.

Each unique IP address becomes a node.

Each aggregated flow becomes a directed edge from source IP to destination IP. Edges are grouped by:

```text
source IP, destination IP, attack label
```

This means that if the same source contacted the same destination many times under the same attack type, those flows become one aggregated edge.

Each edge stores:

```text
flow_count
total_bytes
avg_duration
most_common_protocol
most_common_port
attack label
```

The first three numerical attributes are normalized using Z-score normalization. Z-score normalization means subtracting the mean and dividing by the standard deviation, so features with different scales become easier to compare. Protocol and port are kept as categorical integer-style features.

The node features are 10 centrality measures, such as PageRank, degree, closeness, betweenness, eigenvector centrality, k-core, k-truss, global PageRank, global betweenness, and modularity vitality. These features describe how structurally important each IP is in the communication graph.

The task is **edge classification**, not node classification. This matters because the attack label belongs to the communication behavior between two IPs, not necessarily to the IP address itself.

### Step 1 Results

The processed graph statistics are:

```text
NF-ToN-IoT:    1,501 nodes, 2,127 edges
NF-UNSW-NB15:     49 nodes,   656 edges
```

Step 1 validation passed for both datasets:

```text
ToN-IoT:    passed
UNSW-NB15:  passed
```

---

## 9. Step 2: Knowledge Graph and Semantic Text Construction

In Step 2, each graph edge becomes one knowledge-graph-style row and one natural-language sentence.

The numerical attributes are discretized into:

```text
low / medium / high
```

This is done for:

```text
flow_count
avg_bytes
avg_duration
```

The discretization is performed separately inside each attack class. This avoids comparing all attack types using one global scale. For example, "high" duration for one attack class may not mean the same thing as "high" duration for another class.

The structured KG file includes semantic relation names, for example:

```text
Benign -> communicated with
ddos -> launched ddos against
scanning -> initiated scan against
Reconnaissance -> initiated reconnaissance against
```

However, the natural-language sentences sent into CySecBERT are intentionally **label-free**.

This is a critical design decision. If we include attack names or attack-specific relation verbs in the sentence, the language model would see the answer during training. This is called **label leakage**. Label leakage makes the evaluation invalid because the model is no longer learning from behavior; it is reading the label from the input.

The CySecBERT input looks like:

```text
Observed traffic from source X to destination Y with high connection frequency,
large payloads, long duration, over TCP, to web service port 443.
```

It does not look like:

```text
This is DDoS traffic.
```

### Step 2 Results

Step 2 validation checks that:

- Every aggregated edge has one KG row.
- KG rows align with graph edges by row index.
- Discretized values are only low, medium, or high.
- Semantic relation mapping covers all labels.
- Natural-language sentences are label-free.
- Sentence count equals edge count.

Step 2 passed for both datasets:

```text
ToN-IoT:    passed
UNSW-NB15:  passed
```

---

## 10. Step 3A: GNN Structural Path

The GNN path uses the communication graph from Step 1.

The model is a GATv2-based edge classifier. **GATv2**, or Graph Attention Network v2, is a graph neural network that learns which neighboring nodes and edges deserve more importance during message passing. Attention means the model does not treat every neighbor equally.

The GNN first learns node embeddings using graph structure and edge attributes. An embedding is a learned numerical representation. Then, for each edge, the model builds an edge representation by combining:

```text
source node embedding
destination node embedding
edge attributes
```

This edge representation is passed to an MLP classifier. An MLP, or multilayer perceptron, is a standard neural network classifier.

The GNN learns from:

- IP-to-IP communication topology
- node centrality features
- edge traffic statistics
- communication direction
- edge-level labels

The GNN edge embedding shape is:

```text
[number_of_edges, 64]
```

Training uses shared 5-fold splits and out-of-fold predictions. This means each edge is tested in a fold where it was not used for training.

### GNN Results

```text
NF-ToN-IoT:
GATv2 pooled Macro-F1 = 0.3006

NF-UNSW-NB15:
GATv2 pooled Macro-F1 = 0.5229
```

The GNN performs better on UNSW-NB15 than on ToN-IoT.

---

## 11. Step 3B: CySecBERT Semantic Path

The semantic path uses the label-free natural-language sentences from Step 2.

Each sentence is encoded using:

```text
markusbayer/CySecBERT
```

CySecBERT is a BERT-style language model trained for cybersecurity text. In our pipeline, it is used as a frozen feature extractor. This means CySecBERT converts each sentence into an embedding, and then a smaller classifier learns from those embeddings. We do not fine-tune the full language model.

Freezing CySecBERT is a practical choice because the dataset is small compared with the number of parameters in a language model. Fine-tuning the full model could overfit.

The semantic embedding shape is:

```text
[number_of_edges, 768]
```

The semantic path learns from:

- natural-language traffic descriptions
- discretized connection frequency
- discretized byte volume
- discretized duration
- protocol and port semantics

The same train, validation, and test folds are used as in the GNN path, so the comparison is fair.

### CySecBERT Results

```text
NF-ToN-IoT:
CySecBERT pooled Macro-F1 = 0.3979

NF-UNSW-NB15:
CySecBERT pooled Macro-F1 = 0.5165
```

On ToN-IoT, CySecBERT is stronger than the GNN.

On UNSW-NB15, CySecBERT is close to the GNN, but slightly weaker.

---

## 12. Step 3C: Fusion Path

The fusion path combines both modalities:

```text
GNN structural embedding:      64 dimensions
CySecBERT semantic embedding:  768 dimensions
```

Because the two embeddings have different sizes, both are projected into a shared 128-dimensional space.

The model uses **AGAF: Adaptive Gated Attention Fusion**.

A gate is a learned mechanism that decides how much information to take from each input. In our case, the fusion model learns how much to rely on the GNN representation and how much to rely on the CySecBERT representation for each edge.

Conceptually:

```text
fused representation =
    gate for GNN * GNN representation
  + gate for LLM * CySecBERT representation
```

Then feature-level attention is applied before classification. Feature-level attention lets the model emphasize the most useful dimensions of the fused representation.

The purpose of fusion is not only to combine features. It must prove that the combined model is better than both individual models.

### Fusion Results

For **NF-ToN-IoT**:

```text
GATv2 Only:      Macro-F1 = 0.3006
CySecBERT Only:  Macro-F1 = 0.3979
Fusion Model:    Macro-F1 = 0.3865
```

Fusion improves over GNN:

```text
0.3865 > 0.3006
```

But fusion does not improve over CySecBERT:

```text
0.3865 < 0.3979
```

Therefore, Phase 4 fails for ToN-IoT.

For **NF-UNSW-NB15**:

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

---

## 13. Results and Current Status

| Stage | NF-ToN-IoT | NF-UNSW-NB15 |
|---|---:|---:|
| Phase 1: Graph Construction | Passed | Passed |
| Phase 2: Knowledge Graph / Semantic Text | Passed | Passed |
| Phase 3: GNN Baseline | 0.3006 Macro-F1 | 0.5229 Macro-F1 |
| Phase 3: CySecBERT Baseline | 0.3979 Macro-F1 | 0.5165 Macro-F1 |
| Phase 4: Fusion | 0.3865, below CySecBERT | 0.5705, beats both |

The pipeline is working end to end. The key finding is:

- Fusion works as expected on UNSW-NB15.
- Fusion improves over the GNN on ToN-IoT, but does not beat CySecBERT.

This gives us a clear research problem for the next stage: improving fusion robustness on ToN-IoT.

---

## 14. Important Concepts Explained Simply

### Macro-F1

Macro-F1 calculates the F1-score for each class separately, then averages all classes equally. It is useful for imbalanced datasets because rare attacks matter as much as common classes in the final score.

Short presentation sentence:

**We use Macro-F1 because every attack class should matter equally, even if some attacks appear much less often than others.**

### Cross-Entropy Loss

Cross-Entropy is the standard loss function for multiclass classification. It penalizes the model when it gives low probability to the correct class.

Short presentation sentence:

**Cross-Entropy teaches the model to assign high probability to the correct attack class.**

### Focal Loss

Focal Loss modifies Cross-Entropy so the model focuses more on hard or misclassified examples. This is useful when common classes are easy and rare classes are being ignored.

Short presentation sentence:

**We use Focal Loss because the datasets are imbalanced, and it forces the model to pay more attention to difficult and minority-class examples.**

### Class Weights

Class weights increase the penalty for mistakes on rare classes.

Short presentation sentence:

**Class weights make mistakes on rare attack classes more costly during training.**

### Attention

Attention is a mechanism that lets the model learn which inputs are more important.

Short presentation sentence:

**Attention helps the model decide which neighbors, features, or modalities deserve more importance for a specific prediction.**

### Fusion

Fusion means combining multiple sources of information. In our case, it combines graph structure from the GNN with semantic traffic descriptions from CySecBERT.

Short presentation sentence:

**Fusion combines the structural view and the semantic view so the final classifier can use both kinds of evidence.**

---

## 15. Papers Read and How They Influenced the Work

### FedGATSage

FedGATSage is directly related because it applies graph-based learning to IoT intrusion detection. It supports the idea that network topology is important for detecting coordinated attacks. It also motivates the use of attention-based graph models.

Influence on our work:

- Reinforced the use of graph-based IDS.
- Motivated attention-based GNN modeling.
- Supported the idea that coordinated attacks require structural analysis.

### KLAGE

KLAGE combines knowledge graphs, Graph-BERT, explainability, and LLM-generated reports for network threat detection. It shows that KGs and LLMs are useful for cybersecurity explanation.

Influence on our work:

- Motivated KG-style representation of network logs.
- Supported the use of LLMs for semantic interpretation.
- Helped distinguish our work: we use semantic embeddings during classification, not only after classification for reporting.

### KGNN

KGNN is from drug-drug interaction prediction, not IDS, but it shows how knowledge graphs and GNNs can learn multi-hop relational patterns.

Influence on our work:

- Helped frame the value of KG and GNN reasoning.
- Showed that graph neighborhoods can contain useful semantic and structural information.
- Highlighted that graph expansion can introduce noise if not controlled.

### GENI

GENI estimates node importance in knowledge graphs using GNNs and attention. It is useful for thinking about centrality and node importance.

Influence on our work:

- Supported the use of centrality-based node features.
- Reinforced that node importance can improve graph representation.

### LOGIN

LOGIN uses an LLM as a consultant for uncertain GNN predictions during training. It is important because it shows a more interactive relationship between GNNs and LLMs.

Influence on our work:

- Motivated future work on uncertainty-based GNN and LLM interaction.
- Showed that selective LLM use can be more efficient than querying an LLM for every sample.

### GL-Fusion

GL-Fusion studies how to combine GNNs and LLMs. It is relevant because it shows that the fusion mechanism itself matters.

Influence on our work:

- Reinforced the need to evaluate fusion carefully.
- Supported the idea that structural and semantic embeddings can complement each other.
- Helped frame the current challenge: fusion must beat both baselines, not only the weaker one.

### Graph Meets LLM Survey

This survey organizes the broader research area of graphs and LLMs.

Influence on our work:

- Helped position the project in the GNN and LLM landscape.
- Clarified that many existing methods assume text-attributed graphs.
- Supported our need to create text from numerical network flows.

### EMT-IDNet

EMT-IDNet is an explainable multimodal IDS framework for IoT environments. It uses attention-based fusion across multiple data sources.

Influence on our work:

- Reinforced the value of multimodal IDS.
- Supported the use of attention-based fusion.
- Showed that attention weights can contribute to explainability.

### LLMs for Cybersecurity Review

This review summarizes LLM applications and challenges in cybersecurity.

Influence on our work:

- Confirmed that LLMs are promising for cybersecurity.
- Highlighted risks such as hallucination, computational cost, and security issues.
- Supported our controlled use of CySecBERT embeddings instead of relying on free-form LLM generation.

---

## 16. Problems We Are Facing

The main current problem is not that the code fails. The pipeline runs and validates. The problem is methodological:

**Fusion does not yet outperform the best standalone model on both datasets.**

On UNSW-NB15, fusion beats both baselines:

```text
Fusion = 0.5705
GNN    = 0.5229
LLM    = 0.5165
```

On ToN-IoT, fusion beats the GNN but not CySecBERT:

```text
Fusion = 0.3865
GNN    = 0.3006
LLM    = 0.3979
```

This means the fusion model is learning something useful from structure, but the structural signal is not yet strong enough, or not being weighted carefully enough, to improve over the semantic branch.

---

## 17. Why These Problems Happen

### 1. Class imbalance

ToN-IoT is highly imbalanced after aggregation:

```text
Benign:      1,746 edges, 82.1%
dos:             4 edges, 0.2%
ransomware:      3 edges, 0.1%
scanning:       13 edges, 0.6%
xss:            12 edges, 0.6%
```

This makes Macro-F1 difficult because rare classes count equally in the final average. Missing a rare class can strongly reduce the score.

UNSW-NB15 is more balanced after preprocessing:

```text
Normal: 311 edges, 47.4%
Most attack classes: around 40 edges each
```

This helps explain why fusion is more successful on UNSW-NB15.

### 2. Small graph size after aggregation

The ToN-IoT graph has 2,127 aggregated edges. With 5-fold cross-validation, each fold has a limited number of training examples. This makes deep learning difficult, especially for rare classes.

### 3. ToN-IoT has a weaker structural branch

On ToN-IoT, the GNN baseline is much lower than CySecBERT:

```text
GNN = 0.3006
LLM = 0.3979
```

This suggests that the semantic descriptions are more discriminative than the graph structure for this dataset. If the GNN embedding is weak, fusion may add noise unless the gate learns to rely more on CySecBERT.

### 4. Some attack classes are behaviorally similar

Some classes may share similar traffic behavior. For example, DoS and DDoS can both involve high-volume traffic. Injection and XSS may also be hard to distinguish from simple flow attributes. This makes classification harder.

### 5. Label-free text is correct but harder

Removing attack names from CySecBERT input avoids label leakage, which is necessary for a valid experiment. However, it also makes the task harder because the model must infer the attack type from behavior descriptions only.

### 6. Aggregation may remove temporal detail

Aggregating flows into edges makes the graph smaller and easier to train, but it can hide temporal patterns. Some attacks are defined not only by who communicates with whom, but also by how behavior changes over time.

---

## 18. Next Steps

The next research priority is improving fusion on ToN-IoT.

Possible directions:

- Analyze per-class errors to identify which classes reduce Macro-F1 most.
- Inspect fusion gate weights to see whether the model relies too much on the weaker GNN branch.
- Improve the GNN branch so structural embeddings become more useful.
- Improve fusion so it can fall back to the stronger modality when one branch is unreliable.
- Experiment with confidence-aware or class-specific fusion gates.
- Strengthen imbalance handling for rare classes.
- Add temporal features or temporal aggregation if ToN-IoT requires time-based signals.
- Improve semantic sentence construction while keeping it label-free.
- Later, explore a stronger GNN-LLM feedback loop where uncertain edges are selectively routed for semantic refinement.

---

## 19. Suggested PowerPoint Flow

1. Title and research objective
2. IoT security context
3. Why traditional IDS misses network structure
4. Why GNNs help, and what they still miss
5. Why CySecBERT helps, and what it still misses
6. Research gap
7. Proposed pipeline overview
8. Step 1: communication graph construction
9. Step 2: label-free semantic text construction
10. Step 3: GNN and CySecBERT baselines
11. Fusion model
12. Papers read and how they shaped the work
13. Results on ToN-IoT and UNSW-NB15
14. Current problem: ToN-IoT fusion does not beat CySecBERT
15. Reasons: imbalance, weak structural signal, small graph, similar classes
16. Next steps and conclusion

---

## 20. Closing Statement

The project has reached a meaningful validation point. We successfully built and validated an end-to-end pipeline that converts network flows into a graph, converts graph edges into label-free semantic text, trains GNN and CySecBERT baselines, and evaluates adaptive fusion.

The results show that fusion can improve intrusion detection when structural and semantic signals are complementary, as seen on UNSW-NB15. On ToN-IoT, fusion improves over the GNN but does not yet beat CySecBERT, mainly because the dataset is highly imbalanced and the structural branch is weaker. The next research challenge is to make fusion more reliable across datasets, especially when one modality is stronger than the other.
