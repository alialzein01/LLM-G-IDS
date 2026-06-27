# Presentation Report: LLM-Enhanced Graph-Based Intrusion Detection System
**Author:** Mohamad Salman Ali  
**Supervisor:** Fouad Al Tfaily — CESI LINEACT / Lebanese University  
**Date:** June 2026  

---

## PART 1 — Context: Why Network Security Is a Growing Problem

We live in an era where billions of devices are permanently connected to the internet. IoT (Internet of Things) networks — smart home sensors, industrial controllers, medical devices, connected cameras — generate enormous volumes of network traffic every second. Alongside this growth, cyberattacks have become more frequent, more sophisticated, and more damaging.

An **Intrusion Detection System (IDS)** is a tool that monitors network traffic and raises an alarm when it detects malicious activity. The goal of our research is to build a better IDS — one that is more accurate, especially on the kinds of attacks that are rare but dangerous.

Traditional IDS approaches have evolved significantly over time:

- **Signature-based systems** (early generation): detect attacks by comparing traffic against a database of known attack patterns. They are fast and accurate for known attacks, but completely blind to anything new.
- **Machine learning methods**: learned statistical patterns from labelled traffic data using algorithms like Random Forests and Support Vector Machines. Better at generalization, but treat each flow as an independent data point — they miss relational patterns.
- **Deep learning methods** (CNN, LSTM): can capture more complex patterns but still treat flows as flat sequences or vectors, ignoring the network's communication topology.
- **Graph Neural Networks (GNNs)**: the current state of the art. Instead of treating each flow independently, they model the entire network as a graph — IP addresses as nodes, communications as edges — and learn from the structural relationships between them.

Our supervisor's own published work, **FedGATSage** (Al Tfaily et al., Scientific Reports 2025), sits at this frontier: a federated GNN-based IDS that achieves near-centralized performance while preserving device privacy. It uses Graph Attention Networks (GAT) on the client side and GraphSAGE on the server side. This paper is our direct baseline and the concrete motivation for asking: can we do even better by adding language understanding?

---

## PART 2 — The Gap: What GNNs Cannot Do Alone

GNNs are powerful precisely because they understand graph structure. They can detect a DDoS attack because it produces a hub-and-spoke topology: many source IPs all converging on one victim. A scanning attack looks like a star pattern: one source reaching out to many destinations. These structural fingerprints are exactly what a GNN learns.

**But GNNs are structurally blind to meaning.** When a GNN sees an edge, it sees numbers: byte count, duration, protocol code, port number. It does not know that port 22 is associated with remote access and therefore more suspicious in certain contexts. It does not know that a short-duration, high-frequency TCP flow to port 443 is a signature pattern for certain injection attacks. It has no concept of what these attacks mean semantically.

**Large Language Models (LLMs), on the other hand, are rich with semantic knowledge.** A model like BERT or CySecBERT has been trained on vast amounts of cybersecurity text. It knows what a DDoS is, what SQL injection means, how ransomware typically behaves. But an LLM cannot natively process a graph. It reads text, not topology.

This is the gap our work targets: **GNNs see structure but lack semantic knowledge. LLMs carry semantic knowledge but cannot see graph structure.** No prior published work had applied a bidirectional, jointly-trained GNN+LLM architecture to intrusion detection. This is the novelty we are pursuing.

---

## PART 3 — Our Proposed Solution

Our solution is a hybrid pipeline that represents the same network traffic in two complementary forms and combines both views for classification:

1. **A structural view** (for the GNN): the communication graph, where IP addresses are nodes and aggregated flows are edges. The GNN learns the topology.
2. **A semantic view** (for the LLM): each flow is described in natural language. CySecBERT reads these descriptions and produces a semantic embedding that captures the meaning of the traffic.
3. **A fusion model**: both embeddings are combined using a learned attention mechanism that decides, per edge, how much to trust the structural signal versus the semantic signal.

The key design principle is that **neither modality biases the other**. Both run in parallel. Only at the fusion stage do they meet. This means that if the GNN is uncertain (e.g., a rare attack that doesn't have a clear structural fingerprint), the LLM can compensate — and vice versa.

---

## PART 4 — Papers We Read

### Papers Assigned by the Supervisor

**1. FedGATSage — Al Tfaily et al. (Scientific Reports, 2025)**  
Our supervisor's own paper. This is the direct architectural predecessor of our work. FedGATSage proposes a federated learning architecture where clients run GAT locally and a central server aggregates via GraphSAGE. It achieves strong results on NF-ToN-IoT and CIC-ToN-IoT. We use the same datasets and the same evaluation protocol. This paper defines our GNN baseline and establishes that graph-based IDS is the right approach — our work asks whether adding an LLM can push it further.

**2. KLAGE — Belcastro et al. (Future Generation Computer Systems, 2025)**  
This paper combines Knowledge Graphs, Explainable AI (XAI), and LLMs for network threat detection. It uses Graph-BERT to encode communication patterns into a knowledge graph, then uses LIME (an explanation tool) for transparency, and GPT-4o to generate readable threat reports. Key lesson: knowledge graphs built from network logs can be directly fed into an LLM for explainable detection. Our Step 2 is directly inspired by this idea of building an internal KG from the dataset itself.

**3. KGNN — Lin et al. (IJCAI, 2020)**  
This paper applies Knowledge Graph Neural Networks to drug-drug interaction prediction in pharmacology. While the domain is completely different, the architectural principle is the same: build a KG from domain data, then use a GNN to extract both high-order structural patterns and semantic relations. It introduced us to the idea that KG + GNN is a powerful combination that transcends any single application domain.

**4. GENI — Park et al. (KDD, 2019)**  
This paper estimates node importance in knowledge graphs using GNNs. It uses predicate-aware attention (attention that depends on the type of relationship, not just the presence of a connection). This influenced our node feature design in Step 1, where we use 10 centrality measures (betweenness, PageRank, degree, closeness, etc.) as the structural signature of each IP node. These centrality measures are essentially a quantified version of "how important is this node in the communication network."

**5. LLMs for Cybersecurity — Karras et al. (Information, 2025)**  
A comprehensive systematic review of 235 papers on using LLMs in cybersecurity. This paper gave us the broader picture: LLMs improve detection accuracy and reduce response latency compared to traditional methods, but they are vulnerable to adversarial attacks, suffer from computational overhead, and lack explainability. This review confirmed that the direction of combining LLMs with domain-specific architectures (like GNNs) is well-motivated and under-explored.

### Papers Found Independently

**6. GL-Fusion — Yang et al. (arXiv, December 2024)**  
This paper proposes a deep integration of GNNs and LLMs using Structure-Aware Transformers and Graph-Text Cross-Attention. The core idea: inject GNN message-passing capabilities directly into the transformer layers of an LLM, so structural and textual information are processed simultaneously. This is architecturally more ambitious than our parallel approach — but it requires rich text on every node, which network flow graphs don't have. It confirmed that our flow serialization strategy (converting flows into NL sentences in Step 2) is the right bridge between the graph world and the LLM world.

**7. LOGIN — Qiao et al. (arXiv, 2024)**  
LOGIN proposes "LLMs-as-Consultants": during GNN training, whenever the model is uncertain about a node (high entropy prediction), it queries the LLM for guidance. The LLM either updates the node's features or triggers edge pruning. Closest existing precedent to our uncertainty-gated approach. Key limitation: the LLM consultation loop closes after training — at inference, the GNN runs alone. Our target architecture keeps both paths active at all times.

**8. A Survey of Graph Meets LLM — Li et al.**  
A broad survey organizing all GNN+LLM integration methods into three categories: LLM as an enhancer (enriching node features), LLM as a predictor (replacing the GNN), and LLM as an alignment component (bridging both). Our approach falls in the "enhancer + joint classifier" category, which the survey identifies as the most promising but least explored pattern for non-text-attributed graphs.

**9. EMT-IDNet — Jain et al. (Scientific Reports, 2026)**  
A very recent paper (accepted May 2026) proposing an explainable multimodal temporal deep learning framework for IDS in IoT environments. It combines network traffic, OS logs, and IoT telemetry using modality-specific encoders and cross-modal attention-based fusion. Directly relevant because it validates our choice of multimodal fusion for IDS. The difference is that EMT-IDNet fuses different data sources (traffic + logs + telemetry), while we fuse two different representations of the same traffic data.

**10. GreaseLM, JointLK, BiGTex, CO-EVOLVE, GLEM, GLANCE (Literature Survey)**  
These papers cover the full landscape of GNN+LLM interaction patterns, all applied to Text-Attributed Graphs (graphs where nodes have text descriptions, like academic papers or product reviews). The consistent limitation across all of them: they assume text exists natively on graph nodes. IP addresses have no natural text description — which is precisely why our Step 2 (flow serialization into NL sentences) is a necessary and novel contribution. Without Step 2, none of these architectures could be applied to network IDS.

---

## PART 5 — What We Built: Step by Step

### Step 1 — Communication Graph Construction

**What we did:**  
We took the raw CSV files from two public IoT intrusion detection datasets — NF-ToN-IoT (532 MB) and NF-UNSW-NB15 — and converted them into directed communication graphs.

Each unique IP address in the dataset became a **node**. The node is described by 10 centrality measures computed from the graph structure: betweenness, PageRank, degree, closeness, eigenvector centrality, k-core, k-truss, global betweenness, global PageRank, and modularity vitality. These measures capture how central, how well-connected, and how structurally important each IP is. For example, a victim of a DDoS attack will have extremely high in-degree centrality.

Each aggregated communication between two IPs under the same attack type became a **directed edge**. We grouped by the triple (source IP, destination IP, attack label) — so if the same source contacted the same destination 10,000 times as part of a DDoS campaign, that became one edge with aggregated statistics: flow count, total bytes, average duration, most common protocol, most common destination port. The first three of these were Z-score normalized — meaning we shift them to have mean zero and standard deviation one — so that features with very different scales (e.g., bytes vs. duration) don't dominate the training.

**The result:** NF-ToN-IoT produces a graph with 1,501 nodes and 2,127 edges across 10 attack classes. Each edge carries an attack label from 0 (Benign) to 9 (XSS).

**Why edge classification, not node classification:**  
Most graph learning papers classify nodes. We classify edges. This is because each network flow — each communication event — carries an attack label. The attacker's IP address itself is not inherently malicious (it could be spoofed, or a compromised machine). What is labelled is the specific communication between two IPs, not the IP itself.

**Outcome:** Step 1 passed validation on both datasets.

---

### Step 2 — Knowledge Graph and Semantic Text Construction

**What we did:**  
From the same aggregated edge data, we built a Knowledge Graph (KG). A Knowledge Graph is a structured representation where every relationship is named and carries meaning — not just "A is connected to B" but "A launched a DDoS attack against B with high-frequency, large-byte, TCP traffic to port 80."

We converted each numerical edge attribute into a semantic label. Instead of saying "flow_count = 4,200", we say "high connection frequency" — relative to other flows of the same attack type. This discretization is done per attack class independently, so "high traffic" for a Benign connection is judged against other Benign connections, not against DDoS traffic. This is important because different attack types operate at completely different traffic scales.

We also mapped each attack label to a descriptive relation verb: DDoS → "launched ddos against", Scanning → "initiated scan against", Benign → "communicated with", and so on.

**The critical design decision — label-free sentences:**  
The natural language sentences fed into CySecBERT deliberately do not mention the attack type. The model sees:

*"Observed traffic from source 192.168.1.1 to destination 10.0.0.2 with high connection frequency, large payloads, long duration, over TCP, to web service port 443."*

It does NOT see: *"This is a DDoS attack."*

This is essential to avoid **label leakage** — if the model reads the attack name in the input, it is simply memorizing labels, not learning to classify traffic from its behavioral properties.

**Outcome:** Step 2 passed validation on both datasets. Every edge has exactly one NL sentence. Sentences are verified to be label-free.

---

### Step 3A — GNN Path: Structural Encoder

**What we did:**  
We trained a GATv2-based edge classifier on the graph from Step 1.

**GATv2 (Graph Attention Network v2)** is a neural network designed specifically for graph data. Unlike a regular neural network that treats each data point independently, GATv2 lets each node "listen" to its neighbors and weight their influence. The key idea is **attention**: not all neighbors are equally important. A victim node in a DDoS attack is connected to many sources — the GATv2 learns to give higher attention scores to the suspicious sources and lower scores to the normal ones. The "v2" improvement over the original GAT is that attention scores depend on both the sender and the receiver, not just the sender — which matters because being a victim is as structurally meaningful as being an attacker.

We used two GATv2 layers (to capture two-hop neighborhoods), with 4 attention heads in the first layer. For each edge, we concatenated the source node embedding, the destination node embedding, and the 5 edge attributes, and passed this into an MLP classifier.

**Loss function — why Focal Loss instead of Cross-Entropy Loss:**  
Standard **Cross-Entropy Loss** is the default classification loss. It measures how wrong the model's predictions are. The problem: in an imbalanced dataset, the model quickly learns to always predict "Benign" (which is 95%+ of traffic) because that minimizes the average loss. The rare attack classes — ransomware has fewer than 10 samples — contribute almost nothing to the total loss, so the model ignores them.

**Focal Loss** solves this by adding a modulating factor that down-weights easy examples (predictions the model is already confident about) and focuses the gradient on hard, misclassified examples. In practice, this means the model is forced to pay attention to rare attack classes even when they contribute few training samples.

We also use **class-weighted loss**: each class receives a weight inversely proportional to its frequency. If ransomware appears 50 times and Benign appears 5,000 times, ransomware gets a 100× higher weight. Focal Loss on top of class weights gives us two layers of protection against the imbalance problem.

We trained using **5-fold cross-validation**: the data is split into 5 equal portions, and the model is trained 5 times, each time using a different portion as the held-out test set. This gives more reliable performance estimates and prevents overfitting on a single lucky/unlucky split.

**Primary metric — Macro-F1:**  
Accuracy is misleading for imbalanced datasets. A model that always predicts "Benign" achieves 95% accuracy while completely failing at intrusion detection. **Macro-F1** averages the F1 score across all 10 classes equally. F1 combines precision (of what you flagged as ransomware, how many were actually ransomware?) and recall (of all actual ransomware, how many did you catch?). If you miss ransomware entirely, that class gets F1=0, which drags the macro average down regardless of how well you perform on Benign. This makes Macro-F1 the honest metric for intrusion detection.

**Results:**
- NF-ToN-IoT: pooled Macro-F1 = **0.3006**
- NF-UNSW-NB15: pooled Macro-F1 = **0.5229**

The GNN performs noticeably better on UNSW-NB15, likely because it has better class balance and more structurally distinct attack patterns.

---

### Step 3B — LLM Path: Semantic Encoder

**What we did:**  
We encoded each of the 2,127 NL sentences from Step 2 using **CySecBERT** (markusbayer/CySecBERT), a BERT-based language model pre-trained on cybersecurity-specific text corpora. CySecBERT takes a sentence and produces a 768-dimensional dense vector — a semantic embedding that captures the meaning of the text in a high-dimensional space.

We used CySecBERT as a **frozen feature extractor**: we did not fine-tune the model's weights. We simply passed each sentence through it once, saved the 768-dimensional output vector for each edge, and then trained a small classifier (projection layers + MLP) on top of these frozen embeddings. The reason for freezing: with only 2,127 training samples and 66 million parameters in the language model backbone, fine-tuning would lead to catastrophic overfitting. The pre-trained knowledge is used as-is.

**Why CySecBERT over general BERT:**  
CySecBERT was trained on cybersecurity-domain text, meaning it already understands terms like "TCP session", "ephemeral port", "high-frequency connection" in a cybersecurity context. A general BERT model would need to infer this context from the training data alone, which is insufficient at our dataset scale.

**Results:**
- NF-ToN-IoT: pooled Macro-F1 = **0.3979**
- NF-UNSW-NB15: pooled Macro-F1 = **0.5165**

On ToN-IoT, the LLM path outperforms the GNN path (0.3979 vs 0.3006), suggesting that semantic descriptions of traffic behavior carry more discriminative signal than structural topology for this dataset. On UNSW-NB15, both paths are comparable.

---

### Step 3C — Fusion: Combining Both Worlds

**What we did:**  
The fusion model takes both the 64-dimensional GNN edge embedding and the 768-dimensional CySecBERT semantic embedding for each edge, and combines them into a single classification.

We use **AGAF — Adaptive Gated Attention Fusion**. AGAF first projects both embeddings into the same dimension (128-dim). Then it computes a **gate**: a learned vector that acts as a mixing coefficient. For each edge, the gate decides "how much structural information to use vs. how much semantic information to use." This gate is computed from the combination of both embeddings, considering their difference and their product, making it highly expressive. The final fused representation is then passed through a feature-level attention layer and an MLP classifier.

The fusion model's purpose is not merely to combine features — it must demonstrate that combining both modalities is better than using either alone. We validated this by comparing fusion against both standalone baselines.

**Results:**
- NF-ToN-IoT: GNN alone 0.3006 | CySecBERT alone 0.3979 | **Fusion 0.3865**
- NF-UNSW-NB15: GNN alone 0.5229 | CySecBERT alone 0.5165 | **Fusion 0.5705**

On UNSW-NB15, fusion beats both baselines — this is the expected outcome and validates the approach. On ToN-IoT, fusion beats the GNN but falls short of CySecBERT alone (0.3865 < 0.3979). This is the central open problem.

---

## PART 6 — Problems We Are Facing and Why

### Problem 1 — Class Imbalance (Both Datasets)

The most fundamental problem in both datasets is the severe imbalance between attack classes. In NF-ToN-IoT, Benign traffic accounts for over 90% of all edges. The rarest attack class (ransomware) has fewer than 10 training examples per fold in a 5-fold split. No model can reliably learn a new class from fewer than 10 examples.

We partially address this with Focal Loss and class weights (explained above), and with SMOTE-style oversampling at the embedding level (creating synthetic minority-class examples by interpolating between real ones). But the fundamental data problem remains a challenge, especially for ransomware and MITM attacks.

### Problem 2 — Small Graph Size

After aggregating millions of raw flows into (src_ip, dst_ip, attack) groups, the NF-ToN-IoT graph has only 2,127 edges total. This is the entire dataset. Splitting into 5 folds means roughly 1,700 training edges per fold. Training deep neural networks on 1,700 examples is inherently difficult and prone to variance. This is why we deliberately kept model architectures small (hidden dimension 64 for GNN, 128 for fusion) and used regularization (dropout, weight decay, early stopping).

### Problem 3 — Fusion Failing to Outperform CySecBERT on ToN-IoT

The most pressing current research problem: on NF-ToN-IoT, the fusion model cannot beat CySecBERT alone. This suggests that when integrating two modalities, the GNN contribution is hurting rather than helping — the structural signal adds noise rather than complementary information.

**Why this happens:** The GNN embedding (64 dimensions, trained on a tiny graph) and the CySecBERT embedding (768 dimensions, derived from a pre-trained LLM) operate at very different scales and quality levels. Despite standard scaling, the fusion model may be over-relying on the higher-quality LLM embeddings and treating the GNN signal as noise. On UNSW-NB15, the structural patterns are more distinct (the graph is larger and better balanced), so the GNN contribution is meaningful. On ToN-IoT, the structural signal is weaker.

**What we are investigating:** Improving the fusion architecture or training strategy so that the GNN contribution is weighted more carefully — for example, by increasing the GNN's expressiveness, by entropy-regularizing the gate to prevent it from collapsing to one modality, or by augmenting the GNN training with additional supervision.

---

## PART 7 — Summary of Results and Current Status

| Stage | NF-ToN-IoT | NF-UNSW-NB15 |
|---|---|---|
| Phase 1 — Graph Construction | ✅ Passed | ✅ Passed |
| Phase 2 — Knowledge Graph | ✅ Passed | ✅ Passed |
| Phase 3 — GNN Baseline | ✅ Macro-F1: 0.3006 | ✅ Macro-F1: 0.5229 |
| Phase 3 — LLM Baseline | ✅ Macro-F1: 0.3979 | ✅ Macro-F1: 0.5165 |
| Phase 4 — Fusion | ⚠️ 0.3865 (below LLM) | ✅ 0.5705 (beats both) |

The pipeline is architecturally sound and validated end-to-end. On UNSW-NB15, the full hypothesis is confirmed: fusion outperforms both individual modalities. On ToN-IoT, the remaining challenge is improving the fusion so the GNN contribution is additive rather than dilutive.

---

## PART 8 — Next Steps

The immediate research priority is improving fusion performance on NF-ToN-IoT. Several directions are being explored:

1. **Improving the GNN path on ToN-IoT**: higher-quality structural embeddings are a prerequisite for useful fusion. This may involve graph augmentation, different aggregation strategies, or incorporating temporal information.
2. **Entropy-regularized gate training**: penalizing the fusion gate for collapsing to a single modality — forcing it to use both signals meaningfully.
3. **Exploring the bidirectional feedback loop (Step 4)**: the original proposal includes a Phase 2 where the GNN and LLM interact iteratively — the GNN identifies uncertain flows, routes them to the LLM for semantic re-evaluation, and the LLM's output biases the GNN's next pass. This has not yet been implemented. It is the architectural step that moves from "parallel combination" to "mutual refinement."
4. **Generalization to other datasets**: testing on CIC-IDS-2017/2018 to validate that the approach generalizes beyond NF-formatted datasets.
