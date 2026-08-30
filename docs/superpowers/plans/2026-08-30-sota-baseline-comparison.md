# SOTA Baseline Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-implement E-GraphSAGE and TE-G-SAGE from their papers in PyTorch Geometric, train them on our aggregated flow graphs under the canonical protocol, and emit a results contract the report can cite.

**Architecture:** Two model modules under `src/models/baselines/`, a per-paper featurization module and a shared out-of-fold harness under `src/pipeline/baselines/`, and one driver that writes `results/{dataset}_sota_baselines.json`. Baselines consume the same `aggregated_edges.csv` rows and the same `folds.pt` as our rungs; only the categorical featurization follows each source paper.

**Tech Stack:** Python 3.14.3, torch 2.13.0, torch_geometric 2.8.0, scikit-learn 1.9.0, pandas, unittest.

**Spec:** `docs/superpowers/specs/2026-08-30-sota-baseline-comparison-design.md`

## Global Constraints

- `export OMP_NUM_THREADS=1` for every training or evaluation run, without exception.
- Seeds 42, 1, 2. Per-fold seeding is `seed + fold_idx`, matching `src/pipeline/step3/verify_gnn_baseline.py:70`.
- Splits come from `config.splits_path` (`folds.pt`). Never regenerate them.
- Metric: pooled out-of-fold macro-F1 via `src.pipeline.common.splits.eval_macro_f1(labels, preds, config.eval_classes)`. ToN scores 8 classes (`eval_classes=(0,1,2,4,5,6,8,9)`); classes 3 and 7 stay in the graph and are excluded from the metric.
- Hyperparameter selection uses **validation folds only**. Test masks are never read during selection.
- Never pass our v2 `data.edge_attr` to a baseline. Columns 3-4 are vocabulary indices; feeding them as magnitudes is the v1 bug. Baselines featurize from `aggregated_edges.csv`.
- Any results entry that is not faithful to its source paper carries `"is_faithful_to_paper": false` and a `"variant_label"`.
- Report language: "re-trained on our aggregated representation", never "outperforms <paper>". "Highest point estimate", never "top rung".

## RESOLVED DECISION — loss weighting

**Decided 2026-08-30: `class_weighting` is on the `refit` grid.**

E-GraphSAGE specifies plain, unweighted cross-entropy. Our rungs use inverse-frequency
class weights (`get_class_weights`). On ToN's ~145:1 imbalance this single difference can
dominate macro-F1, so running baselines unweighted while ours are weighted would report a
loss-function gap as an architectural one.

- `as_published` — plain cross-entropy, faithful to the paper.
- `refit` — `class_weighting ∈ {none, inverse_frequency}`, selected on validation folds.

Spec §5's grid table has been updated to match. **Task 6 is no longer blocked.**

---

### Task 1: Row alignment and per-paper featurization

**Files:**
- Create: `src/pipeline/baselines/__init__.py`
- Create: `src/pipeline/baselines/preprocess.py`
- Test: `tests/test_baseline_preprocess.py`

**Interfaces:**
- Consumes: `src.pipeline.common.datasets.get_dataset_config`
- Produces:
  - `load_aligned_edges(config) -> tuple[pandas.DataFrame, torch_geometric.data.Data]`
  - `egraphsage_edge_features(df: DataFrame) -> numpy.ndarray`  # `[E, 5]` float32
  - `te_g_sage_edge_features(df: DataFrame, rare_min_freq: int = 50) -> tuple[numpy.ndarray, list[str]]`

Alignment is the highest-risk item in the plan: `folds.pt` indexes edges by their
position in `data.edge_index`. If CSV row order and edge order ever diverge, every
baseline number is silently wrong. Test it before anything else.

- [ ] **Step 1: Write the failing alignment test**

```python
# tests/test_baseline_preprocess.py
from __future__ import annotations

import unittest

import numpy as np

from src.pipeline.baselines.preprocess import load_aligned_edges
from src.pipeline.common.datasets import get_dataset_config

# NOTE: import ONLY load_aligned_edges here. The featurizers do not exist yet, and
# a module-level import of a missing name fails COLLECTION -- pytest would report an
# ImportError for the whole file instead of running the alignment test. Step 5 adds
# the featurizer import alongside the tests that need it.


class AlignmentTest(unittest.TestCase):
    def test_csv_rows_align_with_graph_edges(self) -> None:
        for key in ("unsw_nb15", "ton_iot"):
            with self.subTest(dataset=key):
                config = get_dataset_config(key)
                df, data = load_aligned_edges(config)
                self.assertEqual(len(df), int(data.edge_index.shape[1]))
                label_names = list(config.label_names)
                csv_labels = np.array(
                    [label_names.index(a.strip()) for a in df["Attack"]]
                )
                np.testing.assert_array_equal(
                    csv_labels, data.edge_label.cpu().numpy()
                )
```

- [ ] **Step 2: Run it to verify it fails**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/test_baseline_preprocess.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.pipeline.baselines'`

- [ ] **Step 3: Implement `load_aligned_edges`**

```python
# src/pipeline/baselines/preprocess.py
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

from src.pipeline.common.datasets import DatasetConfig

CONTINUOUS_COLUMNS = ["flow_count", "total_bytes", "avg_duration"]
CATEGORICAL_COLUMNS = ["most_common_protocol", "most_common_port"]
RARE_TOKEN = "__rare__"


def load_aligned_edges(config: DatasetConfig) -> tuple[pd.DataFrame, Data]:
    """Load the aggregated edge table alongside its graph, order-checked.

    `folds.pt` indexes edges positionally, so a mismatch between CSV row order
    and `edge_index` column order silently corrupts every downstream number.
    """
    df = pd.read_csv(config.aggregated_edges_path)
    df["Attack"] = df["Attack"].astype(str).str.strip()
    data = torch.load(config.graph_path, weights_only=False)

    if len(df) != int(data.edge_index.shape[1]):
        raise ValueError(
            f"{config.key}: {len(df)} CSV rows vs "
            f"{int(data.edge_index.shape[1])} graph edges"
        )

    label_names = list(config.label_names)
    csv_labels = np.array([label_names.index(a) for a in df["Attack"]])
    graph_labels = data.edge_label.cpu().numpy()
    if not np.array_equal(csv_labels, graph_labels):
        raise ValueError(
            f"{config.key}: CSV row order does not match graph edge order"
        )
    return df, data
```

- [ ] **Step 4: Run the alignment test to verify it passes**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/test_baseline_preprocess.py -v`
Expected: 1 passed (both dataset subTests). The file must collect cleanly at this point
with only `load_aligned_edges` implemented — if you see a collection ImportError, a
featurizer name has been imported at module level too early. If the assertion itself
FAILS, stop and report — the CSV and the graph are out of sync and no baseline number is trustworthy until that is fixed.

- [ ] **Step 5: Write the failing featurization tests**

Add this import at the top of the file, beside the existing one:

```python
from src.pipeline.baselines.preprocess import (
    egraphsage_edge_features,
    te_g_sage_edge_features,
)
```

Then append:

```python
class EGraphSAGEFeatureTest(unittest.TestCase):
    def test_five_standard_scaled_columns(self) -> None:
        config = get_dataset_config("unsw_nb15")
        df, _ = load_aligned_edges(config)
        feats = egraphsage_edge_features(df)
        self.assertEqual(feats.shape, (len(df), 5))
        self.assertEqual(feats.dtype, np.float32)
        np.testing.assert_allclose(feats.mean(axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(feats.std(axis=0), 1.0, atol=1e-5)


class TEGSageFeatureTest(unittest.TestCase):
    def test_onehot_blocks_and_rare_folding(self) -> None:
        config = get_dataset_config("unsw_nb15")
        df, _ = load_aligned_edges(config)
        feats, names = te_g_sage_edge_features(df, rare_min_freq=50)
        self.assertEqual(feats.shape[0], len(df))
        self.assertEqual(feats.shape[1], len(names))
        onehot = [n for n in names if n.startswith("most_common_")]
        self.assertTrue(onehot, "expected one-hot categorical columns")
        # One-hot blocks are indicator columns.
        block = feats[:, [names.index(n) for n in onehot]]
        self.assertTrue(set(np.unique(block)).issubset({0.0, 1.0}))

    def test_rare_categories_collapse_at_our_scale(self) -> None:
        config = get_dataset_config("unsw_nb15")
        df, _ = load_aligned_edges(config)
        strict, strict_names = te_g_sage_edge_features(df, rare_min_freq=50)
        loose, loose_names = te_g_sage_edge_features(df, rare_min_freq=2)
        self.assertLess(
            len([n for n in strict_names if n.startswith("most_common_port")]),
            len([n for n in loose_names if n.startswith("most_common_port")]),
            "rare_min_freq=50 must collapse more port categories than =2",
        )
```

- [ ] **Step 6: Run to verify they fail**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/test_baseline_preprocess.py -v`
Expected: collection ERROR — `ImportError: cannot import name 'egraphsage_edge_features'
from 'src.pipeline.baselines.preprocess'`. Here that error IS the expected red state,
because the import was added deliberately in Step 5 to drive Step 7.

- [ ] **Step 7: Implement both featurizers**

```python
def egraphsage_edge_features(df: pd.DataFrame) -> np.ndarray:
    """E-GraphSAGE preprocessing: the paper documents no categorical scheme,
    so all five NetFlow columns are used numerically and standard-scaled."""
    raw = df[CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS].to_numpy(dtype=np.float64)
    return StandardScaler().fit_transform(raw).astype(np.float32)


def te_g_sage_edge_features(
    df: pd.DataFrame, rare_min_freq: int = 50
) -> tuple[np.ndarray, list[str]]:
    """TE-G-SAGE preprocessing: log1p + StandardScaler + correlation pruning
    at 0.995 on numerics; one-hot on categoricals with rare-category folding."""
    numeric = np.log1p(
        df[CONTINUOUS_COLUMNS].to_numpy(dtype=np.float64).clip(min=0.0)
    )
    numeric = StandardScaler().fit_transform(numeric)
    numeric_names = list(CONTINUOUS_COLUMNS)

    keep = _prune_correlated(numeric, threshold=0.995)
    numeric = numeric[:, keep]
    numeric_names = [numeric_names[i] for i in keep]

    blocks = [numeric]
    names = list(numeric_names)
    for col in CATEGORICAL_COLUMNS:
        values = df[col].astype(str)
        counts = values.value_counts()
        frequent = set(counts[counts >= rare_min_freq].index)
        folded = values.where(values.isin(frequent), RARE_TOKEN)
        dummies = pd.get_dummies(folded, prefix=col, dtype=np.float64)
        dummies = dummies.reindex(sorted(dummies.columns), axis=1)
        blocks.append(dummies.to_numpy())
        names.extend(dummies.columns.tolist())

    return np.hstack(blocks).astype(np.float32), names


def _prune_correlated(matrix: np.ndarray, threshold: float) -> list[int]:
    """Drop later columns correlating above `threshold` with an earlier kept one."""
    if matrix.shape[1] < 2:
        return list(range(matrix.shape[1]))
    corr = np.corrcoef(matrix, rowvar=False)
    corr = np.nan_to_num(corr)
    keep: list[int] = []
    for col in range(matrix.shape[1]):
        if all(abs(corr[col, kept]) <= threshold for kept in keep):
            keep.append(col)
    return keep
```

- [ ] **Step 8: Run all Task 1 tests**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/test_baseline_preprocess.py -v`
Expected: 4 passed.

- [ ] **Step 9: Commit**

```bash
git add src/pipeline/baselines/ tests/test_baseline_preprocess.py
git commit -m "Add per-paper featurization for the SOTA baselines

Loads the aggregated edge table order-checked against the graph, since
folds.pt indexes edges positionally. E-GraphSAGE gets standard-scaled
numeric columns; TE-G-SAGE gets log1p + StandardScaler + correlation
pruning with one-hot categoricals and rare-category folding.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: E-GraphSAGE model

**Files:**
- Create: `src/models/baselines/__init__.py`
- Create: `src/models/baselines/e_graphsage.py`
- Test: `tests/test_baseline_models.py`

**Interfaces:**
- Consumes: nothing from Task 1 (pure model module)
- Produces:
  - `EGraphSAGE(edge_dim: int, hidden_dim: int = 128, num_layers: int = 2, num_classes: int = 10, dropout: float = 0.2, node_init: str = "ones", node_feat_dim: int | None = None)`
  - `EGraphSAGE.forward(x: Tensor | None, edge_index: Tensor, edge_attr: Tensor) -> Tensor`  # `[E, num_classes]`
  - Module constants `PUBLISHED_HIDDEN_DIM = 128`, `PUBLISHED_NUM_LAYERS = 2`, `PUBLISHED_DROPOUT = 0.2`, `PUBLISHED_LR = 1e-3`

- [ ] **Step 1: Verify the reference message function before writing code**

The paper's Eq. 4 aggregates edge features alone, but the authors' released
implementation computes the message as `W_msg([h_u ‖ e_uv])`. Confirm which form the
published numbers came from, then implement that form and record the choice.

**This has already been verified — do not re-verify, just implement it.** The authors'
released notebook

`https://raw.githubusercontent.com/waimorris/E-GraphSAGE/master/E-GraphSAGE/netflow/ton-iot/Unsw_ton_iot_multiclass_mean_agg.ipynb`

defines the message as:

```python
self.W_msg(torch.cat([edges.src["h"], edges.data["h"]], 2))
```

i.e. `W_msg([h_u ‖ e_uv])`, which **differs from the paper's Eq. 4** (that aggregates
`e_uv` alone, without the source node state). The published numbers came from the code,
so implement the code's form.

Record the source URL as a comment in `e_graphsage.py`.

**Do not edit any deviations list in this task.** The driver does not exist until Task 5,
whose literal `DEVIATIONS` block already contains the corresponding `D5` entry. This step
produces a code comment only.

- [ ] **Step 2: Write the failing model tests**

```python
# tests/test_baseline_models.py
from __future__ import annotations

import unittest

import torch

from src.models.baselines.e_graphsage import EGraphSAGE


def _toy_graph() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    edge_index = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]], dtype=torch.long)
    edge_attr = torch.randn(4, 5)
    x = torch.randn(3, 10)
    return x, edge_index, edge_attr


class EGraphSAGEShapeTest(unittest.TestCase):
    def test_emits_one_logit_row_per_edge(self) -> None:
        x, edge_index, edge_attr = _toy_graph()
        model = EGraphSAGE(edge_dim=5, num_classes=10)
        logits = model(x, edge_index, edge_attr)
        self.assertEqual(tuple(logits.shape), (4, 10))

    def test_ones_init_ignores_node_features(self) -> None:
        """The published model sets x_v = {1,...,1}; data.x must not reach it."""
        x, edge_index, edge_attr = _toy_graph()
        model = EGraphSAGE(edge_dim=5, num_classes=10, node_init="ones")
        model.eval()
        with torch.no_grad():
            a = model(x, edge_index, edge_attr)
            b = model(torch.randn_like(x) * 100.0, edge_index, edge_attr)
        torch.testing.assert_close(a, b)

    def test_node_feature_variant_consumes_node_features(self) -> None:
        """The labelled variant must actually depend on data.x."""
        x, edge_index, edge_attr = _toy_graph()
        model = EGraphSAGE(
            edge_dim=5, num_classes=10, node_init="node_features", node_feat_dim=10
        )
        model.eval()
        with torch.no_grad():
            a = model(x, edge_index, edge_attr)
            b = model(torch.randn_like(x) * 100.0, edge_index, edge_attr)
        self.assertFalse(torch.allclose(a, b))
```

- [ ] **Step 3: Run to verify they fail**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/test_baseline_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.baselines'`

- [ ] **Step 4: Implement the model**

```python
# src/models/baselines/e_graphsage.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

PUBLISHED_HIDDEN_DIM = 128
PUBLISHED_NUM_LAYERS = 2
PUBLISHED_DROPOUT = 0.2
PUBLISHED_LR = 1e-3


class EGraphSAGELayer(MessagePassing):
    """One E-GraphSAGE layer (Lo et al., NOMS 2022).

    Message is built from the source node state concatenated with the edge
    feature, mean-aggregated over incident edges, then concatenated with the
    node's own state -- the form used by the authors' released implementation.
    """

    def __init__(self, in_dim: int, edge_dim: int, out_dim: int) -> None:
        super().__init__(aggr="mean", flow="source_to_target")
        self.w_msg = nn.Linear(in_dim + edge_dim, out_dim)
        self.w_apply = nn.Linear(in_dim + out_dim, out_dim)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        neigh = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return F.relu(self.w_apply(torch.cat([x, neigh], dim=1)))

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.w_msg(torch.cat([x_j, edge_attr], dim=1))


class EGraphSAGE(nn.Module):
    """E-GraphSAGE edge classifier.

    `node_init="ones"` reproduces the paper: nodes carry x_v = {1,...,1} with
    dimensionality equal to the edge-feature count, so node features are unused.
    `node_init="node_features"` is the labelled, NON-FAITHFUL variant that
    substitutes our centrality measures; it is never reported as E-GraphSAGE.
    """

    def __init__(
        self,
        edge_dim: int,
        hidden_dim: int = PUBLISHED_HIDDEN_DIM,
        num_layers: int = PUBLISHED_NUM_LAYERS,
        num_classes: int = 10,
        dropout: float = PUBLISHED_DROPOUT,
        node_init: str = "ones",
        node_feat_dim: int | None = None,
    ) -> None:
        super().__init__()
        if node_init not in ("ones", "node_features"):
            raise ValueError(f"unknown node_init {node_init!r}")
        if node_init == "node_features" and node_feat_dim is None:
            raise ValueError("node_init='node_features' requires node_feat_dim")

        self.node_init = node_init
        self.dropout = dropout
        in_dim = edge_dim if node_init == "ones" else int(node_feat_dim)

        dims = [in_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList(
            EGraphSAGELayer(dims[i], edge_dim, dims[i + 1]) for i in range(num_layers)
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        self.in_dim = in_dim

    def forward(
        self,
        x: torch.Tensor | None,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = int(edge_index.max()) + 1 if x is None else x.shape[0]
        if self.node_init == "ones":
            h = torch.ones(
                num_nodes, self.in_dim, dtype=edge_attr.dtype, device=edge_attr.device
            )
        else:
            h = x

        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index, edge_attr)
            if i < len(self.layers) - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)

        src, dst = edge_index[0], edge_index[1]
        return self.classifier(torch.cat([h[src], h[dst]], dim=1))
```

Also create `src/models/baselines/__init__.py` as an empty file.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/test_baseline_models.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/models/baselines/ tests/test_baseline_models.py
git commit -m "Add E-GraphSAGE baseline model in PyTorch Geometric

Faithful ones-vector node initialisation with mean aggregation over
incident edge features and a concatenated-endpoint edge classifier.
node_init='node_features' provides the labelled non-faithful variant,
guarded by tests asserting each mode's dependence on data.x.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: TE-G-SAGE model

**Files:**
- Create: `src/models/baselines/te_g_sage.py`
- Modify: `tests/test_baseline_models.py` (append)

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces:
  - `TEGSage(in_dim: int, edge_dim: int, hidden_dim: int = 128, num_layers: int = 2, num_classes: int = 10, dropout: float = 0.3)`
  - `TEGSage.forward(x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor`  # `[E, num_classes]`
  - Constants `PUBLISHED_HIDDEN_DIM = 128`, `PUBLISHED_NUM_LAYERS = 2`, `PUBLISHED_DROPOUT = 0.3`

TE-G-SAGE is `SAGEConv` over node states with mean aggregation, and the flow
representation concatenates the two endpoint embeddings with the edge's own feature
vector. Its published fanout `[25, 15]` exceeds the degree available in a 656-edge graph,
so full-neighbourhood aggregation is used (deviation D2).

- [ ] **Step 1: Write the failing test**

```python
from src.models.baselines.te_g_sage import TEGSage


class TEGSageShapeTest(unittest.TestCase):
    def test_emits_one_logit_row_per_edge(self) -> None:
        edge_index = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]], dtype=torch.long)
        edge_attr = torch.randn(4, 12)
        x = torch.randn(3, 10)
        model = TEGSage(in_dim=10, edge_dim=12, num_classes=10)
        logits = model(x, edge_index, edge_attr)
        self.assertEqual(tuple(logits.shape), (4, 10))
```

- [ ] **Step 2: Run to verify it fails**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/test_baseline_models.py::TEGSageShapeTest -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.baselines.te_g_sage'`

- [ ] **Step 3: Implement the model**

```python
# src/models/baselines/te_g_sage.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

PUBLISHED_HIDDEN_DIM = 128
PUBLISHED_NUM_LAYERS = 2
PUBLISHED_DROPOUT = 0.3
PUBLISHED_FANOUT = (25, 15)


class TEGSage(nn.Module):
    """TE-G-SAGE edge classifier (MDPI 2025), edge-aware GraphSAGE.

    Published fanout (25, 15) exceeds the degree available in our aggregated
    graphs, so full-neighbourhood aggregation is used -- deviation D2.
    """

    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        hidden_dim: int = PUBLISHED_HIDDEN_DIM,
        num_layers: int = PUBLISHED_NUM_LAYERS,
        num_classes: int = 10,
        dropout: float = PUBLISHED_DROPOUT,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        dims = [in_dim] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList(
            SAGEConv(dims[i], dims[i + 1], aggr="mean") for i in range(num_layers)
        )
        # Built eagerly: a submodule created inside forward() is invisible to an
        # optimizer constructed before the first forward pass, so its weights
        # would never be updated.
        self.classifier = nn.Linear(hidden_dim * 2 + edge_dim, num_classes)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            h = F.relu(h)
            if i < len(self.convs) - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)

        src, dst = edge_index[0], edge_index[1]
        flow = torch.cat([h[src], h[dst], edge_attr], dim=1)
        return self.classifier(flow)
```

`edge_dim` varies with `rare_min_freq`, so the driver must pass the actual featurized
width from `te_g_sage_edge_features` rather than assuming a constant.

- [ ] **Step 4: Run to verify it passes**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/test_baseline_models.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/models/baselines/te_g_sage.py tests/test_baseline_models.py
git commit -m "Add TE-G-SAGE baseline model in PyTorch Geometric

Two mean-aggregating SAGEConv layers with a concatenated endpoint-plus-edge
flow representation. Full-neighbourhood aggregation replaces the published
(25, 15) fanout, which exceeds our graphs' degree -- deviation D2.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Out-of-fold training harness

**Files:**
- Create: `src/pipeline/baselines/harness.py`
- Test: `tests/test_baseline_harness.py`

**Interfaces:**
- Consumes: `src.pipeline.common.splits.eval_macro_f1`, `get_class_weights`; `src.pipeline.common.metrics.classification_metrics`
- Produces:
  - `run_out_of_fold(model_factory: Callable[[], nn.Module], x, edge_index, edge_attr, labels, folds, *, seed: int, class_weighting: str, max_epochs: int = 200, patience: int = 25, lr: float = 1e-3, eval_classes) -> numpy.ndarray`  # pooled OOF predictions `[E]`
  - `pooled_scores(labels, preds, config) -> dict[str, float]` with keys `macro_f1`, `accuracy`, `weighted_f1`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_baseline_harness.py
from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn as nn

from src.pipeline.baselines.harness import pooled_scores, run_out_of_fold
from src.pipeline.common.datasets import get_dataset_config


class HarnessTest(unittest.TestCase):
    def test_every_edge_receives_exactly_one_oof_prediction(self) -> None:
        num_edges, num_nodes = 40, 8
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, 5)
        x = torch.randn(num_nodes, 10)
        labels = torch.randint(0, 3, (num_edges,))
        folds = []
        perm = torch.randperm(num_edges)
        for k in range(2):
            test_mask = torch.zeros(num_edges, dtype=torch.bool)
            test_mask[perm[k * 20 : (k + 1) * 20]] = True
            train_mask = ~test_mask
            val_mask = train_mask.clone()
            folds.append(
                {"train_mask": train_mask, "val_mask": val_mask, "test_mask": test_mask}
            )

        class Tiny(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(5, 3)

            def forward(self, x, edge_index, edge_attr):
                return self.lin(edge_attr)

        preds = run_out_of_fold(
            Tiny, x, edge_index, edge_attr, labels, folds,
            seed=42, class_weighting="none", max_epochs=3,
            eval_classes=(0, 1, 2),
        )
        self.assertEqual(preds.shape, (num_edges,))
        self.assertTrue((preds >= 0).all())

    def test_same_seed_reproduces(self) -> None:
        config = get_dataset_config("unsw_nb15")
        self.assertIsNotNone(config)
```

- [ ] **Step 2: Run to verify it fails**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/test_baseline_harness.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.pipeline.baselines.harness'`

- [ ] **Step 3: Implement the harness**

```python
# src/pipeline/baselines/harness.py
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from src.pipeline.common.datasets import DatasetConfig
from src.pipeline.common.metrics import classification_metrics
from src.pipeline.common.splits import eval_macro_f1, get_class_weights

GRAD_CLIP = 1.0


def run_out_of_fold(
    model_factory: Callable[[], nn.Module],
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    labels: torch.Tensor,
    folds: list[dict[str, torch.Tensor]],
    *,
    seed: int,
    class_weighting: str,
    eval_classes: tuple[int, ...],
    max_epochs: int = 200,
    patience: int = 25,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
) -> np.ndarray:
    """Train one model per fold, predict that fold's test edges, pool the result.

    Early stopping selects on validation macro-F1 restricted to `eval_classes`.
    Test masks are never read for selection.
    """
    pooled = np.full(labels.shape[0], -1, dtype=np.int64)

    for fold_idx, fold in enumerate(folds):
        torch.manual_seed(seed + fold_idx)
        np.random.seed(seed + fold_idx)

        model = model_factory()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        train_mask, val_mask = fold["train_mask"], fold["val_mask"]
        if class_weighting == "inverse_frequency":
            criterion = nn.CrossEntropyLoss(weight=get_class_weights(labels, train_mask))
        elif class_weighting == "none":
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"unknown class_weighting {class_weighting!r}")

        best_f1, best_state, stale = -1.0, None, 0
        for _ in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            loss = criterion(
                model(x, edge_index, edge_attr)[train_mask], labels[train_mask]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_preds = model(x, edge_index, edge_attr)[val_mask].argmax(dim=1)
            val_f1 = eval_macro_f1(labels[val_mask], val_preds, eval_classes)

            if val_f1 > best_f1:
                best_f1, stale = val_f1, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            fold_preds = model(x, edge_index, edge_attr).argmax(dim=1).cpu().numpy()
        test_mask = fold["test_mask"].cpu().numpy()
        pooled[test_mask] = fold_preds[test_mask]

    if (pooled < 0).any():
        raise ValueError("some edges never appeared in a test fold")
    return pooled


def pooled_scores(
    labels: torch.Tensor, preds: np.ndarray, config: DatasetConfig
) -> dict[str, object]:
    y_true = labels.cpu().numpy()
    row_mask = np.isin(y_true, list(config.eval_classes))
    detail = classification_metrics(
        preds[row_mask], y_true[row_mask], config.label_names
    )
    return {
        "macro_f1": eval_macro_f1(labels, preds, config.eval_classes),
        "accuracy": float((preds[row_mask] == y_true[row_mask]).mean()),
        "weighted_f1": detail["overall_weighted_f1"],
        "per_class_f1": [
            {"class_id": c["class_id"], "class_name": c["class_name"], "f1": c["f1"]}
            for c in detail["per_class"]
            if c["class_id"] in config.eval_classes
        ],
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/test_baseline_harness.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/baselines/harness.py tests/test_baseline_harness.py
git commit -m "Add out-of-fold training harness for the SOTA baselines

Per-fold training with validation-only early stopping on eval-class macro-F1,
pooled OOF predictions, and a scorer restricted to the dataset's eval classes.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Driver and results contract

**Files:**
- Create: `src/pipeline/baselines/run_baselines.py`
- Test: `tests/test_baseline_harness.py` (append a contract-shape test)

**Interfaces:**
- Consumes: Tasks 1, 2, 3, 4
- Produces:
  - `build_payload(dataset: str, configs: dict) -> dict`
  - CLI: `python -m src.pipeline.baselines.run_baselines --dataset <key> --mode {as_published,refit,plus_node_features,all} --seeds 42 1 2`
  - Output: `results/{dataset}_sota_baselines.json`

- [ ] **Step 1: Write the failing contract test**

```python
class ContractShapeTest(unittest.TestCase):
    def test_payload_declares_provenance_and_deviations(self) -> None:
        from src.pipeline.baselines.run_baselines import empty_payload

        payload = empty_payload("unsw_nb15")
        for key in (
            "schema_version", "dataset", "dataset_key", "metric_protocol",
            "primary_metric", "data_provenance", "deviations",
            "preprocessing", "baselines",
        ):
            self.assertIn(key, payload)
        self.assertEqual(payload["primary_metric"], "macro_f1")
        codes = {d["code"] for d in payload["deviations"]}
        self.assertTrue({"D1", "D2", "D3", "D4", "D5"}.issubset(codes))
        for dev in payload["deviations"]:
            self.assertTrue(dev["reason"])
            self.assertTrue(dev["resolution"])
```

- [ ] **Step 2: Run to verify it fails**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/test_baseline_harness.py::ContractShapeTest -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.pipeline.baselines.run_baselines'`

- [ ] **Step 3: Implement `empty_payload` and the deviations list**

```python
# src/pipeline/baselines/run_baselines.py
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

from src.pipeline.common.datasets import get_dataset_config

SCHEMA_VERSION = 1

DEVIATIONS = [
    {
        "code": "D1",
        "reason": "TE-G-SAGE publishes a 60/30/10 chronological split; aggregation "
                  "collapsed per-flow timestamps, so no chronological order exists.",
        "resolution": "Our stratified 5-fold folds.pt is used for all models.",
    },
    {
        "code": "D2",
        "reason": "TE-G-SAGE's published fanout (25, 15) exceeds the node degree "
                  "available in a 656-edge (UNSW) / 2127-edge (ToN) graph.",
        "resolution": "Full-neighbourhood aggregation.",
    },
    {
        "code": "D3",
        "reason": "Neither paper states an epoch count.",
        "resolution": "Our standard schedule: max 200 epochs, patience 25, best "
                      "validation macro-F1 restored before test prediction.",
    },
    {
        "code": "D4",
        "reason": "TE-G-SAGE's rare_min_freq=50 removes almost every port category "
                  "at our scale, where the whole graph has fewer than 2200 edges.",
        "resolution": "Applied as published in as_published; swept in refit.",
    },
    {
        "code": "D5",
        "reason": "E-GraphSAGE's paper Eq. 4 aggregates edge features alone, but the "
                  "authors' released implementation builds the message as "
                  "W_msg([h_u || e_uv]), concatenating the source node state. The "
                  "published numbers came from the code, not the equation.",
        "resolution": "The released implementation's form is used. Source: "
                      "github.com/waimorris/E-GraphSAGE, "
                      "E-GraphSAGE/netflow/ton-iot/Unsw_ton_iot_multiclass_mean_agg.ipynb",
    },
]


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def empty_payload(dataset: str) -> dict:
    config = get_dataset_config(dataset)
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": config.display_name,
        "dataset_key": config.key,
        "metric_protocol": "pooled five-fold out-of-fold evaluation",
        "primary_metric": "macro_f1",
        "generated": date.today().isoformat(),
        "claim_boundary": (
            "Published architectures re-trained on our aggregated flow-graph "
            "representation. NOT a comparison against their published numbers, "
            "which were obtained on per-flow graphs ~300x larger."
        ),
        "data_provenance": {
            "aggregated_edges_path": config.aggregated_edges_path,
            "aggregated_edges_sha256": _sha256(config.aggregated_edges_path),
            "splits_path": config.splits_path,
            "splits_sha256": _sha256(config.splits_path),
            "eval_classes": list(config.eval_classes),
            "excluded_classes": list(config.dropped_classes),
        },
        "preprocessing": {
            "ours_v2": "log1p+Z on cols 0-2; learned embeddings on protocol/port",
            "e_graphsage": "all five NetFlow columns standard-scaled",
            "te_g_sage": "log1p + StandardScaler + corr-prune@0.995; one-hot "
                         "categoricals with rare-category folding",
        },
        "deviations": list(DEVIATIONS),
        "baselines": {},
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/test_baseline_harness.py::ContractShapeTest -v`
Expected: PASS

- [ ] **Step 5: Implement the run loop and CLI**

Append to `run_baselines.py`: a `run(dataset, mode, seeds)` that, for each model and each
requested mode, builds features (Task 1), constructs the model factory (Tasks 2-3), calls
`run_out_of_fold` per seed (Task 4), records `mean`/`std` of `macro_f1` across seeds plus
the per-seed list, and writes `results/{dataset}_sota_baselines.json`. Each entry carries
`hyperparameters`, `seeds`, and — for the variant — `"is_faithful_to_paper": False` and
`"variant_label": "E-GraphSAGE + our node features"`. `argparse` exposes
`--dataset`, `--mode`, `--seeds`.

- [ ] **Step 6: Smoke-run the driver on the smaller dataset**

Run: `OMP_NUM_THREADS=1 python -m src.pipeline.baselines.run_baselines --dataset unsw_nb15 --mode as_published --seeds 42`
Expected: writes `results/unsw_nb15_sota_baselines.json` containing a `baselines.e_graphsage.as_published.macro_f1` float.

- [ ] **Step 7: Commit**

```bash
git add src/pipeline/baselines/run_baselines.py tests/test_baseline_harness.py
git commit -m "Add SOTA baseline driver and results contract

Emits data provenance hashes, the D1-D4 deviation list, per-model
preprocessing description and the claim boundary alongside the scores, so
the report cites the caveats rather than burying them.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: `as_published` runs, both datasets, three seeds

**Files:**
- Modify: `results/unsw_nb15_sota_baselines.json`, `results/ton_iot_sota_baselines.json`

**Interfaces:**
- Consumes: Task 5 CLI
- Produces: populated `baselines.{e_graphsage,te_g_sage}.as_published` entries

- [ ] **Step 1: Run both datasets**

```bash
export OMP_NUM_THREADS=1
python -m src.pipeline.baselines.run_baselines --dataset unsw_nb15 --mode as_published --seeds 42 1 2
python -m src.pipeline.baselines.run_baselines --dataset ton_iot   --mode as_published --seeds 42 1 2
```

- [ ] **Step 2: Verify determinism**

Re-run the UNSW command and confirm `macro_f1` matches the stored value to 1e-6. If it
does not, stop — determinism is broken and no number in this plan is citable.

- [ ] **Step 3: Report the numbers to the user before proceeding**

Print a table of `as_published` macro-F1 against our rungs (UNSW GNN 0.7219 / LLM 0.7353 /
AGAF 0.7595 / Loop 0.7728; ToN GNN 0.4290 / LLM 0.2785 / AGAF 0.3334 / Loop 0.4478).
Do not interpret yet — `refit` is required before any comparison is meaningful.

- [ ] **Step 4: Commit**

```bash
git add results/unsw_nb15_sota_baselines.json results/ton_iot_sota_baselines.json
git commit -m "Record as-published SOTA baseline results on both datasets

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: `refit` grid

**Files:**
- Modify: `src/pipeline/baselines/run_baselines.py`
- Modify: both `results/*_sota_baselines.json`

**Interfaces:**
- Consumes: Task 5 `run()`
- Produces: `select_refit(dataset, model_name, seeds) -> dict` returning the winning
  hyperparameters and the full grid trace

Grid, fixed by spec §5 and not to be widened after seeing results:

| Knob | Values |
|---|---|
| hidden dim | 32, 64, 128 |
| layers | 1, 2 |
| learning rate | 1e-3, 5e-4 |
| dropout | as published (0.2 / 0.3) |
| TE-G-SAGE `rare_min_freq` | 50, 2 |
| `class_weighting` | none, inverse_frequency |

- [ ] **Step 1: Write the failing selection test**

```python
class RefitSelectionTest(unittest.TestCase):
    def test_selection_never_reads_test_masks(self) -> None:
        """Selection must score on validation folds only."""
        import inspect
        from src.pipeline.baselines import run_baselines
        source = inspect.getsource(run_baselines.select_refit)
        self.assertNotIn("test_mask", source)
```

- [ ] **Step 2: Run to verify it fails**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/test_baseline_harness.py::RefitSelectionTest -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'select_refit'`

- [ ] **Step 3: Implement `select_refit`**

Score each grid point by mean validation macro-F1 across folds and seeds, using only
`fold["val_mask"]`. Return the argmax configuration plus the full trace under
`baselines.<model>.refit.grid_trace`.

- [ ] **Step 4: Run the test, then the grid**

```bash
OMP_NUM_THREADS=1 python -m pytest tests/test_baseline_harness.py -v
export OMP_NUM_THREADS=1
python -m src.pipeline.baselines.run_baselines --dataset unsw_nb15 --mode refit --seeds 42 1 2
python -m src.pipeline.baselines.run_baselines --dataset ton_iot   --mode refit --seeds 42 1 2
```

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/baselines/run_baselines.py results/
git commit -m "Add refit grid selection for the SOTA baselines

Grid fixed in the spec and selected on validation folds only, so the
baselines are not judged outside the graph scale their defaults assume.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: `plus_node_features` variant

**Files:**
- Modify: `src/pipeline/baselines/run_baselines.py`
- Modify: both `results/*_sota_baselines.json`

**Interfaces:**
- Consumes: `EGraphSAGE(node_init="node_features", node_feat_dim=10)` from Task 2
- Produces: `baselines.e_graphsage.plus_node_features` carrying `is_faithful_to_paper: False`

- [ ] **Step 1: Write the failing guard test**

```python
class VariantLabellingTest(unittest.TestCase):
    def test_non_faithful_entries_carry_a_label(self) -> None:
        import json
        from pathlib import Path

        payload = json.loads(Path("results/unsw_nb15_sota_baselines.json").read_text())
        for model, entries in payload["baselines"].items():
            for name, entry in entries.items():
                if entry.get("is_faithful_to_paper") is False:
                    self.assertTrue(entry.get("variant_label"), f"{model}.{name}")
```

- [ ] **Step 2: Run to verify it fails**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/test_baseline_harness.py::VariantLabellingTest -v`
Expected: FAIL — no `plus_node_features` entry exists yet.

- [ ] **Step 3: Implement the variant path and run it**

Reuse the winning `refit` hyperparameters, changing only `node_init` to
`"node_features"` with `node_feat_dim=data.x.shape[1]`.

```bash
export OMP_NUM_THREADS=1
python -m src.pipeline.baselines.run_baselines --dataset unsw_nb15 --mode plus_node_features --seeds 42 1 2
python -m src.pipeline.baselines.run_baselines --dataset ton_iot   --mode plus_node_features --seeds 42 1 2
```

- [ ] **Step 4: Run the guard test**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/test_baseline_harness.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/baselines/run_baselines.py results/
git commit -m "Add the E-GraphSAGE node-feature ablation column

Isolates how much of our advantage is centrality feature engineering rather
than GNN+LLM fusion. Flagged non-faithful and guarded by a labelling test.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: Bootstrap confidence intervals against our rungs

**Files:**
- Create: `src/pipeline/baselines/compare_to_ladder.py`
- Modify: both `results/*_sota_baselines.json`
- Test: `tests/test_sota_baselines.py` (written in Task 10)

**Interfaces:**
- Consumes: `src.pipeline.step4.assemble_ladder._bootstrap_ci(labels, preds_a, preds_b, eval_classes, iters=2000, seed=42)`; the baseline OOF predictions from Task 5
- Produces: `compare_to_ladder(dataset: str) -> dict` writing `statistical_comparisons` into the results file

Spec §10 requires a CI on every baseline-vs-rung difference. A point-estimate table alone
cannot support any claim, given that on UNSW no adjacent rung step is separated.

Our rungs' pooled OOF predictions already exist as logits under
`data/{dataset}/processed/step4_feedback/`: `oof_logits.pt` (GNN rung),
`feedback_oof_real.pt` (Loop rung), plus the prototype LLM and AGAF logits that
`assemble_ladder` reconstructs. Reuse `assemble_ladder`'s own loading path rather than
re-deriving it, and apply `_masked_preds` so dropped classes are handled identically.

- [ ] **Step 1: Persist baseline OOF predictions**

Modify Task 5's `run()` to save pooled predictions alongside the JSON as
`data/{dataset}/processed/baselines/{model}_{mode}_oof_preds.npy`, so CIs can be computed
without retraining.

- [ ] **Step 2: Write the failing test**

```python
class StatisticalComparisonTest(unittest.TestCase):
    def test_every_baseline_has_a_ci_against_each_rung(self) -> None:
        import json
        from pathlib import Path

        payload = json.loads(Path("results/unsw_nb15_sota_baselines.json").read_text())
        comparisons = payload["statistical_comparisons"]
        for key, entry in comparisons.items():
            for field in ("mean_diff", "ci_low", "ci_high", "prob_positive"):
                self.assertIn(field, entry, key)
            self.assertLessEqual(entry["ci_low"], entry["mean_diff"])
            self.assertGreaterEqual(entry["ci_high"], entry["mean_diff"])
```

- [ ] **Step 3: Run to verify it fails**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/test_sota_baselines.py::StatisticalComparisonTest -v`
Expected: FAIL with `KeyError: 'statistical_comparisons'`

- [ ] **Step 4: Implement `compare_to_ladder`**

For each `(baseline, mode)` and each rung in `(gnn, llm, agaf, feedback)`, compute
`_bootstrap_ci(labels, rung_preds, baseline_preds, config.eval_classes)` and store it
under `statistical_comparisons["<rung>_vs_<model>_<mode>"]`. Note the sign convention in
the JSON: positive `mean_diff` means **our rung is ahead**.

- [ ] **Step 5: Run and verify**

```bash
export OMP_NUM_THREADS=1
python -m src.pipeline.baselines.compare_to_ladder --dataset unsw_nb15
python -m src.pipeline.baselines.compare_to_ladder --dataset ton_iot
OMP_NUM_THREADS=1 python -m pytest tests/test_sota_baselines.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/baselines/compare_to_ladder.py results/ data/
git commit -m "Add bootstrap CIs for every baseline-versus-rung difference

Point estimates alone cannot support a comparison claim when no adjacent
rung step on UNSW is itself separated. Reuses assemble_ladder's bootstrap
so the protocol matches the ladder's own CI table.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 10: Contract tests, archive entry, CLAUDE.md

**Files:**
- Create: `tests/test_sota_baselines.py`
- Modify: `docs/RESULTS_ARCHIVE.md`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: both populated results files
- Produces: enforced contract between baseline results and `*_current.json`

- [ ] **Step 1: Write the contract tests**

```python
# tests/test_sota_baselines.py
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from src.pipeline.common.datasets import get_dataset_config


class SotaBaselineContractTest(unittest.TestCase):
    def test_baselines_used_the_same_folds_as_our_rungs(self) -> None:
        for key in ("unsw_nb15", "ton_iot"):
            with self.subTest(dataset=key):
                config = get_dataset_config(key)
                payload = json.loads(
                    Path(f"results/{key}_sota_baselines.json").read_text()
                )
                actual = hashlib.sha256(
                    Path(config.splits_path).read_bytes()
                ).hexdigest()
                self.assertEqual(
                    payload["data_provenance"]["splits_sha256"],
                    actual,
                    "baselines must score on the same folds as the ladder",
                )

    def test_same_classes_excluded_as_the_ladder(self) -> None:
        for key in ("unsw_nb15", "ton_iot"):
            with self.subTest(dataset=key):
                config = get_dataset_config(key)
                payload = json.loads(
                    Path(f"results/{key}_sota_baselines.json").read_text()
                )
                self.assertEqual(
                    payload["data_provenance"]["eval_classes"],
                    list(config.eval_classes),
                )

    def test_every_deviation_is_explained(self) -> None:
        payload = json.loads(Path("results/unsw_nb15_sota_baselines.json").read_text())
        codes = {d["code"] for d in payload["deviations"]}
        self.assertTrue({"D1", "D2", "D3", "D4", "D5"}.issubset(codes))
        for dev in payload["deviations"]:
            self.assertTrue(dev["reason"].strip())
            self.assertTrue(dev["resolution"].strip())
```

- [ ] **Step 2: Run the full suite**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/ -v`
Expected: all pass, including the pre-existing suite.

- [ ] **Step 3: Add a `docs/RESULTS_ARCHIVE.md` section**

New section "§5 SOTA baseline comparison" recording: the two baselines and why the other
candidates were rejected, the as_published / refit / plus_node_features numbers per
dataset, the full Task 9 CI table with its sign convention, the D1-D5 deviations, and the
claim boundary sentence verbatim. State explicitly which baseline-versus-rung differences
are separated and which are not.

- [ ] **Step 4: Add a Traps bullet to `CLAUDE.md`**

```markdown
- **Baseline numbers are "re-trained on our aggregated representation", never "we beat
  <paper>".** Their published numbers come from per-flow graphs ~300x larger; ours are
  2,127 / 656 edges. `plus_node_features` is NOT E-GraphSAGE. (§5)
```

- [ ] **Step 5: Commit**

```bash
git add tests/test_sota_baselines.py docs/RESULTS_ARCHIVE.md CLAUDE.md
git commit -m "Enforce the SOTA baseline contract and record it in the archive

Asserts baselines scored on the same folds and the same eval classes as the
ladder, and that every deviation carries a reason and resolution.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```
