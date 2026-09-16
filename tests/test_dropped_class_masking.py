"""Every stage that predicts must obey the dataset's evaluation protocol.

NF-ToN-IoT excludes `dos` (class 3, 4 edges) and `ransomware` (class 7, 3 edges)
from the metric. Rows whose true label is one of those are not scored, but a
model that *predicts* one of them for a scored edge loses that edge, and macro-F1
with it. Until 2026-09-16 the GNN, semantic and feedback stages masked those
columns before argmax and the fusion stage and the re-trained baselines did not,
so three of five stages could make a mistake the other two were forbidden to
make.

These tests pin the mask at each of the three places it was missing. They
construct logits whose unmasked argmax lands on a dropped class for every row, so
an unmasked implementation fails on every row rather than by luck.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.pipeline.common.splits import NUM_CLASSES

TON_DROPPED = (3, 7)
TON_EVAL = tuple(c for c in range(NUM_CLASSES) if c not in TON_DROPPED)


def logits_favouring_dropped(n_rows: int = 24) -> torch.Tensor:
    """Logits whose plain argmax is a dropped class on every row.

    Classes 3 and 7 get the two largest values, so any argmax that has not had
    those columns driven to -inf returns 3 or 7 for all `n_rows`.
    """
    logits = torch.zeros(n_rows, NUM_CLASSES)
    logits += torch.linspace(-1.0, 1.0, NUM_CLASSES)  # a non-constant background
    logits[:, 3] = 50.0
    logits[:, 7] = 40.0
    return logits


class FusionMaskTest(unittest.TestCase):
    """`src/pipeline/step3/train_fusion.py`."""

    def test_eval_preds_never_returns_a_dropped_class(self) -> None:
        from src.pipeline.step3 import train_fusion

        original = train_fusion.DROPPED_CLASSES
        try:
            train_fusion.DROPPED_CLASSES = TON_DROPPED
            preds = train_fusion._eval_preds(logits_favouring_dropped())
            self.assertEqual(preds.shape[0], 24)
            for cls in TON_DROPPED:
                self.assertFalse(
                    bool((preds == cls).any()),
                    f"fusion predicted excluded class {cls}",
                )
        finally:
            train_fusion.DROPPED_CLASSES = original

    def test_unsw_is_unchanged_because_it_drops_nothing(self) -> None:
        from src.pipeline.step3 import train_fusion

        original = train_fusion.DROPPED_CLASSES
        try:
            train_fusion.DROPPED_CLASSES = ()
            logits = logits_favouring_dropped()
            preds = train_fusion._eval_preds(logits)
            self.assertTrue(bool((preds == 3).all()))
        finally:
            train_fusion.DROPPED_CLASSES = original


class GnnMaskTest(unittest.TestCase):
    """`src/pipeline/step3/train_gnn.py`."""

    def test_eval_preds_never_returns_a_dropped_class(self) -> None:
        from src.pipeline.step3 import train_gnn

        original = train_gnn.DROPPED_CLASSES
        try:
            train_gnn.DROPPED_CLASSES = TON_DROPPED
            preds = train_gnn._eval_preds(logits_favouring_dropped())
            for cls in TON_DROPPED:
                self.assertFalse(
                    bool((preds == cls).any()), f"GNN predicted excluded class {cls}"
                )
        finally:
            train_gnn.DROPPED_CLASSES = original

    def test_macro_f1_scores_only_the_evaluated_classes(self) -> None:
        """The metric is averaged over eight labels, not ten.

        A ten-class average divides by ten and counts two classes this protocol
        never scores, which is the rule NF-ToN-IoT checkpoints were selected
        under before the fix.
        """
        from src.pipeline.step3 import train_gnn

        original_dropped = train_gnn.DROPPED_CLASSES
        original_eval = train_gnn.EVAL_CLASSES
        try:
            train_gnn.DROPPED_CLASSES = TON_DROPPED
            train_gnn.EVAL_CLASSES = TON_EVAL
            # One row per evaluated class, every prediction correct.
            labels = torch.tensor(list(TON_EVAL))
            logits = torch.full((len(TON_EVAL), NUM_CLASSES), -10.0)
            for row, cls in enumerate(TON_EVAL):
                logits[row, cls] = 10.0
            mask = torch.ones(len(TON_EVAL), dtype=torch.bool)
            self.assertAlmostEqual(train_gnn._macro_f1(logits, labels, mask), 1.0)
        finally:
            train_gnn.DROPPED_CLASSES = original_dropped
            train_gnn.EVAL_CLASSES = original_eval


class BaselineHarnessMaskTest(unittest.TestCase):
    """`src/pipeline/baselines/harness.py`."""

    def test_pooled_predictions_never_land_in_a_dropped_class(self) -> None:
        import torch.nn as nn

        from src.pipeline.baselines.harness import run_out_of_fold

        n_edges = 40
        rng = np.random.default_rng(0)
        labels = torch.from_numpy(
            rng.choice(TON_EVAL, size=n_edges).astype(np.int64)
        )
        x = torch.randn(6, 4)
        edge_index = torch.randint(0, 6, (2, n_edges))
        edge_attr = torch.randn(n_edges, 3)

        class AlwaysDropped(nn.Module):
            """Emits dropped-class-favouring logits whatever it is given.

            The parameter contributes nothing to the value and everything to the
            graph: the harness calls `backward()` each epoch, and a constant
            tensor has nothing to differentiate.
            """

            def __init__(self) -> None:
                super().__init__()
                self.scale = nn.Parameter(torch.zeros(1))

            def forward(self, x, edge_index, edge_attr):
                base = logits_favouring_dropped(edge_attr.shape[0])
                return base + self.scale * base

        folds = []
        for start in range(0, n_edges, 20):
            test = torch.zeros(n_edges, dtype=torch.bool)
            test[start:start + 20] = True
            train = ~test
            folds.append({"train_mask": train, "val_mask": train, "test_mask": test})

        preds = run_out_of_fold(
            AlwaysDropped,
            x,
            edge_index,
            edge_attr,
            labels,
            folds,
            seed=0,
            class_weighting="none",
            eval_classes=TON_EVAL,
            dropped_classes=TON_DROPPED,
            max_epochs=1,
            patience=1,
        )
        self.assertEqual(preds.shape[0], n_edges)
        for cls in TON_DROPPED:
            self.assertFalse(
                bool((preds == cls).any()), f"baseline predicted excluded class {cls}"
            )


if __name__ == "__main__":
    unittest.main()
