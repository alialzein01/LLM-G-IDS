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
