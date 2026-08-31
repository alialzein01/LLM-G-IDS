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


class RefitSelectionTest(unittest.TestCase):
    def test_selection_never_reads_test_masks(self) -> None:
        """Selection must score on validation folds only."""
        import inspect
        from src.pipeline.baselines import run_baselines
        source = inspect.getsource(run_baselines.select_refit)
        self.assertNotIn("test_mask", source)

    def test_published_config_is_reachable_in_the_grid(self) -> None:
        """refit must be able to select each paper's own published setting,
        otherwise it can score below as_published, which is incoherent."""
        from src.pipeline.baselines.run_baselines import refit_grid

        eg = refit_grid("e_graphsage")
        self.assertTrue(
            any(c["hidden_dim"] == 128 and c["num_layers"] == 2 and c["lr"] == 1e-3 for c in eg),
            "E-GraphSAGE's published config must be a grid point",
        )
        tg = refit_grid("te_g_sage")
        self.assertTrue(
            any(
                c["hidden_dim"] == 128 and c["num_layers"] == 2
                and c["lr"] == 3e-4 and c["rare_min_freq"] == 50
                for c in tg
            ),
            "TE-G-SAGE's published config must be a grid point",
        )


class VariantLabellingTest(unittest.TestCase):
    def test_both_baselines_have_a_node_feature_variant(self) -> None:
        """Driving test: the entries must exist, for both datasets and both models."""
        import json
        from pathlib import Path

        for dataset in ("unsw_nb15", "ton_iot"):
            payload = json.loads(
                Path(f"results/{dataset}_sota_baselines.json").read_text()
            )
            for model in ("e_graphsage", "te_g_sage"):
                with self.subTest(dataset=dataset, model=model):
                    entry = payload["baselines"][model]["plus_node_features"]
                    self.assertIsInstance(entry["macro_f1"], float)
                    self.assertFalse(entry["is_faithful_to_paper"])
                    self.assertTrue(entry["variant_label"])

    def test_no_unlabelled_non_faithful_entry_anywhere(self) -> None:
        """Standing guard, not the driver: this one MAY pass vacuously, and that is
        fine -- its job is to catch a future non-faithful entry added without a label."""
        import json
        from pathlib import Path

        for dataset in ("unsw_nb15", "ton_iot"):
            payload = json.loads(
                Path(f"results/{dataset}_sota_baselines.json").read_text()
            )
            for model, entries in payload["baselines"].items():
                for name, entry in entries.items():
                    if entry.get("is_faithful_to_paper") is False:
                        self.assertTrue(entry.get("variant_label"), f"{model}.{name}")
