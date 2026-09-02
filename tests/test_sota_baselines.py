from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from src.pipeline.common.datasets import get_dataset_config


class StatisticalComparisonTest(unittest.TestCase):
    def test_every_baseline_has_a_ci_against_each_rung(self) -> None:
        import json
        from pathlib import Path

        for dataset in ("unsw_nb15", "ton_iot"):
            payload = json.loads(
                Path(f"results/{dataset}_sota_baselines.json").read_text()
            )
            comparisons = payload["statistical_comparisons"]
            # Assert the expected keys EXIST. Iterating whatever happens to be
            # present would pass vacuously on an empty dict.
            for rung in ("gnn", "llm", "agaf", "feedback"):
                for model in ("e_graphsage", "te_g_sage"):
                    for mode in ("as_published", "refit", "plus_node_features"):
                        key = f"{rung}_vs_{model}_{mode}"
                        with self.subTest(dataset=dataset, key=key):
                            entry = comparisons[key]
                            for field in ("mean_diff", "ci_low", "ci_high", "prob_positive"):
                                self.assertIn(field, entry, key)
                            self.assertLessEqual(entry["ci_low"], entry["mean_diff"])
                            self.assertGreaterEqual(entry["ci_high"], entry["mean_diff"])
                            # The primary interval MUST propagate baseline seed variance.
                            self.assertEqual(entry["resampled"], "edges_and_baseline_seed")
                            self.assertEqual(entry["baseline_seeds"], 3)
                            self.assertIn(key, payload["seed_matched_comparisons"])
            self.assertTrue(payload["comparison_caveat"].strip())


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
        for key in ("unsw_nb15", "ton_iot"):
            with self.subTest(dataset=key):
                payload = json.loads(
                    Path(f"results/{key}_sota_baselines.json").read_text()
                )
                codes = {d["code"] for d in payload["deviations"]}
                self.assertTrue({"D1", "D2", "D3", "D4", "D5"}.issubset(codes))
                for dev in payload["deviations"]:
                    self.assertTrue(dev["reason"].strip())
                    self.assertTrue(dev["resolution"].strip())

    def test_the_e_graphsage_ceiling_is_recorded(self) -> None:
        """Step 0's known_ceilings block must reach both results files: the report
        quotes it wherever an E-GraphSAGE number appears."""
        for key in ("unsw_nb15", "ton_iot"):
            with self.subTest(dataset=key):
                payload = json.loads(
                    Path(f"results/{key}_sota_baselines.json").read_text()
                )
                ceiling = payload["known_ceilings"]["e_graphsage_endpoint_only"]
                self.assertIn("CONCAT(h_u, h_v)", ceiling["description"])
                self.assertEqual(
                    ceiling["edges_indistinguishable_by_endpoints"],
                    {"unsw_nb15": 385, "ton_iot": 167},
                )
                self.assertTrue(ceiling["reporting_rule"].strip())
                self.assertTrue(ceiling["not_an_epoch_artifact"].strip())
