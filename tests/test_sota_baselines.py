from __future__ import annotations

import unittest


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
