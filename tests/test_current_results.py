from __future__ import annotations

import json
import unittest
from pathlib import Path

from reproduce_ladder import load_expected_accuracy, load_expected_ladder


UNSW_RESULTS_PATH = Path("results/unsw_nb15_current.json")
TON_RESULTS_PATH = Path("results/ton_iot_current.json")
TON_PROTOTYPE_PATH = Path("results/ton_iot_prototype_comparison.json")
COMPARISON_PATH = Path("results/cross_dataset_comparison.json")


class CurrentResultsContractTest(unittest.TestCase):
    def test_authoritative_unsw_ladder_and_configuration(self) -> None:
        payload = json.loads(UNSW_RESULTS_PATH.read_text())
        results = payload["results"]

        self.assertEqual(payload["configuration"]["top_k_percent"], 16.0)
        self.assertEqual(payload["configuration"]["semantic_consultant"], "whitened_prototype")
        self.assertEqual(payload["configuration"]["trained_llm_head"], False)

        macro_f1 = [results[name]["macro_f1"] for name in ("gnn", "llm", "agaf", "feedback")]
        self.assertEqual(macro_f1, sorted(macro_f1))
        self.assertAlmostEqual(results["gnn"]["macro_f1"], 0.5496206583793729)
        self.assertAlmostEqual(results["llm"]["macro_f1"], 0.735318802626)
        self.assertAlmostEqual(results["agaf"]["macro_f1"], 0.7458681287559671)
        self.assertAlmostEqual(results["feedback"]["macro_f1"], 0.7763992869991123)
        self.assertAlmostEqual(results["feedback"]["accuracy"], 0.8185975609756098)

    def test_reproduction_uses_authoritative_manifest(self) -> None:
        expected = load_expected_ladder()
        self.assertAlmostEqual(expected["gnn_alone"], 0.5496206583793729)
        self.assertAlmostEqual(expected["llm_alone"], 0.735318802626)
        self.assertAlmostEqual(expected["agaf"], 0.7458681287559671)
        self.assertAlmostEqual(expected["feedback_loop"], 0.7763992869991123)

        accuracy = load_expected_accuracy()
        self.assertAlmostEqual(accuracy["feedback_loop"], 0.8185975609756098)

    def test_authoritative_ton_iot_aggregated_ladder(self) -> None:
        payload = json.loads(TON_RESULTS_PATH.read_text())
        results = payload["results"]

        self.assertEqual(payload["dataset_key"], "ton_iot")
        self.assertEqual(payload["graph_scope"], "aggregated")
        self.assertEqual(payload["feature_profile"], "structural10")
        self.assertEqual(payload["n_eval_classes"], 8)
        self.assertFalse(payload["ladder_order_holds"])
        self.assertEqual(payload["configuration"]["top_k_percent"], 15.0)
        self.assertEqual(payload["configuration"]["semantic_consultant"], "trained_oof_head")
        self.assertEqual(payload["configuration"]["trained_llm_head"], True)
        self.assertEqual(payload["configuration"]["agaf_head_fusion"], True)

        self.assertAlmostEqual(results["gnn"]["macro_f1"], 0.33984409634016377)
        self.assertAlmostEqual(results["llm"]["macro_f1"], 0.5165429122825544)
        self.assertAlmostEqual(results["agaf"]["macro_f1"], 0.5140964250159155)
        self.assertAlmostEqual(results["feedback"]["macro_f1"], 0.4985978148287057)

        # Head fusion lifted AGAF well clear of the GNN (CI excludes zero) ...
        self.assertGreater(results["agaf"]["macro_f1"], results["gnn"]["macro_f1"])
        self.assertGreater(results["feedback"]["macro_f1"], results["gnn"]["macro_f1"])
        # ... but the top three rungs are statistically tied: every pairwise CI
        # among LLM / AGAF / loop crosses zero, so the strict ladder is NOT shown.
        for pair in ("agaf_vs_llm", "feedback_vs_agaf", "feedback_vs_llm"):
            low, high = payload["statistical_comparisons"][pair]["ci_95"]
            self.assertLess(low, 0.0, pair)
            self.assertGreater(high, 0.0, pair)

        # The loop's selection is real even though its rung is tied: choosing which
        # flows to consult beats both random selection and head-only consultation.
        ablations = payload["feedback_ablations"]
        self.assertGreater(ablations["real"]["macro_f1"], ablations["random"]["macro_f1"])
        self.assertGreater(ablations["real_vs_random"]["ci_95"][0], 0.0)
        self.assertGreater(ablations["real_vs_head_only"]["ci_95"][0], 0.0)

    def test_ton_iot_prototype_comparison_is_marked_non_authoritative(self) -> None:
        """The prototype ladder is kept only so the consultant's effect is auditable."""
        payload = json.loads(TON_PROTOTYPE_PATH.read_text())

        self.assertEqual(payload["role"], "comparison_only")
        self.assertEqual(payload["configuration"]["semantic_consultant"], "whitened_prototype")
        self.assertTrue(payload["ladder_order_holds"])

        # It holds only because the LLM rung is crippled: same embeddings, weaker scorer.
        authoritative = json.loads(TON_RESULTS_PATH.read_text())
        self.assertLess(
            payload["results"]["llm"]["macro_f1"],
            authoritative["results"]["llm"]["macro_f1"],
        )

    def test_cross_dataset_comparison_matches_contracts(self) -> None:
        comparison = json.loads(COMPARISON_PATH.read_text())
        unsw = json.loads(UNSW_RESULTS_PATH.read_text())["results"]
        ton = json.loads(TON_RESULTS_PATH.read_text())["results"]

        for name in ("gnn", "llm", "agaf", "feedback"):
            self.assertAlmostEqual(
                comparison["datasets"]["unsw_nb15"]["macro_f1"][name],
                unsw[name]["macro_f1"],
            )
            self.assertAlmostEqual(
                comparison["datasets"]["ton_iot_aggregated"]["macro_f1"][name],
                ton[name]["macro_f1"],
            )


if __name__ == "__main__":
    unittest.main()
