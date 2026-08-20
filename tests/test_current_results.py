from __future__ import annotations

import json
import unittest
from pathlib import Path

from reproduce_ladder import load_expected_accuracy, load_expected_ladder


UNSW_RESULTS_PATH = Path("results/unsw_nb15_current.json")
UNSW_HEAD_BASELINE_PATH = Path("results/unsw_nb15_head_baseline.json")
TON_HEAD_BASELINE_PATH = Path("results/ton_iot_head_baseline.json")
TON_RESULTS_PATH = Path("results/ton_iot_current.json")
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
        """ToN runs the SAME architecture as UNSW, and the ladder does not hold there."""
        payload = json.loads(TON_RESULTS_PATH.read_text())
        results = payload["results"]

        self.assertEqual(payload["dataset_key"], "ton_iot")
        self.assertEqual(payload["graph_scope"], "aggregated")
        self.assertEqual(payload["feature_profile"], "structural10")
        self.assertEqual(payload["n_eval_classes"], 8)
        self.assertEqual(payload["configuration"]["semantic_consultant"], "whitened_prototype")
        self.assertFalse(payload["configuration"]["trained_llm_head"])
        self.assertFalse(payload["configuration"]["agaf_head_fusion"])

        self.assertAlmostEqual(results["gnn"]["macro_f1"], 0.32711755971083384)
        self.assertAlmostEqual(results["llm"]["macro_f1"], 0.27852446280044896)
        self.assertAlmostEqual(results["agaf"]["macro_f1"], 0.4446525803655099)
        self.assertAlmostEqual(results["feedback"]["macro_f1"], 0.3875657812346581)

        # The ladder does NOT hold on ToN: AGAF is the top rung, the loop sits below it,
        # and the LLM rung sits below the GNN rung. All three facts must stay declared.
        self.assertFalse(payload["ladder_order_holds"])
        self.assertTrue(payload["llm_below_gnn"])
        self.assertLess(results["llm"]["macro_f1"], results["gnn"]["macro_f1"])
        self.assertLess(results["feedback"]["macro_f1"], results["agaf"]["macro_f1"])

        # The loop-AGAF gap is negative but not significant; the CI must still cross zero.
        loop_agaf = payload["statistical_comparisons"]["loop_vs_agaf"]
        self.assertLess(loop_agaf["mean_diff"], 0.0)
        self.assertLess(loop_agaf["ci_low"], 0.0)
        self.assertGreater(loop_agaf["ci_high"], 0.0)

        # AGAF is clearly above both unimodal rungs; that separation IS significant.
        self.assertGreater(payload["statistical_comparisons"]["agaf_vs_gnn"]["ci_low"], 0.0)
        self.assertGreater(payload["statistical_comparisons"]["agaf_vs_llm"]["ci_low"], 0.0)

    def test_both_datasets_declare_the_same_architecture(self) -> None:
        """Parity guard: the two datasets must never drift onto different consultants again.

        UNSW and ToN were previously scored under different semantic consultants, which made
        the cross-dataset comparison unreadable. This test fails the moment they diverge.
        """
        unsw = json.loads(UNSW_RESULTS_PATH.read_text())["configuration"]
        ton = json.loads(TON_RESULTS_PATH.read_text())["configuration"]

        self.assertEqual(unsw["semantic_consultant"], ton["semantic_consultant"])
        self.assertEqual(unsw["trained_llm_head"], ton["trained_llm_head"])
        self.assertEqual(
            unsw["semantic_confidence_fraction"], ton["semantic_confidence_fraction"]
        )
        # top_k_percent is the one quantity allowed to differ: it is selected per dataset
        # on validation folds and must never be carried across datasets.
        self.assertIn("top_k_percent", unsw)
        self.assertIn("top_k_percent", ton)

        comparison = json.loads(COMPARISON_PATH.read_text())
        self.assertTrue(comparison["architecture"]["shared"])
        self.assertEqual(
            comparison["architecture"]["semantic_consultant"], unsw["semantic_consultant"]
        )

    def test_head_baselines_record_that_they_beat_the_full_system(self) -> None:
        """The trained-head LLM-only baseline outscores the full system on both datasets.

        That is a genuine weakness of the result. These contracts exist so it cannot be
        dropped from a write-up by accident.
        """
        for path, canonical_path in (
            (UNSW_HEAD_BASELINE_PATH, UNSW_RESULTS_PATH),
            (TON_HEAD_BASELINE_PATH, TON_RESULTS_PATH),
        ):
            payload = json.loads(path.read_text())
            canonical = json.loads(canonical_path.read_text())
            self.assertEqual(payload["role"], "llm_only_baseline")
            self.assertTrue(payload["must_be_reported"])
            self.assertGreater(
                payload["llm_only_macro_f1"],
                canonical["results"]["feedback"]["macro_f1"],
                f"{path} no longer beats the loop; update the caveat text",
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
