from __future__ import annotations

import json
import unittest
from pathlib import Path

from reproduce_ladder import load_expected_accuracy, load_expected_ladder


UNSW_RESULTS_PATH = Path("results/unsw_nb15_current.json")
UNSW_HEAD_BASELINE_PATH = Path("results/unsw_nb15_head_baseline.json")
TON_HEAD_BASELINE_PATH = Path("results/ton_iot_head_baseline.json")
LOOP_MECHANISM_PATH = Path("results/unsw_nb15_loop_mechanism.json")
TON_RESULTS_PATH = Path("results/ton_iot_current.json")
COMPARISON_PATH = Path("results/cross_dataset_comparison.json")


class CurrentResultsContractTest(unittest.TestCase):
    def test_authoritative_unsw_ladder_and_configuration(self) -> None:
        payload = json.loads(UNSW_RESULTS_PATH.read_text())
        results = payload["results"]

        self.assertEqual(payload["configuration"]["top_k_percent"], 31.0)
        self.assertEqual(payload["configuration"]["injection_scale"], 2.0)
        self.assertEqual(payload["configuration"]["semantic_consultant"], "whitened_prototype")
        self.assertEqual(payload["configuration"]["trained_llm_head"], False)
        self.assertEqual(payload["edge_attr_encoding"], "v2_log_cont_cat_idx")
        # The mechanism must be declared. A contract that omits it gets misread as
        # whatever the CLI default happens to be on the day it is read.
        self.assertEqual(payload["configuration"]["injection_mode"], "edge")
        self.assertEqual(
            payload["selection"]["swept_under_injection_mode"],
            payload["configuration"]["injection_mode"],
            "top_k must be selected under the same mechanism the loop trains with",
        )

        macro_f1 = [results[name]["macro_f1"] for name in ("gnn", "llm", "agaf", "feedback")]
        self.assertEqual(macro_f1, sorted(macro_f1))
        self.assertAlmostEqual(results["gnn"]["macro_f1"], 0.7219365593805762)
        self.assertAlmostEqual(results["llm"]["macro_f1"], 0.7353188026257084)
        self.assertAlmostEqual(results["agaf"]["macro_f1"], 0.7595293220194291)
        self.assertAlmostEqual(results["feedback"]["macro_f1"], 0.772755270351248)
        self.assertAlmostEqual(results["feedback"]["accuracy"], 0.8307926829268293)

        # The ladder holds by ORDER only. The loop-AGAF gap CI crosses zero (P=0.7785),
        # so it must never be described as a significant separation.
        loop_agaf = payload["statistical_comparisons"]["loop_vs_agaf"]
        self.assertGreater(loop_agaf["mean_diff"], 0.0)
        self.assertLess(loop_agaf["ci_low"], 0.0)

    def test_reproduction_uses_authoritative_manifest(self) -> None:
        expected = load_expected_ladder()
        self.assertAlmostEqual(expected["gnn_alone"], 0.7219365593805762)
        self.assertAlmostEqual(expected["llm_alone"], 0.7353188026257084)
        self.assertAlmostEqual(expected["agaf"], 0.7595293220194291)
        self.assertAlmostEqual(expected["feedback_loop"], 0.772755270351248)

        accuracy = load_expected_accuracy()
        self.assertAlmostEqual(accuracy["feedback_loop"], 0.8307926829268293)

    def test_authoritative_ton_iot_aggregated_ladder(self) -> None:
        """ToN runs the SAME architecture as UNSW. The ladder still fails there, but now
        because AGAF regressed below the GNN rung -- not because the loop underperforms."""
        payload = json.loads(TON_RESULTS_PATH.read_text())
        results = payload["results"]

        self.assertEqual(payload["dataset_key"], "ton_iot")
        self.assertEqual(payload["graph_scope"], "aggregated")
        self.assertEqual(payload["feature_profile"], "structural10")
        self.assertEqual(payload["n_eval_classes"], 8)
        self.assertEqual(payload["configuration"]["semantic_consultant"], "whitened_prototype")
        self.assertFalse(payload["configuration"]["trained_llm_head"])
        self.assertFalse(payload["configuration"]["agaf_head_fusion"])
        self.assertEqual(payload["edge_attr_encoding"], "v2_log_cont_cat_idx")

        self.assertAlmostEqual(results["gnn"]["macro_f1"], 0.4289705488338377)
        self.assertAlmostEqual(results["llm"]["macro_f1"], 0.27852446280044896)
        self.assertAlmostEqual(results["agaf"]["macro_f1"], 0.3333694556345273)
        self.assertAlmostEqual(results["feedback"]["macro_f1"], 0.4478180285563262)
        self.assertEqual(payload["configuration"]["top_k_percent"], 25.0)
        self.assertEqual(payload["configuration"]["injection_scale"], 20.0)
        self.assertEqual(payload["configuration"]["injection_mode"], "edge")

        # The ladder does NOT hold on ToN, but the failure moved: AGAF is now BELOW the
        # GNN rung, and the loop is the TOP rung, significantly above AGAF. Both facts
        # must stay declared -- this reverses the pre-v2 finding, do not revert it.
        self.assertFalse(payload["ladder_order_holds"])
        self.assertTrue(payload["llm_below_gnn"])
        self.assertTrue(payload["agaf_below_gnn"])
        self.assertTrue(payload["loop_is_top_rung"])
        self.assertLess(results["llm"]["macro_f1"], results["gnn"]["macro_f1"])
        self.assertLess(results["agaf"]["macro_f1"], results["gnn"]["macro_f1"])
        self.assertEqual(
            results["feedback"]["macro_f1"],
            max(results[name]["macro_f1"] for name in ("gnn", "llm", "agaf", "feedback")),
        )

        # AGAF is significantly BELOW the GNN rung now (P=0.0005) -- this is the one
        # rung that regressed and the sole reason the canonical ladder shape fails here.
        agaf_gnn = payload["statistical_comparisons"]["agaf_vs_gnn"]
        self.assertLess(agaf_gnn["mean_diff"], 0.0)
        self.assertLess(agaf_gnn["ci_high"], 0.0)
        self.assertLess(agaf_gnn["prob_positive"], 0.05)

        # The loop is significantly ABOVE AGAF now (flipped from the pre-v2 finding).
        loop_agaf = payload["statistical_comparisons"]["loop_vs_agaf"]
        self.assertGreater(loop_agaf["mean_diff"], 0.0)
        self.assertGreater(loop_agaf["ci_low"], 0.0)
        self.assertGreater(loop_agaf["prob_positive"], 0.95)

        # AGAF is clearly above the LLM rung; that separation IS significant.
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
        # The mechanism is part of the shared architecture, not a per-dataset knob.
        self.assertEqual(unsw["injection_mode"], ton["injection_mode"])
        # injection_scale and top_k_percent are the two quantities allowed to differ:
        # both are selected per dataset on validation folds and must never be carried
        # across datasets (see PROJECT_NOTES.md -- injection_scale must be re-selected whenever
        # the edge encoding changes, and it was: 2.0 on UNSW vs 20.0 on ToN).
        self.assertIn("top_k_percent", unsw)
        self.assertIn("top_k_percent", ton)
        self.assertIn("injection_scale", unsw)
        self.assertIn("injection_scale", ton)

        comparison = json.loads(COMPARISON_PATH.read_text())
        self.assertTrue(comparison["architecture"]["shared"])
        self.assertEqual(
            comparison["architecture"]["semantic_consultant"], unsw["semantic_consultant"]
        )

    def test_head_baselines_record_that_they_beat_the_full_system(self) -> None:
        """The trained-head LLM-only baseline outscores the full system on both datasets.

        That is a genuine weakness of the result. These contracts exist so it cannot be
        dropped from a write-up by accident. NOTE: these baseline files were generated
        2026-08-20 on the v1 encoding and have not been re-run on v2 -- see PROJECT_NOTES.md.
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

    def test_loop_attention_mechanism_is_recorded_as_inert(self) -> None:
        """The ~8% attention consultation changes zero predictions, in every config.

        todo.md Step 4 calls the bidirectional loop the core novelty. It does not run on
        this graph. This test pins that finding so it cannot be quietly dropped. This is a
        structural finding (co-located edges share node embeddings, so attention -- which
        only speaks through node embeddings -- cannot separate them) and is not encoding-
        dependent, so it is not re-verified against v2 in this pass.
        """
        payload = json.loads(LOOP_MECHANISM_PATH.read_text())
        self.assertTrue(payload["verdict"].startswith("NO"))

        # Churn is exactly zero everywhere: the biased pass reproduces the unbiased pass.
        for config, churn in payload["mechanism_diagnostics"]["churn_by_config"].items():
            self.assertEqual(churn, 0.0, f"{config} churn is no longer zero")

        # Even a full-scale ~1 nat bias flips nothing, so magnitude is not the problem.
        mags = payload["mechanism_diagnostics"]["bias_magnitude_by_config"]
        self.assertGreater(max(mags.values()), 1.0)

        # No configuration beats the no-feedback control.
        control = payload["control_macro_f1"]
        for name, stats in payload["results_multi_seed"].items():
            if name == "control_headonly":
                continue
            self.assertLessEqual(
                stats["mean"], control, f"{name} now beats the control; re-open Phase 2"
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
