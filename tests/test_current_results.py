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
UNSW_ORACLE_V2_PATH = Path("results/unsw_nb15_oracle_ceiling_v2.json")
TON_ORACLE_V2_PATH = Path("results/ton_iot_oracle_ceiling_v2.json")
UNSW_ORACLE_V2_HEAD_PATH = Path(
    "results/unsw_nb15_oracle_ceiling_v2_trained_head.json"
)
TON_ORACLE_V2_HEAD_PATH = Path(
    "results/ton_iot_oracle_ceiling_v2_trained_head.json"
)


class CurrentResultsContractTest(unittest.TestCase):
    """Contracts are 3-training-seed measurements (schema 4 UNSW / 6 ToN, 2026-09-06).

    The headline lives in ``multi_seed``. ``results`` holds the single seed-42 run that
    ``reproduce_ladder.py`` drift-checks against. Nothing is statistically separated on
    either dataset once training-seed variance enters the bootstrap, and these tests pin
    that so no later edit can quietly reintroduce a separation claim.
    """

    RUNGS = ("gnn", "llm", "agaf", "feedback")
    COMPARISONS = ("loop_vs_agaf", "loop_vs_gnn", "agaf_vs_gnn")

    def _assert_multi_seed_contract(self, payload: dict, schema: int) -> None:
        self.assertEqual(payload["schema_version"], schema)
        self.assertEqual(payload["headline"], "multi_seed")
        ms = payload["multi_seed"]
        self.assertEqual(ms["seeds"], [42, 1, 2])
        self.assertEqual(payload["configuration"]["seeds"], [42, 1, 2])
        self.assertTrue(payload["configuration"]["injection_scale_verified_in_run"])
        for rung in self.RUNGS:
            r = ms["rungs"][rung]
            self.assertEqual(sorted(r["macro_f1_per_seed"]), ["1", "2", "42"])
            vals = list(r["macro_f1_per_seed"].values())
            self.assertAlmostEqual(r["macro_f1_mean"], sum(vals) / 3)
            # results = the seed-42 run, for reproduce_ladder
            self.assertAlmostEqual(
                payload["results"][rung]["macro_f1"], r["macro_f1_per_seed"]["42"]
            )
        # The LLM rung is deterministic given folds + frozen embeddings.
        self.assertEqual(ms["rungs"]["llm"]["macro_f1_std"], 0.0)
        # NOTHING is separated: every two-level CI straddles zero.
        for name in self.COMPARISONS:
            c = ms["comparisons"][name]
            self.assertEqual(c["two_level"]["resampled"], "edges_and_training_seed")
            self.assertLess(c["two_level"]["ci_low"], 0.0, name)
            self.assertGreater(c["two_level"]["ci_high"], 0.0, name)
            self.assertFalse(c["separated"], name)
            self.assertEqual(payload["statistical_comparisons"][name], c["two_level"])
        self.assertTrue(ms["nothing_separated"])
        self.assertFalse(payload["ladder_order_holds"])
        self.assertNotIn("loop_is_top_rung", payload)  # banned wording; use *_highest_mean
        # Shared architecture must still be declared.
        cfg = payload["configuration"]
        self.assertEqual(cfg["semantic_consultant"], "whitened_prototype")
        self.assertFalse(cfg["trained_llm_head"])
        self.assertFalse(cfg["agaf_head_fusion"])
        self.assertEqual(cfg["injection_mode"], "edge")
        self.assertEqual(payload["edge_attr_encoding"], "v2_log_cont_cat_idx")
        self.assertEqual(
            payload["selection"]["swept_under_injection_mode"], cfg["injection_mode"]
        )

    def test_authoritative_unsw_ladder_and_configuration(self) -> None:
        payload = json.loads(UNSW_RESULTS_PATH.read_text())
        self._assert_multi_seed_contract(payload, schema=4)
        cfg = payload["configuration"]
        self.assertEqual(cfg["top_k_percent"], 31.0)
        self.assertEqual(cfg["injection_scale"], 2.0)
        ms = payload["multi_seed"]
        # GNN and LLM reproduce the schema-3 seed-42 values exactly; AGAF and loop do not.
        self.assertAlmostEqual(payload["results"]["gnn"]["macro_f1"], 0.7219365593805762)
        self.assertAlmostEqual(payload["results"]["llm"]["macro_f1"], 0.7353188026257084)
        self.assertEqual(payload["supersedes"]["schema_version"], 3)
        self.assertAlmostEqual(
            payload["supersedes"]["results"]["feedback"]["macro_f1"], 0.772755270351248
        )
        # 3-seed ordering by mean: AGAF highest, loop below it at EVERY seed. Not separated.
        self.assertEqual(ms["highest_mean_rung"], "agaf")
        la = ms["comparisons"]["loop_vs_agaf"]
        self.assertTrue(la["sign_stable_across_seeds"])
        self.assertLess(la["two_level"]["mean_diff"], 0.0)
        self.assertTrue(all(v < 0 for v in la["per_seed_diff"].values()))
        # The old multi_seed_caveat was conversation-recorded; it is now resolved by artifact.
        self.assertNotIn("multi_seed_caveat", payload)
        self.assertIn("multi_seed_caveat", payload["supersedes"])

    def test_reproduction_uses_authoritative_manifest(self) -> None:
        payload = json.loads(UNSW_RESULTS_PATH.read_text())
        expected = load_expected_ladder()
        per42 = {r: payload["multi_seed"]["rungs"][r]["macro_f1_per_seed"]["42"] for r in self.RUNGS}
        self.assertAlmostEqual(expected["gnn_alone"], per42["gnn"])
        self.assertAlmostEqual(expected["llm_alone"], per42["llm"])
        self.assertAlmostEqual(expected["agaf"], per42["agaf"])
        self.assertAlmostEqual(expected["feedback_loop"], per42["feedback"])
        accuracy = load_expected_accuracy()
        self.assertAlmostEqual(
            accuracy["feedback_loop"], payload["results"]["feedback"]["accuracy"]
        )

    def test_authoritative_ton_iot_aggregated_ladder(self) -> None:
        """ToN runs the SAME architecture as UNSW. By 3-seed mean the loop is highest and
        AGAF sits below the GNN, but neither ordering is separated -- the schema-5
        'AGAF significantly below GNN (P=0.0005)' was a seed-42 artifact."""
        payload = json.loads(TON_RESULTS_PATH.read_text())
        self._assert_multi_seed_contract(payload, schema=6)
        self.assertEqual(payload["dataset_key"], "ton_iot")
        self.assertEqual(payload["graph_scope"], "aggregated")
        self.assertEqual(payload["feature_profile"], "structural10")
        self.assertEqual(payload["n_eval_classes"], 8)
        cfg = payload["configuration"]
        self.assertEqual(cfg["top_k_percent"], 25.0)
        self.assertEqual(cfg["injection_scale"], 20.0)
        self.assertAlmostEqual(payload["results"]["gnn"]["macro_f1"], 0.4289705488338377)
        self.assertAlmostEqual(payload["results"]["llm"]["macro_f1"], 0.27852446280044896)
        self.assertEqual(payload["supersedes"]["schema_version"], 5)
        ms = payload["multi_seed"]
        self.assertEqual(ms["highest_mean_rung"], "feedback")
        self.assertTrue(payload["loop_highest_mean"])
        self.assertTrue(payload["llm_below_gnn"])
        self.assertTrue(payload["agaf_below_gnn_by_mean"])
        # ...but AGAF-vs-GNN is NOT separated and not even sign-stable across seeds.
        ag = ms["comparisons"]["agaf_vs_gnn"]
        self.assertFalse(ag["sign_stable_across_seeds"])
        self.assertLess(ag["two_level"]["ci_low"], 0.0)
        self.assertGreater(ag["two_level"]["ci_high"], 0.0)

    def test_both_datasets_declare_the_same_architecture(self) -> None:
        """Parity guard: the two datasets must never drift onto different consultants again."""
        unsw_p = json.loads(UNSW_RESULTS_PATH.read_text())
        ton_p = json.loads(TON_RESULTS_PATH.read_text())
        unsw, ton = unsw_p["configuration"], ton_p["configuration"]
        self.assertEqual(unsw["semantic_consultant"], ton["semantic_consultant"])
        self.assertEqual(unsw["trained_llm_head"], ton["trained_llm_head"])
        self.assertEqual(
            unsw["semantic_confidence_fraction"], ton["semantic_confidence_fraction"]
        )
        self.assertEqual(unsw["injection_mode"], ton["injection_mode"])
        self.assertEqual(unsw["seeds"], ton["seeds"])
        # injection_scale and top_k_percent are the two quantities allowed to differ.
        for key in ("top_k_percent", "injection_scale"):
            self.assertIn(key, unsw)
            self.assertIn(key, ton)
        comparison = json.loads(COMPARISON_PATH.read_text())
        self.assertEqual(comparison["schema_version"], 5)
        self.assertTrue(comparison["architecture"]["shared"])
        self.assertEqual(
            comparison["architecture"]["semantic_consultant"], unsw["semantic_consultant"]
        )
        for key in ("unsw_nb15", "ton_iot_aggregated"):
            self.assertTrue(comparison["datasets"][key]["nothing_separated"])
        self.assertEqual(comparison["datasets"]["unsw_nb15"]["highest_mean_rung"], "agaf")
        self.assertEqual(
            comparison["datasets"]["ton_iot_aggregated"]["highest_mean_rung"], "feedback"
        )

    def test_v2_oracle_contracts_declare_shared_architecture_and_evidence(self) -> None:
        """Gate 0 must be comparable across datasets and retain causal evidence."""
        for path in (UNSW_ORACLE_V2_PATH, TON_ORACLE_V2_PATH):
            self.assertTrue(
                path.exists(),
                f"Missing Gate 0 artifact {path}; run the prespecified v2 oracle experiment",
            )

        unsw = json.loads(UNSW_ORACLE_V2_PATH.read_text())
        ton = json.loads(TON_ORACLE_V2_PATH.read_text())
        shared_fields = (
            "edge_attr_encoding",
            "seeds",
            "fold_count",
            "max_iterations",
            "churn_tolerance",
            "bias_confidence_fraction",
            "gate_mode",
            "real_arm_semantic_consultant",
            "trained_llm_head",
            "use_output_fusion",
            "injection_modes",
            "oracle_logit_magnitude",
            "deterministic_cpu",
            "omp_num_threads",
            "mkl_num_threads",
            "torch_num_threads",
        )
        for field in shared_fields:
            self.assertIn(field, unsw["configuration"])
            self.assertEqual(
                unsw["configuration"][field],
                ton["configuration"][field],
                f"Gate 0 architecture drifted on {field}",
            )

        self.assertEqual(unsw["configuration"]["top_k_percent"], 31.0)
        self.assertEqual(ton["configuration"]["top_k_percent"], 25.0)
        self.assertEqual(unsw["configuration"]["injection_scale"], 2.0)
        self.assertEqual(ton["configuration"]["injection_scale"], 20.0)

        expected_arms = {
            "control_head_only",
            "real_prototype_edge",
            "oracle_edge",
            "oracle_attention",
        }
        for payload in (unsw, ton):
            self.assertIn("NEVER REPORTABLE", payload["status"])
            self.assertEqual(set(payload["arms"]), expected_arms)
            self.assertEqual(set(payload["results"]), expected_arms)
            for arm in expected_arms:
                result = payload["results"][arm]
                self.assertEqual(len(result["fold_records"]), 15)
                self.assertIn("iteration_evidence", result)
                for evidence in result["iteration_evidence"].values():
                    self.assertIn("wrong_to_correct", evidence)
                    self.assertIn("correct_to_wrong", evidence)
                    self.assertIn("mean_selection_jaccard_previous", evidence)
                    self.assertFalse(evidence["advice_recomputed"])
            for comparison in payload["paired_comparisons"].values():
                self.assertEqual(comparison["n_pairs"], 15)
                self.assertIn("ci_low", comparison)
                self.assertIn("ci_high", comparison)
            self.assertIn("headroom_below_0_02", payload["decision"])
            self.assertIn("stop_mechanism_surgery", payload["program_decision"])

    def test_gate05_trained_head_is_diagnostic_and_never_a_ladder_rung(self) -> None:
        for path in (UNSW_ORACLE_V2_HEAD_PATH, TON_ORACLE_V2_HEAD_PATH):
            self.assertTrue(
                path.exists(),
                f"Missing Gate 0.5 artifact {path}; regenerate heads and run the diagnostic",
            )

        payloads = [
            json.loads(UNSW_ORACLE_V2_HEAD_PATH.read_text()),
            json.loads(TON_ORACLE_V2_HEAD_PATH.read_text()),
        ]
        shared_fields = (
            "edge_attr_encoding",
            "seeds",
            "fold_count",
            "max_iterations",
            "churn_tolerance",
            "bias_confidence_fraction",
            "gate_mode",
            "use_output_fusion",
            "injection_modes",
            "oracle_logit_magnitude",
            "deterministic_cpu",
        )
        for field in shared_fields:
            self.assertEqual(
                payloads[0]["configuration"][field],
                payloads[1]["configuration"][field],
                f"Gate 0.5 architecture drifted on {field}",
            )

        expected_arms = {
            "control_head_only",
            "real_prototype_edge",
            "head_trained_edge",
            "oracle_edge",
            "oracle_attention",
        }
        for payload in payloads:
            self.assertIn("DIAGNOSTIC ONLY", payload["status"])
            self.assertEqual(set(payload["arms"]), expected_arms)
            head_contract = payload["arms"]["head_trained_edge"]
            self.assertEqual(head_contract["advice_source"], "trained_head")
            self.assertFalse(head_contract["use_output_fusion"])
            self.assertTrue(head_contract["diagnostic_only"])
            self.assertFalse(head_contract["eligible_for_ladder"])
            self.assertTrue(payload["configuration"]["trained_head_artifact_regenerated"])
            self.assertEqual(len(payload["results"]["head_trained_edge"]["fold_records"]), 15)
            comparison = payload["paired_comparisons"][
                "head_trained_edge_vs_control_head_only"
            ]
            self.assertEqual(comparison["n_pairs"], 15)
            self.assertIn("trained_head_captured_share", payload["decision"])

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

        unsw_ms = json.loads(UNSW_RESULTS_PATH.read_text())["multi_seed"]["rungs"]
        ton_ms = json.loads(TON_RESULTS_PATH.read_text())["multi_seed"]["rungs"]
        for name in ("gnn", "llm", "agaf", "feedback"):
            # seed-42 single run == each contract's results block
            self.assertAlmostEqual(
                comparison["datasets"]["unsw_nb15"]["macro_f1"][name],
                unsw[name]["macro_f1"],
            )
            self.assertAlmostEqual(
                comparison["datasets"]["ton_iot_aggregated"]["macro_f1"][name],
                ton[name]["macro_f1"],
            )
            # 3-seed headline == each contract's multi_seed block
            self.assertAlmostEqual(
                comparison["datasets"]["unsw_nb15"]["macro_f1_3seed_mean"][name],
                unsw_ms[name]["macro_f1_mean"],
            )
            self.assertAlmostEqual(
                comparison["datasets"]["ton_iot_aggregated"]["macro_f1_3seed_mean"][name],
                ton_ms[name]["macro_f1_mean"],
            )


if __name__ == "__main__":
    unittest.main()
