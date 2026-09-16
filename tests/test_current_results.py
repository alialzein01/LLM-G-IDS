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
    """Contracts are 3-training-seed measurements (schema 5 UNSW / 7 ToN, 2026-09-07).

    The loop's semantic consultant is now the per-fold trained head; the LLM rung and
    AGAF keep the whitened prototype encoder, and the head alone is carried as its own
    rung because it is BOTH the strongest LLM-only baseline and the thing the loop
    consults. These tests pin what the data shows, including the comparisons that are
    NOT separated -- above all loop-vs-head_alone, which is the one a reader must not
    be allowed to lose sight of.
    """

    RUNGS = ("gnn", "llm", "agaf", "feedback", "head_alone")
    SEPARATED = (
        "feedback_vs_agaf", "feedback_vs_gnn", "feedback_vs_llm", "head_alone_vs_gnn",
    )
    NOT_SEPARATED = ("agaf_vs_gnn", "feedback_vs_head_alone")
    PARITY_RULE = (
        "same encoder, same graph, same folds, same seeds on every rung; the loop "
        "additionally trains a classification head on the semantic embeddings"
    )

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

        # Exactly these comparisons are separated, and exactly these are not.
        self.assertEqual(
            sorted(ms["separated_comparisons_two_level"]), sorted(self.SEPARATED)
        )
        self.assertEqual(
            sorted(ms["not_separated_comparisons_two_level"]),
            sorted(self.NOT_SEPARATED),
        )
        self.assertFalse(ms["nothing_separated"])
        for name in self.SEPARATED:
            c = ms["comparisons"][name]
            self.assertEqual(
                c["two_level"]["resampled"], "edges_and_both_seeds_independently"
            )
            self.assertTrue(c["separated_two_level"], name)
            self.assertGreater(c["two_level"]["mean_diff"], 0.0, name)
        for name in self.NOT_SEPARATED:
            c = ms["comparisons"][name]
            self.assertFalse(c["separated_two_level"], name)
            self.assertLess(c["two_level"]["ci_low"], 0.0, name)
            self.assertGreater(c["two_level"]["ci_high"], 0.0, name)
        for name, c in ms["comparisons"].items():
            self.assertEqual(payload["statistical_comparisons"][name], c["two_level"])

        # The loop does NOT beat the consultant it consults. This is the single
        # fact most likely to be dropped when the ladder is written up, so it is
        # pinned three ways: the interval, the sign instability, and the sentence.
        head = ms["comparisons"]["feedback_vs_head_alone"]
        self.assertFalse(head["separated_two_level"])
        self.assertFalse(head["separated_seed_matched"])
        self.assertFalse(head["sign_stable_across_seeds"])
        self.assertLess(head["two_level"]["mean_diff"], 0.0)
        self.assertIn("NOT separated", payload["loop_vs_head_alone_headline"])
        self.assertEqual(ms["highest_mean_rung"], "head_alone")
        self.assertFalse(payload["loop_highest_mean"])
        self.assertFalse(payload["ladder_order_holds"])
        self.assertNotIn("loop_is_top_rung", payload)  # banned wording

        # Architecture, under the new parity rule.
        cfg = payload["configuration"]
        self.assertEqual(cfg["semantic_consultant"], "trained_llm_head")
        self.assertTrue(cfg["trained_llm_head"])
        self.assertEqual(cfg["llm_rung_consultant"], "whitened_prototype")
        self.assertFalse(cfg["agaf_head_fusion"])
        self.assertEqual(cfg["injection_mode"], "edge")
        self.assertEqual(payload["architecture_parity_rule"], self.PARITY_RULE)
        self.assertEqual(payload["edge_attr_encoding"], "v2_log_cont_cat_idx")
        self.assertEqual(
            payload["selection"]["swept_under_injection_mode"], cfg["injection_mode"]
        )
        # The knob curves are flat; the contract must say so rather than implying
        # the knobs were tuned.
        self.assertTrue(cfg["selection_curve_is_flat"])
        self.assertTrue(payload["selection"]["curve_is_flat"])

        # The superseded prototype ladder must remain recoverable.
        sup = payload["supersedes"]
        self.assertEqual(sup["consultant"], "whitened_prototype_scorer")
        self.assertIn("multi_seed", sup)
        self.assertIn("configuration", sup)

    def test_authoritative_unsw_ladder_and_configuration(self) -> None:
        payload = json.loads(UNSW_RESULTS_PATH.read_text())
        self._assert_multi_seed_contract(payload, schema=5)
        cfg = payload["configuration"]
        self.assertEqual(cfg["top_k_percent"], 29.0)
        self.assertEqual(cfg["injection_scale"], 2.0)
        ms = payload["multi_seed"]
        # GNN, LLM and AGAF are untouched by the consultant change and must equal
        # the values the superseded schema-4 contract recorded.
        sup = payload["supersedes"]["multi_seed"]["rungs"]
        for rung in ("gnn", "llm", "agaf"):
            self.assertAlmostEqual(
                ms["rungs"][rung]["macro_f1_mean"], sup[rung]["macro_f1_mean"]
            )
        self.assertEqual(payload["supersedes"]["schema_version"], 4)
        self.assertGreater(ms["rungs"]["feedback"]["macro_f1_mean"], 0.83)
        self.assertGreater(ms["rungs"]["head_alone"]["macro_f1_mean"],
                           ms["rungs"]["feedback"]["macro_f1_mean"])

    def test_authoritative_ton_iot_aggregated_ladder(self) -> None:
        payload = json.loads(TON_RESULTS_PATH.read_text())
        self._assert_multi_seed_contract(payload, schema=7)
        self.assertEqual(payload["dataset_key"], "ton_iot")
        self.assertEqual(payload["graph_scope"], "aggregated")
        self.assertEqual(payload["n_eval_classes"], 8)
        cfg = payload["configuration"]
        self.assertEqual(cfg["top_k_percent"], 16.0)
        self.assertEqual(cfg["injection_scale"], 20.0)
        # ToN's scale selected on the upper boundary of the swept range; the
        # contract has to carry that, not bury it.
        self.assertTrue(cfg["scale_on_range_boundary"])
        self.assertFalse(cfg["top_k_on_range_boundary"])
        self.assertEqual(payload["supersedes"]["schema_version"], 6)
        ms = payload["multi_seed"]
        sup = payload["supersedes"]["multi_seed"]["rungs"]
        for rung in ("gnn", "llm", "agaf"):
            self.assertAlmostEqual(
                ms["rungs"][rung]["macro_f1_mean"], sup[rung]["macro_f1_mean"]
            )
        # AGAF still sits below the GNN by mean on ToN, and still not separated.
        self.assertLess(ms["rungs"]["agaf"]["macro_f1_mean"],
                        ms["rungs"]["gnn"]["macro_f1_mean"])
        self.assertFalse(ms["comparisons"]["agaf_vs_gnn"]["separated_two_level"])

    def test_the_two_datasets_separate_the_same_comparisons(self) -> None:
        """Not required by anything -- recorded because it is true and surprising:
        after the consultant change the separated/not-separated split is identical
        on both datasets, which it never was under the prototype."""
        unsw = json.loads(UNSW_RESULTS_PATH.read_text())["multi_seed"]
        ton = json.loads(TON_RESULTS_PATH.read_text())["multi_seed"]
        self.assertEqual(
            sorted(unsw["separated_comparisons_two_level"]),
            sorted(ton["separated_comparisons_two_level"]),
        )
        self.assertEqual(
            sorted(unsw["not_separated_comparisons_two_level"]),
            sorted(ton["not_separated_comparisons_two_level"]),
        )

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

    def test_both_datasets_declare_the_same_architecture(self) -> None:
        """Parity guard, under the 2026-09-07 rule: same encoder, same graph, same
        folds, same seeds on every rung; the loop additionally trains a
        classification head on the semantic embeddings. Only `top_k_percent` and
        `injection_scale` may differ between datasets."""
        unsw_p = json.loads(UNSW_RESULTS_PATH.read_text())
        ton_p = json.loads(TON_RESULTS_PATH.read_text())
        unsw, ton = unsw_p["configuration"], ton_p["configuration"]
        shared = (
            "semantic_consultant", "trained_llm_head", "llm_rung_consultant",
            "agaf_head_fusion", "semantic_confidence_fraction", "injection_mode",
            "max_feedback_iterations", "churn_tolerance", "seeds",
            "deterministic_cpu",
        )
        for field in shared:
            self.assertIn(field, unsw, field)
            self.assertEqual(unsw[field], ton[field], f"parity drifted on {field}")
        self.assertEqual(
            unsw_p["architecture_parity_rule"], ton_p["architecture_parity_rule"]
        )
        self.assertEqual(unsw_p["architecture_parity_rule"], self.PARITY_RULE)
        self.assertEqual(
            unsw_p["edge_attr_encoding"], ton_p["edge_attr_encoding"]
        )
        # The two knobs that are allowed to differ, and do.
        self.assertNotEqual(unsw["top_k_percent"], ton["top_k_percent"])
        for key in ("top_k_percent", "injection_scale"):
            self.assertIn(key, unsw)
            self.assertIn(key, ton)

        comparison = json.loads(COMPARISON_PATH.read_text())
        self.assertEqual(comparison["schema_version"], 6)
        arch = comparison["architecture"]
        self.assertTrue(arch["shared"])
        self.assertEqual(arch["parity_rule"], self.PARITY_RULE)
        self.assertEqual(arch["semantic_consultant"], unsw["semantic_consultant"])
        self.assertEqual(arch["llm_rung_consultant"], unsw["llm_rung_consultant"])
        for key in ("unsw_nb15", "ton_iot"):
            d = comparison["datasets"][key]
            self.assertFalse(d["nothing_separated"])
            self.assertEqual(d["highest_mean_rung"], "head_alone")

    def test_cross_dataset_comparison_matches_contracts(self) -> None:
        comparison = json.loads(COMPARISON_PATH.read_text())
        for key, path in (("unsw_nb15", UNSW_RESULTS_PATH),
                          ("ton_iot", TON_RESULTS_PATH)):
            contract = json.loads(path.read_text())
            block = comparison["datasets"][key]
            self.assertEqual(block["schema_version"], contract["schema_version"])
            for rung in self.RUNGS:
                self.assertAlmostEqual(
                    block["rungs"][rung]["mean"],
                    contract["multi_seed"]["rungs"][rung]["macro_f1_mean"],
                )
                self.assertAlmostEqual(
                    block["rungs"][rung]["std"],
                    contract["multi_seed"]["rungs"][rung]["macro_f1_std"],
                )
            self.assertEqual(block["mean_order"], contract["multi_seed"]["mean_order"])
            for name, delta_key in (
                ("agaf", "observed_deltas_feedback_minus_agaf_3seed_mean"),
                ("gnn", "observed_deltas_feedback_minus_gnn_3seed_mean"),
                ("head_alone", "observed_deltas_feedback_minus_head_alone_3seed_mean"),
            ):
                rungs = contract["multi_seed"]["rungs"]
                self.assertAlmostEqual(
                    comparison[delta_key][key],
                    rungs["feedback"]["macro_f1_mean"] - rungs[name]["macro_f1_mean"],
                )
        # The loop trails the head alone on BOTH datasets by 3-seed mean.
        for key in ("unsw_nb15", "ton_iot"):
            self.assertLess(
                comparison["observed_deltas_feedback_minus_head_alone_3seed_mean"][key],
                0.0,
            )
        self.assertEqual(
            comparison["supersedes"]["schema_version"], 5
        )

    def test_the_head_only_baseline_is_carried_as_a_rung_and_is_not_beaten(self) -> None:
        """The strongest LLM-only baseline is now the loop's own consultant, so it
        is a rung in the contract rather than a side file. The claim it supports
        has changed and must be stated as it now is: by 3-seed mean it is above
        the loop on both datasets, but the difference is NOT separated, so the
        honest statement is "indistinguishable", not "beats".

        The v1-era results/*_head_baseline.json files are stale (2026-08-20, v1
        encoding) and must not be used for this comparison any more.
        """
        for path in (UNSW_RESULTS_PATH, TON_RESULTS_PATH):
            payload = json.loads(path.read_text())
            rungs = payload["multi_seed"]["rungs"]
            head, loop = rungs["head_alone"], rungs["feedback"]
            self.assertIn("consultant", head["role"])
            self.assertGreater(head["macro_f1_mean"], loop["macro_f1_mean"], str(path))
            c = payload["multi_seed"]["comparisons"]["feedback_vs_head_alone"]
            self.assertFalse(c["separated_two_level"], str(path))
            self.assertFalse(c["separated_seed_matched"], str(path))
        # ...and at seed 42 on UNSW the ordering actually flips, which is why the
        # mean alone must never be quoted as if it were a result.
        unsw = json.loads(UNSW_RESULTS_PATH.read_text())["results"]
        self.assertGreater(
            unsw["feedback"]["macro_f1"], unsw["head_alone"]["macro_f1"]
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
