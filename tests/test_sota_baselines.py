from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from src.pipeline.common.datasets import get_dataset_config


RUNGS = ("gnn", "llm", "agaf", "feedback", "head_alone")
MODELS = ("e_graphsage", "te_g_sage")
MODES = ("as_published", "refit", "plus_node_features")
BLOCKS = ("statistical_comparisons_3seed", "seed_matched_comparisons_3seed")

# The cells where a rung is NOT separated ABOVE a FAITHFUL baseline, recorded
# from the 2026-09-07 3-seed run. These are facts about the data, pinned so a
# later edit cannot quietly turn a null into a win — or the reverse. UNSW has
# none; ToN's list is the interesting half of the result.
NOT_SEPARATED_ABOVE_FAITHFUL = {
    ("unsw_nb15", "statistical_comparisons_3seed"): set(),
    ("unsw_nb15", "seed_matched_comparisons_3seed"): set(),
    ("ton_iot", "statistical_comparisons_3seed"): {
        "gnn_vs_e_graphsage_as_published",
        "llm_vs_e_graphsage_as_published",
        "agaf_vs_e_graphsage_as_published",
        "gnn_vs_e_graphsage_refit",
        "llm_vs_e_graphsage_refit",
        "agaf_vs_e_graphsage_refit",
    },
    ("ton_iot", "seed_matched_comparisons_3seed"): {
        "gnn_vs_e_graphsage_as_published",
        "llm_vs_e_graphsage_as_published",
        "agaf_vs_e_graphsage_as_published",
        "gnn_vs_e_graphsage_refit",
        "llm_vs_e_graphsage_refit",
        "agaf_vs_e_graphsage_refit",
    },
}

# The LLM (prototype) rung is separated BELOW E-GraphSAGE on ToN, which is why it
# appears above: "not separated above" covers both a null and a separated loss.
LLM_SEPARATED_BELOW_ON_TON = {
    "llm_vs_e_graphsage_as_published",
    "llm_vs_e_graphsage_refit",
    "llm_vs_e_graphsage_plus_node_features",
    "llm_vs_te_g_sage_plus_node_features",
}

# Under the trained-head consultant (contracts schema 5/7, 2026-09-07) the loop and
# the head-alone baseline clear every faithful baseline on BOTH datasets. Recorded
# so a later edit cannot quietly lose that, or invent it where it does not hold.
ALWAYS_SEPARATED_ABOVE_FAITHFUL = ("feedback", "head_alone")


def _payload(dataset: str) -> dict:
    return json.loads(Path(f"results/{dataset}_sota_baselines.json").read_text())


class StatisticalComparison3SeedTest(unittest.TestCase):
    """Both sides are three seeds now. The schema-1 blocks held the rung at seed
    42 only, which made every interval too narrow; they are kept under
    `superseded` and must not be read as current."""

    def test_every_baseline_has_both_intervals_against_each_rung(self) -> None:
        for dataset in ("unsw_nb15", "ton_iot"):
            payload = _payload(dataset)
            self.assertEqual(payload["schema_version"], 2)
            for block in BLOCKS:
                comparisons = payload[block]
                # Assert the expected keys EXIST. Iterating whatever happens to
                # be present would pass vacuously on an empty dict.
                for rung in RUNGS:
                    for model in MODELS:
                        for mode in MODES:
                            key = f"{rung}_vs_{model}_{mode}"
                            with self.subTest(dataset=dataset, block=block, key=key):
                                entry = comparisons[key]
                                for field in ("mean_diff", "ci_low", "ci_high",
                                              "prob_positive", "separated",
                                              "sign_stable_across_seed_pairs"):
                                    self.assertIn(field, entry, key)
                                self.assertLessEqual(entry["ci_low"], entry["mean_diff"])
                                self.assertGreaterEqual(entry["ci_high"], entry["mean_diff"])
                                # Both sides at three seeds is the whole point.
                                self.assertEqual(entry["rung_seeds"], 3)
                                self.assertEqual(entry["baseline_seeds"], 3)
                                self.assertEqual(
                                    entry["separated"],
                                    entry["ci_low"] > 0 or entry["ci_high"] < 0,
                                )
                                self.assertEqual(
                                    sorted(entry["per_seed_pair_diff"]),
                                    sorted(["42", "1", "2"]),
                                )
            self.assertEqual(
                payload["statistical_comparisons_3seed"]["gnn_vs_e_graphsage_refit"][
                    "resampled"
                ],
                "edges_and_both_seeds_independently",
            )
            self.assertEqual(
                payload["seed_matched_comparisons_3seed"]["gnn_vs_e_graphsage_refit"][
                    "resampled"
                ],
                "edges_only_seed_matched",
            )
            self.assertTrue(payload["comparison_caveat"].strip())

    def test_only_the_llm_rung_is_marked_seed_invariant(self) -> None:
        """The LLM rung is an argmax over the frozen prototype scorer, so its
        three rows are identical; the other three rungs genuinely vary."""
        for dataset in ("unsw_nb15", "ton_iot"):
            payload = _payload(dataset)
            for block in BLOCKS:
                for key, entry in payload[block].items():
                    with self.subTest(dataset=dataset, block=block, key=key):
                        self.assertEqual(
                            entry["rung_is_seed_invariant"],
                            key.startswith("llm_vs_"),
                        )

    def test_the_recorded_separations_against_faithful_baselines(self) -> None:
        """Assert what the data says, not what we would like it to say. Every
        rung/faithful-baseline pair NOT in the recorded set must be separated and
        positive; every pair in it must not be."""
        for dataset in ("unsw_nb15", "ton_iot"):
            payload = _payload(dataset)
            faithful = {
                f"{model}_{mode}"
                for model, entries in payload["baselines"].items()
                for mode, entry in entries.items()
                if entry["is_faithful_to_paper"]
            }
            self.assertEqual(len(faithful), 4, "expected 4 faithful configs")
            for block in BLOCKS:
                observed = set()
                for key, entry in payload[block].items():
                    if key.split("_vs_", 1)[1] not in faithful:
                        continue
                    if not (entry["separated"] and entry["mean_diff"] > 0):
                        observed.add(key)
                with self.subTest(dataset=dataset, block=block):
                    self.assertEqual(
                        observed, NOT_SEPARATED_ABOVE_FAITHFUL[(dataset, block)]
                    )

    def test_the_loop_and_head_alone_clear_every_faithful_baseline(self) -> None:
        """The two rungs that carry the system's claim. Both datasets, both
        interval types, all four faithful configurations."""
        for dataset in ("unsw_nb15", "ton_iot"):
            payload = _payload(dataset)
            faithful = {
                f"{model}_{mode}"
                for model, entries in payload["baselines"].items()
                for mode, entry in entries.items()
                if entry["is_faithful_to_paper"]
            }
            for block in BLOCKS:
                for rung in ALWAYS_SEPARATED_ABOVE_FAITHFUL:
                    for config in sorted(faithful):
                        key = f"{rung}_vs_{config}"
                        with self.subTest(dataset=dataset, block=block, key=key):
                            entry = payload[block][key]
                            self.assertTrue(entry["separated"], key)
                            self.assertGreater(entry["mean_diff"], 0.0, key)

    def test_the_llm_rung_is_separated_below_e_graphsage_on_ton(self) -> None:
        payload = _payload("ton_iot")
        for block in BLOCKS:
            for key in LLM_SEPARATED_BELOW_ON_TON:
                with self.subTest(block=block, key=key):
                    entry = payload[block][key]
                    self.assertTrue(entry["separated"], key)
                    self.assertLess(entry["mean_diff"], 0.0, key)

    def test_two_level_and_seed_matched_never_disagree_on_sign(self) -> None:
        """They may disagree on separation — the seed-matched interval is paired
        and therefore tighter — but a sign flip would mean the two are describing
        different comparisons."""
        for dataset in ("unsw_nb15", "ton_iot"):
            payload = _payload(dataset)
            for key, two_level in payload["statistical_comparisons_3seed"].items():
                matched = payload["seed_matched_comparisons_3seed"][key]
                with self.subTest(dataset=dataset, key=key):
                    self.assertEqual(
                        two_level["mean_diff"] > 0, matched["mean_diff"] > 0, key
                    )

    def test_the_single_seed_blocks_are_superseded_not_deleted(self) -> None:
        for dataset in ("unsw_nb15", "ton_iot"):
            payload = _payload(dataset)
            self.assertNotIn("statistical_comparisons", payload)
            self.assertNotIn("seed_matched_comparisons", payload)
            superseded = payload["superseded"]
            self.assertEqual(superseded["schema_version"], 1)
            self.assertIn("rung_seeds was 1", superseded["note"])
            # The schema-1 blocks predate the head_alone rung, so they carry only
            # the four rungs that existed then.
            for rung in ("gnn", "llm", "agaf", "feedback"):
                key = f"{rung}_vs_e_graphsage_refit"
                self.assertEqual(
                    superseded["statistical_comparisons"][key]["rung_seeds"], 1
                )
                self.assertIn(key, superseded["seed_matched_comparisons"])

    def test_the_prototype_consultant_3seed_block_is_kept(self) -> None:
        """Re-running the comparison after the loop's consultant changed must file
        the earlier consultant's intervals, not overwrite them. It overwrote them
        once and they had to be recovered from git."""
        for dataset in ("unsw_nb15", "ton_iot"):
            payload = _payload(dataset)
            prior = payload["superseded"]["whitened_prototype_scorer_3seed"]
            for block in BLOCKS:
                self.assertIn(block, prior)
                entry = prior[block]["feedback_vs_e_graphsage_refit"]
                self.assertEqual(entry["rung_seeds"], 3)
            self.assertEqual(
                payload["comparison_sources_3seed"]["loop_consultant"],
                "trained_llm_head",
            )


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
