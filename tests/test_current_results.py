from __future__ import annotations

import json
import unittest
from pathlib import Path

from reproduce_ladder import load_expected_accuracy, load_expected_ladder


RESULTS_PATH = Path("results/unsw_nb15_current.json")


class CurrentResultsContractTest(unittest.TestCase):
    def test_authoritative_unsw_ladder_and_configuration(self) -> None:
        payload = json.loads(RESULTS_PATH.read_text())
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


if __name__ == "__main__":
    unittest.main()
