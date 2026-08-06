from __future__ import annotations

import unittest

from src.pipeline.common.run_pipeline import run_all


class PipelineRunnerTest(unittest.TestCase):
    def test_structural10_is_default_feature_profile(self) -> None:
        manifest = run_all(dataset="ton_iot", only=["features"], dry_run=True)

        self.assertEqual(manifest["dataset"], "ton_iot")
        self.assertEqual(manifest["feature_profile"], "structural10")
        self.assertIn("features", manifest["stages"])

    def test_fusion_uses_oof_gnn_embeddings(self) -> None:
        manifest = run_all(dataset="ton_iot", only=["fusion"], dry_run=True)

        fusion = manifest["stages"]["fusion"]
        self.assertEqual(fusion["module"], "src.pipeline.step3.train_fusion")
        self.assertIn("--gnn-emb-path", fusion["args"])
        self.assertTrue(
            any(str(arg).endswith("edge_embeddings_oof.pt") for arg in fusion["args"])
        )

    def test_top_k_sweep_uses_validation_range(self) -> None:
        manifest = run_all(dataset="ton_iot", only=["top_k_sweep"], dry_run=True)

        sweep = manifest["stages"]["top_k_sweep"]
        self.assertEqual(sweep["module"], "src.pipeline.step4.sweep_top_k")
        self.assertIn("--min-percent", sweep["args"])
        self.assertIn("15", sweep["args"])
        self.assertIn("--max-percent", sweep["args"])
        self.assertIn("35", sweep["args"])

    def test_enhanced12_is_ton_only(self) -> None:
        with self.assertRaisesRegex(ValueError, "only for ToN-IoT"):
            run_all(
                dataset="unsw_nb15",
                feature_profile="enhanced12",
                only=["features"],
                dry_run=True,
            )


if __name__ == "__main__":
    unittest.main()
