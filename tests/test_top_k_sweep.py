from __future__ import annotations

import unittest

import torch

from src.pipeline.step4.train_feedback import FoldTrainingResult, _build_model
from src.pipeline.step4.feedback_config import (
    load_feedback_config,
    write_selected_feedback_config,
)
from src.pipeline.step4.sweep_top_k import (
    DEFAULT_CANDIDATES,
    _pooled_macro_f1,
    select_best_candidate,
)
from src.pipeline.step4.sweep_semantic_confidence import (
    DEFAULT_CANDIDATES as DEFAULT_CONFIDENCE_CANDIDATES,
    select_best_candidate as select_best_confidence_candidate,
)


class FeedbackTopKSweepTest(unittest.TestCase):
    def test_build_model_propagates_top_k_percent(self) -> None:
        model = _build_model(top_k_percent=27.0)
        self.assertEqual(model.selector.top_k_percent, 27.0)

    def test_fold_training_result_fields(self) -> None:
        result = FoldTrainingResult(
            logits=torch.empty(0, 10),
            iterations=[],
            best_val_macro_f1=0.7,
            test_macro_f1=0.6,
        )
        self.assertEqual(result.best_val_macro_f1, 0.7)
        self.assertEqual(result.test_macro_f1, 0.6)

    def test_default_candidates_cover_every_integer(self) -> None:
        self.assertEqual(DEFAULT_CANDIDATES, tuple(range(15, 36)))

    def test_default_confidence_candidates_cover_tenths(self) -> None:
        self.assertEqual(
            DEFAULT_CONFIDENCE_CANDIDATES,
            tuple(round(value / 10, 1) for value in range(1, 11)),
        )

    def test_confidence_selection_uses_validation_only(self) -> None:
        rows = [
            {
                "bias_confidence_fraction": 0.4,
                "mean_best_val_macro_f1": 0.71,
                "pooled_oof_test_macro_f1": 0.99,
            },
            {
                "bias_confidence_fraction": 0.6,
                "mean_best_val_macro_f1": 0.73,
                "pooled_oof_test_macro_f1": 0.10,
            },
        ]
        selected = select_best_confidence_candidate(rows)
        self.assertEqual(selected["bias_confidence_fraction"], 0.6)

    def test_selection_uses_validation_and_breaks_ties_toward_smaller_n(
        self,
    ) -> None:
        rows = [
            {
                "top_k_percent": 25,
                "mean_best_val_macro_f1": 0.70,
                "pooled_oof_test_macro_f1": 0.99,
            },
            {
                "top_k_percent": 26,
                "mean_best_val_macro_f1": 0.72,
                "pooled_oof_test_macro_f1": 0.10,
            },
            {
                "top_k_percent": 27,
                "mean_best_val_macro_f1": 0.72,
                "pooled_oof_test_macro_f1": 1.00,
            },
        ]
        self.assertEqual(select_best_candidate(rows)["top_k_percent"], 26)

    def test_pooled_macro_f1_masks_dropped_classes(self) -> None:
        labels = torch.tensor([0, 1, 2])
        logits = torch.tensor(
            [
                [3.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 3.0],
            ]
        )

        score = _pooled_macro_f1(
            labels=labels,
            logits=logits,
            eval_classes=(0, 1),
            dropped_classes=(2,),
        )

        self.assertEqual(score, 1.0)

    def test_selected_feedback_config_round_trips_without_test_selection(self) -> None:
        with self.subTest("round_trip"):
            selected = {
                "top_k_percent": 18,
                "mean_best_val_macro_f1": 0.72,
                "std_best_val_macro_f1": 0.03,
                "pooled_oof_test_macro_f1": 0.69,
                "sweep_summary_path": "summary.json",
            }
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                write_selected_feedback_config("ton_iot", selected, root=tmp)
                payload = load_feedback_config("ton_iot", root=tmp)

            self.assertEqual(payload["source"], "validation_sweep")
            self.assertEqual(payload["top_k_percent"], 18.0)
            self.assertFalse(payload["selection_uses_test_labels"])

    def test_selected_feedback_config_records_trained_head(self) -> None:
        selected = {
            "top_k_percent": 21,
            "mean_best_val_macro_f1": 0.74,
            "std_best_val_macro_f1": 0.02,
            "pooled_oof_test_macro_f1": 0.70,
            "sweep_summary_path": "summary.json",
        }
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            write_selected_feedback_config(
                "ton_iot",
                selected,
                root=tmp,
                semantic_consultant="trained_oof_head",
                trained_llm_head=True,
            )
            payload = load_feedback_config("ton_iot", root=tmp)

        self.assertEqual(payload["semantic_consultant"], "trained_oof_head")
        self.assertTrue(payload["trained_llm_head"])


if __name__ == "__main__":
    unittest.main()
