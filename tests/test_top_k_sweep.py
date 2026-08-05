from __future__ import annotations

import unittest

import torch

from src.pipeline.step4.train_feedback import FoldTrainingResult, _build_model
from src.pipeline.step4.sweep_top_k import (
    DEFAULT_CANDIDATES,
    select_best_candidate,
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
        self.assertEqual(DEFAULT_CANDIDATES, tuple(range(25, 36)))

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


if __name__ == "__main__":
    unittest.main()
