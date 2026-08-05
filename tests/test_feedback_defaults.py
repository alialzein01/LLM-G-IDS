from __future__ import annotations

import unittest

import torch

from src.models.feedback_classifier import UncertaintySelector
from src.pipeline.step4.train_feedback import (
    BIAS_CONFIDENCE_FRAC,
    TOP_K_PERCENT,
    _build_benchmark_summary,
    _build_model,
)


class FeedbackDefaultsTest(unittest.TestCase):
    def test_validation_selected_entropy_percentage_is_default(self) -> None:
        self.assertEqual(TOP_K_PERCENT, 16.0)
        self.assertEqual(_build_model().selector.top_k_percent, 16.0)
        self.assertEqual(UncertaintySelector().top_k_percent, 16.0)

    def test_benchmark_summary_records_accuracy_and_configuration(self) -> None:
        labels = torch.tensor([0, 1])
        logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
        summary = _build_benchmark_summary(
            dataset="unsw_nb15",
            labels=labels,
            oof_by_mode={"real": logits},
            pooled_f1={"real": 0.75},
            target_agaf=0.70,
        )

        self.assertEqual(summary["pooled_cv_accuracy"], 1.0)
        self.assertEqual(summary["top_k_percent"], TOP_K_PERCENT)
        self.assertEqual(summary["bias_confidence_fraction"], BIAS_CONFIDENCE_FRAC)
        self.assertEqual(summary["semantic_consultant"], "whitened_prototype_scorer")
        self.assertEqual(summary["target_agaf"], 0.70)

        head_summary = _build_benchmark_summary(
            dataset="unsw_nb15",
            labels=labels,
            oof_by_mode={"real": logits},
            pooled_f1={"real": 0.75},
            target_agaf=0.70,
            use_llm_head=True,
        )
        self.assertEqual(head_summary["semantic_consultant"], "trained_llm_head")
        self.assertEqual(head_summary["trained_llm_head"], True)


if __name__ == "__main__":
    unittest.main()
