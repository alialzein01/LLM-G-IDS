from __future__ import annotations

import unittest

import torch

from src.models.feedback_classifier import UncertaintySelector
from src.pipeline.step4.train_feedback import (
    BIAS_CONFIDENCE_FRAC,
    DEFAULT_GATE_MODE,
    TOP_K_PERCENT,
    _build_benchmark_summary,
    _build_model,
)


class FeedbackDefaultsTest(unittest.TestCase):
    def test_validation_selected_entropy_percentage_is_default(self) -> None:
        self.assertEqual(TOP_K_PERCENT, 16.0)
        self.assertEqual(_build_model().selector.top_k_percent, 16.0)
        self.assertEqual(UncertaintySelector().top_k_percent, 16.0)
        self.assertEqual(_build_model().gate_mode, "confidence")
        self.assertEqual(DEFAULT_GATE_MODE, "confidence")

    def test_build_model_rejects_invalid_confidence_fraction(self) -> None:
        with self.assertRaises(ValueError):
            _build_model(bias_confidence_fraction=0.0)

    def test_build_model_rejects_invalid_gate_mode(self) -> None:
        with self.assertRaises(ValueError):
            _build_model(gate_mode="invalid")

    def test_gate_modes_rank_flagged_edges(self) -> None:
        flagged = torch.arange(4)
        llm_probs = torch.tensor(
            [
                [0.90, 0.05, 0.05],
                [0.10, 0.85, 0.05],
                [0.30, 0.60, 0.10],
                [0.45, 0.40, 0.15],
            ]
        )
        semantic_logits = llm_probs.log()
        gnn_probs = torch.tensor([[0.8, 0.1, 0.1]]).repeat(4, 1)

        confidence_model = _build_model(
            bias_confidence_fraction=0.5, gate_mode="confidence"
        )
        old_confidence = semantic_logits[flagged].softmax(dim=-1).max(dim=-1).values
        old_selection = flagged[torch.topk(old_confidence, 2).indices]
        self.assertTrue(
            torch.equal(
                confidence_model._gate_flagged(
                    flagged, semantic_logits, gnn_probs
                ),
                old_selection,
            )
        )

        disagreement_model = _build_model(
            bias_confidence_fraction=0.5, gate_mode="disagreement"
        )
        both_model = _build_model(bias_confidence_fraction=0.5, gate_mode="both")
        self.assertEqual(
            set(disagreement_model._gate_flagged(
                flagged, semantic_logits, gnn_probs
            ).tolist()),
            {1, 2},
        )
        self.assertEqual(
            set(both_model._gate_flagged(
                flagged, semantic_logits, gnn_probs
            ).tolist()),
            {1, 2},
        )
        mean_disagreement, overlap = disagreement_model._gate_diagnostics(
            flagged, semantic_logits, gnn_probs
        )
        self.assertAlmostEqual(mean_disagreement, 0.5625)
        self.assertAlmostEqual(overlap, 0.5)

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
        self.assertEqual(summary["gate_mode"], "confidence")
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
