"""Gate 0 contracts for the v2 mechanism-only oracle-ceiling experiment.

These tests deliberately precede the implementation.  They pin the four arms and
the only permitted route for deliberately leaky oracle labels: the semantic
``head_logits`` channel with output fusion disabled.
"""

from __future__ import annotations

from collections.abc import Mapping
import json

import numpy as np
import pytest
import torch

import src.pipeline.step4.mechanism_only_edge_injection as experiment
import src.pipeline.step4.train_feedback as training


EXPECTED_ARMS = {
    "control_head_only": {
        "feedback_mode": "head_only",
        "injection_mode": "edge",
        "advice_source": "none",
    },
    "real_prototype_edge": {
        "feedback_mode": "real",
        "injection_mode": "edge",
        "advice_source": "prototype",
    },
    "head_trained_edge": {
        "feedback_mode": "real",
        "injection_mode": "edge",
        "advice_source": "trained_head",
    },
    "oracle_edge": {
        "feedback_mode": "real",
        "injection_mode": "edge",
        "advice_source": "oracle",
    },
    "oracle_attention": {
        "feedback_mode": "real",
        "injection_mode": "attention",
        "advice_source": "oracle",
    },
}


def test_gate0_preserves_prespecified_seeds_and_dataset_settings() -> None:
    assert experiment.SEEDS == (42, 1, 2)
    assert experiment.SETUP == {
        "unsw_nb15": {"top_k": 31.0, "scale": 2.0},
        "ton_iot": {"top_k": 25.0, "scale": 20.0},
    }


def test_gate05_declares_exactly_five_explicit_arm_specs() -> None:
    assert isinstance(experiment.ARMS, Mapping), (
        "Gate 0 arms must be named specifications, not implicit string branches"
    )
    assert set(experiment.ARMS) == set(EXPECTED_ARMS)

    actual = {
        name: {
            "feedback_mode": spec.feedback_mode,
            "injection_mode": spec.injection_mode,
            "advice_source": spec.advice_source,
        }
        for name, spec in experiment.ARMS.items()
    }
    assert actual == EXPECTED_ARMS
    for spec in experiment.ARMS.values():
        with pytest.raises((AttributeError, TypeError)):
            spec.feedback_mode = "random"


def test_oracle_logits_encode_true_class_at_plus_minus_four_nats() -> None:
    labels = torch.tensor([0, 2, 1], dtype=torch.long)

    logits = experiment._oracle_logits(labels, num_classes=4)

    expected = torch.tensor(
        [
            [4.0, -4.0, -4.0, -4.0],
            [-4.0, -4.0, 4.0, -4.0],
            [-4.0, 4.0, -4.0, -4.0],
        ]
    )
    assert logits.shape == (3, 4)
    assert logits.device == labels.device
    assert logits.dtype == torch.get_default_dtype()
    assert torch.equal(logits, expected)


@pytest.mark.parametrize("magnitude", [0.0, -1.0])
def test_oracle_logits_reject_non_positive_magnitude(magnitude: float) -> None:
    with pytest.raises(ValueError, match="magnitude"):
        experiment._oracle_logits(
            torch.tensor([0, 1], dtype=torch.long),
            num_classes=2,
            magnitude=magnitude,
        )


@pytest.mark.parametrize(
    "labels,num_classes",
    [
        (torch.tensor([-1, 0], dtype=torch.long), 2),
        (torch.tensor([0, 2], dtype=torch.long), 2),
        (torch.tensor([[0, 1]], dtype=torch.long), 2),
        (torch.tensor([0.0, 1.0]), 2),
        (torch.tensor([0, 1], dtype=torch.long), 1),
    ],
)
def test_oracle_logits_reject_invalid_label_contract(
    labels: torch.Tensor, num_classes: int
) -> None:
    with pytest.raises(ValueError):
        experiment._oracle_logits(labels, num_classes=num_classes)


@pytest.mark.parametrize(
    "arm_name,expected_source",
    [
        ("control_head_only", "none"),
        ("real_prototype_edge", "prototype"),
        ("head_trained_edge", "trained_head"),
        ("oracle_edge", "oracle"),
        ("oracle_attention", "oracle"),
    ],
)
def test_train_arm_fold_uses_only_the_semantic_channel(
    monkeypatch: pytest.MonkeyPatch,
    arm_name: str,
    expected_source: str,
) -> None:
    captured: dict = {}
    sentinel = object()

    def fake_train_one_fold(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(experiment.FB, "_train_one_fold", fake_train_one_fold)
    labels = torch.tensor([1, 0, 2], dtype=torch.long)
    trained_head_logits = torch.stack(
        [torch.full((3, 3), float(fold_idx)) for fold_idx in range(5)]
    )

    result = experiment._train_arm_fold(
        arm_name,
        labels=labels,
        num_classes=3,
        fold_idx=4,
        trained_head_logits=trained_head_logits,
    )

    spec = EXPECTED_ARMS[arm_name]
    assert result is sentinel
    assert captured["mode"] == spec["feedback_mode"]
    assert captured["injection_mode"] == spec["injection_mode"]
    assert captured["use_output_fusion"] is False
    assert captured["fold_idx"] == 4

    if expected_source == "oracle":
        expected = experiment._oracle_logits(labels, num_classes=3)
        assert torch.equal(captured["head_logits"], expected)
    elif expected_source == "trained_head":
        assert torch.equal(captured["head_logits"], trained_head_logits[4])
    else:
        assert captured["head_logits"] is None


@pytest.mark.parametrize(
    "trained_head_logits,fold_idx,error",
    [
        (torch.zeros(5, 3, 2), 0, "shape"),
        (torch.zeros(5, 3, 3), -1, "fold_idx"),
        (torch.zeros(5, 3, 3), 5, "fold_idx"),
    ],
)
def test_trained_head_arm_rejects_wrong_shape_or_fold_index(
    monkeypatch: pytest.MonkeyPatch,
    trained_head_logits: torch.Tensor,
    fold_idx: int,
    error: str,
) -> None:
    monkeypatch.setattr(experiment.FB, "_train_one_fold", lambda **_: None)
    with pytest.raises(ValueError, match=error):
        experiment._train_arm_fold(
            "head_trained_edge",
            labels=torch.tensor([0, 1, 2]),
            num_classes=3,
            fold_idx=fold_idx,
            trained_head_logits=trained_head_logits,
        )


def test_iteration_rows_report_transitions_relative_to_iteration_one() -> None:
    labels = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    mask = torch.ones(4, dtype=torch.bool)

    def logits(predictions: list[int]) -> torch.Tensor:
        result = torch.full((4, 2), -2.0)
        result[torch.arange(4), torch.tensor(predictions)] = 2.0
        return result

    trace = [
        {
            "iter": 1,
            "churn": float("nan"),
            "mean_entropy": 0.4,
            "mean_disagreement": 0.3,
            "flagged_overlap": 0.5,
            "selected_count": 2,
            "selection_jaccard_previous": float("nan"),
            "advice_recomputed": False,
            "logits": logits([0, 0, 1, 1]),
        },
        {
            "iter": 2,
            "churn": 0.5,
            "mean_entropy": 0.3,
            "mean_disagreement": 0.2,
            "flagged_overlap": 0.5,
            "selected_count": 2,
            "selection_jaccard_previous": 1 / 3,
            "advice_recomputed": False,
            "logits": logits([0, 1, 0, 1]),
        },
        {
            "iter": 3,
            "churn": 0.5,
            "mean_entropy": 0.2,
            "mean_disagreement": 0.1,
            "flagged_overlap": 1.0,
            "selected_count": 2,
            "selection_jaccard_previous": 1.0,
            "advice_recomputed": False,
            "logits": logits([0, 1, 1, 0]),
        },
    ]

    rows = training._build_iteration_rows(
        trace, labels, mask, eval_classes=(0, 1)
    )

    assert [(row["wrong_to_correct"], row["correct_to_wrong"]) for row in rows] == [
        (0, 0),
        (1, 1),
        (2, 0),
    ]
    assert rows[1]["selection_jaccard_previous"] == pytest.approx(1 / 3)
    assert all(row["advice_recomputed"] is False for row in rows)
    assert rows[2]["test_macro_f1"] == pytest.approx(1.0)


def _fold_rows(scores: list[float]) -> list[dict]:
    return [
        {"seed": seed, "fold": fold, "test_macro_f1": scores[index]}
        for index, (seed, fold) in enumerate(
            (pair for seed in (42, 1, 2) for pair in ((seed, 0), (seed, 1)))
        )
    ]


def test_paired_fold_ci_is_zero_for_identical_pairs() -> None:
    rows = _fold_rows([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    result = experiment._paired_fold_ci(rows, rows, iters=500, seed=7)

    assert result == {
        "mean_diff": 0.0,
        "ci_low": 0.0,
        "ci_high": 0.0,
        "prob_positive": 0.0,
        "n_pairs": 6,
    }


def test_paired_fold_ci_detects_uniform_improvement() -> None:
    control = _fold_rows([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    candidate = _fold_rows([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

    result = experiment._paired_fold_ci(candidate, control, iters=500, seed=7)

    assert result["mean_diff"] == pytest.approx(0.1)
    assert result["ci_low"] > 0.0
    assert result["prob_positive"] == 1.0


def test_paired_fold_ci_rejects_unmatched_and_duplicate_pairs() -> None:
    rows = _fold_rows([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    with pytest.raises(ValueError, match="identical"):
        experiment._paired_fold_ci(rows, rows[:-1], iters=10)
    with pytest.raises(ValueError, match="duplicate"):
        experiment._paired_fold_ci(rows + [rows[0]], rows, iters=10)


def test_atomic_json_dump_is_strict_and_replaces_existing_file(tmp_path) -> None:
    path = tmp_path / "nested" / "result.json"
    path.parent.mkdir()
    path.write_text('{"stale": true}\n')

    experiment._atomic_json_dump(
        {"finite": 1.0, "nan": float("nan"), "numpy": np.float64(2.5)}, path
    )

    assert json.loads(path.read_text()) == {
        "finite": 1.0,
        "nan": None,
        "numpy": 2.5,
    }
    assert not path.with_suffix(".json.tmp").exists()


def test_iteration_evidence_keeps_improvement_and_selection_separate() -> None:
    fold_records = [
        {
            "iterations": [
                {
                    "iter": 1,
                    "test_macro_f1": 0.4,
                    "wrong_to_correct": 0,
                    "correct_to_wrong": 0,
                    "selected_count": 3,
                    "selection_jaccard_previous": float("nan"),
                    "advice_recomputed": False,
                },
                {
                    "iter": 2,
                    "test_macro_f1": 0.5,
                    "wrong_to_correct": 4,
                    "correct_to_wrong": 1,
                    "selected_count": 3,
                    "selection_jaccard_previous": 0.5,
                    "advice_recomputed": False,
                },
            ]
        }
    ]

    summary = experiment._iteration_evidence(fold_records)

    assert summary["2"]["mean_change_from_iteration_1"] == pytest.approx(0.1)
    assert summary["2"]["wrong_to_correct"] == 4
    assert summary["2"]["correct_to_wrong"] == 1
    assert summary["2"]["mean_selection_jaccard_previous"] == 0.5
    assert summary["2"]["advice_recomputed"] is False


def test_run_restores_training_seed_when_dataset_run_fails(monkeypatch) -> None:
    original_seed = training.SEED

    def fail(*_args, **_kwargs):
        training.SEED = 999
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(experiment, "_run_dataset", fail)
    with pytest.raises(RuntimeError, match="synthetic"):
        experiment.run()

    assert training.SEED == original_seed


def test_shared_architecture_declares_every_gate0_knob() -> None:
    architecture = experiment._shared_architecture(fold_count=5)

    assert architecture["edge_attr_encoding"] == "v2_log_cont_cat_idx"
    assert architecture["seeds"] == [42, 1, 2]
    assert architecture["fold_count"] == 5
    assert architecture["use_output_fusion"] is False
    assert architecture["real_arm_semantic_consultant"] == "whitened_prototype"
    assert architecture["trained_llm_head"] is False
    assert architecture["oracle_logit_magnitude"] == 4.0
    assert architecture["injection_modes"] == {
        name: values["injection_mode"] for name, values in EXPECTED_ARMS.items()
    }
    assert architecture["omp_num_threads"] == 1
    assert architecture["mkl_num_threads"] == 1
    assert architecture["torch_num_threads"] == 1


def _fake_dataset_result(dataset: str, *, below_threshold: bool) -> dict:
    return {
        "dataset": dataset,
        "dataset_key": dataset,
        "configuration": {
            **experiment._shared_architecture(5),
            "top_k_percent": 31.0 if dataset == "unsw_nb15" else 25.0,
            "injection_scale": 2.0 if dataset == "unsw_nb15" else 20.0,
        },
        "arms": {name: {} for name in EXPECTED_ARMS},
        "paired_comparisons": {},
        "decision": {
            "oracle_edge_headroom": 0.01 if below_threshold else 0.03,
            "prototype_gain": 0.001,
            "prototype_captured_share": 0.1,
            "headroom_below_0_02": below_threshold,
        },
        "elapsed_seconds": 1.0,
    }


def test_run_builds_one_program_decision_and_writes_from_same_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    captured: dict = {}

    def fake_run_dataset(dataset, _setup):
        return _fake_dataset_result(dataset, below_threshold=True)

    def fake_write_outputs(raw, output_path, contract_paths):
        captured["raw"] = raw
        captured["output_path"] = output_path
        captured["contract_paths"] = contract_paths

    monkeypatch.setattr(experiment, "_run_dataset", fake_run_dataset)
    monkeypatch.setattr(experiment, "_write_outputs", fake_write_outputs)
    output_path = tmp_path / "raw.json"
    contract_paths = {
        dataset: tmp_path / f"{dataset}.json" for dataset in experiment.SETUP
    }

    raw = experiment.run(output_path, contract_paths)

    assert captured["raw"] is raw
    assert captured["output_path"] == output_path
    assert captured["contract_paths"] == contract_paths
    assert raw["program_decision"] == {
        "rule": "stop P1'-P4 if oracle_edge headroom < 0.02 on both datasets",
        "stop_mechanism_surgery": True,
        "outcome": "stop",
        "gate05_is_diagnostic_only": True,
        "trained_head_is_ladder_rung": False,
    }
    assert raw["arms"]["oracle_edge"]["diagnostic_label_leakage"] is True
    assert raw["arms"]["real_prototype_edge"]["diagnostic_label_leakage"] is False
    assert raw["arms"]["head_trained_edge"]["diagnostic_only"] is True
    assert raw["arms"]["head_trained_edge"]["eligible_for_ladder"] is False


def test_write_outputs_derives_both_contracts_from_raw_payload(tmp_path) -> None:
    raw = {
        "schema_version": 2,
        "experiment": "oracle_consultant_ceiling_v2",
        "generated": "2026-08-30",
        "status": "DIAGNOSTIC ONLY - NEVER REPORTABLE",
        "question": "test question",
        "primary_metric": "pooled_5_fold_oof_macro_f1",
        "raw_output": str(tmp_path / "raw" / "oracle.json"),
        "arms": experiment._arm_contracts(),
        "datasets": {
            dataset: _fake_dataset_result(dataset, below_threshold=True)
            for dataset in experiment.SETUP
        },
        "program_decision": {
            "rule": "test",
            "stop_mechanism_surgery": True,
            "outcome": "stop",
        },
        "reproduce": "test command",
    }
    raw_path = tmp_path / "raw" / "oracle.json"
    contract_paths = {
        dataset: tmp_path / "results" / f"{dataset}.json"
        for dataset in experiment.SETUP
    }

    experiment._write_outputs(raw, raw_path, contract_paths)

    assert json.loads(raw_path.read_text())["datasets"].keys() == raw["datasets"].keys()
    for dataset, path in contract_paths.items():
        contract = json.loads(path.read_text())
        assert contract["dataset_key"] == dataset
        assert contract["configuration"] == raw["datasets"][dataset]["configuration"]
        assert contract["results"] == raw["datasets"][dataset]["arms"]
        assert contract["program_decision"] == raw["program_decision"]
