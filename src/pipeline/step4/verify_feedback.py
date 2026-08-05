"""Per-phase dashboards for Step 4.

Each phase writes its own dashboard under
`data/{dataset}/processed/step4_feedback/reports/phase_4_{N}_dashboard.{md,json}`.

Run one phase at a time; do not proceed to Phase 4.(N+1) until the
current phase's acceptance test PASSes.

Usage:
    python -m src.pipeline.step4.verify_feedback --phase 1 --dataset ton_iot
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch_geometric.data import Data

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.models.feedback_classifier import (
    CYSECBERT_EMBED_DIM,
    BiasedGATv2Layer,
    LiveCySecBERTScorer,
    SemanticAttentionBias,
    UncertaintySelector,
    WhitenedPrototypeScorer,
)
from src.pipeline.common.datasets import get_dataset_config
from src.pipeline.common.splits import NUM_CLASSES
from src.pipeline.step2.knowledge_graph import assert_label_free
from src.pipeline.step4.train_feedback import load_nl_lines


K_SWEEP = [20.0, 30.0, 40.0, 50.0]
DEFAULT_K = 30.0
MIN_CLASSES_PASSING = 8  # acceptance: uncertain_acc < confident_acc for ≥ 8/10 classes


def _step4_root(dataset: str) -> Path:
    return Path(f"data/{dataset}/processed/step4_feedback")


def _load_oof(dataset: str) -> tuple[torch.Tensor, Data, list[str]]:
    root = _step4_root(dataset)
    oof_path = root / "oof_logits.pt"
    if not oof_path.exists():
        raise FileNotFoundError(
            f"OOF logits not found at {oof_path}. Run: "
            f"python -m src.pipeline.step4.build_oof_predictions --dataset {dataset}"
        )
    config = get_dataset_config(dataset)
    logits = torch.load(oof_path, weights_only=False)
    data: Data = torch.load(config.graph_path, weights_only=False)
    return logits, data, list(config.label_names)


def _per_class_row(
    mask: torch.Tensor,
    probs: torch.Tensor,
    preds: torch.Tensor,
    labels: torch.Tensor,
    entropy: torch.Tensor,
) -> list[dict[str, float]]:
    """One row per class. `mask` is the uncertain-edge mask at some k."""
    rows: list[dict[str, float]] = []
    for c in range(NUM_CLASSES):
        cls_mask = labels == c
        total = int(cls_mask.sum())
        if total == 0:
            rows.append(
                {
                    "class": c,
                    "count_total": 0,
                    "count_uncertain": 0,
                    "mean_entropy_uncertain": float("nan"),
                    "gnn_acc_uncertain": float("nan"),
                    "gnn_acc_confident": float("nan"),
                }
            )
            continue

        uncertain_cls = cls_mask & mask
        confident_cls = cls_mask & (~mask)
        n_unc = int(uncertain_cls.sum())
        n_conf = int(confident_cls.sum())

        mean_h = (
            float(entropy[uncertain_cls].mean()) if n_unc > 0 else float("nan")
        )
        acc_unc = (
            float((preds[uncertain_cls] == c).float().mean())
            if n_unc > 0
            else float("nan")
        )
        acc_conf = (
            float((preds[confident_cls] == c).float().mean())
            if n_conf > 0
            else float("nan")
        )
        rows.append(
            {
                "class": c,
                "count_total": total,
                "count_uncertain": n_unc,
                "mean_entropy_uncertain": mean_h,
                "gnn_acc_uncertain": acc_unc,
                "gnn_acc_confident": acc_conf,
            }
        )
    return rows


def _pooled_macro_f1_on_mask(
    preds: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> float:
    if mask.sum() == 0:
        return float("nan")
    return float(
        f1_score(
            labels[mask].numpy(),
            preds[mask].numpy(),
            average="macro",
            labels=list(range(NUM_CLASSES)),
            zero_division=0,
        )
    )


def _try_plot_threshold_sensitivity(
    ks: list[float],
    macro_f1s: list[float],
    output_path: Path,
) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, macro_f1s, marker="o", linewidth=2)
    ax.set_xlabel("top-k% uncertain edges")
    ax.set_ylabel("GNN macro-F1 on the flagged subset")
    ax.set_title("Phase 4.1 — GNN performance on flagged (uncertain) edges")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def dashboard_phase_4_1(dataset: str) -> Path:
    logits, data, label_names = _load_oof(dataset)
    labels = data.edge_label
    probs = logits.softmax(dim=-1)
    preds = probs.argmax(dim=-1)
    entropy = UncertaintySelector.entropy(probs)

    reports_dir = _step4_root(dataset) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    sweep_payload: dict[str, dict] = {}
    sweep_pooled_f1: list[float] = []

    for k in K_SWEEP:
        selector = UncertaintySelector(top_k_percent=k)
        mask = selector(probs)
        per_class = _per_class_row(mask, probs, preds, labels, entropy)
        pooled_f1 = _pooled_macro_f1_on_mask(preds, labels, mask)
        sweep_pooled_f1.append(pooled_f1)
        sweep_payload[f"k={k:.0f}%"] = {
            "num_flagged": int(mask.sum()),
            "gnn_macro_f1_on_flagged": pooled_f1,
            "per_class": per_class,
        }

    default_selector = UncertaintySelector(top_k_percent=DEFAULT_K)
    default_mask = default_selector(probs)
    default_rows = _per_class_row(default_mask, probs, preds, labels, entropy)

    passing_classes = [
        r
        for r in default_rows
        if r["count_uncertain"] > 0
        and r["count_total"] - r["count_uncertain"] > 0
        and not np.isnan(r["gnn_acc_uncertain"])
        and not np.isnan(r["gnn_acc_confident"])
        and r["gnn_acc_uncertain"] < r["gnn_acc_confident"]
    ]
    acceptance_pass = len(passing_classes) >= MIN_CLASSES_PASSING

    cm = confusion_matrix(
        labels[default_mask].numpy(),
        preds[default_mask].numpy(),
        labels=list(range(NUM_CLASSES)),
    )

    plot_path = _try_plot_threshold_sensitivity(
        K_SWEEP,
        sweep_pooled_f1,
        reports_dir / "phase_4_1_threshold_sensitivity.png",
    )

    json_payload = {
        "phase": "4.1",
        "component": "UncertaintySelector",
        "dataset": dataset,
        "num_edges": int(labels.shape[0]),
        "acceptance": {
            "rule": (
                f"at k={DEFAULT_K:.0f}%, gnn_acc_uncertain[c] < gnn_acc_confident[c] "
                f"for at least {MIN_CLASSES_PASSING}/{NUM_CLASSES} classes"
            ),
            "passing_classes": [r["class"] for r in passing_classes],
            "num_passing": len(passing_classes),
            "result": "PASS" if acceptance_pass else "FAIL",
        },
        "k_sweep": sweep_payload,
        "confusion_matrix_at_default_k": {
            "k": DEFAULT_K,
            "labels": label_names,
            "matrix": cm.tolist(),
        },
        "threshold_sensitivity_plot": str(plot_path) if plot_path else None,
    }

    with open(reports_dir / "phase_4_1_dashboard.json", "w") as f:
        json.dump(json_payload, f, indent=2)

    md_lines: list[str] = []
    md_lines.append("# Phase 4.1 Dashboard — Uncertain-Edge Selector")
    md_lines.append("")
    md_lines.append(f"**Dataset:** {dataset}")
    md_lines.append(f"**Total edges:** {labels.shape[0]}")
    md_lines.append(f"**Signal source:** OOF logits (`step4_feedback/oof_logits.pt`)")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    md_lines.append("## Acceptance test")
    md_lines.append("")
    md_lines.append(
        f"**Rule:** at k = {DEFAULT_K:.0f}%, GNN accuracy on flagged edges must be "
        f"strictly lower than on non-flagged edges, for at least "
        f"{MIN_CLASSES_PASSING}/{NUM_CLASSES} classes."
    )
    md_lines.append("")
    md_lines.append(
        f"**Result:** **{'PASS' if acceptance_pass else 'FAIL'}** "
        f"({len(passing_classes)}/{NUM_CLASSES} classes passing: {sorted(r['class'] for r in passing_classes)})"
    )
    md_lines.append("")

    md_lines.append(f"## Per-class breakdown at default k = {DEFAULT_K:.0f}%")
    md_lines.append("")
    headers = [
        "class",
        "name",
        "total",
        "n_uncertain",
        "mean_H (nats)",
        "acc_uncertain",
        "acc_confident",
        "gap",
    ]
    rows = []
    for r in default_rows:
        gap = (
            r["gnn_acc_confident"] - r["gnn_acc_uncertain"]
            if not (
                np.isnan(r["gnn_acc_uncertain"]) or np.isnan(r["gnn_acc_confident"])
            )
            else float("nan")
        )
        rows.append(
            [
                str(r["class"]),
                label_names[r["class"]],
                str(r["count_total"]),
                str(r["count_uncertain"]),
                f"{r['mean_entropy_uncertain']:.3f}"
                if not np.isnan(r["mean_entropy_uncertain"])
                else "—",
                f"{r['gnn_acc_uncertain']:.3f}"
                if not np.isnan(r["gnn_acc_uncertain"])
                else "—",
                f"{r['gnn_acc_confident']:.3f}"
                if not np.isnan(r["gnn_acc_confident"])
                else "—",
                f"{gap:+.3f}" if not np.isnan(gap) else "—",
            ]
        )
    md_lines.append(_md_table(headers, rows))
    md_lines.append("")

    md_lines.append("## Threshold sweep")
    md_lines.append("")
    md_lines.append(
        "GNN macro-F1 on the flagged subset — should stay well below the "
        "overall macro-F1 (otherwise entropy is picking easy edges)."
    )
    md_lines.append("")
    sweep_headers = ["k (%)", "n_flagged", "macro-F1 on flagged"]
    sweep_rows = [
        [
            f"{k:.0f}",
            str(sweep_payload[f'k={k:.0f}%']['num_flagged']),
            f"{sweep_payload[f'k={k:.0f}%']['gnn_macro_f1_on_flagged']:.4f}",
        ]
        for k in K_SWEEP
    ]
    md_lines.append(_md_table(sweep_headers, sweep_rows))
    md_lines.append("")
    if plot_path is not None:
        md_lines.append(f"![threshold sensitivity]({plot_path.name})")
        md_lines.append("")

    md_lines.append(f"## Confusion matrix on flagged edges at k = {DEFAULT_K:.0f}%")
    md_lines.append("")
    md_lines.append(
        "Rows = true class, columns = GNN predicted class. Off-diagonal mass "
        "shows which class pairs the LLM will need to disambiguate."
    )
    md_lines.append("")
    cm_headers = ["true \\ pred"] + label_names
    cm_rows = [
        [label_names[i]] + [str(int(v)) for v in cm[i]]
        for i in range(NUM_CLASSES)
    ]
    md_lines.append(_md_table(cm_headers, cm_rows))
    md_lines.append("")

    with open(reports_dir / "phase_4_1_dashboard.md", "w") as f:
        f.write("\n".join(md_lines))

    print(f"Wrote {reports_dir / 'phase_4_1_dashboard.md'}")
    print(f"Wrote {reports_dir / 'phase_4_1_dashboard.json'}")
    if plot_path:
        print(f"Wrote {plot_path}")
    print(f"Acceptance: {'PASS' if acceptance_pass else 'FAIL'}")
    return reports_dir / "phase_4_1_dashboard.md"


CYSECBERT_MODEL = "markusbayer/CySecBERT"
CYSECBERT_MAX_LENGTH = 128
NL_AUDIT_SAMPLE = 10
NL_AUDIT_SEED = 42


def dashboard_phase_4_2(dataset: str) -> Path:
    logits, data, label_names = _load_oof(dataset)
    labels = data.edge_label
    probs = logits.softmax(dim=-1)
    preds = probs.argmax(dim=-1)

    nl_lines = load_nl_lines(dataset)
    if len(nl_lines) != labels.shape[0]:
        raise RuntimeError(
            f"Row-alignment failure: {len(nl_lines)} NL lines vs "
            f"{labels.shape[0]} edges. Re-run Step 2 for {dataset}."
        )

    selector = UncertaintySelector(top_k_percent=DEFAULT_K)
    mask = selector(probs)
    uncertain_indices = mask.nonzero(as_tuple=True)[0].tolist()
    nl_subset = [nl_lines[i] for i in uncertain_indices]

    label_leak_result: dict[str, object]
    try:
        assert_label_free(nl_subset, set(label_names))
        label_leak_result = {"result": "PASS", "offending": []}
    except ValueError as exc:
        label_leak_result = {"result": "FAIL", "message": str(exc)}

    rng = np.random.default_rng(NL_AUDIT_SEED)
    sample_size = min(NL_AUDIT_SAMPLE, len(uncertain_indices))
    sample_positions = rng.choice(len(uncertain_indices), size=sample_size, replace=False)
    audit_rows: list[dict[str, object]] = []
    for pos in sample_positions:
        edge_idx = uncertain_indices[int(pos)]
        src = int(data.edge_index[0, edge_idx].item())
        dst = int(data.edge_index[1, edge_idx].item())
        idx_to_node = getattr(data, "idx_to_node", None)
        src_name = idx_to_node.get(src, str(src)) if isinstance(idx_to_node, dict) else str(src)
        dst_name = idx_to_node.get(dst, str(dst)) if isinstance(idx_to_node, dict) else str(dst)
        audit_rows.append(
            {
                "edge_index": edge_idx,
                "src": src_name,
                "dst": dst_name,
                "true_class": label_names[int(labels[edge_idx])],
                "pred_class": label_names[int(preds[edge_idx])],
                "nl": nl_lines[edge_idx],
            }
        )

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(CYSECBERT_MODEL)
        token_counts = [
            len(tokenizer.encode(s, add_special_tokens=True)) for s in nl_subset
        ]
        tokenizer_available = True
    except Exception as exc:
        token_counts = []
        tokenizer_available = False
        tokenizer_error = str(exc)

    if tokenizer_available:
        bins = [(0, 32), (33, 64), (65, 96), (97, 128), (129, 10_000)]
        length_hist = []
        for lo, hi in bins:
            n = sum(1 for c in token_counts if lo <= c <= hi)
            length_hist.append({"range": f"{lo}-{hi if hi < 10_000 else 'inf'}", "count": n})
        truncation_count = sum(1 for c in token_counts if c > CYSECBERT_MAX_LENGTH)
        max_len = max(token_counts) if token_counts else 0
        mean_len = float(np.mean(token_counts)) if token_counts else 0.0
    else:
        length_hist = []
        truncation_count = None
        max_len = None
        mean_len = None

    per_class_nl_count = []
    for c in range(NUM_CLASSES):
        cls_mask = labels == c
        n_flagged = int((cls_mask & mask).sum())
        per_class_nl_count.append(
            {"class": c, "name": label_names[c], "count": n_flagged}
        )

    reports_dir = _step4_root(dataset) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    acceptance_pass = label_leak_result["result"] == "PASS"

    json_payload = {
        "phase": "4.2",
        "component": "NL slice + label-leak check",
        "dataset": dataset,
        "num_uncertain": len(uncertain_indices),
        "acceptance": {
            "rule": (
                "assert_label_free PASSes on the uncertain NL slice AND "
                "the 10-row audit table is visually confirmed by the user"
            ),
            "label_leak": label_leak_result,
            "result": "PASS" if acceptance_pass else "FAIL",
        },
        "audit_sample": audit_rows,
        "token_length": {
            "tokenizer_available": tokenizer_available,
            "max_length_cap": CYSECBERT_MAX_LENGTH,
            "max_observed": max_len,
            "mean_observed": mean_len,
            "num_truncated": truncation_count,
            "histogram": length_hist,
        },
        "per_class_nl_count": per_class_nl_count,
    }

    with open(reports_dir / "phase_4_2_dashboard.json", "w") as f:
        json.dump(json_payload, f, indent=2)

    md_lines: list[str] = []
    md_lines.append("# Phase 4.2 Dashboard — NL Slice for Uncertain Flows")
    md_lines.append("")
    md_lines.append(f"**Dataset:** {dataset}")
    md_lines.append(f"**Uncertain edges at k={DEFAULT_K:.0f}%:** {len(uncertain_indices)}")
    md_lines.append(f"**NL source:** `data/{dataset}/processed/step2/kg_triples_nl.txt`")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    md_lines.append("## Acceptance test")
    md_lines.append("")
    md_lines.append(
        "**Rule:** `assert_label_free` PASSes on the uncertain NL slice AND "
        "the row-alignment audit table below is visually confirmed."
    )
    md_lines.append("")
    md_lines.append(f"**Label-leak check:** {label_leak_result['result']}")
    if label_leak_result["result"] == "FAIL":
        md_lines.append("")
        md_lines.append(f"> {label_leak_result.get('message', '')}")
    md_lines.append("")

    md_lines.append(f"## Row-alignment audit ({sample_size} random uncertain edges)")
    md_lines.append("")
    md_lines.append(
        "Check that each row's NL sentence describes the src → dst flow shown "
        "in the same row. Any mismatch means the NL file drifted from the graph."
    )
    md_lines.append("")
    headers = ["edge_idx", "src", "dst", "true", "pred", "NL sentence"]
    rows = [
        [
            str(r["edge_index"]),
            str(r["src"]),
            str(r["dst"]),
            str(r["true_class"]),
            str(r["pred_class"]),
            str(r["nl"]),
        ]
        for r in audit_rows
    ]
    md_lines.append(_md_table(headers, rows))
    md_lines.append("")

    md_lines.append("## Token-length distribution (CySecBERT tokenizer)")
    md_lines.append("")
    if tokenizer_available:
        md_lines.append(
            f"Max length cap for CySecBERT: **{CYSECBERT_MAX_LENGTH}**. "
            f"Observed on the uncertain slice: mean = {mean_len:.1f}, "
            f"max = {max_len}, truncated = **{truncation_count}**."
        )
        md_lines.append("")
        md_lines.append(_md_table(
            ["token range", "count"],
            [[h["range"], str(h["count"])] for h in length_hist],
        ))
    else:
        md_lines.append(
            "Tokenizer not available in this environment — skipped. "
            f"Error: {tokenizer_error}"
        )
    md_lines.append("")

    md_lines.append(f"## Per-class NL count at k = {DEFAULT_K:.0f}%")
    md_lines.append("")
    md_lines.append(
        "This must equal `count_uncertain[c]` from Phase 4.1's dashboard "
        "(same mask, same edges)."
    )
    md_lines.append("")
    md_lines.append(_md_table(
        ["class", "name", "count"],
        [[str(r["class"]), r["name"], str(r["count"])] for r in per_class_nl_count],
    ))
    md_lines.append("")

    with open(reports_dir / "phase_4_2_dashboard.md", "w") as f:
        f.write("\n".join(md_lines))

    print(f"Wrote {reports_dir / 'phase_4_2_dashboard.md'}")
    print(f"Wrote {reports_dir / 'phase_4_2_dashboard.json'}")
    print(f"Acceptance: {'PASS' if acceptance_pass else 'FAIL'}")
    return reports_dir / "phase_4_2_dashboard.md"


MIN_WHITENED_SEPARATION = 0.3
TEMPERATURE_SWEEP = [1.0, 5.0, 10.0, 20.0]
LIVE_MATCH_SAMPLE = 100
LIVE_MATCH_TOLERANCE = 1e-4


def _off_diag_mean(matrix: torch.Tensor) -> float:
    n = matrix.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    valid = matrix[mask]
    return float(valid.mean())


def _prototype_similarity(prototypes: torch.Tensor) -> torch.Tensor:
    p = torch.nn.functional.normalize(prototypes, dim=-1)
    return p @ p.T


def _per_class_separation(prototypes: torch.Tensor) -> float:
    """Higher = classes are further apart in cosine terms."""
    sim = _prototype_similarity(prototypes)
    return 1.0 - _off_diag_mean(sim)


def _pooled_llm_predictions(
    dataset: str,
    scorer_temperature: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Return per-edge LLM-alone predictions using precomputed embeddings.

    For each fold: build a WhitenedPrototypeScorer with that fold's
    prototypes; run it on the fold's test edges (using the precomputed
    step3_llm embeddings). Stitch OOF-style so each edge is predicted by
    the fold that held it out.

    Also returns per-edge cosine (before temperature) and per-fold
    LLM-alone macro-F1 details in `meta`.

    `scorer_temperature=None` means "use whatever the scorer is
    initialized to (log(10) → 10)". When set, overrides log_temperature
    to that scalar for all classes (used by the temperature sweep).
    """
    config = get_dataset_config(dataset)
    root = _step4_root(dataset)
    proto_path = root / "prototypes.pt"
    if not proto_path.exists():
        raise FileNotFoundError(
            f"Prototypes not found at {proto_path}. Run: "
            f"python -m src.pipeline.step4.build_prototypes --dataset {dataset}"
        )

    prototypes_bundle = torch.load(proto_path, weights_only=False)
    embeddings = torch.load(config.llm_embedding_path, weights_only=False).float()
    data = torch.load(config.graph_path, weights_only=False)
    folds = torch.load(config.splits_path, weights_only=False)
    labels = data.edge_label

    num_edges = embeddings.shape[0]
    preds = torch.full((num_edges,), -1, dtype=torch.long)
    cosine_matrix = torch.zeros(num_edges, NUM_CLASSES)
    fold_f1s: list[float] = []

    for fold_idx, fold in enumerate(folds):
        state = prototypes_bundle["folds"][fold_idx]
        scorer = WhitenedPrototypeScorer(
            num_classes=NUM_CLASSES,
            embed_dim=prototypes_bundle["embed_dim"],
        )
        scorer.load_fold_state(
            mean=state["mean"],
            whitener=state["whitener"],
            prototypes=state["prototypes_whitened"],
        )
        if scorer_temperature is not None:
            with torch.no_grad():
                scorer.log_temperature.fill_(math.log(scorer_temperature))

        test_mask = fold["test_mask"]
        test_emb = embeddings[test_mask]
        with torch.no_grad():
            logits = scorer(test_emb)
            whitened = scorer.whiten(test_emb)
            cos = scorer.cosine(whitened)
        fold_preds = logits.argmax(dim=-1)
        preds[test_mask] = fold_preds
        cosine_matrix[test_mask] = cos

        from sklearn.metrics import f1_score

        fold_f1 = float(
            f1_score(
                labels[test_mask].numpy(),
                fold_preds.numpy(),
                average="macro",
                labels=list(range(NUM_CLASSES)),
                zero_division=0,
            )
        )
        fold_f1s.append(fold_f1)

    from sklearn.metrics import f1_score

    pooled_f1 = float(
        f1_score(
            labels.numpy(),
            preds.numpy(),
            average="macro",
            labels=list(range(NUM_CLASSES)),
            zero_division=0,
        )
    )
    meta = {
        "per_fold_macro_f1": fold_f1s,
        "pooled_macro_f1": pooled_f1,
    }
    return preds, cosine_matrix, labels, meta


def _live_vs_precomputed_check(dataset: str) -> dict:
    """Encode `LIVE_MATCH_SAMPLE` random NL sentences live and compare to
    the precomputed step3_llm embeddings. Must match to `LIVE_MATCH_TOLERANCE`.
    """
    config = get_dataset_config(dataset)
    precomputed = torch.load(config.llm_embedding_path, weights_only=False).float()
    nl_lines = load_nl_lines(dataset)
    num_edges = precomputed.shape[0]

    rng = np.random.default_rng(NL_AUDIT_SEED)
    sample_size = min(LIVE_MATCH_SAMPLE, num_edges)
    sample_idx = rng.choice(num_edges, size=sample_size, replace=False)

    scorer = LiveCySecBERTScorer()
    sample_sents = [nl_lines[i] for i in sample_idx]
    live_emb = scorer.encode(
        edge_ids=sample_idx.tolist(),
        sentences=sample_sents,
        fold_index=-1,
    )
    diff = (live_emb - precomputed[sample_idx]).abs()
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    return {
        "sample_size": sample_size,
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "tolerance": LIVE_MATCH_TOLERANCE,
        "result": "PASS" if max_diff < LIVE_MATCH_TOLERANCE else "FAIL",
    }


def dashboard_phase_4_3(dataset: str) -> Path:
    reports_dir = _step4_root(dataset) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    proto_path = _step4_root(dataset) / "prototypes.pt"
    if not proto_path.exists():
        raise FileNotFoundError(
            f"Prototypes not found at {proto_path}. Run: "
            f"python -m src.pipeline.step4.build_prototypes --dataset {dataset}"
        )
    prototypes_bundle = torch.load(proto_path, weights_only=False)
    config = get_dataset_config(dataset)
    label_names = list(config.label_names)

    separations = []
    for state in prototypes_bundle["folds"]:
        active_classes = state["class_counts"] > 0
        raw_active = state["prototypes_raw"][active_classes]
        whitened_active = state["prototypes_whitened"][active_classes]
        sep_raw = _per_class_separation(raw_active)
        sep_whitened = _per_class_separation(whitened_active)
        separations.append(
            {"raw": sep_raw, "whitened": sep_whitened, "num_active": int(active_classes.sum())}
        )
    mean_raw_sep = float(np.mean([s["raw"] for s in separations]))
    mean_whitened_sep = float(np.mean([s["whitened"] for s in separations]))

    fold0_state = prototypes_bundle["folds"][0]
    fold0_sim = _prototype_similarity(fold0_state["prototypes_whitened"])

    logits, data, _ = _load_oof(dataset)
    labels = data.edge_label
    probs = logits.softmax(dim=-1)
    gnn_preds = probs.argmax(dim=-1)
    selector = UncertaintySelector(top_k_percent=DEFAULT_K)
    uncertain_mask = selector(probs)

    llm_preds, cosine_matrix, _, llm_meta = _pooled_llm_predictions(dataset)

    from sklearn.metrics import f1_score

    llm_f1_on_uncertain = float(
        f1_score(
            labels[uncertain_mask].numpy(),
            llm_preds[uncertain_mask].numpy(),
            average="macro",
            labels=list(range(NUM_CLASSES)),
            zero_division=0,
        )
    )
    gnn_f1_on_uncertain = float(
        f1_score(
            labels[uncertain_mask].numpy(),
            gnn_preds[uncertain_mask].numpy(),
            average="macro",
            labels=list(range(NUM_CLASSES)),
            zero_division=0,
        )
    )

    llm_per_class_acc: list[dict] = []
    for c in range(NUM_CLASSES):
        cls_mask = (labels == c) & uncertain_mask
        n = int(cls_mask.sum())
        if n == 0:
            llm_per_class_acc.append(
                {"class": c, "name": label_names[c], "n_uncertain": 0, "acc": float("nan")}
            )
            continue
        acc = float((llm_preds[cls_mask] == c).float().mean())
        llm_per_class_acc.append(
            {"class": c, "name": label_names[c], "n_uncertain": n, "acc": acc}
        )

    temperature_sweep_results: list[dict] = []
    for t in TEMPERATURE_SWEEP:
        preds_t, _, _, meta_t = _pooled_llm_predictions(dataset, scorer_temperature=t)
        f1_t = float(
            f1_score(
                labels[uncertain_mask].numpy(),
                preds_t[uncertain_mask].numpy(),
                average="macro",
                labels=list(range(NUM_CLASSES)),
                zero_division=0,
            )
        )
        temperature_sweep_results.append(
            {"temperature": t, "llm_macro_f1_on_uncertain": f1_t}
        )

    live_check = _live_vs_precomputed_check(dataset)

    whitening_pass = mean_whitened_sep > MIN_WHITENED_SEPARATION
    llm_beats_gnn = llm_f1_on_uncertain > gnn_f1_on_uncertain
    live_match_pass = live_check["result"] == "PASS"
    acceptance_pass = whitening_pass and llm_beats_gnn and live_match_pass

    json_payload = {
        "phase": "4.3",
        "component": "LLM semantic scorer (prototype + whitener + live encoder)",
        "dataset": dataset,
        "acceptance": {
            "rule": (
                f"mean whitened separation > {MIN_WHITENED_SEPARATION}, "
                "AND LLM-alone macro-F1 on uncertain > GNN macro-F1 on uncertain, "
                "AND live re-encoding matches precomputed to "
                f"{LIVE_MATCH_TOLERANCE}"
            ),
            "whitening_separation_pass": whitening_pass,
            "llm_beats_gnn_pass": llm_beats_gnn,
            "live_match_pass": live_match_pass,
            "result": "PASS" if acceptance_pass else "FAIL",
        },
        "whitening_effect": {
            "mean_separation_raw": mean_raw_sep,
            "mean_separation_whitened": mean_whitened_sep,
            "per_fold": separations,
            "target": MIN_WHITENED_SEPARATION,
        },
        "llm_alone_on_uncertain": {
            "llm_macro_f1": llm_f1_on_uncertain,
            "gnn_macro_f1": gnn_f1_on_uncertain,
            "delta": llm_f1_on_uncertain - gnn_f1_on_uncertain,
            "per_class_acc": llm_per_class_acc,
        },
        "temperature_sweep": temperature_sweep_results,
        "live_match": live_check,
        "prototype_similarity_fold_0": {
            "labels": label_names,
            "matrix": fold0_sim.tolist(),
        },
        "pooled_llm_metrics": llm_meta,
    }

    with open(reports_dir / "phase_4_3_dashboard.json", "w") as f:
        json.dump(json_payload, f, indent=2)

    md_lines: list[str] = []
    md_lines.append("# Phase 4.3 Dashboard — LLM Semantic Scorer")
    md_lines.append("")
    md_lines.append(f"**Dataset:** {dataset}")
    md_lines.append(
        f"**Uncertain edges at k = {DEFAULT_K:.0f}%:** {int(uncertain_mask.sum())}"
    )
    md_lines.append(f"**CySecBERT:** {prototypes_bundle['model_id']}")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    md_lines.append("## Acceptance test")
    md_lines.append("")
    md_lines.append(
        f"**Rule:** mean whitened separation > **{MIN_WHITENED_SEPARATION}** "
        "AND LLM macro-F1 on uncertain > GNN macro-F1 on uncertain "
        f"AND live-vs-precomputed max abs diff < {LIVE_MATCH_TOLERANCE}."
    )
    md_lines.append("")
    md_lines.append(
        f"| check | value | pass |\n| --- | --- | --- |\n"
        f"| whitened separation | {mean_whitened_sep:.4f} (target > {MIN_WHITENED_SEPARATION}) | "
        f"{'PASS' if whitening_pass else 'FAIL'} |\n"
        f"| LLM vs GNN F1 on uncertain | {llm_f1_on_uncertain:.4f} vs {gnn_f1_on_uncertain:.4f} "
        f"(Δ = {llm_f1_on_uncertain - gnn_f1_on_uncertain:+.4f}) | "
        f"{'PASS' if llm_beats_gnn else 'FAIL'} |\n"
        f"| live vs precomputed | max |Δ| = {live_check['max_abs_diff']:.2e} | "
        f"{'PASS' if live_match_pass else 'FAIL'} |"
    )
    md_lines.append("")
    md_lines.append(
        f"**Overall:** **{'PASS' if acceptance_pass else 'FAIL'}**"
    )
    md_lines.append("")

    md_lines.append("## Whitening effect")
    md_lines.append("")
    md_lines.append(
        f"Mean per-class separation across folds: raw = {mean_raw_sep:.4f}, "
        f"whitened = {mean_whitened_sep:.4f}. Whitening should INCREASE separation."
    )
    md_lines.append("")
    md_lines.append(_md_table(
        ["fold", "active classes", "raw sep", "whitened sep", "delta"],
        [
            [
                str(i),
                str(s["num_active"]),
                f"{s['raw']:.4f}",
                f"{s['whitened']:.4f}",
                f"{s['whitened'] - s['raw']:+.4f}",
            ]
            for i, s in enumerate(separations)
        ],
    ))
    md_lines.append("")

    md_lines.append("## LLM-alone accuracy on uncertain subset (per class)")
    md_lines.append("")
    md_lines.append(
        "Prototype-scored predictions on the same uncertain edges Phase 4.1 flagged, "
        "OOF (each fold's prototypes evaluated on its test-fold uncertain edges)."
    )
    md_lines.append("")
    md_lines.append(_md_table(
        ["class", "name", "n_uncertain", "LLM acc"],
        [
            [
                str(r["class"]),
                r["name"],
                str(r["n_uncertain"]),
                f"{r['acc']:.3f}" if not np.isnan(r["acc"]) else "—",
            ]
            for r in llm_per_class_acc
        ],
    ))
    md_lines.append("")

    md_lines.append("## Temperature sweep (LLM-only macro-F1 on uncertain)")
    md_lines.append("")
    md_lines.append(_md_table(
        ["temperature", "LLM macro-F1"],
        [
            [f"{r['temperature']:.1f}", f"{r['llm_macro_f1_on_uncertain']:.4f}"]
            for r in temperature_sweep_results
        ],
    ))
    md_lines.append("")

    md_lines.append("## Prototype similarity matrix (fold 0, whitened, cosine)")
    md_lines.append("")
    md_lines.append(
        "Diagonal is 1.0. Off-diagonal cells close to 1 = two classes CySecBERT "
        "cannot distinguish; close to 0 = well separated in whitened space."
    )
    md_lines.append("")
    sim_headers = ["c1 \\ c2"] + label_names
    sim_rows = [
        [label_names[i]] + [f"{fold0_sim[i, j].item():+.2f}" for j in range(NUM_CLASSES)]
        for i in range(NUM_CLASSES)
    ]
    md_lines.append(_md_table(sim_headers, sim_rows))
    md_lines.append("")

    md_lines.append("## Live vs precomputed CySecBERT encoding")
    md_lines.append("")
    md_lines.append(
        f"Sampled {live_check['sample_size']} random edges, encoded live with "
        f"`LiveCySecBERTScorer`, compared to the precomputed embeddings in "
        f"`step3_llm/edge_embeddings.pt`. Max abs diff = {live_check['max_abs_diff']:.2e}, "
        f"mean abs diff = {live_check['mean_abs_diff']:.2e}. Result: **{live_check['result']}**."
    )
    md_lines.append("")

    with open(reports_dir / "phase_4_3_dashboard.md", "w") as f:
        f.write("\n".join(md_lines))

    print(f"Wrote {reports_dir / 'phase_4_3_dashboard.md'}")
    print(f"Wrote {reports_dir / 'phase_4_3_dashboard.json'}")
    print(f"Acceptance: {'PASS' if acceptance_pass else 'FAIL'}")
    return reports_dir / "phase_4_3_dashboard.md"


DEFAULT_BIAS_DIM = 1
DEFAULT_HEADS = 8
BIAS_STRENGTH_INJECTION_MAGNITUDE = 1.0
ATTN_SHIFT_TOP_N = 5


def _test_a_zero_on_unflagged(
    num_classes: int, num_edges: int, num_flagged: int, seed: int = 0
) -> dict:
    """Test (a): SemanticAttentionBias must produce exact-zero bias on
    unflagged edges regardless of projection weights.
    """
    torch.manual_seed(seed)
    sab = SemanticAttentionBias(num_classes=num_classes, bias_dim=DEFAULT_BIAS_DIM)
    # Manually break the zero-init to make the test meaningful — after this,
    # a non-flagged edge should still get 0 bias only because of masking, not
    # because the projection happens to output 0.
    with torch.no_grad():
        sab.projection.weight.copy_(torch.randn_like(sab.projection.weight))
        sab.projection.bias.copy_(torch.randn_like(sab.projection.bias))

    flagged_indices = torch.arange(num_flagged)
    semantic_logits = torch.randn(num_flagged, num_classes)
    full_bias = sab(semantic_logits, flagged_indices, num_edges)

    unflagged_slice = full_bias[num_flagged:]
    max_unflagged = float(unflagged_slice.abs().max()) if unflagged_slice.numel() else 0.0
    flagged_slice = full_bias[:num_flagged]
    min_flagged = float(flagged_slice.abs().min()) if flagged_slice.numel() else 0.0

    passed = max_unflagged == 0.0 and min_flagged > 0.0
    return {
        "name": "test_a_zero_on_unflagged",
        "description": (
            "After randomising the projection, unflagged edges' bias must be "
            "exactly zero while flagged edges' bias must be non-zero."
        ),
        "max_abs_bias_on_unflagged": max_unflagged,
        "min_abs_bias_on_flagged": min_flagged,
        "result": "PASS" if passed else "FAIL",
    }


def _test_b_gradient_connected(
    num_classes: int, num_flagged: int, seed: int = 0
) -> dict:
    """Test (b): a scalar loss on the semantic bias output must produce a
    non-None, non-zero gradient on `WhitenedPrototypeScorer.log_temperature`
    when there is at least one flagged edge.
    """
    torch.manual_seed(seed)
    scorer = WhitenedPrototypeScorer(num_classes=num_classes)
    with torch.no_grad():
        scorer.prototypes.copy_(torch.randn_like(scorer.prototypes))
        scorer.whitener.copy_(torch.randn_like(scorer.whitener) * 0.01)
    sab = SemanticAttentionBias(num_classes=num_classes, bias_dim=DEFAULT_BIAS_DIM)
    with torch.no_grad():
        sab.projection.weight.copy_(torch.randn_like(sab.projection.weight))
        sab.projection.bias.copy_(torch.randn_like(sab.projection.bias))

    emb = torch.randn(num_flagged, scorer.embed_dim, requires_grad=False)
    semantic_logits = scorer(emb)
    bias = sab(
        semantic_logits, torch.arange(num_flagged), num_edges=num_flagged
    )
    loss = bias.pow(2).sum()
    loss.backward()

    grad = scorer.log_temperature.grad
    has_grad = grad is not None
    nonzero_grad_norm = float(grad.abs().sum()) if has_grad else 0.0
    passed = has_grad and nonzero_grad_norm > 0.0
    return {
        "name": "test_b_gradient_connected",
        "description": (
            "Backprop from the semantic bias must reach "
            "WhitenedPrototypeScorer.log_temperature with non-zero gradient."
        ),
        "grad_present": has_grad,
        "grad_L1_norm": nonzero_grad_norm,
        "result": "PASS" if passed else "FAIL",
    }


def _test_c_output_localized(seed: int = 0) -> dict:
    """Test (c): on a toy graph, a non-zero bias on a subset of edges must
    only change `BiasedGATv2Layer` output at target nodes of flagged edges.
    Unchanged target nodes must have identical output to the zero-bias case.
    """
    torch.manual_seed(seed)
    # Toy graph: 6 nodes, 8 directed edges.
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 0, 2],
            [1, 2, 3, 4, 5, 0, 3, 5],
        ],
        dtype=torch.long,
    )
    num_edges = edge_index.shape[1]
    num_nodes = 6
    x = torch.randn(num_nodes, 10)
    edge_attr = torch.randn(num_edges, 5)

    flagged_indices = torch.tensor([1, 4], dtype=torch.long)
    flagged_targets = set(edge_index[1, flagged_indices].tolist())
    unaffected_nodes = [n for n in range(num_nodes) if n not in flagged_targets]

    layer = BiasedGATv2Layer(
        in_channels=10,
        out_channels=8,
        edge_attr_dim=5,
        bias_dim=DEFAULT_BIAS_DIM,
        heads=2,
        concat=False,
    )
    layer.eval()

    zero_bias = torch.zeros(num_edges, DEFAULT_BIAS_DIM)
    nonzero_bias = zero_bias.clone()
    nonzero_bias[flagged_indices] = BIAS_STRENGTH_INJECTION_MAGNITUDE

    with torch.no_grad():
        out_baseline = layer(x, edge_index, edge_attr, zero_bias)
        out_biased = layer(x, edge_index, edge_attr, nonzero_bias)

    diff = (out_baseline - out_biased).abs()
    unaffected_diff = float(diff[unaffected_nodes].max()) if unaffected_nodes else 0.0
    affected_diff = float(diff[list(flagged_targets)].max()) if flagged_targets else 0.0

    passed = unaffected_diff < 1e-6 and affected_diff > 1e-6
    return {
        "name": "test_c_output_localized",
        "description": (
            "Non-zero bias on flagged edges must change output only at their "
            "target nodes; nodes that don't receive a flagged edge must be identical."
        ),
        "flagged_target_nodes": sorted(flagged_targets),
        "unaffected_nodes": unaffected_nodes,
        "max_abs_change_on_unaffected_nodes": unaffected_diff,
        "max_abs_change_on_affected_nodes": affected_diff,
        "result": "PASS" if passed else "FAIL",
    }


def _bias_magnitude_and_attention_shift_diagnostic(dataset: str) -> dict:
    """On the real UNSW graph and Phase 4.3 prototypes/uncertain edges,
    measure (i) the per-head bias magnitude distribution after a random
    projection, (ii) the shift in GATv2 attention weights between the
    unbiased and biased forward pass, per class.

    This is not part of acceptance — it's the "does the semantic channel
    actually move attention?" interpretability check.
    """
    config = get_dataset_config(dataset)
    data = torch.load(config.graph_path, weights_only=False)
    logits, _, _ = _load_oof(dataset)
    probs = logits.softmax(dim=-1)
    selector = UncertaintySelector(top_k_percent=DEFAULT_K)
    uncertain_mask = selector(probs)
    flagged_indices = uncertain_mask.nonzero(as_tuple=True)[0]

    prototypes_bundle = torch.load(
        _step4_root(dataset) / "prototypes.pt", weights_only=False
    )
    embeddings = torch.load(config.llm_embedding_path, weights_only=False).float()
    labels = data.edge_label

    scorer = WhitenedPrototypeScorer(
        num_classes=NUM_CLASSES,
        embed_dim=prototypes_bundle["embed_dim"],
    )
    scorer.load_fold_state(
        mean=prototypes_bundle["folds"][0]["mean"],
        whitener=prototypes_bundle["folds"][0]["whitener"],
        prototypes=prototypes_bundle["folds"][0]["prototypes_whitened"],
    )
    sab = SemanticAttentionBias(num_classes=NUM_CLASSES, bias_dim=DEFAULT_BIAS_DIM)
    with torch.no_grad():
        torch.manual_seed(0)
        sab.projection.weight.copy_(
            torch.randn_like(sab.projection.weight) * 0.5
        )
        sab.projection.bias.copy_(torch.zeros_like(sab.projection.bias))

    flagged_emb = embeddings[flagged_indices]
    with torch.no_grad():
        semantic_logits = scorer(flagged_emb)
        edge_attn_bias = sab(
            semantic_logits, flagged_indices, num_edges=data.edge_index.shape[1]
        )

    bias_on_flagged = edge_attn_bias[flagged_indices].squeeze(-1)
    magnitude_stats = {
        "mean_abs_bias_on_flagged": float(bias_on_flagged.abs().mean()),
        "max_abs_bias_on_flagged": float(bias_on_flagged.abs().max()),
        "std_bias_on_flagged": float(bias_on_flagged.std()),
    }

    layer = BiasedGATv2Layer(
        in_channels=data.x.shape[1],
        out_channels=8,
        edge_attr_dim=data.edge_attr.shape[1],
        bias_dim=DEFAULT_BIAS_DIM,
        heads=DEFAULT_HEADS,
        concat=True,
    )
    layer.eval()

    with torch.no_grad():
        _, (edge_index_att_zero, alpha_zero) = layer(
            data.x, data.edge_index, data.edge_attr,
            torch.zeros_like(edge_attn_bias),
            return_attention_weights=True,
        )
        _, (edge_index_att_bias, alpha_bias) = layer(
            data.x, data.edge_index, data.edge_attr, edge_attn_bias,
            return_attention_weights=True,
        )

    if alpha_zero.dim() == 2:
        alpha_zero_flat = alpha_zero.mean(dim=-1)
        alpha_bias_flat = alpha_bias.mean(dim=-1)
    else:
        alpha_zero_flat = alpha_zero
        alpha_bias_flat = alpha_bias

    original_edge_count = data.edge_index.shape[1]
    alpha_zero_orig = alpha_zero_flat[:original_edge_count]
    alpha_bias_orig = alpha_bias_flat[:original_edge_count]

    shift = (alpha_bias_orig - alpha_zero_orig).abs()
    per_class_shift: list[dict] = []
    for c in range(NUM_CLASSES):
        cls_uncertain = (labels == c) & uncertain_mask
        n = int(cls_uncertain.sum())
        if n == 0:
            per_class_shift.append(
                {"class": c, "n_uncertain": 0, "mean_attn_shift": float("nan")}
            )
            continue
        mean_shift = float(shift[cls_uncertain].mean())
        per_class_shift.append(
            {"class": c, "n_uncertain": n, "mean_attn_shift": mean_shift}
        )

    return {
        "bias_magnitude": magnitude_stats,
        "per_class_attention_shift": per_class_shift,
        "mean_attn_shift_on_flagged": float(shift[uncertain_mask].mean()),
        "mean_attn_shift_on_unflagged": float(shift[~uncertain_mask].mean()),
    }


def dashboard_phase_4_4(dataset: str) -> Path:
    reports_dir = _step4_root(dataset) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    config = get_dataset_config(dataset)
    label_names = list(config.label_names)

    test_a = _test_a_zero_on_unflagged(num_classes=NUM_CLASSES, num_edges=200, num_flagged=30)
    test_b = _test_b_gradient_connected(num_classes=NUM_CLASSES, num_flagged=30)
    test_c = _test_c_output_localized()
    diagnostic = _bias_magnitude_and_attention_shift_diagnostic(dataset)

    core_tests = [test_a, test_b, test_c]
    acceptance_pass = all(t["result"] == "PASS" for t in core_tests)

    json_payload = {
        "phase": "4.4",
        "component": "SemanticAttentionBias + BiasedGATv2Layer",
        "dataset": dataset,
        "acceptance": {
            "rule": "tests (a), (b), (c) all PASS",
            "tests": core_tests,
            "result": "PASS" if acceptance_pass else "FAIL",
        },
        "diagnostic": diagnostic,
        "config": {
            "bias_dim": DEFAULT_BIAS_DIM,
            "heads": DEFAULT_HEADS,
        },
    }
    with open(reports_dir / "phase_4_4_dashboard.json", "w") as f:
        json.dump(json_payload, f, indent=2)

    md_lines: list[str] = []
    md_lines.append("# Phase 4.4 Dashboard — Attention-Bias Injection")
    md_lines.append("")
    md_lines.append(f"**Dataset:** {dataset}")
    md_lines.append(f"**Injection path:** additive bias on the attention logits, pre-softmax (`BiasedGATv2Conv`), bias_dim = {DEFAULT_BIAS_DIM}")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    md_lines.append("## Acceptance test — three isolation checks")
    md_lines.append("")
    for t in core_tests:
        md_lines.append(f"### {t['name']} — **{t['result']}**")
        md_lines.append("")
        md_lines.append(t["description"])
        md_lines.append("")
        details = {k: v for k, v in t.items() if k not in {"name", "description", "result"}}
        for k, v in details.items():
            md_lines.append(f"- `{k}`: {v}")
        md_lines.append("")

    md_lines.append(f"**Overall:** **{'PASS' if acceptance_pass else 'FAIL'}**")
    md_lines.append("")

    md_lines.append("## Bias magnitude on flagged edges (random projection)")
    md_lines.append("")
    md_lines.append(
        "With a randomised `SemanticAttentionBias.projection` (weights N(0, 0.5)) "
        "run on the 197 UNSW uncertain edges through fold-0 prototypes:"
    )
    md_lines.append("")
    for k, v in diagnostic["bias_magnitude"].items():
        md_lines.append(f"- `{k}`: {v:.4f}")
    md_lines.append("")

    md_lines.append("## Attention shift on the real UNSW graph")
    md_lines.append("")
    md_lines.append(
        "Difference in per-edge attention weights between a zero-bias and a "
        "non-zero-bias forward pass through a fresh `BiasedGATv2Layer` "
        "(heads averaged). Confirms the bias channel actually moves GATv2's "
        "attention distribution."
    )
    md_lines.append("")
    md_lines.append(
        f"- Mean |Δα| on flagged edges: {diagnostic['mean_attn_shift_on_flagged']:.4e}"
    )
    md_lines.append(
        f"- Mean |Δα| on unflagged edges: {diagnostic['mean_attn_shift_on_unflagged']:.4e}"
    )
    md_lines.append("")
    md_lines.append("### Per-class attention shift on flagged edges")
    md_lines.append("")
    md_lines.append(_md_table(
        ["class", "name", "n_uncertain", "mean |Δα|"],
        [
            [
                str(r["class"]),
                label_names[r["class"]],
                str(r["n_uncertain"]),
                f"{r['mean_attn_shift']:.4e}"
                if not np.isnan(r["mean_attn_shift"]) else "—",
            ]
            for r in diagnostic["per_class_attention_shift"]
        ],
    ))
    md_lines.append("")

    with open(reports_dir / "phase_4_4_dashboard.md", "w") as f:
        f.write("\n".join(md_lines))

    print(f"Wrote {reports_dir / 'phase_4_4_dashboard.md'}")
    print(f"Wrote {reports_dir / 'phase_4_4_dashboard.json'}")
    print(f"Acceptance: {'PASS' if acceptance_pass else 'FAIL'}")
    return reports_dir / "phase_4_4_dashboard.md"


def _load_feedback_artifacts(dataset: str) -> dict:
    root = _step4_root(dataset)
    bench_path = root / "benchmark_summary.json"
    if not bench_path.exists():
        raise FileNotFoundError(
            f"Feedback results not found at {bench_path}. Run: "
            f"python -m src.pipeline.step4.train_feedback --dataset {dataset}"
        )
    with open(bench_path) as f:
        benchmark = json.load(f)
    with open(root / "ablation_summary.json") as f:
        ablation = json.load(f)
    traces = {}
    for mode in ("real", "random", "head_only"):
        tpath = root / f"feedback_trace_{mode}.json"
        if tpath.exists():
            with open(tpath) as f:
                traces[mode] = json.load(f)
    return {"benchmark": benchmark, "ablation": ablation, "traces": traces}


def _mean_per_iter(trace: dict, key: str) -> list[float]:
    """Average a per-iteration scalar (e.g. test_macro_f1) across folds,
    aligned by iteration index (folds may early-stop at different lengths)."""
    by_iter: dict[int, list[float]] = {}
    for fold in trace["folds"]:
        for row in fold["iterations"]:
            by_iter.setdefault(row["iter"], []).append(row[key])
    return [float(np.mean(by_iter[i])) for i in sorted(by_iter)]


def _mean_per_class_f1_last_iter(trace: dict) -> list[float]:
    """Average the last-iteration per-class F1 across folds."""
    acc = np.zeros(NUM_CLASSES)
    n = 0
    for fold in trace["folds"]:
        if not fold["iterations"]:
            continue
        acc += np.array(fold["iterations"][-1]["per_class_f1"])
        n += 1
    return list(acc / max(n, 1))


def dashboard_phase_4_5(dataset: str) -> Path:
    reports_dir = _step4_root(dataset) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    label_names = list(get_dataset_config(dataset).label_names)

    art = _load_feedback_artifacts(dataset)
    pooled = art["benchmark"]["per_mode_pooled_macro_f1"]
    ablation = art["ablation"]
    traces = art["traces"]

    real_f1 = pooled.get("real", float("nan"))
    target = art["benchmark"].get("target_agaf", 0.6225)

    beats_random = (
        "real_vs_random" in ablation and ablation["real_vs_random"]["ci_low"] > 0
    )
    beats_head = pooled.get("real", 0) > pooled.get("head_only", 1)
    hits_target = real_f1 >= target
    acceptance = beats_random and beats_head and hits_target

    json_payload = {
        "phase": "4.5",
        "component": "FeedbackLoopClassifier (outer loop + convergence)",
        "dataset": dataset,
        "per_mode_pooled_macro_f1": pooled,
        "ablation": ablation,
        "acceptance": {
            "rule": "pooled macro-F1 >= target AND real>random (bootstrap CI>0) AND real>head_only",
            "pooled_macro_f1": real_f1,
            "target": target,
            "hits_target": hits_target,
            "beats_random_ci_gt_0": beats_random,
            "beats_head_only": beats_head,
            "result": "PASS" if acceptance else "FAIL",
        },
    }
    with open(reports_dir / "phase_4_5_dashboard.json", "w") as f:
        json.dump(json_payload, f, indent=2)

    md: list[str] = []
    md.append("# Phase 4.5 Dashboard — Outer Loop with Convergence")
    md.append("")
    md.append(f"**Dataset:** {dataset}")
    md.append("**Mechanism:** true pre-softmax attention bias (`BiasedGATv2Conv`), "
              "3-variant fusion, OOF cross-validation.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Acceptance test")
    md.append("")
    md.append("**Rule:** pooled macro-F1 ≥ target **and** real beats random "
              "(bootstrap CI > 0) **and** real beats head-only.")
    md.append("")
    md.append(_md_table(
        ["check", "value", "pass"],
        [
            [f"pooled macro-F1 ≥ {target}", f"{real_f1:.4f}", "PASS" if hits_target else "FAIL"],
            ["real > random (CI low > 0)",
             f"CI=[{ablation.get('real_vs_random', {}).get('ci_low', float('nan')):+.4f}, "
             f"{ablation.get('real_vs_random', {}).get('ci_high', float('nan')):+.4f}]",
             "PASS" if beats_random else "FAIL"],
            ["real > head_only",
             f"{pooled.get('real', float('nan')):.4f} vs {pooled.get('head_only', float('nan')):.4f}",
             "PASS" if beats_head else "FAIL"],
        ],
    ))
    md.append("")
    md.append(f"**Overall:** **{'PASS' if acceptance else 'FAIL'}**")
    md.append("")

    md.append("## Pooled CV macro-F1 by feedback mode")
    md.append("")
    md.append(_md_table(
        ["mode", "pooled CV macro-F1", "meaning"],
        [
            ["real", f"{pooled.get('real', float('nan')):.4f}", "LLM semantic logits drive the bias"],
            ["random", f"{pooled.get('random', float('nan')):.4f}", "fixed per-edge noise drives the bias"],
            ["head_only", f"{pooled.get('head_only', float('nan')):.4f}", "bias disabled (plain GNN in a loop)"],
        ],
    ))
    md.append("")
    if "real_vs_random" in ablation:
        r = ablation["real_vs_random"]
        md.append(f"- **real − random:** Δ={r['mean_diff']:+.4f}, "
                  f"95% CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}], "
                  f"P(real>random)={r['prob_positive']:.3f}")
    if "real_vs_head_only" in ablation:
        r = ablation["real_vs_head_only"]
        md.append(f"- **real − head_only:** Δ={r['mean_diff']:+.4f}, "
                  f"95% CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}], "
                  f"P(real>head_only)={r['prob_positive']:.3f}")
    md.append("")

    # per-iteration convergence (real mode)
    if "real" in traces:
        md.append("## Per-iteration convergence (real mode, mean across folds)")
        md.append("")
        f1s = _mean_per_iter(traces["real"], "test_macro_f1")
        churns = _mean_per_iter(traces["real"], "churn")
        ents = _mean_per_iter(traces["real"], "mean_entropy")
        md.append(_md_table(
            ["iter", "test macro-F1", "churn", "mean entropy"],
            [
                [str(i + 1), f"{f1s[i]:.4f}",
                 "—" if np.isnan(churns[i]) else f"{churns[i]:.4f}",
                 f"{ents[i]:.4f}"]
                for i in range(len(f1s))
            ],
        ))
        md.append("")
        md.append("## Per-class F1 at the final iteration (mean across folds)")
        md.append("")
        rows = []
        real_pc = _mean_per_class_f1_last_iter(traces["real"])
        head_pc = _mean_per_class_f1_last_iter(traces["head_only"]) if "head_only" in traces else [float("nan")] * NUM_CLASSES
        for c in range(NUM_CLASSES):
            rows.append([str(c), label_names[c], f"{real_pc[c]:.3f}", f"{head_pc[c]:.3f}",
                         f"{real_pc[c] - head_pc[c]:+.3f}"])
        md.append(_md_table(["class", "name", "real F1", "head_only F1", "Δ (real−head)"], rows))
        md.append("")

    with open(reports_dir / "phase_4_5_dashboard.md", "w") as f:
        f.write("\n".join(md))
    print(f"Wrote {reports_dir / 'phase_4_5_dashboard.md'}")
    print(f"Wrote {reports_dir / 'phase_4_5_dashboard.json'}")
    print(f"Acceptance: {'PASS' if acceptance else 'FAIL'}")
    return reports_dir / "phase_4_5_dashboard.md"


PHASES = {
    "1": dashboard_phase_4_1,
    "2": dashboard_phase_4_2,
    "3": dashboard_phase_4_3,
    "4": dashboard_phase_4_4,
    "5": dashboard_phase_4_5,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=sorted(PHASES))
    parser.add_argument("--dataset", default="ton_iot", choices=["ton_iot", "unsw_nb15"])
    args = parser.parse_args()
    PHASES[args.phase](args.dataset)


if __name__ == "__main__":
    main()
