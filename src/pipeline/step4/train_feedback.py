"""Step 4 — Phase 4.5 feedback-loop training driver.

Trains the `FeedbackLoopClassifier` per CV fold and stitches out-of-fold
(OOF) predictions, exactly like `build_oof_predictions.py` does for the
plain GNN, so the loop is evaluated only on edges it was not trained on.

Three feedback modes are trained independently and compared — this is the
honest ablation the reverted attempt lacked:

  - ``real``      — LLM semantic logits drive the attention bias;
  - ``random``    — the bias is driven by fixed per-edge noise;
  - ``head_only`` — the bias is disabled (plain GNN, run in a loop).

Outputs (under ``data/{dataset}/processed/step4_feedback/``):
  - ``feedback_oof_{mode}.pt``    — [E, C] OOF logits per mode;
  - ``feedback_trace_{mode}.json``— per-iteration pooled/per-class F1;
  - ``benchmark_summary.json``    — real-mode pooled CV macro-F1;
  - ``ablation_summary.json``     — real vs random vs head-only + bootstrap CI.

Run:
    python -m src.pipeline.step4.train_feedback --dataset unsw_nb15
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch_geometric.data import Data

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.models.feedback_classifier import (
    FEEDBACK_MODES,
    GATE_MODES,
    FeedbackLoopClassifier,
    WhitenedPrototypeScorer,
)
from src.models.gnn_classifier import VARIANT_NAMES
from src.pipeline.common.datasets import DATASETS, get_dataset_config
from src.pipeline.common.splits import (
    oversample_minority_embeddings,
    class_weights_from_labels,
    NUM_CLASSES,
    FocalLoss,
    eval_macro_f1,
    get_class_weights,
    mask_dropped_logits,
)
from src.pipeline.step4.feedback_config import (
    DEFAULT_BIAS_CONFIDENCE_FRAC,
    DEFAULT_CHURN_TOLERANCE,
    DEFAULT_GATE_MODE,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_TOP_K_PERCENT,
    load_feedback_config,
)


# --- hyperparameters (mirror build_oof_predictions.py; fewer epochs since
#     each forward runs the GAT `max_iterations` times) -------------------
IN_DIM = 10
HIDDEN_DIM = 64
EDGE_ATTR_DIM = 5
HEADS = 8
DROPOUT = 0.2
AUX_LOSS_WEIGHT = 0.30

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
MAX_EPOCHS = 300
EARLY_STOPPING_PATIENCE = 40
GRAD_CLIP = 1.0
LOG_EVERY = 60
SEED = 42

TOP_K_PERCENT = DEFAULT_TOP_K_PERCENT
MAX_ITERATIONS = DEFAULT_MAX_ITERATIONS
CHURN_TOL = DEFAULT_CHURN_TOLERANCE
BIAS_CONFIDENCE_FRAC = DEFAULT_BIAS_CONFIDENCE_FRAC  # only bias the top-half most-confident flagged edges
# Multiplier on the projected attention bias. 1.5 is the spec's ~1.5-nat order; the
# Phase-2 mechanism sweep raises it to test whether the bias is simply too quiet to
# change which neighbours the GAT attends to.
DEFAULT_BIAS_STRENGTH = 1.5

BOOTSTRAP_ITERS = 2000


@dataclass
class FoldTrainingResult:
    """Artifacts and selection metrics from one independently trained fold."""

    logits: torch.Tensor
    iterations: list[dict]
    best_val_macro_f1: float
    test_macro_f1: float
    bias_diagnostics: dict | None = None


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _preds_from_logits(
    logits: torch.Tensor,
    dropped_classes: tuple[int, ...] | list[int] = (),
) -> torch.Tensor:
    return mask_dropped_logits(logits, dropped_classes).argmax(dim=1)


def _macro_f1(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    eval_classes: tuple[int, ...] | list[int] | None = None,
    dropped_classes: tuple[int, ...] | list[int] = (),
) -> float:
    preds = _preds_from_logits(logits[mask], dropped_classes)
    return eval_macro_f1(labels[mask], preds, eval_classes)


def _build_iteration_rows(
    trace: list[dict],
    labels: torch.Tensor,
    test_mask: torch.Tensor,
    eval_classes: tuple[int, ...] | list[int] | None = None,
    dropped_classes: tuple[int, ...] | list[int] = (),
) -> list[dict]:
    """Convert tensor-bearing model traces into JSON-safe test-fold evidence."""
    if not trace:
        return []

    baseline_pred = _preds_from_logits(trace[0]["logits"][test_mask], dropped_classes)
    targets = labels[test_mask]
    baseline_correct = baseline_pred == targets
    rows: list[dict] = []
    for item in trace:
        it_logits = item["logits"]
        current_pred = _preds_from_logits(it_logits[test_mask], dropped_classes)
        current_correct = current_pred == targets
        rows.append(
            {
                "iter": item["iter"],
                "churn": item["churn"],
                "mean_entropy": item["mean_entropy"],
                "mean_disagreement": item["mean_disagreement"],
                "flagged_overlap": item["flagged_overlap"],
                "selected_count": item["selected_count"],
                "selection_jaccard_previous": item[
                    "selection_jaccard_previous"
                ],
                "advice_recomputed": item["advice_recomputed"],
                "test_macro_f1": _macro_f1(
                    it_logits, labels, test_mask, eval_classes, dropped_classes
                ),
                "wrong_to_correct": int(
                    ((~baseline_correct) & current_correct).sum()
                ),
                "correct_to_wrong": int(
                    (baseline_correct & (~current_correct)).sum()
                ),
                "per_class_f1": _per_class_f1(it_logits, labels, test_mask),
            }
        )
    return rows


def _build_model(
    top_k_percent: float = TOP_K_PERCENT,
    bias_confidence_fraction: float = BIAS_CONFIDENCE_FRAC,
    gate_mode: str = DEFAULT_GATE_MODE,
    use_no_regret_floor: bool = False,
    use_output_fusion: bool = True,
    bias_strength: float = DEFAULT_BIAS_STRENGTH,
    bias_dim: int | None = None,
    max_iterations: int = MAX_ITERATIONS,
    bias_init: str | None = None,
    injection_mode: str = "edge",
    injection_scale: float = 10.0,
    data=None,
) -> FeedbackLoopClassifier:
    # Single source of truth for the mode-dependent bias shape. Edge injection needs a
    # bias exactly as wide as edge_attr and a live projection; attention injection wants
    # the 1-wide zero-init bias that reproduces stock GATv2. Callers that do not resolve
    # these themselves (sweep_top_k) would otherwise silently sweep under a different
    # mechanism than the one the feedback stage then trains with.
    if bias_dim is None:
        bias_dim = EDGE_ATTR_DIM if injection_mode == "edge" else 1
    if bias_init is None:
        bias_init = "xavier" if injection_mode == "edge" else "zeros"
    if not 0.0 < bias_confidence_fraction <= 1.0:
        raise ValueError(
            "bias_confidence_fraction must be in (0, 1], got "
            f"{bias_confidence_fraction}"
        )
    return FeedbackLoopClassifier(
        in_dim=IN_DIM,
        hidden_dim=HIDDEN_DIM,
        edge_attr_dim=EDGE_ATTR_DIM,
        num_classes=NUM_CLASSES,
        heads=HEADS,
        dropout=DROPOUT,
        top_k_percent=top_k_percent,
        max_iterations=max_iterations,
        churn_tol=CHURN_TOL,
        bias_dim=bias_dim,
        initial_log_bias_strength=math.log(bias_strength),
        bias_init=bias_init,
        bias_confidence_frac=bias_confidence_fraction,
        gate_mode=gate_mode,
        use_no_regret_floor=use_no_regret_floor,
        use_output_fusion=use_output_fusion,
        injection_mode=injection_mode,
        injection_scale=injection_scale,
        **({} if data is None or getattr(data, "num_protocols", None) is None
           else {"num_protocols": data.num_protocols, "num_ports": data.num_ports}),
    )


def _train_one_fold(
    data: Data,
    emb: torch.Tensor,
    fold: dict[str, torch.Tensor],
    fold_state: dict,
    fold_idx: int,
    mode: str,
    head_logits: torch.Tensor | None = None,
    top_k_percent: float = TOP_K_PERCENT,
    bias_confidence_fraction: float = BIAS_CONFIDENCE_FRAC,
    gate_mode: str = DEFAULT_GATE_MODE,
    eval_classes: tuple[int, ...] | list[int] | None = None,
    dropped_classes: tuple[int, ...] | list[int] = (),
    use_no_regret_floor: bool = False,
    use_output_fusion: bool = True,
    bias_strength: float = DEFAULT_BIAS_STRENGTH,
    bias_dim: int | None = None,
    max_iterations: int = MAX_ITERATIONS,
    bias_init: str | None = None,
    oversample_ratio: float = 0.0,
    injection_mode: str = "edge",
    injection_scale: float = 10.0,
) -> FoldTrainingResult:
    """Train one fold in one feedback mode; return full-graph logits from the
    best-val checkpoint plus the per-iteration trace on the test edges."""
    _set_seed(SEED + fold_idx)
    model = _build_model(
        top_k_percent=top_k_percent,
        bias_confidence_fraction=bias_confidence_fraction,
        gate_mode=gate_mode,
        use_no_regret_floor=use_no_regret_floor,
        use_output_fusion=use_output_fusion,
        bias_strength=bias_strength,
        bias_dim=bias_dim,
        max_iterations=max_iterations,
        bias_init=bias_init,
        injection_mode=injection_mode,
        injection_scale=injection_scale,
        data=data,
    )
    model.load_fold_state(
        fold_state["mean"], fold_state["whitener"], fold_state["prototypes_whitened"]
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    train_mask = fold["train_mask"]
    val_mask = fold["val_mask"]
    labels = data.edge_label

    class_weights = get_class_weights(labels, train_mask)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    best_val_f1 = -1.0
    best_state: dict | None = None
    no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        logits, aux = model(
            data.x, data.edge_index, data.edge_attr, emb, feedback_mode=mode,
            head_logits=head_logits,
        )
        loss = criterion(logits[train_mask], labels[train_mask])
        aux_loss = torch.stack(
            [criterion(aux[n][train_mask], labels[train_mask]) for n in VARIANT_NAMES]
        ).mean()
        total_loss = loss + AUX_LOSS_WEIGHT * aux_loss
        # GraphSMOTE-style balancing on TRAIN-fold edge representations only. The
        # graph topology is never touched; only the classifier head sees synthetic
        # points, which is what keeps them on the learned manifold.
        if oversample_ratio > 0.0 and model._last_edge_emb is not None:
            aug_repr, aug_labels = oversample_minority_embeddings(
                model._last_edge_emb[train_mask], labels[train_mask],
                target_ratio=oversample_ratio, seed=SEED + epoch,
            )
            if aug_repr.shape[0] > 0:
                # One objective over real+synthetic, class weights recomputed on the
                # balanced distribution (reusing imbalanced weights double-corrects).
                all_logits = torch.cat(
                    [logits[train_mask], model.classify_repr(aug_repr)], dim=0
                )
                all_labels = torch.cat([labels[train_mask], aug_labels], dim=0)
                balanced = FocalLoss(
                    alpha=class_weights_from_labels(all_labels), gamma=2.0
                )
                total_loss = (
                    balanced(all_logits, all_labels) + AUX_LOSS_WEIGHT * aux_loss
                )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_logits, _ = model(
                data.x, data.edge_index, data.edge_attr, emb, feedback_mode=mode,
                head_logits=head_logits,
            )
            val_f1 = _macro_f1(
                eval_logits, labels, val_mask, eval_classes, dropped_classes
            )

        if epoch % LOG_EVERY == 0 or epoch == 1:
            print(
                f"    [{mode}] fold {fold_idx} epoch {epoch:3d} "
                f"loss={loss.item():.4f} val_f1={val_f1:.4f}"
            )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOPPING_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    test_mask = fold["test_mask"]
    with torch.no_grad():
        eval_logits, _, trace = model(
            data.x, data.edge_index, data.edge_attr, emb,
            feedback_mode=mode, collect_trace=True, head_logits=head_logits,
        )

    iter_rows = _build_iteration_rows(
        trace, labels, test_mask, eval_classes, dropped_classes
    )

    test_f1 = _macro_f1(eval_logits, labels, test_mask, eval_classes, dropped_classes)

    # --- mechanism diagnostics -------------------------------------------
    # SemanticAttentionBias initialises its projection to all zeros, so the
    # attention bias starts at exactly zero and has to learn its way off that
    # init. These numbers say whether it ever did: `projection_weight_absmean`
    # near zero means the bias never fires, and `flagged_bias_absmean` is the
    # magnitude actually added to GATv2's attention logits (which sit on a
    # ~1 nat scale, so anything <<1 cannot reorder a softmax).
    with torch.no_grad():
        bm = model.bias_module
        semantic = model._semantic_logits(emb, mode, None, head_logits)
        if semantic is None:
            bias_diag = {"mode_has_no_semantic_signal": True}
        else:
            probs = eval_logits.softmax(dim=-1)
            flagged = model.selector(probs).nonzero(as_tuple=False).squeeze(-1)
            flagged = model._gate_flagged(flagged, semantic, probs)
            full_bias = bm(semantic[flagged], flagged, data.edge_index.shape[1])
            fb = full_bias[flagged]
            bias_diag = {
                "learned_bias_strength": float(bm.log_bias_strength.exp()),
                "projection_weight_absmean": float(bm.projection.weight.abs().mean()),
                "projection_weight_absmax": float(bm.projection.weight.abs().max()),
                "flagged_bias_absmean": float(fb.abs().mean()) if fb.numel() else 0.0,
                "flagged_bias_absmax": float(fb.abs().max()) if fb.numel() else 0.0,
                "n_flagged_edges": int(flagged.numel()),
                "iterations_run": len(trace),
                "final_churn": trace[-1]["churn"] if trace else float("nan"),
            }
    print(
        f"    [{mode}] fold {fold_idx} DONE best_val={best_val_f1:.4f} "
        f"test={test_f1:.4f} iters={len(trace)}"
        + (f" bias|.|={bias_diag['flagged_bias_absmean']:.2e}"
           if "flagged_bias_absmean" in bias_diag else "")
    )
    return FoldTrainingResult(
        logits=eval_logits.detach().cpu(),
        iterations=iter_rows,
        best_val_macro_f1=best_val_f1,
        test_macro_f1=test_f1,
        bias_diagnostics=bias_diag,
    )


def _per_class_f1(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> list[float]:
    preds = logits[mask].argmax(dim=1).cpu().numpy()
    targets = labels[mask].cpu().numpy()
    per = f1_score(
        targets, preds, average=None,
        labels=list(range(NUM_CLASSES)), zero_division=0,
    )
    return [float(x) for x in per]


def _bootstrap_ci(
    labels: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    eval_classes: tuple[int, ...] | list[int] | None = None,
    iters: int = BOOTSTRAP_ITERS,
    seed: int = SEED,
) -> dict:
    """Bootstrap CI of macro-F1(a) - macro-F1(b) over resampled edges."""
    rng = np.random.default_rng(seed)
    labs = list(eval_classes) if eval_classes is not None else list(range(NUM_CLASSES))
    row_mask = np.isin(labels, labs)
    labels = labels[row_mask]
    preds_a = preds_a[row_mask]
    preds_b = preds_b[row_mask]
    n = len(labels)
    diffs = []
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        fa = f1_score(labels[idx], preds_a[idx], average="macro", labels=labs, zero_division=0)
        fb = f1_score(labels[idx], preds_b[idx], average="macro", labels=labs, zero_division=0)
        diffs.append(fa - fb)
    diffs = np.array(diffs)
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "prob_positive": float((diffs > 0).mean()),
    }


def _train_loop(
    model, data, emb, fold, criterion, mode, epochs, patience,
    freeze_backbone=False, gen=None, eval_classes=None, dropped_classes=(),
):
    """Shared inner training loop. Returns (best_state, best_val_f1)."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_mask, val_mask, labels = fold["train_mask"], fold["val_mask"], data.edge_label

    best_val_f1, best_state, no_improve = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        if freeze_backbone:
            model.backbone_eval()  # keep frozen BN/dropout fixed
        optimizer.zero_grad()
        logits, aux = model(data.x, data.edge_index, data.edge_attr, emb, feedback_mode=mode)
        loss = criterion(logits[train_mask], labels[train_mask])
        aux_loss = torch.stack(
            [criterion(aux[n][train_mask], labels[train_mask]) for n in VARIANT_NAMES]
        ).mean()
        (loss + AUX_LOSS_WEIGHT * aux_loss).backward()
        torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_logits, _ = model(data.x, data.edge_index, data.edge_attr, emb, feedback_mode=mode)
            val_f1 = _macro_f1(
                eval_logits, labels, val_mask, eval_classes, dropped_classes
            )
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    return best_state, best_val_f1


def _train_one_fold_frozen(
    data, emb, fold, fold_state, fold_idx, eval_classes=None, dropped_classes=(),
    gate_mode: str = DEFAULT_GATE_MODE,
):
    """Two-phase frozen training:
      Phase 1 — train the GAT backbone alone (feedback off) → the canonical
                GNN baseline.
      Phase 2 — FREEZE that backbone and train only the feedback modules on
                top (attention bias + LLM fusion), a pure residual correction.
    Returns (gnn_alone_logits, frozen_feedback_logits), both full-graph."""
    _set_seed(SEED + fold_idx)
    model = _build_model(data=data, gate_mode=gate_mode)
    model.load_fold_state(fold_state["mean"], fold_state["whitener"], fold_state["prototypes_whitened"])
    labels = data.edge_label
    criterion = FocalLoss(alpha=get_class_weights(labels, fold["train_mask"]), gamma=2.0)

    # Phase 1 — backbone (GNN alone)
    best_state, val1 = _train_loop(
        model, data, emb, fold, criterion, "head_only", MAX_EPOCHS,
        EARLY_STOPPING_PATIENCE, eval_classes=eval_classes,
        dropped_classes=dropped_classes,
    )
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        gnn_logits, _ = model(data.x, data.edge_index, data.edge_attr, emb, feedback_mode="head_only")

    # Phase 2 — freeze backbone, train the correction only
    model.freeze_backbone()
    best_state2, val2 = _train_loop(
        model, data, emb, fold, criterion, "real", MAX_EPOCHS, EARLY_STOPPING_PATIENCE,
        freeze_backbone=True, eval_classes=eval_classes,
        dropped_classes=dropped_classes,
    )
    if best_state2 is not None:
        model.load_state_dict(best_state2)
    model.eval()
    with torch.no_grad():
        fb_logits, _ = model(data.x, data.edge_index, data.edge_attr, emb, feedback_mode="real")

    test_mask = fold["test_mask"]
    print(
        f"  [frozen] fold {fold_idx}: GNN-alone val={val1:.4f} "
        f"test={_macro_f1(gnn_logits, labels, test_mask, eval_classes, dropped_classes):.4f} | "
        f"frozen-loop val={val2:.4f} "
        f"test={_macro_f1(fb_logits, labels, test_mask, eval_classes, dropped_classes):.4f}"
    )
    return gnn_logits.detach().cpu(), fb_logits.detach().cpu()


def _llm_alone_oof(data, emb, folds, protos):
    """LLM path alone: per-fold prototype scorer, OOF-stitched predictions."""
    labels = data.edge_label
    oof = torch.full((labels.shape[0], NUM_CLASSES), float("nan"))
    for fi, fold in enumerate(folds):
        st = protos["folds"][fi]
        scorer = WhitenedPrototypeScorer(num_classes=NUM_CLASSES, embed_dim=protos["embed_dim"])
        scorer.load_fold_state(st["mean"], st["whitener"], st["prototypes_whitened"])
        tm = fold["test_mask"]
        with torch.no_grad():
            oof[tm] = scorer(emb[tm])
    return oof


def train_feedback_frozen(
    dataset: str, gate_mode: str = DEFAULT_GATE_MODE
) -> Path:
    """Run the frozen-backbone ladder: GNN alone · LLM alone · AGAF · frozen loop."""
    config = get_dataset_config(dataset)
    eval_classes = config.eval_classes
    dropped_classes = config.dropped_classes
    root = Path(f"data/{dataset}/processed/step4_feedback")
    data = torch.load(config.graph_path, weights_only=False)
    global IN_DIM
    IN_DIM = data.x.shape[1]  # derive from graph (supports pruned node features)
    emb = torch.load(config.llm_embedding_path, weights_only=False).float()
    folds = torch.load(config.splits_path, weights_only=False)
    protos = torch.load(root / "prototypes.pt", weights_only=False)
    labels = data.edge_label
    ne = labels.shape[0]

    gnn_oof = torch.full((ne, NUM_CLASSES), float("nan"))
    fb_oof = torch.full((ne, NUM_CLASSES), float("nan"))
    print("===== FROZEN-BACKBONE LADDER =====")
    for fi, fold in enumerate(folds):
        g, f = _train_one_fold_frozen(
            data, emb, fold, protos["folds"][fi], fi,
            eval_classes=eval_classes, dropped_classes=dropped_classes,
            gate_mode=gate_mode,
        )
        gnn_oof[fold["test_mask"]] = g[fold["test_mask"]]
        fb_oof[fold["test_mask"]] = f[fold["test_mask"]]

    llm_oof = _llm_alone_oof(data, emb, folds, protos)

    def pooled(oof):
        preds = _preds_from_logits(oof, dropped_classes)
        return eval_macro_f1(labels, preds, eval_classes)

    # Fair AGAF number: require the re-run benchmark on the current enriched NL.
    agaf_bench = Path(f"data/{dataset}/processed/step3_fusion/benchmark_summary.json")
    if not agaf_bench.exists():
        raise FileNotFoundError(
            f"Missing current AGAF benchmark: {agaf_bench}. Run Step 3 fusion first."
        )
    with open(agaf_bench) as f:
        agaf_ref = float(json.load(f)["pooled_cv_macro_f1"])

    ladder = {
        "dataset": dataset,
        "gnn_alone": pooled(gnn_oof),
        "llm_alone": pooled(llm_oof),
        "agaf_reference": agaf_ref,
        "feedback_loop_frozen": pooled(fb_oof),
        "eval_classes": list(eval_classes),
        "dropped_classes": list(dropped_classes),
        "gate_mode": gate_mode,
        "vs_agaf_bootstrap": _bootstrap_ci(
            labels.numpy(),
            _preds_from_logits(fb_oof, dropped_classes).numpy(),
            _preds_from_logits(gnn_oof, dropped_classes).numpy(),
            eval_classes,
        ),
    }
    torch.save(fb_oof, root / "feedback_oof_frozen.pt")
    with open(root / "ladder_summary.json", "w") as f:
        json.dump(ladder, f, indent=2)

    print("\n=== COMPARISON LADDER (pooled CV macro-F1) ===")
    print(f"  GNN path alone        : {ladder['gnn_alone']:.4f}")
    print(f"  LLM path alone        : {ladder['llm_alone']:.4f}")
    print(f"  AGAF (reference)      : {ladder['agaf_reference']:.4f}")
    print(f"  Feedback loop (frozen): {ladder['feedback_loop_frozen']:.4f}")
    r = ladder["vs_agaf_bootstrap"]
    print(f"  frozen-loop − GNN-alone: Δ={r['mean_diff']:+.4f} CI[{r['ci_low']:+.4f},{r['ci_high']:+.4f}]")
    return root / "ladder_summary.json"


def _build_benchmark_summary(
    dataset: str,
    labels: torch.Tensor,
    oof_by_mode: dict[str, torch.Tensor],
    pooled_f1: dict[str, float],
    target_agaf: float | None,
    top_k_percent: float = TOP_K_PERCENT,
    eval_classes: tuple[int, ...] | list[int] | None = None,
    dropped_classes: tuple[int, ...] | list[int] = (),
    selected_config: dict | None = None,
    use_llm_head: bool = False,
    bias_confidence_fraction: float = BIAS_CONFIDENCE_FRAC,
    gate_mode: str = DEFAULT_GATE_MODE,
    use_output_fusion: bool = True,
    injection_mode: str = "edge",
    injection_scale: float = 10.0,
    trace_summary_by_mode: dict[str, dict[str, float]] | None = None,
) -> dict:
    """Build the canonical feedback result payload from OOF predictions."""
    pooled_accuracy = {
        mode: float(
            (_preds_from_logits(logits, dropped_classes) == labels).sum().item()
            / labels.numel()
        )
        for mode, logits in oof_by_mode.items()
    }
    eval_labels = (
        list(eval_classes) if eval_classes is not None else list(range(NUM_CLASSES))
    )
    label_mask = np.isin(labels.cpu().numpy(), eval_labels)
    pooled_weighted_f1 = {
        mode: float(
            f1_score(
                labels.cpu().numpy()[label_mask],
                _preds_from_logits(logits, dropped_classes).cpu().numpy()[label_mask],
                average="weighted",
                labels=eval_labels,
                zero_division=0,
            )
        )
        for mode, logits in oof_by_mode.items()
    }
    trace_summary_by_mode = trace_summary_by_mode or {}
    return {
        "dataset": dataset,
        "eval_classes": list(eval_classes) if eval_classes is not None else list(range(NUM_CLASSES)),
        "dropped_classes": list(dropped_classes),
        "pooled_cv_macro_f1": pooled_f1.get("real"),
        "pooled_cv_accuracy": pooled_accuracy.get("real"),
        "pooled_cv_weighted_f1": pooled_weighted_f1.get("real"),
        "per_mode_pooled_macro_f1": pooled_f1,
        "per_mode_pooled_accuracy": pooled_accuracy,
        "per_mode_pooled_weighted_f1": pooled_weighted_f1,
        "mean_churn": trace_summary_by_mode.get("real", {}).get("mean_churn"),
        "mean_disagreement": trace_summary_by_mode.get("real", {}).get(
            "mean_disagreement"
        ),
        "per_mode_trace_summary": trace_summary_by_mode,
        "top_k_percent": top_k_percent,
        "bias_confidence_fraction": bias_confidence_fraction,
        "gate_mode": gate_mode,
        "effective_feedback_percent": top_k_percent * bias_confidence_fraction,
        "semantic_consultant": (
            "trained_llm_head" if use_llm_head else "whitened_prototype_scorer"
        ),
        "trained_llm_head": use_llm_head,
        "use_output_fusion": use_output_fusion,
        "injection_mode": injection_mode,
        "injection_scale": injection_scale,
        "llm_access": (
            f"{injection_mode}_injection_on_flagged_edges"
            + (" + output_fusion_on_all_edges" if use_output_fusion else "_only")
        ),
        "target_agaf": target_agaf,
        "selected_config": selected_config,
    }


def train_feedback(
    dataset: str,
    modes: list[str] | None = None,
    use_llm_head: bool = False,
    top_k_percent: float | None = None,
    output_dir: str | Path | None = None,
    prototypes_path: str | Path | None = None,
    head_logits_path: str | Path | None = None,
    agaf_benchmark_path: str | Path | None = None,
    bias_confidence_fraction: float | None = None,
    gate_mode: str = DEFAULT_GATE_MODE,
    use_no_regret_floor: bool = False,
    use_output_fusion: bool = True,
    bias_strength: float = DEFAULT_BIAS_STRENGTH,
    bias_dim: int | None = None,
    max_iterations: int = MAX_ITERATIONS,
    bias_init: str | None = None,
    oversample_ratio: float = 0.0,
    seed: int = SEED,
    injection_mode: str = "edge",
    injection_scale: float = 10.0,
) -> Path:
    modes = modes or list(FEEDBACK_MODES)
    # Edge injection needs a bias exactly as wide as edge_attr and a live
    # projection; attention injection wants the 1-wide zero-init bias that
    # reproduces stock GATv2. Resolve from the mode unless the caller was explicit.
    if bias_dim is None:
        bias_dim = EDGE_ATTR_DIM if injection_mode == "edge" else 1
    if bias_init is None:
        bias_init = "xavier" if injection_mode == "edge" else "zeros"
    global SEED
    SEED = seed
    config = get_dataset_config(dataset)
    canonical_root = Path(f"data/{dataset}/processed/step4_feedback")
    root = Path(output_dir) if output_dir is not None else canonical_root
    root.mkdir(parents=True, exist_ok=True)
    selected_config = load_feedback_config(
        dataset,
        root=root if output_dir is not None else None,
    )
    resolved_top_k = (
        float(top_k_percent)
        if top_k_percent is not None
        else float(selected_config["top_k_percent"])
    )
    resolved_confidence_fraction = (
        float(bias_confidence_fraction)
        if bias_confidence_fraction is not None
        else float(selected_config["bias_confidence_fraction"])
    )

    data: Data = torch.load(config.graph_path, weights_only=False)
    global IN_DIM
    IN_DIM = data.x.shape[1]  # derive from graph (supports pruned node features)
    emb = torch.load(config.llm_embedding_path, weights_only=False).float()
    folds = torch.load(config.splits_path, weights_only=False)
    resolved_prototypes_path = Path(prototypes_path) if prototypes_path else canonical_root / "prototypes.pt"
    protos = torch.load(resolved_prototypes_path, weights_only=False)
    # The loop's semantic consultant is the whitened-prototype scorer, which
    # fits no parameters. `use_llm_head` swaps in per-fold trained MLP head
    # logits instead; it is off by default so the reported ladder measures the
    # prototype path end to end.
    head_logits_all = None
    if use_llm_head:
        head_path = Path(head_logits_path) if head_logits_path else canonical_root / "llm_head_logits.pt"
        if not head_path.exists():
            raise FileNotFoundError(
                f"use_llm_head=True but {head_path} is missing. Run "
                f"`python -m src.pipeline.step4.build_llm_heads --dataset {dataset}`."
            )
        head_logits_all = torch.load(head_path, weights_only=False)
        print(f"Using trained LLM head logits from {head_path.name}")
    else:
        print("Semantic consultant: whitened-prototype scorer (no trained head)")
    labels = data.edge_label
    num_edges = labels.shape[0]
    eval_classes = config.eval_classes
    dropped_classes = config.dropped_classes

    oof_by_mode: dict[str, torch.Tensor] = {}
    pooled_f1: dict[str, float] = {}
    trace_summary_by_mode: dict[str, dict[str, float]] = {}

    for mode in modes:
        print(f"\n===== MODE: {mode} =====")
        oof = torch.full((num_edges, NUM_CLASSES), float("nan"))
        trace_by_fold = []
        for fold_idx, fold in enumerate(folds):
            fold_head = head_logits_all[fold_idx] if head_logits_all is not None else None
            fold_result = _train_one_fold(
                data, emb, fold, protos["folds"][fold_idx], fold_idx, mode,
                head_logits=fold_head, top_k_percent=resolved_top_k,
                bias_confidence_fraction=resolved_confidence_fraction,
                gate_mode=gate_mode,
                eval_classes=eval_classes, dropped_classes=dropped_classes,
                use_no_regret_floor=use_no_regret_floor,
                use_output_fusion=use_output_fusion,
                bias_strength=bias_strength,
                bias_dim=bias_dim,
                max_iterations=max_iterations,
                bias_init=bias_init,
                oversample_ratio=oversample_ratio,
                injection_mode=injection_mode,
                injection_scale=injection_scale,
            )
            oof[fold["test_mask"]] = fold_result.logits[fold["test_mask"]]
            trace_by_fold.append(
                {
                    "fold": fold_idx,
                    "iterations": fold_result.iterations,
                    "bias_diagnostics": fold_result.bias_diagnostics,
                }
            )

        if torch.isnan(oof).any():
            raise RuntimeError(f"[{mode}] some edges uncovered by any test fold.")
        oof_by_mode[mode] = oof
        preds = _preds_from_logits(oof, dropped_classes)
        pooled = eval_macro_f1(labels, preds, eval_classes)
        pooled_f1[mode] = pooled
        trace_summary_by_mode[mode] = {}
        for metric, summary_key in (
            ("churn", "mean_churn"),
            ("mean_disagreement", "mean_disagreement"),
            ("flagged_overlap", "mean_flagged_overlap"),
        ):
            values = [
                float(row[metric])
                for fold_row in trace_by_fold
                for row in fold_row["iterations"]
                if math.isfinite(float(row[metric]))
            ]
            trace_summary_by_mode[mode][summary_key] = (
                float(np.mean(values)) if values else float("nan")
            )
        torch.save(oof, root / f"feedback_oof_{mode}.pt")
        with open(root / f"feedback_trace_{mode}.json", "w") as f:
            json.dump({
                "dataset": dataset,
                "mode": mode,
                "eval_classes": list(eval_classes),
                "dropped_classes": list(dropped_classes),
                "top_k_percent": resolved_top_k,
                "bias_confidence_fraction": resolved_confidence_fraction,
                "gate_mode": gate_mode,
                "pooled_cv_macro_f1": pooled,
                "trace_summary": trace_summary_by_mode[mode],
                "folds": trace_by_fold,
            }, f, indent=2)
        print(f"[{mode}] pooled CV macro-F1 = {pooled:.4f}")

    # --- summaries ---------------------------------------------------------
    labels_np = labels.numpy()
    resolved_agaf_benchmark_path = (
        Path(agaf_benchmark_path)
        if agaf_benchmark_path is not None
        else Path(config.fusion_output_dir) / "benchmark_summary.json"
    )
    target_agaf = None
    if resolved_agaf_benchmark_path.exists():
        agaf_payload = json.loads(resolved_agaf_benchmark_path.read_text())
        agaf_macro_f1 = agaf_payload.get("pooled_cv_macro_f1")
        if agaf_macro_f1 is not None:
            target_agaf = float(agaf_macro_f1)
    benchmark = _build_benchmark_summary(
        dataset=dataset,
        labels=labels,
        oof_by_mode=oof_by_mode,
        pooled_f1=pooled_f1,
        target_agaf=target_agaf,
        top_k_percent=resolved_top_k,
        eval_classes=eval_classes,
        dropped_classes=dropped_classes,
        selected_config=selected_config,
        use_llm_head=use_llm_head,
        bias_confidence_fraction=resolved_confidence_fraction,
        gate_mode=gate_mode,
        use_output_fusion=use_output_fusion,
        injection_mode=injection_mode,
        injection_scale=injection_scale,
        trace_summary_by_mode=trace_summary_by_mode,
    )
    with open(root / "benchmark_summary.json", "w") as f:
        json.dump(benchmark, f, indent=2)

    ablation: dict = {
        "dataset": dataset,
        "eval_classes": list(eval_classes),
        "dropped_classes": list(dropped_classes),
        "top_k_percent": resolved_top_k,
        "bias_confidence_fraction": resolved_confidence_fraction,
        "gate_mode": gate_mode,
        "per_mode_pooled_macro_f1": pooled_f1,
    }
    if "real" in oof_by_mode and "random" in oof_by_mode:
        ablation["real_vs_random"] = _bootstrap_ci(
            labels_np,
            _preds_from_logits(oof_by_mode["real"], dropped_classes).numpy(),
            _preds_from_logits(oof_by_mode["random"], dropped_classes).numpy(),
            eval_classes,
        )
    if "real" in oof_by_mode and "head_only" in oof_by_mode:
        ablation["real_vs_head_only"] = _bootstrap_ci(
            labels_np,
            _preds_from_logits(oof_by_mode["real"], dropped_classes).numpy(),
            _preds_from_logits(oof_by_mode["head_only"], dropped_classes).numpy(),
            eval_classes,
        )
    with open(root / "ablation_summary.json", "w") as f:
        json.dump(ablation, f, indent=2)

    print("\n=== SUMMARY ===")
    for m, v in pooled_f1.items():
        print(f"  {m:10} pooled CV macro-F1 = {v:.4f}")
    if "real_vs_random" in ablation:
        r = ablation["real_vs_random"]
        print(f"  real - random: Δ={r['mean_diff']:+.4f} "
              f"CI[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] P(>0)={r['prob_positive']:.3f}")
    return root / "benchmark_summary.json"


def load_nl_lines(dataset: str) -> list[str]:
    """Load `kg_triples_nl.txt` for `dataset`, one line per edge.

    The file is row-aligned to `data.edge_index` (Step 2 writes it in
    the same order as `aggregated_edges.csv`, which Step 1 built by
    grouping the raw flow CSV in a stable order).
    """
    config = get_dataset_config(dataset)
    path = Path(config.kg_nl_path)
    if not path.exists():
        raise FileNotFoundError(
            f"KG NL file not found at {path}. Re-run Step 2 for {dataset}."
        )
    lines = path.read_text().splitlines()
    return [line for line in lines if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="unsw_nb15", choices=sorted(DATASETS))
    parser.add_argument(
        "--injection-mode", choices=("attention", "edge"), default="edge",
        help="Where the semantic advice is delivered. 'edge' (default) puts the "
             "advice on the edge features. 'attention' is todo.md Step 4 "
             "verbatim and is MEASURED INERT (churn 0.0000); 'edge' puts it on the "
             "edge features, the only term that separates co-located edges.",
    )
    parser.add_argument(
        "--injection-scale", type=float, default=10.0,
        help="Multiplier on the advice when --injection-mode edge. Selected on "
             "validation folds (5/10/20 grid).",
    )
    parser.add_argument(
        "--modes", nargs="+", default=list(FEEDBACK_MODES), choices=list(FEEDBACK_MODES)
    )
    parser.add_argument(
        "--frozen", action="store_true",
        help="Run the frozen-backbone ladder (GNN alone · LLM alone · AGAF · frozen loop).",
    )
    parser.add_argument(
        "--use-llm-head", action="store_true",
        help="Use trained per-fold MLP head logits as the semantic consultant "
             "instead of the whitened-prototype scorer.",
    )
    parser.add_argument(
        "--top-k-percent", type=float, default=None,
        help="Override the validation-selected top-k percentage for this run.",
    )
    parser.add_argument(
        "--output-dir",
        help="Write feedback artifacts here instead of the canonical Step 4 directory.",
    )
    parser.add_argument("--prototypes-path")
    parser.add_argument("--head-logits-path")
    parser.add_argument("--agaf-benchmark-path")
    parser.add_argument("--bias-confidence-fraction", type=float)
    parser.add_argument(
        "--gate-mode", choices=GATE_MODES, default=DEFAULT_GATE_MODE,
        help="Rank entropy-flagged edges by LLM confidence, GNN/LLM "
             "disagreement, or their product before keeping the configured fraction.",
    )
    parser.add_argument(
        "--oversample-ratio", type=float, default=0.0,
        help="GraphSMOTE-style balancing: bring each minority class up to this "
             "fraction of the majority class by interpolating TRAIN-fold edge "
             "representations. 0.0 (default) disables it entirely.",
    )
    parser.add_argument(
        "--bias-init", choices=("zeros", "xavier"), default=None,
        help="Init for the bias projection. 'zeros' makes the biased GAT reproduce "
             "stock GATv2 exactly but starts the mechanism inert; 'xavier' starts it "
             "live. Default: 'zeros' for --injection-mode attention, 'xavier' for edge.",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="Training seed. Vary it to measure how much of a between-config gap is "
             "just run-to-run noise.",
    )
    parser.add_argument(
        "--bias-strength", type=float, default=DEFAULT_BIAS_STRENGTH,
        help="Multiplier on the projected attention bias (default 1.5, the spec's "
             "~1.5-nat order). Raise it to test whether the bias is too quiet to change "
             "which neighbours the GAT attends to.",
    )
    parser.add_argument(
        "--bias-dim", type=int, default=None,
        help="Width of the bias: 1 broadcasts one value across all attention heads; "
             "8 gives each head its own. Default: 1 for --injection-mode attention, "
             f"{EDGE_ATTR_DIM} (edge_attr_dim) for edge, which is required there.",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=MAX_ITERATIONS,
        help="Feedback rounds before the loop stops (default 3).",
    )
    parser.add_argument(
        "--no-output-fusion", action="store_true",
        help="Disable the per-edge output fusion of the LLM branch, so the LLM "
             "reaches the loop ONLY through the attention bias on the flagged "
             "top-k edges (~effective_feedback_percent of edges).",
    )
    parser.add_argument(
        "--no-regret-floor", action="store_true",
        help="Fuse as floor(confidence-routed GNN/head) + bounded correction "
             "instead of the unconstrained gated blend, so the loop can't "
             "score below the better of its two branches on average.",
    )
    args = parser.parse_args()
    # Edge injection rides on the edge-feature vector, so the bias must be exactly
    # as wide as edge_attr, and a zero-init projection has nothing to push against
    # in the first pass. Fill both in unless the caller set them explicitly.
    if args.bias_dim is None:
        args.bias_dim = EDGE_ATTR_DIM if args.injection_mode == "edge" else 1
    if args.bias_init is None:
        args.bias_init = "xavier" if args.injection_mode == "edge" else "zeros"
    if args.frozen:
        train_feedback_frozen(args.dataset, gate_mode=args.gate_mode)
    else:
        train_feedback(
            args.dataset, args.modes, use_llm_head=args.use_llm_head,
            top_k_percent=args.top_k_percent,
            output_dir=args.output_dir,
            prototypes_path=args.prototypes_path,
            head_logits_path=args.head_logits_path,
            agaf_benchmark_path=args.agaf_benchmark_path,
            bias_confidence_fraction=args.bias_confidence_fraction,
            gate_mode=args.gate_mode,
            use_no_regret_floor=args.no_regret_floor,
            use_output_fusion=not args.no_output_fusion,
            bias_strength=args.bias_strength,
            bias_dim=args.bias_dim,
            max_iterations=args.max_iterations,
            bias_init=args.bias_init,
            oversample_ratio=args.oversample_ratio,
            seed=args.seed,
            injection_mode=args.injection_mode,
            injection_scale=args.injection_scale,
        )


if __name__ == "__main__":
    main()
