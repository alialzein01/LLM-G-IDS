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
    FeedbackLoopClassifier,
    WhitenedPrototypeScorer,
)
from src.models.gnn_classifier import VARIANT_NAMES
from src.pipeline.common.datasets import DATASETS, get_dataset_config
from src.pipeline.common.splits import NUM_CLASSES, FocalLoss, get_class_weights


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

TOP_K_PERCENT = 16.0
MAX_ITERATIONS = 3
CHURN_TOL = 0.01
BIAS_CONFIDENCE_FRAC = 0.5  # only bias the top-half most-confident flagged edges

BOOTSTRAP_ITERS = 2000


@dataclass
class FoldTrainingResult:
    """Artifacts and selection metrics from one independently trained fold."""

    logits: torch.Tensor
    iterations: list[dict]
    best_val_macro_f1: float
    test_macro_f1: float


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _macro_f1(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    preds = logits[mask].argmax(dim=1).cpu().numpy()
    targets = labels[mask].cpu().numpy()
    return float(
        f1_score(
            targets, preds, average="macro",
            labels=list(range(NUM_CLASSES)), zero_division=0,
        )
    )


def _build_model(
    top_k_percent: float = TOP_K_PERCENT,
) -> FeedbackLoopClassifier:
    return FeedbackLoopClassifier(
        in_dim=IN_DIM,
        hidden_dim=HIDDEN_DIM,
        edge_attr_dim=EDGE_ATTR_DIM,
        num_classes=NUM_CLASSES,
        heads=HEADS,
        dropout=DROPOUT,
        top_k_percent=top_k_percent,
        max_iterations=MAX_ITERATIONS,
        churn_tol=CHURN_TOL,
        bias_confidence_frac=BIAS_CONFIDENCE_FRAC,
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
) -> FoldTrainingResult:
    """Train one fold in one feedback mode; return full-graph logits from the
    best-val checkpoint plus the per-iteration trace on the test edges."""
    _set_seed(SEED + fold_idx)
    model = _build_model(top_k_percent=top_k_percent)
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
        (loss + AUX_LOSS_WEIGHT * aux_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_logits, _ = model(
                data.x, data.edge_index, data.edge_attr, emb, feedback_mode=mode,
                head_logits=head_logits,
            )
            val_f1 = _macro_f1(eval_logits, labels, val_mask)

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

    # per-iteration test-fold metrics
    iter_rows = []
    for t in trace:
        it_logits = t["logits"]
        iter_rows.append(
            {
                "iter": t["iter"],
                "churn": t["churn"],
                "mean_entropy": t["mean_entropy"],
                "test_macro_f1": _macro_f1(it_logits, labels, test_mask),
                "per_class_f1": _per_class_f1(it_logits, labels, test_mask),
            }
        )

    test_f1 = _macro_f1(eval_logits, labels, test_mask)
    print(
        f"    [{mode}] fold {fold_idx} DONE best_val={best_val_f1:.4f} "
        f"test={test_f1:.4f} iters={len(trace)}"
    )
    return FoldTrainingResult(
        logits=eval_logits.detach().cpu(),
        iterations=iter_rows,
        best_val_macro_f1=best_val_f1,
        test_macro_f1=test_f1,
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
    iters: int = BOOTSTRAP_ITERS,
    seed: int = SEED,
) -> dict:
    """Bootstrap CI of macro-F1(a) - macro-F1(b) over resampled edges."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    diffs = []
    labs = list(range(NUM_CLASSES))
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
    freeze_backbone=False, gen=None,
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
            val_f1 = _macro_f1(eval_logits, labels, val_mask)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    return best_state, best_val_f1


def _train_one_fold_frozen(data, emb, fold, fold_state, fold_idx):
    """Two-phase frozen training:
      Phase 1 — train the GAT backbone alone (feedback off) → the canonical
                GNN baseline.
      Phase 2 — FREEZE that backbone and train only the feedback modules on
                top (attention bias + LLM fusion), a pure residual correction.
    Returns (gnn_alone_logits, frozen_feedback_logits), both full-graph."""
    _set_seed(SEED + fold_idx)
    model = _build_model()
    model.load_fold_state(fold_state["mean"], fold_state["whitener"], fold_state["prototypes_whitened"])
    labels = data.edge_label
    criterion = FocalLoss(alpha=get_class_weights(labels, fold["train_mask"]), gamma=2.0)

    # Phase 1 — backbone (GNN alone)
    best_state, val1 = _train_loop(
        model, data, emb, fold, criterion, "head_only", MAX_EPOCHS, EARLY_STOPPING_PATIENCE
    )
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        gnn_logits, _ = model(data.x, data.edge_index, data.edge_attr, emb, feedback_mode="head_only")

    # Phase 2 — freeze backbone, train the correction only
    model.freeze_backbone()
    best_state2, val2 = _train_loop(
        model, data, emb, fold, criterion, "real", MAX_EPOCHS, EARLY_STOPPING_PATIENCE,
        freeze_backbone=True,
    )
    if best_state2 is not None:
        model.load_state_dict(best_state2)
    model.eval()
    with torch.no_grad():
        fb_logits, _ = model(data.x, data.edge_index, data.edge_attr, emb, feedback_mode="real")

    test_mask = fold["test_mask"]
    print(
        f"  [frozen] fold {fold_idx}: GNN-alone val={val1:.4f} test={_macro_f1(gnn_logits, labels, test_mask):.4f} | "
        f"frozen-loop val={val2:.4f} test={_macro_f1(fb_logits, labels, test_mask):.4f}"
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


def train_feedback_frozen(dataset: str) -> Path:
    """Run the frozen-backbone ladder: GNN alone · LLM alone · AGAF · frozen loop."""
    config = get_dataset_config(dataset)
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
        g, f = _train_one_fold_frozen(data, emb, fold, protos["folds"][fi], fi)
        gnn_oof[fold["test_mask"]] = g[fold["test_mask"]]
        fb_oof[fold["test_mask"]] = f[fold["test_mask"]]

    llm_oof = _llm_alone_oof(data, emb, folds, protos)

    def pooled(oof):
        return float(f1_score(labels.numpy(), oof.argmax(1).numpy(), average="macro",
                              labels=list(range(NUM_CLASSES)), zero_division=0))

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
        "vs_agaf_bootstrap": _bootstrap_ci(labels.numpy(), fb_oof.argmax(1).numpy(), gnn_oof.argmax(1).numpy()),
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
    use_llm_head: bool = False,
) -> dict:
    """Build the canonical feedback result payload from OOF predictions."""
    pooled_accuracy = {
        mode: float(
            (logits.argmax(dim=1) == labels).sum().item() / labels.numel()
        )
        for mode, logits in oof_by_mode.items()
    }
    return {
        "dataset": dataset,
        "pooled_cv_macro_f1": pooled_f1.get("real"),
        "pooled_cv_accuracy": pooled_accuracy.get("real"),
        "per_mode_pooled_macro_f1": pooled_f1,
        "per_mode_pooled_accuracy": pooled_accuracy,
        "top_k_percent": TOP_K_PERCENT,
        "bias_confidence_fraction": BIAS_CONFIDENCE_FRAC,
        "effective_feedback_percent": TOP_K_PERCENT * BIAS_CONFIDENCE_FRAC,
        "semantic_consultant": (
            "trained_llm_head" if use_llm_head else "whitened_prototype_scorer"
        ),
        "trained_llm_head": use_llm_head,
        "target_agaf": target_agaf,
    }


def train_feedback(
    dataset: str,
    modes: list[str] | None = None,
    use_llm_head: bool = False,
) -> Path:
    modes = modes or list(FEEDBACK_MODES)
    config = get_dataset_config(dataset)
    root = Path(f"data/{dataset}/processed/step4_feedback")
    root.mkdir(parents=True, exist_ok=True)

    data: Data = torch.load(config.graph_path, weights_only=False)
    global IN_DIM
    IN_DIM = data.x.shape[1]  # derive from graph (supports pruned node features)
    emb = torch.load(config.llm_embedding_path, weights_only=False).float()
    folds = torch.load(config.splits_path, weights_only=False)
    protos = torch.load(root / "prototypes.pt", weights_only=False)
    # The loop's semantic consultant is the whitened-prototype scorer, which
    # fits no parameters. `use_llm_head` swaps in per-fold trained MLP head
    # logits instead; it is off by default so the reported ladder measures the
    # prototype path end to end.
    head_logits_all = None
    if use_llm_head:
        head_path = root / "llm_head_logits.pt"
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

    oof_by_mode: dict[str, torch.Tensor] = {}
    pooled_f1: dict[str, float] = {}

    for mode in modes:
        print(f"\n===== MODE: {mode} =====")
        oof = torch.full((num_edges, NUM_CLASSES), float("nan"))
        trace_by_fold = []
        for fold_idx, fold in enumerate(folds):
            fold_head = head_logits_all[fold_idx] if head_logits_all is not None else None
            fold_result = _train_one_fold(
                data, emb, fold, protos["folds"][fold_idx], fold_idx, mode,
                head_logits=fold_head,
            )
            oof[fold["test_mask"]] = fold_result.logits[fold["test_mask"]]
            trace_by_fold.append(
                {"fold": fold_idx, "iterations": fold_result.iterations}
            )

        if torch.isnan(oof).any():
            raise RuntimeError(f"[{mode}] some edges uncovered by any test fold.")
        oof_by_mode[mode] = oof
        preds = oof.argmax(dim=1)
        pooled = float(
            f1_score(labels.numpy(), preds.numpy(), average="macro",
                     labels=list(range(NUM_CLASSES)), zero_division=0)
        )
        pooled_f1[mode] = pooled
        torch.save(oof, root / f"feedback_oof_{mode}.pt")
        with open(root / f"feedback_trace_{mode}.json", "w") as f:
            json.dump({"dataset": dataset, "mode": mode,
                       "pooled_cv_macro_f1": pooled, "folds": trace_by_fold}, f, indent=2)
        print(f"[{mode}] pooled CV macro-F1 = {pooled:.4f}")

    # --- summaries ---------------------------------------------------------
    labels_np = labels.numpy()
    agaf_benchmark_path = Path(config.fusion_output_dir) / "benchmark_summary.json"
    target_agaf = None
    if agaf_benchmark_path.exists():
        agaf_payload = json.loads(agaf_benchmark_path.read_text())
        agaf_macro_f1 = agaf_payload.get("pooled_cv_macro_f1")
        if agaf_macro_f1 is not None:
            target_agaf = float(agaf_macro_f1)
    benchmark = _build_benchmark_summary(
        dataset=dataset,
        labels=labels,
        oof_by_mode=oof_by_mode,
        pooled_f1=pooled_f1,
        target_agaf=target_agaf,
        use_llm_head=use_llm_head,
    )
    with open(root / "benchmark_summary.json", "w") as f:
        json.dump(benchmark, f, indent=2)

    ablation: dict = {"dataset": dataset, "per_mode_pooled_macro_f1": pooled_f1}
    if "real" in oof_by_mode and "random" in oof_by_mode:
        ablation["real_vs_random"] = _bootstrap_ci(
            labels_np,
            oof_by_mode["real"].argmax(1).numpy(),
            oof_by_mode["random"].argmax(1).numpy(),
        )
    if "real" in oof_by_mode and "head_only" in oof_by_mode:
        ablation["real_vs_head_only"] = _bootstrap_ci(
            labels_np,
            oof_by_mode["real"].argmax(1).numpy(),
            oof_by_mode["head_only"].argmax(1).numpy(),
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
    args = parser.parse_args()
    if args.frozen:
        train_feedback_frozen(args.dataset)
    else:
        train_feedback(args.dataset, args.modes, use_llm_head=args.use_llm_head)


if __name__ == "__main__":
    main()
