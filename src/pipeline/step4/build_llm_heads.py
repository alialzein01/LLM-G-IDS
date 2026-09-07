"""Build per-fold trained LLM classification heads (OOF-safe).

The feedback loop originally consulted a whitened-prototype scorer for its
semantic signal. That scorer computes class-mean prototypes, which collapse
under severe class imbalance (a 3-sample class has a meaningless prototype) —
on ToN-IoT it scores only 0.28 macro-F1 while a trained MLP on the *same* LLM
embeddings scores ~0.43-0.50. Feeding that weak signal into the loop starves
it of the LLM's real strength.

This builds the strong analogue: for each CV fold, train an
`UnimodalEdgeClassifier` on the TRAIN-edge LLM embeddings (class-weighted focal
loss, per-fold `StandardScaler`), then store that head's logits over the WHOLE
graph. Row `f` of the output is fold `f`'s head applied to every edge:
  - train/val edges -> in-fold logits (used to shape the loop during training);
  - test edges       -> out-of-fold logits (used at inference).
The loop indexes the current fold's row; OOF discipline is preserved because a
fold's head never trained on its own test edges.

Output: `data/{dataset}/processed/step4_feedback/llm_head_logits.pt`
        tensor [num_folds, E, num_classes]

Run:
    python -m src.pipeline.step4.build_llm_heads --dataset ton_iot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.models.fusion_classifier import UnimodalEdgeClassifier
from src.pipeline.common.datasets import DATASETS, get_dataset_config
from src.pipeline.common.splits import (
    NUM_CLASSES,
    FocalLoss,
    eval_macro_f1,
    get_class_weights,
    mask_dropped_logits,
)

PROJ_DIM = 128
HIDDEN_DIM = 128
DROPOUT = 0.2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
MAX_EPOCHS = 300
EARLY_STOPPING_PATIENCE = 40
GRAD_CLIP = 1.0
SEED = 42
FOCAL_GAMMA = 2.0
DEFAULT_CROSSFIT_K = 5


def _train_head(emb, labels, train_mask, val_mask, seed, eval_classes):
    """Train one head on `train_mask`, early-stopping on `val_mask`; return its
    logits over ALL edges plus the best validation macro-F1."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    scaler = StandardScaler().fit(emb[train_mask].numpy())
    emb_s = torch.tensor(scaler.transform(emb.numpy()), dtype=torch.float32)

    model = UnimodalEdgeClassifier(
        in_dim=emb.shape[1], proj_dim=PROJ_DIM, hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES, dropout=DROPOUT,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = FocalLoss(alpha=get_class_weights(labels, train_mask), gamma=FOCAL_GAMMA)

    best_f1, best_state, bad = -1.0, None, 0
    for _ in range(MAX_EPOCHS):
        model.train()
        opt.zero_grad()
        logits, _ = model(emb_s[train_mask])
        loss = criterion(logits, labels[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        model.eval()
        with torch.no_grad():
            vlogits, _ = model(emb_s[val_mask])
        f1 = eval_macro_f1(labels[val_mask], vlogits.argmax(1), eval_classes)
        if f1 > best_f1:
            best_f1, bad = f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= EARLY_STOPPING_PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        full_logits, _ = model(emb_s)  # logits over ALL edges
    return full_logits.detach(), float(best_f1)


def _train_fold(emb, labels, fold, fold_idx, eval_classes):
    """Today's behaviour: one head per outer fold, its logits over every edge."""
    full_logits, best_f1 = _train_head(
        emb, labels, fold["train_mask"], fold["val_mask"], SEED + fold_idx, eval_classes
    )
    test_f1 = eval_macro_f1(
        labels[fold["test_mask"]],
        mask_dropped_logits(full_logits, [])[fold["test_mask"]].argmax(1),
        eval_classes,
    )
    print(f"  fold {fold_idx}: val_f1={best_f1:.4f} test_f1={test_f1:.4f}")
    return full_logits


def _inner_splits(labels, train_mask, k, seed):
    """`k` stratified inner splits of the OUTER TRAIN edges.

    Classes too small to appear in every inner fold are pooled into one `rare`
    stratum: stratifying on a 3-edge class is impossible, and letting
    StratifiedKFold fail there would push the whole split to unstratified.
    """
    idx = train_mask.nonzero(as_tuple=True)[0].numpy()
    y = labels[train_mask].numpy()
    counts = np.bincount(y, minlength=NUM_CLASSES)
    strata = np.where(counts[y] >= k, y, -1)
    if np.count_nonzero(strata == -1) and np.count_nonzero(strata == -1) < k:
        # Even the pooled rare stratum is too small to stratify on.
        strata = np.where(strata == -1, strata[strata != -1][0], strata)
    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return [idx[held] for _, held in splitter.split(idx, strata)]


def _load_inputs(config):
    """Graph labels, embeddings and folds. One seam so a test can substitute
    labels without patching `torch.load` globally."""
    data = torch.load(config.graph_path, weights_only=False)
    emb = torch.load(config.llm_embedding_path, weights_only=False).float()
    folds = torch.load(config.splits_path, weights_only=False)
    return data.edge_label, emb, folds


def build_llm_heads_crossfit(
    dataset: str,
    k: int = DEFAULT_CROSSFIT_K,
    output_path: str | Path | None = None,
) -> Path:
    """Per-fold head logits whose TRAIN rows are out-of-fold too.

    `build_llm_heads` writes, for outer fold f, the fold-f head applied to every
    edge. On f's own train edges those logits are IN-FOLD: the head saw those
    labels, so it is far more accurate there than on the test edges the loop is
    scored on. A loop trained against that file learns to trust a consultant
    that does not exist at test time.

    Here each outer fold's TRAIN rows are filled by inner cross-fitting — an
    inner head trained on 4/5 of the outer train split scores the held-out 1/5 —
    so the advice the loop trains on is as reliable as the advice it is scored
    with. VAL and TEST rows are unchanged: they already came from a head that
    never saw them, so this file's out-of-fold predictions, and therefore its
    pooled OOF macro-F1, are identical to the in-fold file's.

    No outer test label reaches any head that scores it, before or after.
    """
    if k < 2:
        raise ValueError(f"crossfit K must be >= 2, got {k}")
    config = get_dataset_config(dataset)
    labels, emb, folds = _load_inputs(config)
    ec = config.eval_classes

    num_edges = labels.shape[0]
    per_fold = torch.zeros(len(folds), num_edges, NUM_CLASSES)
    print(f"Building CROSS-FITTED per-fold LLM heads for {dataset} "
          f"({num_edges} edges, {len(folds)} outer folds, K={k} inner)")
    for fi, fold in enumerate(folds):
        # Outer val/test rows: the full-outer-train head, exactly as today.
        full_logits, best_f1 = _train_head(
            emb, labels, fold["train_mask"], fold["val_mask"], SEED + fi, ec
        )
        per_fold[fi] = full_logits

        # Outer train rows: replaced by inner out-of-fold logits.
        held_out = _inner_splits(labels, fold["train_mask"], k, SEED + fi)
        for ki, held in enumerate(held_out):
            inner_train = fold["train_mask"].clone()
            inner_train[held] = False
            inner_logits, _ = _train_head(
                emb, labels, inner_train, fold["val_mask"], SEED + fi * k + ki, ec
            )
            per_fold[fi, held] = inner_logits[held]

        tm = fold["train_mask"]
        infold_acc = float(
            (mask_dropped_logits(full_logits, config.dropped_classes)[tm].argmax(1)
             == labels[tm]).float().mean()
        )
        crossfit_acc = float(
            (mask_dropped_logits(per_fold[fi], config.dropped_classes)[tm].argmax(1)
             == labels[tm]).float().mean()
        )
        te = fold["test_mask"]
        test_acc = float(
            (mask_dropped_logits(full_logits, config.dropped_classes)[te].argmax(1)
             == labels[te]).float().mean()
        )
        print(f"  fold {fi}: val_f1={best_f1:.4f} | train acc in-fold={infold_acc:.4f} "
              f"cross-fit={crossfit_acc:.4f} | test acc={test_acc:.4f}")

    oof = torch.zeros(num_edges, NUM_CLASSES)
    for fi, fold in enumerate(folds):
        oof[fold["test_mask"]] = per_fold[fi][fold["test_mask"]]
    oof_f1 = eval_macro_f1(
        labels, mask_dropped_logits(oof, config.dropped_classes).argmax(1), ec
    )
    print(f"\nPooled OOF macro-F1 ({len(ec)} classes): {oof_f1:.4f} "
          "(must equal the in-fold file's — test rows are untouched)")

    out = Path(output_path) if output_path else Path(
        f"data/{dataset}/processed/step4_feedback/llm_head_logits_crossfit.pt"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(per_fold, out)
    print(f"Saved -> {out}  [{len(folds)}, {num_edges}, {NUM_CLASSES}]")
    return out


def build_llm_heads(dataset: str) -> Path:
    config = get_dataset_config(dataset)
    data = torch.load(config.graph_path, weights_only=False)
    emb = torch.load(config.llm_embedding_path, weights_only=False).float()
    folds = torch.load(config.splits_path, weights_only=False)
    labels = data.edge_label
    ec = config.eval_classes

    num_edges = labels.shape[0]
    per_fold = torch.zeros(len(folds), num_edges, NUM_CLASSES)
    oof = torch.zeros(num_edges, NUM_CLASSES)
    print(f"Building per-fold LLM heads for {dataset} ({num_edges} edges, {len(folds)} folds)")
    for fi, fold in enumerate(folds):
        full_logits = _train_fold(emb, labels, fold, fi, ec)
        per_fold[fi] = full_logits
        oof[fold["test_mask"]] = full_logits[fold["test_mask"]]

    oof_pred = mask_dropped_logits(oof, config.dropped_classes).argmax(1)
    oof_f1 = eval_macro_f1(labels, oof_pred, ec)
    print(f"\nPooled OOF macro-F1 ({len(ec)} classes): {oof_f1:.4f}")

    out = Path(f"data/{dataset}/processed/step4_feedback/llm_head_logits.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(per_fold, out)
    print(f"Saved per-fold LLM head logits -> {out}  [{len(folds)}, {num_edges}, {NUM_CLASSES}]")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ton_iot", choices=sorted(DATASETS))
    parser.add_argument(
        "--crossfit", type=int, nargs="?", const=DEFAULT_CROSSFIT_K, default=None,
        help="Cross-fit the TRAIN rows with K inner folds (default K=5) and write "
             "llm_head_logits_crossfit.pt. Off by default: the canonical file keeps "
             "its in-fold train rows.",
    )
    args = parser.parse_args()
    if args.crossfit:
        build_llm_heads_crossfit(args.dataset, args.crossfit)
    else:
        build_llm_heads(args.dataset)


if __name__ == "__main__":
    main()
