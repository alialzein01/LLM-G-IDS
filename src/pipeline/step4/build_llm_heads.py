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
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.models.fusion_classifier import UnimodalEdgeClassifier
from src.pipeline.common.datasets import get_dataset_config
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


def _train_fold(emb, labels, fold, fold_idx, eval_classes):
    torch.manual_seed(SEED + fold_idx)
    np.random.seed(SEED + fold_idx)
    train_mask, val_mask = fold["train_mask"], fold["val_mask"]

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
    test_f1 = eval_macro_f1(
        labels[fold["test_mask"]],
        mask_dropped_logits(full_logits, [])[fold["test_mask"]].argmax(1),
        eval_classes,
    )
    print(f"  fold {fold_idx}: val_f1={best_f1:.4f} test_f1={test_f1:.4f}")
    return full_logits.detach()


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
    parser.add_argument("--dataset", default="ton_iot", choices=["ton_iot", "unsw_nb15"])
    args = parser.parse_args()
    build_llm_heads(args.dataset)


if __name__ == "__main__":
    main()
