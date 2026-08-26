"""Build OUT-OF-FOLD GNN edge embeddings for a leakage-free AGAF fusion.

`train_gnn.py` saves `step3_gnn/edge_embeddings.pt` from a model trained on
ALL edges (it self-labels this "fit-on-all-edges results (not benchmark
performance)"). Feeding those embeddings to AGAF leaks: the GNN features on a
CV test edge were produced by a network that had already trained on that
edge's label — so AGAF's cross-validation is not truly out-of-fold and its
score is inflated (0.798 → an honest 0.7465 on UNSW-NB15).

This script produces the honest analogue: for each fold, train a fresh
`GATEdgeClassifier` on the TRAIN edges only, take that fold's TEST-edge
embeddings, and stitch them out-of-fold. AGAF (and any other consumer that
benchmarks on these embeddings) should read this file, not the fit-on-all one.

Output: `data/{dataset}/processed/step3_gnn/edge_embeddings_oof.pt`  [E, 64]

Hyperparameters mirror `src.pipeline.step3.train_gnn` /
`src.pipeline.step4.build_oof_predictions` so the OOF backbone matches the
standalone GNN baseline.

Run:
    python -m src.pipeline.step3.build_oof_gnn_embeddings --dataset unsw_nb15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.models.gnn_classifier import GATEdgeClassifier, VARIANT_NAMES
from src.pipeline.common.datasets import DATASETS, get_dataset_config
from src.pipeline.common.splits import NUM_CLASSES, FocalLoss, get_class_weights

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
SEED = 42


def _macro_f1(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    from sklearn.metrics import f1_score

    return float(
        f1_score(
            labels[mask].numpy(), logits[mask].argmax(1).numpy(),
            average="macro", labels=list(range(NUM_CLASSES)), zero_division=0,
        )
    )


def _train_fold_capture_embeddings(data: Data, fold: dict, fold_idx: int) -> torch.Tensor:
    torch.manual_seed(SEED + fold_idx)
    np.random.seed(SEED + fold_idx)
    model = GATEdgeClassifier(
        IN_DIM, HIDDEN_DIM, EDGE_ATTR_DIM, NUM_CLASSES, HEADS, DROPOUT,
        **({} if getattr(data, "num_protocols", None) is None
           else {"num_protocols": data.num_protocols, "num_ports": data.num_ports}),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_mask, val_mask, labels = fold["train_mask"], fold["val_mask"], data.edge_label
    criterion = FocalLoss(alpha=get_class_weights(labels, train_mask), gamma=2.0)

    best_f1, best_state, bad = -1.0, None, 0
    for _ in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        logits, _, aux = model.forward_with_aux(data.x, data.edge_index, data.edge_attr)
        loss = criterion(logits[train_mask], labels[train_mask]) + AUX_LOSS_WEIGHT * torch.stack(
            [criterion(aux[n][train_mask], labels[train_mask]) for n in VARIANT_NAMES]
        ).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            eval_logits, _ = model(data.x, data.edge_index, data.edge_attr)
            f1 = _macro_f1(eval_logits, labels, val_mask)
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
        emb = model.encode_edges(data.x, data.edge_index, data.edge_attr)
        eval_logits, _ = model(data.x, data.edge_index, data.edge_attr)
    print(
        f"  fold {fold_idx}: val_f1={best_f1:.4f} "
        f"test_f1={_macro_f1(eval_logits, labels, fold['test_mask']):.4f}"
    )
    return emb.detach().cpu()


def build_oof_gnn_embeddings(dataset: str) -> Path:
    config = get_dataset_config(dataset)
    data: Data = torch.load(config.graph_path, weights_only=False)
    global IN_DIM
    IN_DIM = data.x.shape[1]  # derive from graph (supports pruned node features)
    folds = torch.load(config.splits_path, weights_only=False)
    num_edges = data.edge_label.shape[0]

    oof = torch.full((num_edges, HIDDEN_DIM), float("nan"))
    print(f"Building OOF GNN embeddings for {dataset} ({num_edges} edges, {len(folds)} folds)")
    for fold_idx, fold in enumerate(folds):
        emb = _train_fold_capture_embeddings(data, fold, fold_idx)
        oof[fold["test_mask"]] = emb[fold["test_mask"]]

    if torch.isnan(oof).any():
        raise RuntimeError("Some edges were not covered by any fold's test set.")

    out = Path(config.gnn_embedding_path).with_name("edge_embeddings_oof.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(oof, out)
    print(f"\nSaved OOF GNN embeddings -> {out}")
    print("Point AGAF at this file (gnn_emb_path) for a leakage-free benchmark.")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="unsw_nb15", choices=sorted(DATASETS))
    args = parser.parse_args()
    build_oof_gnn_embeddings(args.dataset)


if __name__ == "__main__":
    main()
