#!/usr/bin/env python3
"""Consultant complementarity on the flagged set, closing the report's second [GAP].

The report tabulates how accurate each consultant is on the edges the loop actually
consults, and gives, for NF-ToN-IoT with the trained head only, the split of the
disagreements between consultant and graph encoder. The equivalent split for
NF-UNSW-NB15, and for the prototype consultant on either dataset, was marked as a
gap.

Two problems with the figures that were there, both found on 2026-09-13 while
closing the gap:

1. The flagged and gated counts quoted as the realisation of the canonical
   consultation rates (204 of 656 and 532 of 2,127) are the counts at the
   entropy percentiles in force BEFORE the knobs were re-selected, k = 31 and
   k = 25. At the canonical k = 29 and k = 16 the counts are 190 of 656 and
   341 of 2,127.
2. The flagged-set accuracies and the disagreement split could not be reproduced
   from any committed artifact. They were computed inside a training run, from
   the model's live per-fold probabilities, and that intermediate was never saved.

This script recomputes all of it from committed artifacts, by one definition
applied identically to both datasets and both consultants, so every cell is
reproducible:

  flagged   the top k% of edges by Shannon entropy of the graph encoder's pooled
            out-of-fold softmax, at the canonical k for that dataset;
  gated     the top half of the flagged set by the consultant's maximum softmax
            probability, which is the canonical `confidence` gate mode;
  accuracy  of the graph encoder, the consultant and the resulting loop, over the
            flagged set;
  split     among flagged edges where consultant and graph encoder disagree, how
            often the consultant is right, over the flagged set and again inside
            the gated half.

Because it works from the pooled out-of-fold tensors rather than from live
training state, it measures the same quantity at the same knobs but not through
the same code path, so its numbers replace the earlier ones rather than
reconciling with them.

Writes `results/consultant_complementarity.json`.

Run:
    OMP_NUM_THREADS=1 python scripts/close_complementarity_split.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.common.datasets import get_dataset_config
from src.pipeline.common.splits import mask_dropped_logits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "consultant_complementarity.json"
SEED = 42
GATE_FRACTION = 0.5          # canonical bias_confidence_fraction
CANONICAL_K = {"unsw_nb15": 29.0, "ton_iot": 16.0}


def entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * (probs + 1e-12).log()).sum(-1)


def trained_head_logits(dataset: str, config) -> torch.Tensor:
    """Pooled out-of-fold logits of the per-fold trained head, assembled as the
    head-alone baseline assembles them."""
    stacked = torch.load(
        ROOT / f"data/{dataset}/processed/step4_feedback/llm_head_logits.pt",
        weights_only=False,
    ).float()
    folds = torch.load(config.splits_path, weights_only=False)
    pooled = torch.full((stacked.shape[1], stacked.shape[2]), float("nan"))
    for fi, fold in enumerate(folds):
        pooled[fold["test_mask"]] = stacked[fi][fold["test_mask"]]
    return torch.nan_to_num(pooled, nan=-1e9)


def prototype_logits(dataset: str, config) -> torch.Tensor:
    """Pooled out-of-fold logits of the whitened prototype scorer, the semantic rung."""
    from src.pipeline.step4.train_feedback import _llm_alone_oof

    data = torch.load(config.graph_path, weights_only=False)
    emb = torch.load(config.llm_embedding_path, weights_only=False).float()
    folds = torch.load(config.splits_path, weights_only=False)
    protos = torch.load(
        ROOT / f"data/{dataset}/processed/step4_feedback/prototypes.pt",
        weights_only=False,
    )
    return _llm_alone_oof(data, emb, folds, protos)


def analyse(dataset: str) -> dict:
    config = get_dataset_config(dataset)
    labels = torch.load(config.graph_path, weights_only=False).edge_label
    k = CANONICAL_K[dataset]

    gnn = mask_dropped_logits(
        torch.load(
            ROOT / f"results/multiseed_v2/head/{dataset}_seed{SEED}/oof_logits.pt",
            weights_only=False,
        ),
        config.dropped_classes,
    )
    probs = gnn.softmax(-1)
    gnn_argmax = probs.argmax(-1)
    h = entropy(probs)
    flagged = (h > torch.quantile(h, 1.0 - k / 100.0)).nonzero(as_tuple=True)[0]

    consultants = {
        "trained_head": (
            trained_head_logits(dataset, config),
            f"results/multiseed_v2/head/{dataset}_seed{SEED}/feedback_oof_real.pt",
        ),
        "prototype": (
            prototype_logits(dataset, config),
            f"results/multiseed_v2/legacy/{dataset}_seed{SEED}/feedback_oof_real.pt",
        ),
    }

    out = {
        "dataset_key": dataset,
        "n_edges": int(labels.shape[0]),
        "top_k_percent": k,
        "gate_fraction": GATE_FRACTION,
        "n_flagged": int(flagged.numel()),
        "flagged_share": float(flagged.numel() / labels.shape[0]),
        "graph_encoder_accuracy_on_flagged": float(
            (gnn_argmax[flagged] == labels[flagged]).float().mean()
        ),
        "consultants": {},
    }

    for name, (raw, loop_path) in consultants.items():
        cons = mask_dropped_logits(raw, config.dropped_classes)
        cons_argmax = cons.argmax(-1)
        loop = mask_dropped_logits(
            torch.load(ROOT / loop_path, weights_only=False), config.dropped_classes
        ).argmax(-1)

        conf = cons[flagged].softmax(-1).max(-1).values
        n_gate = max(1, round(GATE_FRACTION * flagged.numel()))
        gated = flagged[torch.topk(conf, n_gate).indices]

        def split(idx):
            dis = idx[gnn_argmax[idx] != cons_argmax[idx]]
            right = int((cons_argmax[dis] == labels[dis]).sum())
            return {
                "n_disagreements": int(dis.numel()),
                "consultant_right": right,
                "consultant_wrong": int(dis.numel()) - right,
            }

        out["consultants"][name] = {
            "n_gated": int(gated.numel()),
            "gated_share": float(gated.numel() / labels.shape[0]),
            "accuracy_on_flagged": {
                "graph_encoder": out["graph_encoder_accuracy_on_flagged"],
                "consultant": float((cons_argmax[flagged] == labels[flagged]).float().mean()),
                "loop": float((loop[flagged] == labels[flagged]).float().mean()),
            },
            "disagreement_split_flagged": split(flagged),
            "disagreement_split_gated": split(gated),
        }
    return out


def main() -> None:
    out = {
        "schema_version": 1,
        "seed": SEED,
        "definition": (
            "Flagged is the top k% of edges by Shannon entropy of the graph encoder's "
            "pooled out-of-fold softmax at the canonical k. Gated is the top half of "
            "that set by the consultant's maximum softmax probability. All quantities "
            "are computed from committed pooled out-of-fold tensors, not from live "
            "training state."
        ),
        "supersedes": (
            "The flagged-set accuracies and the NF-ToN-IoT disagreement split previously "
            "quoted (0.603 / 0.770 / 0.750, 0.359 / 0.759 / 0.724, and 245 right against "
            "32 wrong with 145 against 4 inside the gate) came from trained-head runs at "
            "the pre-reselection knobs k = 31 and k = 25, computed inside training from "
            "per-fold live probabilities; that intermediate was never saved and the "
            "figures are not reproducible from any committed artifact."
        ),
        "closes": (
            "report [GAP] in sections/04_results.tex and sections/07_appendices.tex, the "
            "consultant-versus-graph-encoder disagreement split"
        ),
    }
    for dataset in ("unsw_nb15", "ton_iot"):
        r = analyse(dataset)
        out[dataset] = r
        print(f"{dataset}  k={r['top_k_percent']}  flagged={r['n_flagged']}/{r['n_edges']}"
              f"  graph encoder acc={r['graph_encoder_accuracy_on_flagged']:.3f}")
        for name, c in r["consultants"].items():
            a = c["accuracy_on_flagged"]
            f, g = c["disagreement_split_flagged"], c["disagreement_split_gated"]
            print(f"   {name:13s} gated={c['n_gated']:4d}  acc consultant={a['consultant']:.3f}"
                  f" loop={a['loop']:.3f}  disagree {f['consultant_right']}/{f['consultant_wrong']}"
                  f"  gated {g['consultant_right']}/{g['consultant_wrong']}")
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
