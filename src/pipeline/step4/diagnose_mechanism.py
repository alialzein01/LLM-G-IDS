"""Step 4 — mechanism diagnostics for a finished feedback-loop run.

Answers "what did the feedback mechanism actually *do*?" from the artifacts a
`train_feedback` run already writes. It changes no model and trains nothing: it
re-derives the consultation set from the run's own `head_only` OOF probabilities
and then scores every rung on it.

The questions it answers, in order:

  1. **Consultation rate** — how many edges the entropy selector flags at the
     run's `top_k_percent`, and how many survive the semantic-confidence gate.
  2. **Who is right where** — accuracy and per-class F1 for the GNN
     (`head_only`), the prototype consultant, the trained LLM head (when
     `llm_head_logits.pt` exists) and the loop (`real`), each on three subsets:
     all test edges, flagged edges, confidence-gated edges. A consultant that is
     no better than the GNN *on the edges it is consulted about* cannot help,
     however good the channel is.
  3. **Did anything move** — prediction churn loop-vs-`head_only` on
     flagged / unflagged / all edges, the wrong→correct and correct→wrong counts
     on flagged edges, and the mean |Δlogit| / |Δprob| on flagged vs unflagged.
     Churn concentrated off the flagged set is not the mechanism working.
  4. **The trace** — per-fold churn per iteration, iterations actually run, and
     `flagged_bias_absmean`, the size of the injected advice.

Selector caveat: the model runs the selector per fold on that fold's own
predictions. Here it is re-run once on the pooled OOF probabilities, which is
the same rule applied to a leakage-free stitch of the same predictions. The
flagged *count* matches the model's (`top_k_percent` of all edges); the
membership can differ on a few edges near the entropy threshold.

Run:
    python -m src.pipeline.step4.diagnose_mechanism \\
        --dataset unsw_nb15 --run-dir results/dev/task0/unsw_nb15
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.models.feedback_classifier import (
    UncertaintySelector,
    WhitenedPrototypeScorer,
)
from src.pipeline.common.datasets import DATASETS, get_dataset_config
from src.pipeline.common.splits import NUM_CLASSES, mask_dropped_logits
from src.pipeline.step4.feedback_config import (
    DEFAULT_BIAS_CONFIDENCE_FRAC,
    load_feedback_config,
)

LEGACY_TEMPERATURE = 10.0


def _finite_rows(logits: torch.Tensor) -> torch.Tensor:
    """Rows of a pooled OOF tensor that actually carry a prediction."""
    return torch.isfinite(logits).all(dim=1)


def _preds(logits: torch.Tensor, dropped: tuple[int, ...] | list[int]) -> torch.Tensor:
    return mask_dropped_logits(logits, dropped).argmax(dim=1)


def _subset_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    subset: torch.Tensor,
    eval_classes: tuple[int, ...] | list[int] | None,
) -> dict:
    """Accuracy + macro-F1 + per-class F1 restricted to `subset` (a bool mask)."""
    n = int(subset.sum())
    if n == 0:
        return {"n": 0, "accuracy": None, "macro_f1": None, "per_class_f1": {}}
    y = labels[subset].numpy()
    p = preds[subset].numpy()
    classes = list(eval_classes) if eval_classes else list(range(NUM_CLASSES))
    per_class = f1_score(y, p, labels=classes, average=None, zero_division=0)
    # Macro over the classes actually present in this subset's labels — a class
    # with no support here would otherwise drag the mean toward zero and make
    # small subsets incomparable to the pooled number.
    present = [c for c in classes if (y == c).any()]
    macro = float(
        f1_score(y, p, labels=present, average="macro", zero_division=0)
    ) if present else None
    return {
        "n": n,
        "accuracy": float((p == y).mean()),
        "macro_f1_classes_present": macro,
        "n_classes_present": len(present),
        "per_class_f1": {int(c): float(v) for c, v in zip(classes, per_class)},
    }


def _consultant_logits(
    dataset_cfg,
    folds,
    protos: dict,
    emb: torch.Tensor,
    num_edges: int,
    legacy_temperature: bool,
) -> torch.Tensor:
    """Pooled OOF prototype-consultant logits: fold f's scorer on fold f's test
    edges. Mirrors how the loop's consultant is fitted — prototypes from the
    train fold only — so this is leakage-free the same way the rungs are."""
    out = torch.full((num_edges, NUM_CLASSES), float("nan"))
    for f, fold in enumerate(folds):
        state = protos["folds"][f]
        scorer = WhitenedPrototypeScorer(
            num_classes=NUM_CLASSES, embed_dim=emb.shape[1]
        )
        scorer.load_fold_state(
            state["mean"], state["whitener"], state["prototypes_whitened"]
        )
        with torch.no_grad():
            if legacy_temperature:
                scorer.log_temperature.fill_(math.log(LEGACY_TEMPERATURE))
            else:
                scorer.calibrate_temperature(emb[fold["train_mask"]])
            test = fold["test_mask"]
            out[test] = scorer(emb[test]).float()
    return out


def _trace_summary(trace: dict) -> dict:
    folds = []
    for item in trace.get("folds", []):
        diag = item.get("bias_diagnostics") or {}
        folds.append(
            {
                "fold": item.get("fold"),
                "churn_per_iteration": [
                    it.get("churn") for it in item.get("iterations", [])
                ],
                "test_macro_f1_per_iteration": [
                    it.get("test_macro_f1") for it in item.get("iterations", [])
                ],
                "iterations_run": diag.get("iterations_run"),
                "flagged_bias_absmean": diag.get("flagged_bias_absmean"),
                "flagged_bias_absmax": diag.get("flagged_bias_absmax"),
                "n_flagged_edges": diag.get("n_flagged_edges"),
                "consultant_temperature": diag.get("consultant_temperature"),
                "final_churn": diag.get("final_churn"),
            }
        )
    runs = [f["iterations_run"] for f in folds if f["iterations_run"] is not None]
    biases = [
        f["flagged_bias_absmean"] for f in folds
        if f["flagged_bias_absmean"] is not None
    ]
    return {
        "per_fold": folds,
        "mean_iterations_run": float(np.mean(runs)) if runs else None,
        "mean_flagged_bias_absmean": float(np.mean(biases)) if biases else None,
    }


def diagnose(dataset: str, run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    cfg = get_dataset_config(dataset)
    canonical = Path(f"data/{dataset}/processed/step4_feedback")

    real_path = run_dir / "feedback_oof_real.pt"
    head_only_path = run_dir / "feedback_oof_head_only.pt"
    trace_path = run_dir / "feedback_trace_real.json"
    for p in (real_path, head_only_path, trace_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} is missing — is {run_dir} a finished run?")

    # Run configuration, from the run's own benchmark if it wrote one. Reading
    # the knobs off the run instead of off the current config file is what keeps
    # this honest when the config has since moved.
    bench_path = run_dir / "benchmark_summary.json"
    bench = json.loads(bench_path.read_text()) if bench_path.exists() else {}
    selected = load_feedback_config(
        dataset, root=run_dir if (run_dir / "selected_feedback_config.json").exists()
        else None,
    )
    top_k = float(bench.get("top_k_percent", selected["top_k_percent"]))
    conf_frac = float(
        bench.get("bias_confidence_fraction",
                  selected.get("bias_confidence_fraction",
                              DEFAULT_BIAS_CONFIDENCE_FRAC))
    )
    legacy_temperature = bool(bench.get("legacy_temperature", True))

    data = torch.load(cfg.graph_path, weights_only=False)
    emb = torch.load(cfg.llm_embedding_path, weights_only=False).float()
    folds = torch.load(cfg.splits_path, weights_only=False)
    protos_path = (run_dir / "prototypes.pt")
    if not protos_path.exists():
        protos_path = canonical / "prototypes.pt"
    protos = torch.load(protos_path, weights_only=False)

    labels = data.edge_label
    num_edges = labels.shape[0]
    dropped = tuple(cfg.dropped_classes)
    eval_classes = cfg.eval_classes

    real = torch.load(real_path, weights_only=False).float()
    head_only = torch.load(head_only_path, weights_only=False).float()

    # --- 1. consultation rate -------------------------------------------------
    test_rows = _finite_rows(head_only)
    # The model ranks entropy on the RAW softmax over all NUM_CLASSES, and gates
    # on the raw consultant softmax — dropped classes are masked at prediction
    # time only. Masking here instead would flag a different edge set on ToN
    # (which has two dropped classes) and silently stop describing the run.
    gnn_probs = torch.nan_to_num(head_only, nan=0.0).softmax(dim=-1)
    selector = UncertaintySelector(top_k_percent=top_k)
    flagged_mask = selector(gnn_probs)

    consultant = _consultant_logits(
        cfg, folds, protos, emb, num_edges, legacy_temperature
    )
    flagged_idx = flagged_mask.nonzero(as_tuple=True)[0]
    llm_probs = consultant[flagged_idx].softmax(dim=-1)
    confidence = llm_probs.max(dim=-1).values
    k_gate = max(1, int(round(conf_frac * flagged_idx.numel())))
    gated_idx = flagged_idx[torch.topk(confidence, k_gate).indices]
    gated_mask = torch.zeros_like(flagged_mask)
    gated_mask[gated_idx] = True

    consultation = {
        "top_k_percent": top_k,
        "bias_confidence_fraction": conf_frac,
        "n_edges": num_edges,
        "n_test_edges": int(test_rows.sum()),
        "n_flagged": int(flagged_mask.sum()),
        "flagged_over_all_edges": float(flagged_mask.sum() / num_edges),
        "flagged_over_test_edges": float(
            (flagged_mask & test_rows).sum() / max(1, int(test_rows.sum()))
        ),
        "n_confidence_gated": int(gated_mask.sum()),
        "gated_over_all_edges": float(gated_mask.sum() / num_edges),
        "gated_over_test_edges": float(
            (gated_mask & test_rows).sum() / max(1, int(test_rows.sum()))
        ),
        "consultant_temperature_mode": (
            "legacy_T10" if legacy_temperature else "per_fold_calibrated"
        ),
    }

    # --- 2. who is right where ------------------------------------------------
    head_logits_path = canonical / "llm_head_logits.pt"
    trained_head = None
    if head_logits_path.exists():
        stacked = torch.load(head_logits_path, weights_only=False).float()
        trained_head = torch.full((num_edges, NUM_CLASSES), float("nan"))
        for f, fold in enumerate(folds):
            trained_head[fold["test_mask"]] = stacked[f][fold["test_mask"]]

    rungs = {
        "gnn_head_only": head_only,
        "prototype_consultant": consultant,
        "loop_real": real,
    }
    if trained_head is not None:
        rungs["trained_llm_head"] = trained_head

    subsets = {
        "all_test_edges": test_rows,
        "flagged": flagged_mask & test_rows,
        "confidence_gated": gated_mask & test_rows,
        "unflagged": (~flagged_mask) & test_rows,
    }
    by_rung = {}
    for name, logits in rungs.items():
        preds = _preds(torch.nan_to_num(logits, nan=-1e9), dropped)
        by_rung[name] = {
            sname: _subset_metrics(preds, labels, smask, eval_classes)
            for sname, smask in subsets.items()
        }

    # --- 3. did anything move -------------------------------------------------
    gnn_pred = _preds(torch.nan_to_num(head_only, nan=-1e9), dropped)
    loop_pred = _preds(torch.nan_to_num(real, nan=-1e9), dropped)
    changed = (gnn_pred != loop_pred) & test_rows
    gnn_correct = (gnn_pred == labels) & test_rows
    loop_correct = (loop_pred == labels) & test_rows

    def _churn(mask: torch.Tensor) -> float:
        n = int(mask.sum())
        return float((changed & mask).sum() / n) if n else float("nan")

    churn = {
        "all_test_edges": _churn(test_rows),
        "flagged": _churn(subsets["flagged"]),
        "unflagged": _churn(subsets["unflagged"]),
        "confidence_gated": _churn(subsets["confidence_gated"]),
        "n_changed_total": int(changed.sum()),
        "n_changed_on_flagged": int((changed & subsets["flagged"]).sum()),
        "share_of_changes_on_flagged": (
            float((changed & subsets["flagged"]).sum() / changed.sum())
            if int(changed.sum()) else None
        ),
    }

    flips = {}
    for sname in ("flagged", "confidence_gated", "unflagged", "all_test_edges"):
        m = subsets[sname]
        flips[sname] = {
            "wrong_to_correct": int(((~gnn_correct) & loop_correct & m).sum()),
            "correct_to_wrong": int((gnn_correct & (~loop_correct) & m).sum()),
            "net": int(((~gnn_correct) & loop_correct & m).sum())
            - int((gnn_correct & (~loop_correct) & m).sum()),
        }

    d_logit = (real - head_only).abs()
    d_prob = (real.softmax(dim=-1) - head_only.softmax(dim=-1)).abs()
    magnitude = {}
    for sname in ("flagged", "unflagged", "confidence_gated", "all_test_edges"):
        m = subsets[sname]
        magnitude[sname] = {
            "mean_abs_delta_logit": float(d_logit[m].mean()) if int(m.sum()) else None,
            "mean_abs_delta_prob": float(d_prob[m].mean()) if int(m.sum()) else None,
        }

    # --- 4. the trace ---------------------------------------------------------
    trace = json.loads(trace_path.read_text())

    out = {
        "dataset": dataset,
        "run_dir": str(run_dir),
        "artifacts": {
            "real": str(real_path),
            "head_only": str(head_only_path),
            "trace": str(trace_path),
            "prototypes": str(protos_path),
            "trained_head": str(head_logits_path) if trained_head is not None else None,
        },
        "run_config": {
            "top_k_percent": top_k,
            "bias_confidence_fraction": conf_frac,
            "injection_scale": bench.get("injection_scale"),
            "selector_head_loss_weight": bench.get("selector_head_loss_weight"),
            "legacy_temperature": legacy_temperature,
            "semantic_consultant": bench.get("semantic_consultant"),
            "trained_llm_head": bench.get("trained_llm_head"),
        },
        "pooled_macro_f1": bench.get("per_mode_pooled_macro_f1"),
        "consultation": consultation,
        "subset_performance": by_rung,
        "churn_loop_vs_head_only": churn,
        "flips_loop_vs_head_only": flips,
        "delta_magnitude_loop_vs_head_only": magnitude,
        "trace": _trace_summary(trace),
    }

    out_path = run_dir / "mechanism_diagnostics.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    _print(out)
    print(f"\nWrote {out_path}")
    return out


def _fmt(v, nd=4):
    return "  n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def _print(d: dict) -> None:
    c = d["consultation"]
    print(f"\n=== mechanism diagnostics — {d['dataset']} @ {d['run_dir']} ===")
    rc = d["run_config"]
    print(f"top_k={rc['top_k_percent']} scale={rc['injection_scale']} "
          f"conf_frac={rc['bias_confidence_fraction']} "
          f"selector_head_loss_weight={rc['selector_head_loss_weight']} "
          f"legacy_temperature={rc['legacy_temperature']}")
    if d["pooled_macro_f1"]:
        print("pooled macro-F1: " + "  ".join(
            f"{k}={v:.4f}" for k, v in d["pooled_macro_f1"].items()))

    print(f"\n-- consultation rate --")
    print(f"flagged           {c['n_flagged']}/{c['n_edges']} edges "
          f"({c['flagged_over_all_edges']:.3f})   "
          f"{c['flagged_over_test_edges']:.3f} of {c['n_test_edges']} test edges")
    print(f"confidence-gated  {c['n_confidence_gated']}/{c['n_edges']} edges "
          f"({c['gated_over_all_edges']:.3f})   consultant T: "
          f"{c['consultant_temperature_mode']}")

    print(f"\n-- accuracy by subset --")
    subs = ("all_test_edges", "flagged", "confidence_gated", "unflagged")
    print(f"{'rung':<22}" + "".join(f"{s:>19}" for s in subs))
    for rung, per in d["subset_performance"].items():
        row = "".join(
            f"{_fmt(per[s]['accuracy']):>19}" for s in subs
        )
        print(f"{rung:<22}{row}")
    print(f"{'(n)':<22}" + "".join(
        f"{d['subset_performance']['gnn_head_only'][s]['n']:>19}" for s in subs))

    print(f"\n-- macro-F1 (classes present in subset) --")
    print(f"{'rung':<22}" + "".join(f"{s:>19}" for s in subs))
    for rung, per in d["subset_performance"].items():
        print(f"{rung:<22}" + "".join(
            f"{_fmt(per[s]['macro_f1_classes_present']):>19}" for s in subs))

    ch = d["churn_loop_vs_head_only"]
    print(f"\n-- churn loop vs head_only --")
    print(f"all {_fmt(ch['all_test_edges'])}  flagged {_fmt(ch['flagged'])}  "
          f"unflagged {_fmt(ch['unflagged'])}  gated {_fmt(ch['confidence_gated'])}")
    print(f"{ch['n_changed_total']} predictions changed; "
          f"{ch['n_changed_on_flagged']} of them on flagged edges "
          f"({_fmt(ch['share_of_changes_on_flagged'], 3)} share)")

    print(f"\n-- flips loop vs head_only --")
    for sname, f in d["flips_loop_vs_head_only"].items():
        print(f"{sname:<20} wrong→correct {f['wrong_to_correct']:>4}   "
              f"correct→wrong {f['correct_to_wrong']:>4}   net {f['net']:>+4}")

    print(f"\n-- |Δ| loop vs head_only --")
    for sname, m in d["delta_magnitude_loop_vs_head_only"].items():
        print(f"{sname:<20} |Δlogit| {_fmt(m['mean_abs_delta_logit'])}   "
              f"|Δprob| {_fmt(m['mean_abs_delta_prob'])}")

    t = d["trace"]
    print(f"\n-- trace (real) --")
    print(f"mean iterations run {_fmt(t['mean_iterations_run'], 2)}   "
          f"mean flagged_bias_absmean {_fmt(t['mean_flagged_bias_absmean'])}")
    for f in t["per_fold"]:
        churns = ", ".join(_fmt(x, 4) for x in f["churn_per_iteration"])
        print(f"fold {f['fold']}: iters={f['iterations_run']} "
              f"bias_absmean={_fmt(f['flagged_bias_absmean'])} "
              f"n_flagged={f['n_flagged_edges']} churn/iter=[{churns}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument(
        "--run-dir", required=True,
        help="Directory holding feedback_oof_real.pt, feedback_oof_head_only.pt "
             "and feedback_trace_real.json.",
    )
    args = parser.parse_args()
    diagnose(args.dataset, args.run_dir)


if __name__ == "__main__":
    main()
