"""Complete RQ2 using saved predictions; no training or contract replacement.

Run from the repository root with OMP_NUM_THREADS=1.
Uses the same primary and secondary bootstrap as aggregate_multiseed.
Checks every reconstructed score against the existing multi-seed artifact.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pipeline.common.datasets import get_dataset_config
from src.pipeline.common.splits import eval_macro_f1
from src.pipeline.step4.aggregate_multiseed import (
    SEEDS, BOOTSTRAP_ITERS, BOOTSTRAP_SEED, _llm_alone_preds,
    _load_seed_preds, independent_seed_two_level, _seed_matched_bootstrap,
)


def main():
    reference = json.loads((ROOT / "results/multiseed_ladder_v2_head.json").read_text())
    out = {
        "schema_version": 1,
        "metric": "pooled_oof_macro_f1",
        "seeds": list(SEEDS),
        "bootstrap_iters": BOOTSTRAP_ITERS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "protocol": "Fixed folds; primary resamples edges and independently selects each arm's training seed; secondary resamples edges and averages seed-matched differences. Prototype predictions are deterministic. No retraining.",
        "source_artifacts": ["results/multiseed_v2/head/<dataset>_seed<S>/metrics.json (or agaf_predictions.json)", "data/<dataset>/processed/step4_feedback/prototypes.pt", "graph, embeddings and folds from get_dataset_config", "results/multiseed_ladder_v2_head.json"],
    }
    for dataset in ("unsw_nb15", "ton_iot"):
        config = get_dataset_config(dataset)
        labels = torch.load(config.graph_path, weights_only=False).edge_label.cpu().numpy()
        semantic = _llm_alone_preds(dataset, config)
        per_seed = {s: {"agaf": _load_seed_preds(dataset, s, ROOT / "results/multiseed_v2/head")["agaf"], "llm": semantic} for s in SEEDS}
        scores = {arm: {str(s): float(eval_macro_f1(labels, per_seed[s][arm], config.eval_classes)) for s in SEEDS} for arm in ("agaf", "llm")}
        for s in SEEDS:
            assert np.isclose(scores["agaf"][str(s)], reference[dataset]["rungs"]["agaf"]["per_seed"][str(s)], atol=1e-12, rtol=0)
        # Verify the deterministic prototype against its recorded score.
        assert np.isclose(scores["llm"]["42"], reference[dataset]["rungs"]["llm"]["mean"], atol=1e-12, rtol=0)
        two = independent_seed_two_level(labels, per_seed, "agaf", "llm", config.eval_classes)
        matched = _seed_matched_bootstrap(labels, per_seed, "agaf", "llm", config.eval_classes)
        differences = [scores["agaf"][str(s)] - scores["llm"][str(s)] for s in SEEDS]
        out[dataset] = {"per_seed_macro_f1": scores, "comparisons": {"agaf_vs_llm": {
            "two_level": two, "seed_matched": matched,
            "point_difference": float(np.mean(differences)),
            "sign_stable_across_seeds": bool(all(d > 0 for d in differences) or all(d < 0 for d in differences)),
        }}}
        print(dataset, out[dataset], flush=True)
    (ROOT / "results/rq2_fusion_semantic_interval.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
