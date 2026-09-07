#!/usr/bin/env bash
# Re-run ONLY the feedback stage, per (seed, dataset), for one condition.
#
# Tasks 1-2 changed train_feedback's loss and the prototype scorer's temperature.
# Neither touches the GNN rung, the AGAF rung, or the LLM rung (that one is an argmax
# over frozen prototypes, invariant to the temperature), so the per-seed
# oof_emb / fusion / oof captures already in results/multiseed/ stay valid and are
# reused rather than recomputed.
#
# assemble_ladder is deliberately NOT run per seed: the data/ tree holds seed-42
# fusion and oof artifacts, so a per-seed ladder would pair a seed-1 loop against a
# seed-42 AGAF. aggregate_multiseed does the pairing instead, from the captures.
#
# Each job writes to its own scratch dir, so jobs are safe to run in parallel and the
# canonical data/*/processed/step4_feedback/ tree is left untouched.
#
# Usage:  scripts/run_multiseed_feedback.sh <condition>
#   condition: legacy            selector-head loss off + T=10 (pre-Task-1/2 behaviour)
#              fixed             defaults (Tasks 1+2)
#              fixed_reselected  defaults, after top_k / injection_scale re-selection
#              head              legacy signals + the TRAINED HEAD as consultant, at
#                                the head's own selected knobs (2026-09-07 canonical)
# Env:    JOBS=<n>   parallel jobs (default 4)
#         SEEDS="1 2 42"  seed order; 42 last so the tree ends on the canonical seed
set -euo pipefail

COND="${1:?usage: $0 <legacy|fixed|fixed_reselected>}"
JOBS="${JOBS:-4}"
SEEDS="${SEEDS:-1 2 42}"
DATASETS="${DATASETS:-unsw_nb15 ton_iot}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

case "$COND" in
  # `legacy` predates the --calibrate-temperature rename; both spellings of the
  # condition-A signals are the current defaults, so no flags are needed for it.
  legacy)                    EXTRA="" ;;
  fixed|fixed_reselected)    EXTRA="--selector-head-loss-weight 1.0 --calibrate-temperature" ;;
  head)                      EXTRA="--use-llm-head" ;;
  *) echo "unknown condition: $COND" >&2; exit 2 ;;
esac

export OMP_NUM_THREADS=1
export PYTHONPATH="$REPO"
SCRATCH="${SCRATCH:-tmp/multiseed_v2}"
DEST="results/multiseed_v2/$COND"
mkdir -p "$SCRATCH" "$DEST"

run_one() {
  local ds="$1" seed="$2"
  local canon="data/$ds/processed/step4_feedback"
  local work="$SCRATCH/$COND/${ds}_seed${seed}"
  local out="$DEST/${ds}_seed${seed}"
  local prior="results/multiseed/${ds}_seed${seed}"
  rm -rf "$work"; mkdir -p "$work" "$out"

  # The scale and top-k come from the selected config, never from a flag default.
  cp "$canon/selected_feedback_config.json" "$work/"

  python3 -m src.pipeline.step4.train_feedback \
      --dataset "$ds" --modes real random head_only --seed "$seed" \
      --output-dir "$work" --prototypes-path "$canon/prototypes.pt" \
      $EXTRA > "$out/train.log" 2>&1

  # head_only and random are needed by the per-seed ablation block, and the
  # head condition's loop is compared against them, so carry them too.
  for f in feedback_oof_head_only.pt feedback_oof_random.pt; do
    [ -f "$work/$f" ] && cp "$work/$f" "$out/"
  done

  cp "$work/feedback_oof_real.pt"      "$out/"
  cp "$work/ablation_summary.json"     "$out/"
  cp "$work/benchmark_summary.json"    "$out/"
  cp "$work/feedback_trace_real.json"  "$out/"
  cp "$work/selected_feedback_config.json" "$out/"
  # Rungs this condition does not re-run. ladder_summary.json is carried only for its
  # llm_alone_prototype field; nothing else in it is read by aggregate_multiseed.
  cp "$prior/oof_logits.pt" "$prior/metrics.json" "$prior/ladder_summary.json" "$out/"
  echo "done $COND $ds seed $seed -> $out"
}
export -f run_one
export COND EXTRA SCRATCH DEST

for seed in $SEEDS; do for ds in $DATASETS; do echo "$ds $seed"; done; done \
  | xargs -P "$JOBS" -n 2 bash -c 'run_one "$0" "$1"'

echo "condition $COND complete -> $DEST"
