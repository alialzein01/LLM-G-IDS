#!/usr/bin/env bash
# Re-select top_k_percent and injection_scale per dataset, validation folds only,
# seeds 42/1/2. Stage 1 sweeps top_k at the dataset's current scale; stage 2 sweeps
# the scale at the k stage 1 chose; stage 3 writes the config with the old values
# under "superseded". Test-fold numbers are recorded by the sweep but never consulted.
#
# Env: JOBS=<n> parallel jobs (default 4)
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export OMP_NUM_THREADS=1
export PYTHONPATH="$REPO"
JOBS="${JOBS:-4}"
SEEDS="${SEEDS:-42 1 2}"
DATASETS="${DATASETS:-unsw_nb15 ton_iot}"
# USE_LLM_HEAD=1 selects knobs for the trained-head consultant instead of the
# prototype. The two consultants pick different knobs, so their sweeps write to
# separate trees and finalize keeps the prototype block rather than losing it.
HEAD_FLAG=""
if [ "${USE_LLM_HEAD:-0}" = "1" ]; then HEAD_FLAG="--use-llm-head"; fi
export HEAD_FLAG

pairs() { for ds in $DATASETS; do for s in $SEEDS; do echo "$ds $s"; done; done; }

echo "=== stage 1: top_k 15..35, ${SEEDS} ${HEAD_FLAG} ==="
pairs | xargs -P "$JOBS" -n 2 bash -c \
  'python3 -m src.pipeline.step4.select_feedback_knobs --dataset "$0" --stage top_k --seed "$1" $HEAD_FLAG >/dev/null'

echo "=== stage 2: injection_scale at the selected k ==="
pairs | xargs -P "$JOBS" -n 2 bash -c \
  'python3 -m src.pipeline.step4.select_feedback_knobs --dataset "$0" --stage scale --seed "$1" $HEAD_FLAG >/dev/null'

echo "=== stage 3: finalize ==="
for ds in $DATASETS; do
  python3 -m src.pipeline.step4.select_feedback_knobs --dataset "$ds" --stage finalize $HEAD_FLAG
done
