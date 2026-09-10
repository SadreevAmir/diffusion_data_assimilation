#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ID="${COARSE_RESIDUAL_STATS_ID:?COARSE_RESIDUAL_STATS_ID is required}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ ]]; then
  echo "unsafe COARSE_RESIDUAL_STATS_ID" >&2
  exit 2
fi
CONFIG_PATH="config/experiments/train_direct_dynamics_cascade_coarse_residual_v1.json"
RESULT_ROOT="/home/autoresearch_results/direct_dynamics_cascade_v1/residual_statistics"
OUTPUT_DIR="$RESULT_ROOT/$RUN_ID"
if ! mkdir -m 700 "$OUTPUT_DIR"; then
  echo "refusing to reuse residual-statistics output directory" >&2
  exit 3
fi

started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git rev-parse HEAD)"
printf '{"status":"running","run_id":"%s","started_at":"%s","code_commit":"%s"}\n' \
  "$RUN_ID" "$started" "$commit" > "$OUTPUT_DIR/status.json"
finish() {
  code=$?
  finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '{"run_id":"%s","controller_exit_code":%d,"finished_at":"%s","code_commit":"%s"}\n' \
    "$RUN_ID" "$code" "$finished" "$commit" > "$OUTPUT_DIR/exit.json"
  exit "$code"
}
trap finish EXIT

export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6
export TORCH_NUM_INTEROP_THREADS=1
timeout --signal=TERM --kill-after=30s 3600s python -m \
  assim_lib.direct_dynamics_cascade_residual_stats \
  --config "$CONFIG_PATH" \
  --output "$OUTPUT_DIR/statistics.json"
