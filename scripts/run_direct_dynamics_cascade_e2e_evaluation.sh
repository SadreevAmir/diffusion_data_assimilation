#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RUN_ID="${CASCADE_E2E_RUN_ID:?CASCADE_E2E_RUN_ID is required}"
CONFIG="${CASCADE_E2E_CONFIG:-config/experiments/evaluate_direct_dynamics_cascade_e2e_v1.json}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ ]]; then
  echo "unsafe CASCADE_E2E_RUN_ID" >&2
  exit 2
fi
case "$CONFIG" in
  config/experiments/evaluate_direct_dynamics_cascade_e2e_v1.json|\
  config/experiments/evaluate_direct_dynamics_cascade_e2e_fine2048_v2.json|\
  config/experiments/evaluate_direct_dynamics_cascade_e2e_fine6474_v3.json) ;;
  *)
    echo "unsupported CASCADE_E2E_CONFIG" >&2
    exit 2
    ;;
esac
RESULT_ROOT="/home/autoresearch_results/direct_dynamics_cascade_v2"
STATUS_DIR="$RESULT_ROOT/launches/$RUN_ID"
mkdir -p "$RESULT_ROOT/launches"
if ! mkdir -m 700 "$STATUS_DIR"; then
  echo "refusing to reuse cascade evaluation status" >&2
  exit 3
fi
OUTPUT="$RESULT_ROOT/e2e_evaluation/$RUN_ID"
if [[ -e "$OUTPUT" || -L "$OUTPUT" ]]; then
  echo "refusing to reuse cascade evaluation output" >&2
  exit 4
fi
started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git rev-parse HEAD)"
printf '{"status":"admission_check","run_id":"%s","started_at":"%s","code_commit":"%s"}\n' \
  "$RUN_ID" "$started" "$commit" > "$STATUS_DIR/status.json"
finish() {
  code=$?
  printf '{"run_id":"%s","controller_exit_code":%d,"finished_at":"%s","code_commit":"%s"}\n' \
    "$RUN_ID" "$code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$commit" > "$STATUS_DIR/exit.json"
  exit "$code"
}
trap finish EXIT

exec 9>"/home/autoresearch_results/direct_dynamics_all_hours_v1/launches/.gpu_job.lock"
if ! flock -n 9; then
  echo "another project launch owns the GPU lock" >&2
  exit 5
fi
GPU_COUNT="$(nvidia-smi --query-gpu=count --format=csv,noheader | wc -l | tr -d ' ')"
GPU_MEMORY_USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
GPU_UTILIZATION="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
if [[ "$GPU_COUNT" != "1" || ! "$GPU_MEMORY_USED" =~ ^[0-9]+$ || ! "$GPU_UTILIZATION" =~ ^[0-9]+$ ]]; then
  echo "invalid single-GPU inventory" >&2
  exit 6
fi
if [[ "$GPU_MEMORY_USED" -gt 1024 || "$GPU_UTILIZATION" -ge 5 ]]; then
  echo "target GPU is not immediately free" >&2
  exit 7
fi
export CASCADE_E2E_STATUS_PATH="$STATUS_DIR/status.json"
export CLEARML_REQUIRE_ONLINE=1
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6
timeout --signal=TERM --kill-after=2m 14280s python -m \
  assim_lib.direct_dynamics_cascade_e2e_evaluation \
  --config "$CONFIG" \
  --output "$OUTPUT"
