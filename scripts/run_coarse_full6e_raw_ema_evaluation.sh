#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RUN_ID="${COARSE_CHECKPOINT_EVAL_RUN_ID:?COARSE_CHECKPOINT_EVAL_RUN_ID is required}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ ]]; then
  echo "unsafe COARSE_CHECKPOINT_EVAL_RUN_ID" >&2
  exit 2
fi
RESULT_ROOT="/home/autoresearch_results/direct_dynamics_cascade_v1"
STATUS_DIR="$RESULT_ROOT/launches/$RUN_ID"
mkdir -p "$RESULT_ROOT/launches"
if ! mkdir -m 700 "$STATUS_DIR"; then
  echo "refusing to reuse evaluation status" >&2
  exit 3
fi
commit="$(git rev-parse HEAD)"
printf '{"status":"admission_check","run_id":"%s","code_commit":"%s"}\n' "$RUN_ID" "$commit" > "$STATUS_DIR/status.json"
finish() {
  code=$?
  printf '{"run_id":"%s","controller_exit_code":%d,"finished_at":"%s","code_commit":"%s"}\n' \
    "$RUN_ID" "$code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$commit" > "$STATUS_DIR/exit.json"
  exit "$code"
}
trap finish EXIT

GPU_LOCK_ROOT="/home/autoresearch_results/direct_dynamics_all_hours_v1/launches"
mkdir -p "$GPU_LOCK_ROOT"
exec 9>"$GPU_LOCK_ROOT/.gpu_job.lock"
if ! flock -n 9; then
  echo "another project launch owns the GPU lock" >&2
  exit 4
fi
observation="$(nvidia-smi --id=0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
if [[ ! "$observation" =~ ^[0-9]+,[0-9]+$ ]]; then
  echo "malformed GPU observation" >&2
  exit 5
fi
memory="${observation%%,*}"
if [[ "$memory" -gt 1024 ]]; then
  for sample in {1..11}; do
    observation="$(nvidia-smi --id=0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
    utilization="${observation##*,}"
    if [[ "$utilization" -ge 5 ]]; then
      echo "occupied GPU reached utilization=$utilization%" >&2
      exit 6
    fi
    if [[ "$sample" -lt 11 ]]; then sleep 30; fi
  done
fi

OUTPUT="$RESULT_ROOT/coarse_raw_ema_evaluation/$RUN_ID"
if [[ -e "$OUTPUT" || -L "$OUTPUT" ]]; then
  echo "refusing to reuse evaluation output" >&2
  exit 7
fi
export COARSE_CHECKPOINT_EVAL_STATUS_PATH="$STATUS_DIR/status.json"
export CLEARML_REQUIRE_ONLINE=1
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6
timeout --signal=TERM --kill-after=30s 3570s python -m \
  assim_lib.direct_dynamics_cascade_checkpoint_evaluation \
  --config config/experiments/evaluate_coarse_full6e_raw_ema_v1.json \
  --output "$OUTPUT"
