#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ID="${COARSE_CASCADE_LAUNCH_ID:?COARSE_CASCADE_LAUNCH_ID is required}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ ]]; then
  echo "unsafe COARSE_CASCADE_LAUNCH_ID" >&2
  exit 2
fi
CONFIG_PATH="${COARSE_CASCADE_CONFIG:-config/experiments/train_direct_dynamics_cascade_coarse_mechanics_v1.json}"
case "$CONFIG_PATH" in
  config/experiments/train_direct_dynamics_cascade_coarse_mechanics_v1.json|\
  config/experiments/train_direct_dynamics_cascade_coarse_learning_curve_v1.json|\
  config/experiments/train_direct_dynamics_cascade_coarse_residual_v1.json|\
  config/experiments/train_direct_dynamics_cascade_coarse_standardized_residual_v1.json) ;;
  *)
    echo "unsafe COARSE_CASCADE_CONFIG" >&2
    exit 2
    ;;
esac

RESULT_ROOT="/home/autoresearch_results/direct_dynamics_cascade_v1"
STATUS_ROOT="$RESULT_ROOT/launches"
STATUS_DIR="$STATUS_ROOT/$RUN_ID"
mkdir -p "$STATUS_ROOT"
if ! mkdir -m 700 "$STATUS_DIR"; then
  echo "refusing to reuse coarse-cascade launch status directory" >&2
  exit 3
fi
started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '{"status":"admission_check","run_id":"%s","started_at":"%s","code_commit":"%s"}\n' \
  "$RUN_ID" "$started" "$(git rev-parse HEAD)" > "$STATUS_DIR/status.json"
finish() {
  code=$?
  finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '{"run_id":"%s","controller_exit_code":%d,"finished_at":"%s","code_commit":"%s"}\n' \
    "$RUN_ID" "$code" "$finished" "$(git rev-parse HEAD)" > "$STATUS_DIR/exit.json"
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
GPU_COUNT="$(nvidia-smi --query-gpu=count --format=csv,noheader | wc -l | tr -d ' ')"
if [[ ! "$GPU_COUNT" =~ ^[0-9]+$ || "$GPU_COUNT" != "1" ]]; then
  echo "GPU admission failed: visible=$GPU_COUNT, expected exactly one" >&2
  exit 5
fi

read_gpu_into() {
  local observation
  if ! observation="$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"; then
    echo "GPU admission failed: nvidia-smi observation failed" >&2
    return 1
  fi
  if [[ ! "$observation" =~ ^[0-9]+,[0-9]+$ ]]; then
    echo "GPU admission failed: malformed observation '$observation'" >&2
    return 1
  fi
  GPU_MEMORY="${observation%%,*}"
  GPU_UTILIZATION="${observation##*,}"
}
read_gpu_into || exit 6
memory="$GPU_MEMORY"
utilization="$GPU_UTILIZATION"
observe_occupied_gpu() {
  for sample in {1..11}; do
    read_gpu_into || exit 7
    observed_memory="$GPU_MEMORY"
    observed_utilization="$GPU_UTILIZATION"
    if [[ "$observed_utilization" -ge 5 ]]; then
      echo "GPU admission failed: another workload reached utilization=$observed_utilization%" >&2
      exit 8
    fi
    if [[ "$sample" -lt 11 ]]; then sleep 30; fi
  done
}
if [[ "$memory" -gt 1024 ]]; then
  observe_occupied_gpu
fi
read_gpu_into || exit 9
final_memory="$GPU_MEMORY"
final_utilization="$GPU_UTILIZATION"
if [[ "$memory" -le 1024 && "$final_memory" -gt 1024 ]]; then
  observe_occupied_gpu
  read_gpu_into || exit 10
  final_memory="$GPU_MEMORY"
  final_utilization="$GPU_UTILIZATION"
fi
if [[ "$final_memory" -gt 1024 && "$final_utilization" -ge 5 ]]; then
  echo "GPU admission failed at final observation" >&2
  exit 11
fi

printf '{"status":"python_preflight","run_id":"%s","started_at":"%s","code_commit":"%s"}\n' \
  "$RUN_ID" "$started" "$(git rev-parse HEAD)" > "$STATUS_DIR/status.json"
export COARSE_CASCADE_STATUS_PATH="$STATUS_DIR/status.json"
export CLEARML_REQUIRE_ONLINE=1
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6
timeout --signal=TERM --kill-after=2m 7080s python -m assim_lib.direct_dynamics_cascade_coarse_training \
  --config "$CONFIG_PATH"
