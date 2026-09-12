#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_MODE=()
if [[ "$#" -eq 1 && "$1" == "--preflight-only" ]]; then
  PYTHON_MODE=("--preflight-only")
elif [[ "$#" -ne 0 ]]; then
  echo "usage: $0 [--preflight-only]" >&2
  exit 2
fi

RUN_ID="${FINE_CASCADE_LAUNCH_ID:?FINE_CASCADE_LAUNCH_ID is required}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ ]]; then
  echo "unsafe FINE_CASCADE_LAUNCH_ID" >&2
  exit 2
fi
CONFIG_PATH="${FINE_CASCADE_CONFIG:-config/experiments/train_direct_dynamics_cascade_fine_mechanics_v1.json}"
case "$CONFIG_PATH" in
  config/experiments/train_direct_dynamics_cascade_fine_mechanics_v1.json|\
  config/experiments/train_direct_dynamics_cascade_fine_compact_v2.json|\
  config/experiments/train_direct_dynamics_cascade_fine_compact_2048_v3.json|\
  config/experiments/train_direct_dynamics_cascade_fine_matched_scale_2048_v5.json|\
  config/experiments/train_direct_dynamics_cascade_fine_variance_preconditioned_512_v6.json|\
  config/experiments/train_direct_dynamics_cascade_fine_colored_preconditioned_512_v7.json|\
  config/experiments/train_direct_dynamics_cascade_fine_colored_preconditioned_2048_v8.json|\
  config/experiments/train_direct_dynamics_cascade_fine_full_epoch_v4.json) ;;
  *)
    echo "unsafe FINE_CASCADE_CONFIG" >&2
    exit 2
    ;;
esac

RESULT_ROOT="/home/autoresearch_results/direct_dynamics_cascade_v2"
STATUS_ROOT="$RESULT_ROOT/launches"
STATUS_DIR="$STATUS_ROOT/$RUN_ID"
mkdir -p "$STATUS_ROOT"
if ! mkdir -m 700 "$STATUS_DIR"; then
  echo "refusing to reuse fine-cascade launch status directory" >&2
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

GPU_UUID="$(scripts/require_single_gpu_uuid.sh)"
readonly GPU_UUID

read_gpu_into() {
  local observation
  if ! observation="$(nvidia-smi --id=0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"; then
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
GPU_MEMORY_USED="$GPU_MEMORY"
observe_occupied_gpu_for_five_minutes() {
  samples=0
  while [[ "$samples" -lt 11 ]]; do
    read_gpu_into || exit 7
    if [[ "$GPU_UTILIZATION" -ge 5 ]]; then
      echo "GPU admission failed: another workload reached utilization=$GPU_UTILIZATION%" >&2
      exit 8
    fi
    samples=$((samples + 1))
    if [[ "$samples" -lt 11 ]]; then
      sleep 30
    fi
  done
}
if [[ "$GPU_MEMORY_USED" -gt 1024 ]]; then
  observe_occupied_gpu_for_five_minutes
fi

read_gpu_into || exit 9
FINAL_MEMORY_USED="$GPU_MEMORY"
FINAL_UTILIZATION="$GPU_UTILIZATION"
if [[ "$GPU_MEMORY_USED" -le 1024 && "$FINAL_MEMORY_USED" -gt 1024 ]]; then
  observe_occupied_gpu_for_five_minutes
  read_gpu_into || exit 10
  FINAL_MEMORY_USED="$GPU_MEMORY"
  FINAL_UTILIZATION="$GPU_UTILIZATION"
fi
if [[ "$FINAL_MEMORY_USED" -gt 1024 && "$FINAL_UTILIZATION" -ge 5 ]]; then
  echo "GPU admission failed at final observation: memory_used_mib=$FINAL_MEMORY_USED utilization=$FINAL_UTILIZATION" >&2
  exit 11
fi

printf '{"status":"running","run_id":"%s","started_at":"%s","code_commit":"%s"}\n' \
  "$RUN_ID" "$started" "$(git rev-parse HEAD)" > "$STATUS_DIR/status.json"

export FINE_CASCADE_STATUS_PATH="$STATUS_DIR/status.json"
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export CLEARML_REQUIRE_ONLINE=1
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6
timeout --foreground --signal=TERM --kill-after=2m 14280s python -m assim_lib.direct_dynamics_cascade_fine_training \
  --config "$CONFIG_PATH" "${PYTHON_MODE[@]}"
