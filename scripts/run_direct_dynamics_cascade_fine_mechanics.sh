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
  config/experiments/train_direct_dynamics_cascade_fine_compact_2048_v3.json) ;;
  *)
    echo "unsafe FINE_CASCADE_CONFIG" >&2
    exit 2
    ;;
esac

RESULT_ROOT="/home/autoresearch_results/direct_dynamics_cascade_v1"
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
exec 9>"/home/autoresearch_results/direct_dynamics_all_hours_v1/launches/.gpu_job.lock"
if ! flock -n 9; then
  echo "another project launch owns the GPU lock" >&2
  exit 4
fi

GPU_COUNT="$(nvidia-smi --query-gpu=count --format=csv,noheader | wc -l | tr -d ' ')"
if [[ ! "$GPU_COUNT" =~ ^[0-9]+$ || "$GPU_COUNT" != "1" ]]; then
  echo "GPU admission failed: visible=$GPU_COUNT, expected exactly one" >&2
  exit 5
fi
GPU_MEMORY_USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
if [[ ! "$GPU_MEMORY_USED" =~ ^[0-9]+$ ]]; then
  echo "GPU admission failed: invalid memory reading '$GPU_MEMORY_USED'" >&2
  exit 6
fi
observe_occupied_gpu_for_five_minutes() {
  samples=0
  while [[ "$samples" -lt 11 ]]; do
    utilization="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    if [[ ! "$utilization" =~ ^[0-9]+$ ]]; then
      echo "GPU admission failed: invalid utilization reading '$utilization'" >&2
      exit 7
    fi
    if [[ "$utilization" -ge 5 ]]; then
      echo "GPU admission failed: another workload reached utilization=$utilization%" >&2
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

FINAL_MEMORY_USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
FINAL_UTILIZATION="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
if [[ ! "$FINAL_MEMORY_USED" =~ ^[0-9]+$ || ! "$FINAL_UTILIZATION" =~ ^[0-9]+$ ]]; then
  echo "GPU admission failed: invalid final observation" >&2
  exit 9
fi
if [[ "$GPU_MEMORY_USED" -le 1024 && "$FINAL_MEMORY_USED" -gt 1024 ]]; then
  observe_occupied_gpu_for_five_minutes
  FINAL_MEMORY_USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  FINAL_UTILIZATION="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  if [[ ! "$FINAL_MEMORY_USED" =~ ^[0-9]+$ || ! "$FINAL_UTILIZATION" =~ ^[0-9]+$ ]]; then
    echo "GPU admission failed: invalid post-window observation" >&2
    exit 10
  fi
fi
if [[ "$FINAL_MEMORY_USED" -gt 1024 && "$FINAL_UTILIZATION" -ge 5 ]]; then
  echo "GPU admission failed at final observation: memory_used_mib=$FINAL_MEMORY_USED utilization=$FINAL_UTILIZATION" >&2
  exit 11
fi

printf '{"status":"running","run_id":"%s","started_at":"%s","code_commit":"%s"}\n' \
  "$RUN_ID" "$started" "$(git rev-parse HEAD)" > "$STATUS_DIR/status.json"

export FINE_CASCADE_STATUS_PATH="$STATUS_DIR/status.json"
export CLEARML_REQUIRE_ONLINE=1
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6
timeout --signal=TERM --kill-after=2m 14280s python -m assim_lib.direct_dynamics_cascade_fine_training \
  --config "$CONFIG_PATH" "${PYTHON_MODE[@]}"
