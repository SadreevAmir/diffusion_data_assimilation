#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ID="${COARSE_BUDGET_PREFLIGHT_RUN_ID:?COARSE_BUDGET_PREFLIGHT_RUN_ID is required}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ ]]; then
  echo "unsafe COARSE_BUDGET_PREFLIGHT_RUN_ID" >&2
  exit 2
fi
CONFIG="${COARSE_BUDGET_PREFLIGHT_CONFIG:-config/experiments/preflight_direct_dynamics_coarse_budget_actual_gpu_v1.json}"
EXPECTED_CONFIG="config/experiments/preflight_direct_dynamics_coarse_budget_actual_gpu_v1.json"
if [[ "$CONFIG" != "$EXPECTED_CONFIG" || ! -f "$CONFIG" ]]; then
  echo "unreviewed coarse-budget GPU preflight config" >&2
  exit 2
fi

OUTPUT_ROOT="/home/autoresearch_results/direct_dynamics_cascade_v2/coarse_budget_actual_gpu_preflight"
LAUNCH_ROOT="/home/autoresearch_results/direct_dynamics_cascade_v2/coarse_budget_actual_gpu_preflight_launches"
OUTPUT="$OUTPUT_ROOT/$RUN_ID"
mkdir -p "$LAUNCH_ROOT"
if [[ -e "$OUTPUT" || -L "$OUTPUT" ]]; then
  echo "refusing to reuse coarse-budget GPU preflight output" >&2
  exit 3
fi
if ! mkdir -m 700 "$LAUNCH_ROOT/$RUN_ID"; then
  echo "refusing to reuse coarse-budget GPU preflight launch" >&2
  exit 3
fi
printf '{"status":"gpu_admission","run_id":"%s","code_commit":"%s"}\n' \
  "$RUN_ID" "$(git rev-parse HEAD)" > "$LAUNCH_ROOT/$RUN_ID/status.json"
finish() {
  code=$?
  trap - EXIT
  printf '{"run_id":"%s","controller_exit_code":%d,"finished_at":"%s","code_commit":"%s"}\n' \
    "$RUN_ID" "$code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
    > "$LAUNCH_ROOT/$RUN_ID/exit.json"
  if [[ "$code" -eq 0 ]]; then
    printf '{"status":"complete","run_id":"%s","code_commit":"%s"}\n' \
      "$RUN_ID" "$(git rev-parse HEAD)" > "$LAUNCH_ROOT/$RUN_ID/status.json"
  else
    printf '{"status":"failed","run_id":"%s","exit_code":%d,"code_commit":"%s"}\n' \
      "$RUN_ID" "$code" "$(git rev-parse HEAD)" > "$LAUNCH_ROOT/$RUN_ID/status.json"
  fi
  exit "$code"
}
trap finish EXIT

exec 9>"/home/autoresearch_results/direct_dynamics_all_hours_v1/launches/.gpu_job.lock"
if ! flock -n 9; then
  echo "another project launch owns the GPU lock" >&2
  exit 4
fi
GPU_UUID="$(scripts/require_single_gpu_uuid.sh)"
readonly GPU_UUID

read_gpu() {
  local observation
  observation="$(nvidia-smi --id=0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')" || return 1
  [[ "$observation" =~ ^[0-9]+,[0-9]+$ ]] || return 1
  GPU_MEMORY="${observation%%,*}"
  GPU_UTILIZATION="${observation##*,}"
}
read_gpu || exit 5
initial_memory="$GPU_MEMORY"
if [[ "$GPU_MEMORY" -gt 1024 ]]; then
  for sample in {1..11}; do
    read_gpu || exit 6
    if [[ "$GPU_UTILIZATION" -ge 5 ]]; then
      echo "GPU busy: utilization=${GPU_UTILIZATION}%" >&2
      exit 7
    fi
    [[ "$sample" -eq 11 ]] || sleep 30
  done
fi
read_gpu || exit 8
if [[ "$initial_memory" -le 1024 && "$GPU_MEMORY" -gt 1024 ]]; then
  for sample in {1..11}; do
    read_gpu || exit 9
    if [[ "$GPU_UTILIZATION" -ge 5 ]]; then
      echo "GPU became busy: utilization=${GPU_UTILIZATION}%" >&2
      exit 10
    fi
    [[ "$sample" -eq 11 ]] || sleep 30
  done
  read_gpu || exit 11
fi
if [[ "$GPU_MEMORY" -gt 1024 && "$GPU_UTILIZATION" -ge 5 ]]; then
  echo "GPU busy at final observation" >&2
  exit 12
fi

export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export CLEARML_REQUIRE_ONLINE=1
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6
timeout --foreground --signal=TERM --kill-after=60s 1740s \
  python -m assim_lib.direct_dynamics_coarse_budget_actual_gpu_preflight \
  --config "$CONFIG" --output "$OUTPUT"
