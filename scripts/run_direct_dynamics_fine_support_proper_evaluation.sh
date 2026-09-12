#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ID="${FINE_SUPPORT_EVAL_RUN_ID:?FINE_SUPPORT_EVAL_RUN_ID is required}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ ]]; then
  echo "unsafe fine-support evaluation identifier" >&2
  exit 2
fi
CONFIG="config/experiments/evaluate_direct_dynamics_fine_support_proper_v1.json"
OUTPUT_ROOT="/home/autoresearch_results/direct_dynamics_cascade_v2/fine_support_proper_evaluation"
LAUNCH_ROOT="/home/autoresearch_results/direct_dynamics_cascade_v2/fine_support_proper_evaluation_launches"
OUTPUT="$OUTPUT_ROOT/$RUN_ID"
LAUNCH="$LAUNCH_ROOT/$RUN_ID"
if [[ -e "$OUTPUT" || -L "$OUTPUT" || -e "$LAUNCH" || -L "$LAUNCH" ]]; then
  echo "refusing to reuse fine-support evaluation output" >&2
  exit 3
fi
mkdir -p "$LAUNCH_ROOT"
mkdir -m 700 "$LAUNCH"
export FINE_SUPPORT_EVAL_STATUS_PATH="$LAUNCH/status.json"
printf '{"status":"gpu_admission","run_id":"%s","code_commit":"%s"}\n' \
  "$RUN_ID" "$(git rev-parse HEAD)" > "$FINE_SUPPORT_EVAL_STATUS_PATH"
finish() {
  code=$?
  trap - EXIT
  printf '{"run_id":"%s","exit_code":%d,"finished_at":"%s","code_commit":"%s"}\n' \
    "$RUN_ID" "$code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
    > "$LAUNCH/exit.json"
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
  python -m assim_lib.direct_dynamics_fine_support_proper_evaluation \
  --config "$CONFIG" --output "$OUTPUT"
