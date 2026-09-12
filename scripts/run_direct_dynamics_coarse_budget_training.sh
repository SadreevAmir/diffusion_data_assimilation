#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ID="${COARSE_BUDGET_TRAIN_RUN_ID:?COARSE_BUDGET_TRAIN_RUN_ID is required}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ ]] || exit 2
CONFIG="${COARSE_BUDGET_TRAIN_CONFIG:-config/experiments/train_direct_dynamics_coarse_budget_64_v1.json}"
EXPECTED_CONFIG="config/experiments/train_direct_dynamics_coarse_budget_64_v1.json"
[[ "$CONFIG" == "$EXPECTED_CONFIG" && -f "$CONFIG" ]] || exit 2

GROUP="/home/autoresearch_results/direct_dynamics_cascade_v2/coarse_budget_train64"
LAUNCHES="/home/autoresearch_results/direct_dynamics_cascade_v2/coarse_budget_train64_launches"
OUTPUT="$GROUP/$RUN_ID"
mkdir -p "$LAUNCHES"
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || exit 3
mkdir -m 700 "$LAUNCHES/$RUN_ID" || exit 3
printf '{"status":"gpu_admission","run_id":"%s","code_commit":"%s"}\n' \
  "$RUN_ID" "$(git rev-parse HEAD)" > "$LAUNCHES/$RUN_ID/status.json"
finish() {
  code=$?
  trap - EXIT
  printf '{"run_id":"%s","controller_exit_code":%d,"finished_at":"%s","code_commit":"%s"}\n' \
    "$RUN_ID" "$code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
    > "$LAUNCHES/$RUN_ID/exit.json"
  if [[ "$code" -eq 0 ]]; then
    printf '{"status":"complete","run_id":"%s","code_commit":"%s"}\n' \
      "$RUN_ID" "$(git rev-parse HEAD)" > "$LAUNCHES/$RUN_ID/status.json"
  else
    printf '{"status":"failed","run_id":"%s","exit_code":%d,"code_commit":"%s"}\n' \
      "$RUN_ID" "$code" "$(git rev-parse HEAD)" > "$LAUNCHES/$RUN_ID/status.json"
  fi
  exit "$code"
}
trap finish EXIT

exec 9>"/home/autoresearch_results/direct_dynamics_all_hours_v1/launches/.gpu_job.lock"
flock -n 9 || exit 4
GPU_UUID="$(scripts/require_single_gpu_uuid.sh)"
readonly GPU_UUID
read_gpu() {
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
    [[ "$GPU_UTILIZATION" -lt 5 ]] || exit 7
    [[ "$sample" -eq 11 ]] || sleep 30
  done
fi
read_gpu || exit 8
if [[ "$initial_memory" -le 1024 && "$GPU_MEMORY" -gt 1024 ]]; then
  for sample in {1..11}; do
    read_gpu || exit 9
    [[ "$GPU_UTILIZATION" -lt 5 ]] || exit 10
    [[ "$sample" -eq 11 ]] || sleep 30
  done
  read_gpu || exit 11
fi
[[ "$GPU_MEMORY" -le 1024 || "$GPU_UTILIZATION" -lt 5 ]] || exit 12

export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export CLEARML_REQUIRE_ONLINE=1
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6
timeout --foreground --signal=TERM --kill-after=60s 10800s \
  python -m assim_lib.direct_dynamics_coarse_budget_training \
  --config "$CONFIG" --output "$OUTPUT"
