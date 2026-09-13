#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RUN_ID="${GEOMETRY_OBJECTIVE_AB_RUN_ID:?GEOMETRY_OBJECTIVE_AB_RUN_ID is required}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ ]]; then
  echo "unsafe GEOMETRY_OBJECTIVE_AB_RUN_ID" >&2
  exit 2
fi
CONFIG="${GEOMETRY_OBJECTIVE_AB_CONFIG:-config/experiments/train_direct_dynamics_geometry_objective_ab_v1.json}"
if [[ "$CONFIG" != "config/experiments/train_direct_dynamics_geometry_objective_ab_v1.json" || ! -f "$CONFIG" ]]; then
  echo "unreviewed geometry-objective A/B config" >&2
  exit 2
fi
OUTPUT_ROOT="/home/autoresearch_results/direct_dynamics_geometry_objective_v1/training"
LAUNCH_ROOT="/home/autoresearch_results/direct_dynamics_geometry_objective_v1/training_launches"
OUTPUT="$OUTPUT_ROOT/$RUN_ID"
LAUNCH="$LAUNCH_ROOT/$RUN_ID"
mkdir -p "$LAUNCH_ROOT"
if [[ -e "$OUTPUT" || -L "$OUTPUT" ]] || ! mkdir -m 700 "$LAUNCH"; then
  echo "refusing to reuse geometry-objective A/B output or launch" >&2
  exit 3
fi
finish() {
  code=$?
  trap - EXIT
  printf '{"run_id":"%s","controller_exit_code":%d,"finished_at":"%s","code_commit":"%s"}\n' \
    "$RUN_ID" "$code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" > "$LAUNCH/exit.json"
  if [[ "$code" -eq 0 ]]; then status=complete; else status=failed; fi
  printf '{"status":"%s","run_id":"%s","exit_code":%d,"code_commit":"%s"}\n' \
    "$status" "$RUN_ID" "$code" "$(git rev-parse HEAD)" > "$LAUNCH/status.json"
  exit "$code"
}
trap finish EXIT
printf '{"status":"gpu_admission","run_id":"%s","code_commit":"%s"}\n' \
  "$RUN_ID" "$(git rev-parse HEAD)" > "$LAUNCH/status.json"
exec 9>"/home/autoresearch_results/direct_dynamics_all_hours_v1/launches/.gpu_job.lock"
flock -n 9 || { echo "another project launch owns the shared GPU lock" >&2; exit 4; }
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
    [[ "$GPU_UTILIZATION" -lt 5 ]] || { echo "GPU busy" >&2; exit 7; }
    [[ "$sample" -eq 11 ]] || sleep 30
  done
fi
read_gpu || exit 8
[[ "$initial_memory" -gt 1024 || "$GPU_MEMORY" -le 1024 ]] || { echo "GPU ownership changed" >&2; exit 9; }
[[ "$GPU_MEMORY" -le 1024 || "$GPU_UTILIZATION" -lt 5 ]] || { echo "GPU busy at final admission" >&2; exit 10; }
printf '{"status":"running","run_id":"%s","code_commit":"%s","gpu_uuid":"%s"}\n' \
  "$RUN_ID" "$(git rev-parse HEAD)" "$GPU_UUID" > "$LAUNCH/status.json"
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export CLEARML_REQUIRE_ONLINE=1
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6
timeout --foreground --signal=TERM --kill-after=120s 28680s \
  python -m assim_lib.direct_dynamics_geometry_training --config "$CONFIG" --output "$OUTPUT"
