#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ID="${GEOMETRY_FULL_CFM_RUN_ID:?GEOMETRY_FULL_CFM_RUN_ID is required}"
MODE="${GEOMETRY_FULL_CFM_MODE:?GEOMETRY_FULL_CFM_MODE preflight|train is required}"
EXPECTED_COMMIT="${GEOMETRY_FULL_CFM_EXPECTED_COMMIT:?GEOMETRY_FULL_CFM_EXPECTED_COMMIT is required}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ ]]; then
  echo "unsafe GEOMETRY_FULL_CFM_RUN_ID" >&2
  exit 2
fi
if [[ "$MODE" != "preflight" && "$MODE" != "train" ]]; then
  echo "GEOMETRY_FULL_CFM_MODE must be preflight or train" >&2
  exit 2
fi
CONFIG="${GEOMETRY_FULL_CFM_CONFIG:-config/experiments/train_direct_dynamics_geometry_full_cfm_ab_v1.json}"
EXPECTED_CONFIG="config/experiments/train_direct_dynamics_geometry_full_cfm_ab_v1.json"
if [[ "$CONFIG" != "$EXPECTED_CONFIG" || ! -f "$CONFIG" ]]; then
  echo "unreviewed matched full-CFM config" >&2
  exit 2
fi
CURRENT_COMMIT="$(git rev-parse HEAD)"
if [[ "$CURRENT_COMMIT" != "$EXPECTED_COMMIT" ]]; then
  echo "exact commit mismatch" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "matched full-CFM launch requires a clean worktree" >&2
  exit 2
fi

RESULT_ROOT="/home/autoresearch_results/direct_dynamics_geometry_cfm_v1"
OUTPUT="$RESULT_ROOT/$MODE/$RUN_ID"
LAUNCH_ROOT="$RESULT_ROOT/${MODE}_launches"
LAUNCH="$LAUNCH_ROOT/$RUN_ID"
mkdir -p "$LAUNCH_ROOT"
if [[ -e "$OUTPUT" || -L "$OUTPUT" ]] || ! mkdir -m 700 "$LAUNCH"; then
  echo "refusing to reuse matched full-CFM output or launch" >&2
  exit 3
fi
printf '{"status":"gpu_admission","run_id":"%s","mode":"%s","code_commit":"%s"}\n' \
  "$RUN_ID" "$MODE" "$CURRENT_COMMIT" > "$LAUNCH/status.json"
finish() {
  code=$?
  trap - EXIT
  printf '{"run_id":"%s","mode":"%s","controller_exit_code":%d,"finished_at":"%s","code_commit":"%s"}\n' \
    "$RUN_ID" "$MODE" "$code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$CURRENT_COMMIT" \
    > "$LAUNCH/exit.json"
  if [[ "$code" -eq 0 ]]; then status=complete; else status=failed; fi
  printf '{"status":"%s","run_id":"%s","mode":"%s","exit_code":%d,"code_commit":"%s"}\n' \
    "$status" "$RUN_ID" "$MODE" "$code" "$CURRENT_COMMIT" > "$LAUNCH/status.json"
  exit "$code"
}
trap finish EXIT

exec 9>"/home/autoresearch_results/direct_dynamics_all_hours_v1/launches/.gpu_job.lock"
flock -n 9 || { echo "another project launch owns the shared GPU lock" >&2; exit 4; }
GPU_UUID="$(scripts/require_single_gpu_uuid.sh)"
readonly GPU_UUID
EXPECTED_GPU_UUID="GPU-40ff1cbb-07cc-992e-23cb-8fe181972b76"
if [[ "$GPU_UUID" != "$EXPECTED_GPU_UUID" ]]; then
  echo "unexpected visible GPU UUID: $GPU_UUID" >&2
  exit 5
fi
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
    [[ "$GPU_UTILIZATION" -lt 5 ]] || { echo "GPU busy: utilization=${GPU_UTILIZATION}%" >&2; exit 7; }
    [[ "$sample" -eq 11 ]] || sleep 30
  done
fi
read_gpu || exit 8
[[ "$initial_memory" -gt 1024 || "$GPU_MEMORY" -le 1024 ]] || { echo "GPU ownership changed during admission" >&2; exit 9; }
[[ "$GPU_MEMORY" -le 1024 || "$GPU_UTILIZATION" -lt 5 ]] || { echo "GPU busy at final observation" >&2; exit 10; }

printf '{"status":"running","run_id":"%s","mode":"%s","code_commit":"%s","gpu_uuid":"%s"}\n' \
  "$RUN_ID" "$MODE" "$CURRENT_COMMIT" "$GPU_UUID" > "$LAUNCH/status.json"
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export CLEARML_REQUIRE_ONLINE=1
unset CLEARML_OFFLINE_MODE
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6
if [[ "$MODE" == "preflight" ]]; then
  TIMEOUT_SECONDS=3540
  KILL_AFTER_SECONDS=60
  timeout --foreground --signal=TERM --kill-after="${KILL_AFTER_SECONDS}s" "${TIMEOUT_SECONDS}s" \
    python -m assim_lib.direct_dynamics_geometry_cfm_training \
    --config "$CONFIG" --mode preflight --output-dir "$OUTPUT"
else
  TIMEOUT_SECONDS=28680
  KILL_AFTER_SECONDS=120
  timeout --foreground --signal=TERM --kill-after="${KILL_AFTER_SECONDS}s" "${TIMEOUT_SECONDS}s" \
    python -m assim_lib.direct_dynamics_geometry_cfm_transactional_training \
    --config "$CONFIG" --output-dir "$OUTPUT"
fi
