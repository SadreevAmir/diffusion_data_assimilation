#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VALIDATION_ID="${GEOMETRY_FULL_CFM_VALIDATION_ID:?GEOMETRY_FULL_CFM_VALIDATION_ID is required}"
EXPECTED_COMMIT="${GEOMETRY_FULL_CFM_VALIDATION_EXPECTED_COMMIT:?GEOMETRY_FULL_CFM_VALIDATION_EXPECTED_COMMIT is required}"
if [[ ! "$VALIDATION_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ ]]; then
  echo "unsafe GEOMETRY_FULL_CFM_VALIDATION_ID" >&2
  exit 2
fi
CONFIG="${GEOMETRY_FULL_CFM_VALIDATION_CONFIG:-config/experiments/evaluate_direct_dynamics_geometry_full_cfm_ab_v1.json}"
EXPECTED_CONFIG="config/experiments/evaluate_direct_dynamics_geometry_full_cfm_ab_v1.json"
if [[ "$CONFIG" != "$EXPECTED_CONFIG" || ! -f "$CONFIG" ]]; then
  echo "unreviewed full-CFM validation config" >&2
  exit 2
fi
CURRENT_COMMIT="$(git rev-parse HEAD)"
if [[ "$CURRENT_COMMIT" != "$EXPECTED_COMMIT" ]]; then
  echo "exact commit mismatch" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "full-CFM validation requires a clean worktree" >&2
  exit 2
fi

RESULT_ROOT="/home/autoresearch_results/direct_dynamics_geometry_cfm_v1"
OUTPUT="$RESULT_ROOT/validation/$VALIDATION_ID"
LAUNCH_ROOT="$RESULT_ROOT/validation_launches"
LAUNCH="$LAUNCH_ROOT/$VALIDATION_ID"
mkdir -p "$LAUNCH_ROOT"
if [[ -e "$OUTPUT" || -L "$OUTPUT" ]] || ! mkdir -m 700 "$LAUNCH"; then
  echo "refusing to reuse full-CFM validation output or launch" >&2
  exit 3
fi
printf '{"status":"gpu_admission","validation_id":"%s","code_commit":"%s"}\n' \
  "$VALIDATION_ID" "$CURRENT_COMMIT" > "$LAUNCH/status.json"
finish() {
  code=$?
  trap - EXIT
  printf '{"validation_id":"%s","controller_exit_code":%d,"finished_at":"%s","code_commit":"%s"}\n' \
    "$VALIDATION_ID" "$code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$CURRENT_COMMIT" \
    > "$LAUNCH/exit.json"
  if [[ "$code" -eq 0 ]]; then status=complete; else status=failed; fi
  printf '{"status":"%s","validation_id":"%s","exit_code":%d,"code_commit":"%s"}\n' \
    "$status" "$VALIDATION_ID" "$code" "$CURRENT_COMMIT" > "$LAUNCH/status.json"
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

printf '{"status":"running","validation_id":"%s","started_at":"%s","code_commit":"%s","gpu_uuid":"%s"}\n' \
  "$VALIDATION_ID" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$CURRENT_COMMIT" "$GPU_UUID" \
  > "$LAUNCH/status.json"
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export CLEARML_REQUIRE_ONLINE=1
unset CLEARML_OFFLINE_MODE
export PYTHONUNBUFFERED=1
# The checkpoint gate touches thousands of small tensors.  One CPU thread is
# markedly faster than repeatedly starting a six-thread reduction; sampling is
# GPU-bound and does not benefit from the larger pools.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
timeout --foreground --signal=TERM --kill-after=120s 21600s \
  python -m assim_lib.direct_dynamics_geometry_full_cfm_validation \
  --config "$CONFIG" --output "$OUTPUT"
