#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RUN_ID="${IDEA_F1_TRAIN_RUN_ID:?IDEA_F1_TRAIN_RUN_ID is required}"
EXPECTED_COMMIT="${IDEA_F1_TRAIN_EXPECTED_COMMIT:?IDEA_F1_TRAIN_EXPECTED_COMMIT is required}"
CONFIG="config/experiments/train_direct_dynamics_mixed_support_mask_ab_v1.json"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ ]] || exit 2
CURRENT_COMMIT="$(git rev-parse HEAD)"
[[ "$CURRENT_COMMIT" == "$EXPECTED_COMMIT" ]] || { echo "exact commit mismatch" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || { echo "clean worktree required" >&2; exit 2; }
RESULT_ROOT="/home/autoresearch_results/direct_dynamics_mixed_support_mask_ab_training_v1"
OUTPUT="$RESULT_ROOT/$RUN_ID"
LAUNCH="$RESULT_ROOT/launches/$RUN_ID"
mkdir -p "$RESULT_ROOT/launches"
if [[ -e "$OUTPUT" || -L "$OUTPUT" ]] || ! mkdir -m 700 "$LAUNCH"; then exit 3; fi
printf '{"status":"gpu_admission","run_id":"%s","code_commit":"%s"}\n' "$RUN_ID" "$CURRENT_COMMIT" > "$LAUNCH/status.json"
finish() {
  code=$?; trap - EXIT
  printf '{"run_id":"%s","controller_exit_code":%d,"finished_at":"%s","code_commit":"%s"}\n' "$RUN_ID" "$code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$CURRENT_COMMIT" > "$LAUNCH/exit.json"
  if [[ "$code" -eq 0 ]]; then status=complete; else status=failed; fi
  printf '{"status":"%s","run_id":"%s","exit_code":%d,"code_commit":"%s"}\n' "$status" "$RUN_ID" "$code" "$CURRENT_COMMIT" > "$LAUNCH/status.json"
  exit "$code"
}
trap finish EXIT
exec 9>"/home/autoresearch_results/direct_dynamics_all_hours_v1/launches/.gpu_job.lock"
flock -n 9 || { echo "shared GPU lock busy" >&2; exit 4; }
GPU_UUID="$(scripts/require_single_gpu_uuid.sh)"
[[ "$GPU_UUID" == "GPU-40ff1cbb-07cc-992e-23cb-8fe181972b76" ]] || exit 5
read_gpu() {
  observation="$(nvidia-smi --id=0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')" || return 1
  [[ "$observation" =~ ^[0-9]+,[0-9]+$ ]] || return 1
  GPU_MEMORY="${observation%%,*}"; GPU_UTILIZATION="${observation##*,}"
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
[[ "$initial_memory" -gt 1024 || "$GPU_MEMORY" -le 1024 ]] || exit 9
[[ "$GPU_MEMORY" -le 1024 || "$GPU_UTILIZATION" -lt 5 ]] || exit 10
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export IDEA_F1_TRAIN_CODE_COMMIT="$CURRENT_COMMIT"
export CLEARML_REQUIRE_ONLINE=1
unset CLEARML_OFFLINE_MODE
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
timeout --foreground --signal=TERM --kill-after=60s 7140s \
  python -m assim_lib.direct_dynamics_mixed_support_mask_training "$CONFIG" "$OUTPUT"
