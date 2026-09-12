#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RUN_ID="${CASCADE_E2E_RUN_ID:?CASCADE_E2E_RUN_ID is required}"
CONFIG="${CASCADE_E2E_CONFIG:-config/experiments/evaluate_direct_dynamics_cascade_e2e_v1.json}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ ]]; then
  echo "unsafe CASCADE_E2E_RUN_ID" >&2
  exit 2
fi
case "$CONFIG" in
  config/experiments/evaluate_direct_dynamics_cascade_e2e_v1.json|\
  config/experiments/evaluate_direct_dynamics_cascade_e2e_fine2048_v2.json|\
  config/experiments/evaluate_direct_dynamics_cascade_e2e_fine6474_v3.json|\
  config/experiments/evaluate_direct_dynamics_cascade_e2e_matched2048_v5.json|\
  config/experiments/evaluate_direct_dynamics_cascade_colored2048_rk33_v1.json|\
  config/experiments/evaluate_direct_dynamics_cascade_colored2048_rk65_v1.json|\
  config/experiments/evaluate_direct_dynamics_cascade_e2e_fine4096_v4.json|\
  config/experiments/evaluate_direct_dynamics_cascade_proper_refinement_v1.json|\
  config/experiments/evaluate_direct_dynamics_cascade_ordinary64_colored2048_control_v1.json|\
  config/experiments/evaluate_direct_dynamics_cascade_multiscale_refinement_v1.json|\
  config/experiments/evaluate_direct_dynamics_cascade_proper_refinement_confirmation_v1.json|\
  config/experiments/evaluate_direct_dynamics_cascade_threshold_weighted_refinement_v1.json|\
  config/experiments/evaluate_direct_dynamics_cascade_colored_gate_v1.json|\
  config/experiments/evaluate_direct_dynamics_cascade_sit_support_confirmation_v1.json|\
  config/experiments/evaluate_direct_dynamics_cascade_sit_support_final_test_2023_v1.json) ;;
  *)
    echo "unsupported CASCADE_E2E_CONFIG" >&2
    exit 2
    ;;
esac
MODULE="assim_lib.direct_dynamics_cascade_e2e_evaluation"
if [[ "$CONFIG" == "config/experiments/evaluate_direct_dynamics_cascade_colored_gate_v1.json" ]]; then
  MODULE="assim_lib.direct_dynamics_cascade_colored_gate"
fi
case "$CONFIG" in
  config/experiments/evaluate_direct_dynamics_cascade_proper_refinement_v1.json|\
  config/experiments/evaluate_direct_dynamics_cascade_ordinary64_colored2048_control_v1.json|\
  config/experiments/evaluate_direct_dynamics_cascade_multiscale_refinement_v1.json|\
  config/experiments/evaluate_direct_dynamics_cascade_proper_refinement_confirmation_v1.json|\
  config/experiments/evaluate_direct_dynamics_cascade_threshold_weighted_refinement_v1.json|\
  config/experiments/evaluate_direct_dynamics_cascade_sit_support_confirmation_v1.json|\
  config/experiments/evaluate_direct_dynamics_cascade_sit_support_final_test_2023_v1.json)
    MODULE="assim_lib.direct_dynamics_cascade_proper_refinement_evaluation"
    ;;
esac
if [[ "$CONFIG" == "config/experiments/evaluate_direct_dynamics_cascade_sit_support_final_test_2023_v1.json" && \
      "${FINAL_TEST_2023_AUTHORIZATION:-}" != "EXPLICIT_USER_APPROVAL_RECORDED" ]]; then
  echo "frozen final test remains sealed without explicit user approval" >&2
  exit 14
fi
if [[ "$CONFIG" == "config/experiments/evaluate_direct_dynamics_cascade_sit_support_final_test_2023_v1.json" && \
      ( -e "/home/autoresearch_results/direct_dynamics_cascade_v2/launches/.sit_support_final_test_2023_v1.consumed.json" || \
        -L "/home/autoresearch_results/direct_dynamics_cascade_v2/launches/.sit_support_final_test_2023_v1.consumed.json" ) ]]; then
  echo "frozen final test has already been consumed" >&2
  exit 15
fi
if [[ "${CASCADE_E2E_RESOLVE_ONLY:-0}" == "1" ]]; then
  printf '%s\n' "$MODULE"
  exit 0
fi
RESULT_ROOT="/home/autoresearch_results/direct_dynamics_cascade_v2"
STATUS_DIR="$RESULT_ROOT/launches/$RUN_ID"
mkdir -p "$RESULT_ROOT/launches"
if ! mkdir -m 700 "$STATUS_DIR"; then
  echo "refusing to reuse cascade evaluation status" >&2
  exit 3
fi
OUTPUT="$RESULT_ROOT/e2e_evaluation/$RUN_ID"
if [[ -e "$OUTPUT" || -L "$OUTPUT" ]]; then
  echo "refusing to reuse cascade evaluation output" >&2
  exit 4
fi
started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git rev-parse HEAD)"
printf '{"status":"admission_check","run_id":"%s","started_at":"%s","code_commit":"%s"}\n' \
  "$RUN_ID" "$started" "$commit" > "$STATUS_DIR/status.json"
finish() {
  code=$?
  printf '{"run_id":"%s","controller_exit_code":%d,"finished_at":"%s","code_commit":"%s"}\n' \
    "$RUN_ID" "$code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$commit" > "$STATUS_DIR/exit.json"
  exit "$code"
}
trap finish EXIT

exec 9>"/home/autoresearch_results/direct_dynamics_all_hours_v1/launches/.gpu_job.lock"
if ! flock -n 9; then
  echo "another project launch owns the GPU lock" >&2
  exit 5
fi
GPU_UUID="$(scripts/require_single_gpu_uuid.sh)"
readonly GPU_UUID
read_gpu() {
  local observation
  if ! observation="$(nvidia-smi --id=0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"; then
    return 1
  fi
  [[ "$observation" =~ ^[0-9]+,[0-9]+$ ]] || return 1
  GPU_MEMORY_USED="${observation%%,*}"
  GPU_UTILIZATION="${observation##*,}"
}
read_gpu || exit 6
initial_memory="$GPU_MEMORY_USED"
if [[ "$GPU_MEMORY_USED" -gt 1024 ]]; then
  for sample in {1..11}; do
    read_gpu || exit 7
    if [[ "$GPU_UTILIZATION" -ge 5 ]]; then
      echo "GPU busy: utilization=${GPU_UTILIZATION}%" >&2
      exit 8
    fi
    [[ "$sample" -eq 11 ]] || sleep 30
  done
fi
read_gpu || exit 9
if [[ "$initial_memory" -le 1024 && "$GPU_MEMORY_USED" -gt 1024 ]]; then
  for sample in {1..11}; do
    read_gpu || exit 10
    if [[ "$GPU_UTILIZATION" -ge 5 ]]; then
      echo "GPU became busy: utilization=${GPU_UTILIZATION}%" >&2
      exit 11
    fi
    [[ "$sample" -eq 11 ]] || sleep 30
  done
  read_gpu || exit 12
fi
if [[ "$GPU_MEMORY_USED" -gt 1024 && "$GPU_UTILIZATION" -ge 5 ]]; then
  echo "GPU busy at final observation" >&2
  exit 13
fi
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export CASCADE_E2E_STATUS_PATH="$STATUS_DIR/status.json"
export CASCADE_COLORED_GATE_STATUS_PATH="$STATUS_DIR/status.json"
export CLEARML_REQUIRE_ONLINE=1
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6
timeout --foreground --signal=TERM --kill-after=60s 2640s python -m "$MODULE" \
  --config "$CONFIG" \
  --output "$OUTPUT"
