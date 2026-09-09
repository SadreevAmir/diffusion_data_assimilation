#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RUN_ID="${CENSORED_JOINT_MULTISCALE_ID:?CENSORED_JOINT_MULTISCALE_ID is required}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ ]]; then
  echo "unsafe CENSORED_JOINT_MULTISCALE_ID" >&2
  exit 2
fi
RESULT_ROOT="/home/autoresearch_results/direct_dynamics_all_hours_v1"
OUTPUT="$RESULT_ROOT/censored_joint_multiscale/$RUN_ID"
STATUS_DIR="$RESULT_ROOT/censored_joint_multiscale_launches/$RUN_ID"
BASELINE_ZERO="$RESULT_ROOT/censored_joint_heldout/heldout_energy_118d003_20260909T193003Z/model_step_0000.pth"
BASELINE_ZERO_SHA="8a89a40bb335738124a26e823166bca85a8e288705a48ee30bfe1106337c1802"
mkdir -p "$RESULT_ROOT/censored_joint_multiscale" \
  "$RESULT_ROOT/censored_joint_multiscale_launches" "$RESULT_ROOT/launches"
if ! mkdir -m 700 "$STATUS_DIR"; then
  echo "refusing to reuse censored joint multiscale status directory" >&2
  exit 3
fi
exec 9>"$RESULT_ROOT/launches/.gpu_job.lock"
if ! flock -n 9; then
  echo "another direct-dynamics job owns the GPU lock" >&2
  exit 4
fi
GPU_COUNT="$(nvidia-smi --query-gpu=count --format=csv,noheader | wc -l | tr -d ' ')"
GPU_MEMORY_USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
if [[ "$GPU_COUNT" != "1" || "$GPU_MEMORY_USED" -gt 1024 ]]; then
  echo "GPU admission failed: visible=$GPU_COUNT memory_used_mib=$GPU_MEMORY_USED" >&2
  exit 5
fi
if [[ "$(sha256sum "$BASELINE_ZERO" | cut -d' ' -f1)" != "$BASELINE_ZERO_SHA" ]]; then
  echo "frozen baseline step-zero checkpoint differs" >&2
  exit 6
fi
started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '{"status":"running","run_id":"%s","started_at":"%s"}\n' \
  "$RUN_ID" "$started" > "$STATUS_DIR/status.json"
finish() {
  code=$?
  finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '{"status":"finished","run_id":"%s","exit_code":%d,"finished_at":"%s"}\n' \
    "$RUN_ID" "$code" "$finished" > "$STATUS_DIR/status.json"
  exit "$code"
}
trap finish EXIT
export CLEARML_REQUIRE_ONLINE=1
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6
timeout --signal=TERM --kill-after=120s 7080s \
  python -m assim_lib.censored_joint_energy_heldout \
  --config config/experiments/train_direct_dynamics_all_hours_v1.json \
  --output "$OUTPUT" \
  --objective multiscale \
  --step-zero-checkpoint "$BASELINE_ZERO" \
  --expected-step-zero-sha256 "$BASELINE_ZERO_SHA"
