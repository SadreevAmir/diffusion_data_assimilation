#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
EXPERIMENT_ID="${DIRECT_DYNAMICS_AFFINE_ID:?DIRECT_DYNAMICS_AFFINE_ID is required}"
if [[ ! "$EXPERIMENT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ ]]; then
  echo "unsafe DIRECT_DYNAMICS_AFFINE_ID" >&2
  exit 2
fi

RESULT_ROOT="/home/autoresearch_results/direct_dynamics_all_hours_v1"
OUTPUT="$RESULT_ROOT/affine_calibration/$EXPERIMENT_ID"
STATUS_DIR="$RESULT_ROOT/affine_calibration_launches/$EXPERIMENT_ID"
mkdir -p "$RESULT_ROOT/affine_calibration" "$RESULT_ROOT/affine_calibration_launches"
if ! mkdir -m 700 "$STATUS_DIR"; then
  echo "refusing to reuse affine-calibration status directory" >&2
  exit 3
fi
exec 9>"$RESULT_ROOT/launches/.gpu_job.lock"
if ! flock -n 9; then
  echo "another direct-dynamics job owns the GPU lock" >&2
  exit 4
fi
GPU_COUNT="$(nvidia-smi --query-gpu=count --format=csv,noheader | wc -l | tr -d ' ')"
GPU_MEMORY_USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
if [[ "$GPU_COUNT" != "1" || "$GPU_MEMORY_USED" -gt 1024 ]]; then
  echo "GPU admission failed: visible=$GPU_COUNT memory_used_mib=$GPU_MEMORY_USED" >&2
  exit 5
fi
started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '{"status":"running","experiment_id":"%s","started_at":"%s"}\n' "$EXPERIMENT_ID" "$started" > "$STATUS_DIR/status.json"
finish() {
  code=$?
  finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '{"status":"finished","experiment_id":"%s","exit_code":%d,"finished_at":"%s"}\n' "$EXPERIMENT_ID" "$code" "$finished" > "$STATUS_DIR/status.json"
  exit "$code"
}
trap finish EXIT

export CLEARML_REQUIRE_ONLINE=1
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6
timeout --signal=TERM --kill-after=2m 14280s python -m assim_lib.direct_dynamics_affine_calibration \
  --run-dir "$RESULT_ROOT/training/seed1701-night_20260909_v1" \
  --output "$OUTPUT"
