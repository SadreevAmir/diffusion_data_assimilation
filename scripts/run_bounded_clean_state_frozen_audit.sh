#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
AUDIT_ID="${BOUNDED_FROZEN_AUDIT_ID:?BOUNDED_FROZEN_AUDIT_ID is required}"
if [[ ! "$AUDIT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ ]]; then
  echo "unsafe BOUNDED_FROZEN_AUDIT_ID" >&2
  exit 2
fi
RESULT_ROOT="/home/autoresearch_results/direct_dynamics_all_hours_v1"
OUTPUT="$RESULT_ROOT/bounded_frozen_audit/$AUDIT_ID"
STATUS_DIR="$RESULT_ROOT/bounded_frozen_audit_launches/$AUDIT_ID"
mkdir -p "$RESULT_ROOT/bounded_frozen_audit" "$RESULT_ROOT/bounded_frozen_audit_launches"
if ! mkdir -m 700 "$STATUS_DIR"; then
  echo "refusing to reuse bounded frozen audit status directory" >&2
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
started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '{"status":"running","audit_id":"%s","started_at":"%s"}\n' \
  "$AUDIT_ID" "$started" > "$STATUS_DIR/status.json"
finish() {
  code=$?
  finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '{"status":"finished","audit_id":"%s","exit_code":%d,"finished_at":"%s"}\n' \
    "$AUDIT_ID" "$code" "$finished" > "$STATUS_DIR/status.json"
  exit "$code"
}
trap finish EXIT
export CLEARML_REQUIRE_ONLINE=1
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6
timeout --signal=TERM --kill-after=60s 3540s python -m assim_lib.bounded_clean_state_frozen_audit \
  --config config/experiments/train_direct_dynamics_all_hours_v1.json \
  --output "$OUTPUT"
