#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ID="${DIRECT_DYNAMICS_LAUNCH_ID:?DIRECT_DYNAMICS_LAUNCH_ID is required}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ ]]; then
  echo "unsafe DIRECT_DYNAMICS_LAUNCH_ID" >&2
  exit 2
fi

STATUS_ROOT="/home/autoresearch_results/direct_dynamics_all_hours_v1/launches"
STATUS_DIR="$STATUS_ROOT/$RUN_ID"
mkdir -p "$STATUS_ROOT"
if ! mkdir -m 700 "$STATUS_DIR"; then
  echo "refusing to reuse launch status directory: $STATUS_DIR" >&2
  exit 3
fi
exec 9>"$STATUS_ROOT/.gpu_job.lock"
if ! flock -n 9; then
  echo "another direct-dynamics launch owns the GPU lock" >&2
  exit 4
fi

GPU_COUNT="$(nvidia-smi --query-gpu=count --format=csv,noheader | wc -l | tr -d ' ')"
GPU_MEMORY_USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
if [[ "$GPU_COUNT" != "1" || "$GPU_MEMORY_USED" -gt 1024 ]]; then
  echo "GPU admission failed: visible=$GPU_COUNT memory_used_mib=$GPU_MEMORY_USED" >&2
  exit 5
fi

STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '{"status":"running","launch_id":"%s","started_at":"%s"}\n' "$RUN_ID" "$STARTED_AT" > "$STATUS_DIR/status.json"
finish() {
  code=$?
  finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '{"status":"finished","launch_id":"%s","exit_code":%d,"finished_at":"%s"}\n' "$RUN_ID" "$code" "$finished_at" > "$STATUS_DIR/status.json"
  exit "$code"
}
trap finish EXIT

export CLEARML_REQUIRE_ONLINE=1
timeout --signal=TERM --kill-after=5m 20h python -m assim_lib.direct_dynamics_training \
  --config config/experiments/train_direct_dynamics_all_hours_v1.json
