#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TOY_ID="${CENSORED_JOINT_TOY_ID:?CENSORED_JOINT_TOY_ID is required}"
if [[ ! "$TOY_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ ]]; then
  echo "unsafe CENSORED_JOINT_TOY_ID" >&2
  exit 2
fi
RESULT_ROOT="/home/autoresearch_results/direct_dynamics_all_hours_v1"
OUTPUT="$RESULT_ROOT/censored_joint_energy_toy/$TOY_ID"
STATUS_DIR="$RESULT_ROOT/censored_joint_energy_toy_launches/$TOY_ID"
mkdir -p "$RESULT_ROOT/censored_joint_energy_toy" \
  "$RESULT_ROOT/censored_joint_energy_toy_launches" \
  "$RESULT_ROOT/launches"
if ! mkdir -m 700 "$STATUS_DIR"; then
  echo "refusing to reuse censored joint toy status directory" >&2
  exit 3
fi
exec 9>"$RESULT_ROOT/launches/.censored_joint_energy_toy_cpu.lock"
if ! flock -n 9; then
  echo "another censored joint CPU toy owns the lock" >&2
  exit 4
fi
started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '{"status":"running","toy_id":"%s","started_at":"%s"}\n' \
  "$TOY_ID" "$started" > "$STATUS_DIR/status.json"
finish() {
  code=$?
  finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '{"status":"finished","toy_id":"%s","exit_code":%d,"finished_at":"%s"}\n' \
    "$TOY_ID" "$code" "$finished" > "$STATUS_DIR/status.json"
  exit "$code"
}
trap finish EXIT
export CUDA_VISIBLE_DEVICES=""
export CLEARML_REQUIRE_ONLINE=1
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6
timeout --signal=TERM --kill-after=30s 1200s python -m assim_lib.censored_joint_energy_toy \
  --output "$OUTPUT"
