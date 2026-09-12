#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RUN_ID="${SPATIAL_AUDIT_RUN_ID:?SPATIAL_AUDIT_RUN_ID is required}"
CONFIG="${SPATIAL_AUDIT_CONFIG:-config/experiments/audit_direct_dynamics_cascade_spatial_calibration_v1.json}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ ]]; then
  echo "unsafe SPATIAL_AUDIT_RUN_ID" >&2
  exit 2
fi
case "$CONFIG" in
  config/experiments/audit_direct_dynamics_cascade_spatial_calibration_v1.json|\
  config/experiments/audit_direct_dynamics_cascade_colored2048_stage_v1.json) ;;
  *) echo "unsupported spatial calibration audit config" >&2; exit 2 ;;
esac
OUTPUT="/home/autoresearch_results/direct_dynamics_cascade_v2/spatial_calibration/$RUN_ID"
if [[ -e "$OUTPUT" || -L "$OUTPUT" ]]; then
  echo "refusing to reuse spatial calibration output" >&2
  exit 3
fi
exec 9>"/home/autoresearch_results/direct_dynamics_all_hours_v1/launches/.gpu_job.lock"
if ! flock -n 9; then
  echo "another project launch owns the GPU lock" >&2
  exit 4
fi
export CLEARML_REQUIRE_ONLINE=1
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6
timeout --signal=TERM --kill-after=2m 14280s python -m \
  assim_lib.direct_dynamics_cascade_spatial_calibration_audit \
  --config "$CONFIG" \
  --output "$OUTPUT"
