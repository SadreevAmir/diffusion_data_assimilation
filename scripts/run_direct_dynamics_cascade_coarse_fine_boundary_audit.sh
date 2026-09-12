#!/usr/bin/env bash
set -euo pipefail

CONFIG="${COARSE_FINE_AUDIT_CONFIG:-config/experiments/audit_direct_dynamics_cascade_coarse_fine_boundary_v1.json}"
OUTPUT="${COARSE_FINE_AUDIT_OUTPUT:?COARSE_FINE_AUDIT_OUTPUT must be an explicit new JSON path}"
EXPECTED_CONFIG="config/experiments/audit_direct_dynamics_cascade_coarse_fine_boundary_v1.json"
if [[ "$CONFIG" != "$EXPECTED_CONFIG" ]]; then
  echo "unreviewed coarse/fine audit config: $CONFIG" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=""
export CLEARML_REQUIRE_ONLINE=1
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6

set +e
timeout --signal=TERM --kill-after=30s 30m \
python -m assim_lib.direct_dynamics_cascade_coarse_fine_boundary_audit \
  --config "$CONFIG" \
  --output "$OUTPUT"
exit_code=$?
set -e
exit "$exit_code"
