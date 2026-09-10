#!/usr/bin/env bash
set -euo pipefail

CONFIG="${RESIDUAL_LAW_AUDIT_CONFIG:-config/experiments/audit_direct_dynamics_cascade_residual_law_v1.json}"
OUTPUT="${RESIDUAL_LAW_AUDIT_OUTPUT:?RESIDUAL_LAW_AUDIT_OUTPUT must be an explicit new JSON path}"

python -m assim_lib.direct_dynamics_cascade_residual_law_audit \
  --config "$CONFIG" \
  --output "$OUTPUT"
