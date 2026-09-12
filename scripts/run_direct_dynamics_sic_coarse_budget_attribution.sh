#!/usr/bin/env bash
set -euo pipefail

CONFIG="${SIC_BUDGET_ATTRIBUTION_CONFIG:-config/experiments/audit_direct_dynamics_sic_coarse_budget_attribution_v1.json}"
OUTPUT="${SIC_BUDGET_ATTRIBUTION_OUTPUT:?SIC_BUDGET_ATTRIBUTION_OUTPUT must be an explicit new JSON path}"
EXPECTED_CONFIG="config/experiments/audit_direct_dynamics_sic_coarse_budget_attribution_v1.json"
if [[ "$CONFIG" != "$EXPECTED_CONFIG" ]]; then
  echo "unreviewed SIC coarse-budget attribution config: $CONFIG" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=""
export CLEARML_REQUIRE_ONLINE=1
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6
set +e
timeout --signal=TERM --kill-after=30s 20m \
python -m assim_lib.direct_dynamics_sic_coarse_budget_attribution \
  --config "$CONFIG" --output "$OUTPUT"
exit_code=$?
set -e
exit "$exit_code"
