#!/usr/bin/env bash
set -euo pipefail

CONFIG="${SIC_SUPPORT_SCORING_CONFIG:-config/experiments/audit_direct_dynamics_sic_support_decoder_scoring_v1.json}"
OUTPUT="${SIC_SUPPORT_SCORING_OUTPUT:?SIC_SUPPORT_SCORING_OUTPUT must be an explicit new JSON path}"
EXPECTED_CONFIG="config/experiments/audit_direct_dynamics_sic_support_decoder_scoring_v1.json"
if [[ "$CONFIG" != "$EXPECTED_CONFIG" ]]; then
  echo "unreviewed SIC support decoder scoring config: $CONFIG" >&2
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
python -m assim_lib.direct_dynamics_sic_support_decoder_scoring \
  --config "$CONFIG" --output "$OUTPUT"
exit_code=$?
set -e
exit "$exit_code"
