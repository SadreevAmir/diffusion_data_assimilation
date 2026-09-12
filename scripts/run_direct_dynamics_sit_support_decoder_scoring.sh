#!/usr/bin/env bash
set -euo pipefail

CONFIG="${SIT_SUPPORT_DECODER_SCORING_CONFIG:-config/experiments/audit_direct_dynamics_sit_support_decoder_scoring_v1.json}"
OUTPUT="${SIT_SUPPORT_DECODER_SCORING_OUTPUT:?SIT_SUPPORT_DECODER_SCORING_OUTPUT is required}"
EXPECTED="config/experiments/audit_direct_dynamics_sit_support_decoder_scoring_v1.json"
if [[ "$CONFIG" != "$EXPECTED" ]]; then
  echo "refusing unreviewed support decoder scoring config: $CONFIG" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=""
export CLEARML_REQUIRE_ONLINE=1
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6
timeout --signal=TERM --kill-after=30s 45m python -m assim_lib.direct_dynamics_sit_support_decoder_scoring --config "$CONFIG" --output "$OUTPUT"
