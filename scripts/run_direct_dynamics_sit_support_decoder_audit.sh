#!/usr/bin/env bash
set -euo pipefail
CONFIG="${SIT_SUPPORT_DECODER_CONFIG:-config/experiments/audit_direct_dynamics_sit_support_decoder_v1.json}"
OUTPUT="${SIT_SUPPORT_DECODER_OUTPUT:?SIT_SUPPORT_DECODER_OUTPUT must be an explicit new JSON path}"
EXPECTED="config/experiments/audit_direct_dynamics_sit_support_decoder_v1.json"
if [[ "$CONFIG" != "$EXPECTED" ]]; then echo "unreviewed SIT support decoder config: $CONFIG" >&2; exit 2; fi
export CUDA_VISIBLE_DEVICES="" CLEARML_REQUIRE_ONLINE=1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6 MPLCONFIGDIR="/tmp/mpl-support-decoder"
set +e
timeout --signal=TERM --kill-after=30s 30m python -m assim_lib.direct_dynamics_sit_support_decoder_audit --config "$CONFIG" --output "$OUTPUT"
exit_code=$?
set -e
exit "$exit_code"
