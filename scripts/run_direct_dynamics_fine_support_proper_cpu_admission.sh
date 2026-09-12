#!/usr/bin/env bash
set -euo pipefail

CONFIG="${FINE_SUPPORT_ADMISSION_CONFIG:-config/experiments/admit_direct_dynamics_fine_support_proper_cpu_v1.json}"
OUTPUT="${FINE_SUPPORT_ADMISSION_OUTPUT:?FINE_SUPPORT_ADMISSION_OUTPUT must be an explicit new JSON path}"
EXPECTED_CONFIG="config/experiments/admit_direct_dynamics_fine_support_proper_cpu_v1.json"
if [[ "$CONFIG" != "$EXPECTED_CONFIG" ]]; then
  echo "unreviewed fine support-aware CPU admission config: $CONFIG" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=""
export CLEARML_REQUIRE_ONLINE=1
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6
set +e
timeout --signal=TERM --kill-after=30s 10m \
python -m assim_lib.direct_dynamics_fine_support_proper_admission \
  --config "$CONFIG" --output "$OUTPUT"
exit_code=$?
set -e
exit "$exit_code"
