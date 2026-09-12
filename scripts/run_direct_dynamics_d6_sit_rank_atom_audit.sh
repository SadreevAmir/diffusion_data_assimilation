#!/usr/bin/env bash
set -euo pipefail

CONFIG="${D6_SIT_RANK_ATOM_CONFIG:-config/experiments/audit_direct_dynamics_d6_sit_rank_atom_v1.json}"
OUTPUT="${D6_SIT_RANK_ATOM_OUTPUT:?D6_SIT_RANK_ATOM_OUTPUT must be an explicit new JSON path}"
EXPECTED_CONFIG="config/experiments/audit_direct_dynamics_d6_sit_rank_atom_v1.json"
if [[ "$CONFIG" != "$EXPECTED_CONFIG" ]]; then
  echo "unreviewed d6 SIT rank-atom audit config: $CONFIG" >&2
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
python -m assim_lib.direct_dynamics_d6_sit_rank_atom_audit \
  --config "$CONFIG" --output "$OUTPUT"
exit_code=$?
set -e
exit "$exit_code"
