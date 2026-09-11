#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RUN_ID="${COARSE_COVARIANCE_GATE_RUN_ID:?COARSE_COVARIANCE_GATE_RUN_ID is required}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ ]]; then
  echo "unsafe COARSE_COVARIANCE_GATE_RUN_ID" >&2
  exit 2
fi
OUTPUT="/home/autoresearch_results/direct_dynamics_cascade_v2/coarse_covariance/$RUN_ID/coarse_covariance_gate.json"
if [[ -e "$OUTPUT" || -L "$OUTPUT" ]]; then
  echo "refusing to reuse coarse covariance gate output" >&2
  exit 3
fi
export CUDA_VISIBLE_DEVICES=""
export CLEARML_REQUIRE_ONLINE=1
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6
timeout --signal=TERM --kill-after=2m 21480s python -m \
  assim_lib.direct_dynamics_coarse_covariance_gate \
  --config config/experiments/audit_direct_dynamics_coarse_covariance_gate_v1.json \
  --output "$OUTPUT"
