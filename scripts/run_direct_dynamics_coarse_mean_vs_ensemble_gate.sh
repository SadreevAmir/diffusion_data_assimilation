#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RUN_ID="${COARSE_MEAN_VS_ENSEMBLE_RUN_ID:?COARSE_MEAN_VS_ENSEMBLE_RUN_ID is required}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ ]]; then exit 2; fi
OUTPUT="/home/autoresearch_results/direct_dynamics_cascade_v2/coarse_mean_vs_ensemble/$RUN_ID"
if [[ -e "$OUTPUT" || -L "$OUTPUT" ]]; then exit 3; fi
export CUDA_VISIBLE_DEVICES=""
export CLEARML_REQUIRE_ONLINE=1
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6
timeout --signal=TERM --kill-after=30s 3570s python -m \
  assim_lib.direct_dynamics_coarse_mean_vs_ensemble_gate \
  --config config/experiments/audit_direct_dynamics_coarse_mean_vs_ensemble_v1.json \
  --output "$OUTPUT"
