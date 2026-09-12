#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ID="${SIC_COARSE_BUDGET_RUN_ID:?SIC_COARSE_BUDGET_RUN_ID is required}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ ]]; then
  echo "unsafe SIC coarse-budget probe identifier" >&2
  exit 2
fi
OUTPUT_ROOT="/home/autoresearch_results/direct_dynamics_cascade_v2/sic_coarse_budget_sensitivity"
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_ID"
if [[ -e "$OUTPUT_DIR" || -L "$OUTPUT_DIR" ]]; then
  echo "refusing to reuse SIC coarse-budget probe output" >&2
  exit 3
fi
mkdir -p "$OUTPUT_ROOT"
mkdir -m 700 "$OUTPUT_DIR"
export CUDA_VISIBLE_DEVICES=""
export CLEARML_REQUIRE_ONLINE=1
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6
timeout --foreground --signal=TERM --kill-after=30s 1200s \
  python -m assim_lib.direct_dynamics_sic_coarse_budget_sensitivity \
  --config config/experiments/audit_sic_coarse_budget_sensitivity_v1.json \
  --output "$OUTPUT_DIR/status.json"
