#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RUN_ID="${SIT_SUPPORT_ATTRIBUTION_RUN_ID:?SIT_SUPPORT_ATTRIBUTION_RUN_ID is required}"
CONFIG="${SIT_SUPPORT_ATTRIBUTION_CONFIG:-config/experiments/audit_direct_dynamics_sit_support_attribution_v1.json}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ ]]; then
  echo "unsafe SIT support attribution run id" >&2
  exit 2
fi
if [[ "$CONFIG" != "config/experiments/audit_direct_dynamics_sit_support_attribution_v1.json" ]]; then
  echo "unsupported SIT support attribution config" >&2
  exit 2
fi
OUTPUT_ROOT="/home/autoresearch_results/direct_dynamics_cascade_v2/sit_support_attribution/$RUN_ID"
if [[ -e "$OUTPUT_ROOT" || -L "$OUTPUT_ROOT" ]]; then
  echo "refusing to reuse SIT support attribution output" >&2
  exit 3
fi
export CUDA_VISIBLE_DEVICES=""
export CLEARML_REQUIRE_ONLINE=1
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6
timeout --foreground --signal=TERM --kill-after=30s 1200s python -m \
  assim_lib.direct_dynamics_sit_support_attribution \
  --config "$CONFIG" \
  --output "$OUTPUT_ROOT/status.json"
