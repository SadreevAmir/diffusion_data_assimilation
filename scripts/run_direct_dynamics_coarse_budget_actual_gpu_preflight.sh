#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_DIR" >&2
  exit 2
fi

python -m assim_lib.direct_dynamics_coarse_budget_actual_gpu_preflight \
  --config config/experiments/preflight_direct_dynamics_coarse_budget_actual_gpu_v1.json \
  --output "$1"
