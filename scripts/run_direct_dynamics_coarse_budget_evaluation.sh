#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 CONFIG_PATH OUTPUT_DIR" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CLEARML_REQUIRE_ONLINE=1

python -m assim_lib.direct_dynamics_coarse_budget_evaluation \
  --config "$1" \
  --output "$2"
