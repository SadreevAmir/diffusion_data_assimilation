#!/usr/bin/env bash
set -euo pipefail

config="${1:?config path is required}"
mode="${2:?mode preflight|train is required}"
output_dir="${3:?new output directory is required}"

if [[ "$mode" != "preflight" && "$mode" != "train" ]]; then
  echo "mode must be preflight or train" >&2
  exit 2
fi
if [[ -e "$output_dir" ]]; then
  echo "refusing to reuse output directory: $output_dir" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
python -m assim_lib.direct_dynamics_geometry_cfm_training \
  --config "$config" \
  --mode "$mode" \
  --output-dir "$output_dir"
