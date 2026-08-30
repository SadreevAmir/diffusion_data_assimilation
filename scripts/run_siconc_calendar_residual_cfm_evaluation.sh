#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_dir="${1:?usage: run_siconc_calendar_residual_cfm_evaluation.sh RUN_DIR OUTPUT_DIR}"
output_dir="${2:?usage: run_siconc_calendar_residual_cfm_evaluation.sh RUN_DIR OUTPUT_DIR}"

cd "$repo_dir"
python -m assim_lib.evaluate \
  --config config/experiments/evaluate_siconc_calendar_residual_cfm.json \
  --run-dir "$run_dir" \
  --output-dir "$output_dir"
