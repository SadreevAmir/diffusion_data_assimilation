#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_DIR" >&2
  exit 2
fi

output_dir=$1
mkdir -p "$output_dir"
python3 -m assim_lib.two_stage_native_speed_admission --output-dir "$output_dir"
