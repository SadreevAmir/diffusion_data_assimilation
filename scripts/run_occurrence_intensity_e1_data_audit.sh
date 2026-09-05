#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-config/experiments/occurrence_intensity_e1_data_audit.json}"
OUTPUT_DIR="${2:-/tmp/occurrence_intensity_e1_data_audit}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" -m assim_lib.occurrence_intensity_e1_data_audit_cli "$CONFIG_PATH" "$OUTPUT_DIR"
"$PYTHON_BIN" paper/validate_occurrence_intensity_e1_data_audit.py "$CONFIG_PATH" "$OUTPUT_DIR"
