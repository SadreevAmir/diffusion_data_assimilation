#!/usr/bin/env bash
set -euo pipefail

CALLER_DIR="$PWD"
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"

CONFIG_PATH="${1:-$REPO_ROOT/config/experiments/occurrence_intensity_e1_data_audit.json}"
OUTPUT_DIR="${2:-/tmp/occurrence_intensity_e1_data_audit}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

case "$CONFIG_PATH" in
  /*) ;;
  *) CONFIG_PATH="$CALLER_DIR/$CONFIG_PATH" ;;
esac
case "$OUTPUT_DIR" in
  /*) ;;
  *) OUTPUT_DIR="$CALLER_DIR/$OUTPUT_DIR" ;;
esac

cd "$REPO_ROOT"
"$PYTHON_BIN" -m assim_lib.occurrence_intensity_e1_data_audit_cli "$CONFIG_PATH" "$OUTPUT_DIR"
"$PYTHON_BIN" paper/validate_occurrence_intensity_e1_data_audit.py "$CONFIG_PATH" "$OUTPUT_DIR"
