#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-${MODE:-}}"
REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
SOURCE_EXPERIMENT="${SOURCE_EXPERIMENT:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_DEVICE_VALUE="${CUDA_DEVICE:-0}"

if [[ "$MODE" != "validation_swag_tail_weight_posterior_sampling" ]]; then
  echo "[swag-tail] untrusted mode: $MODE" >&2
  exit 2
fi
if [[ -z "$OUTPUT_DIR" || -z "$SOURCE_EXPERIMENT" ]]; then
  echo "[swag-tail] OUTPUT_DIR and SOURCE_EXPERIMENT are required" >&2
  exit 2
fi
if [[ ! "$CUDA_DEVICE_VALUE" =~ ^[0-9]+$ ]]; then
  echo "[swag-tail] CUDA_DEVICE must name exactly one numeric GPU" >&2
  exit 2
fi

cd "$REPO_DIR"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE_VALUE"
exec "$PYTHON_BIN" -m assim_lib.swag_tail_ensemble \
  --output-dir "$OUTPUT_DIR" \
  --source-experiment "$SOURCE_EXPERIMENT"
