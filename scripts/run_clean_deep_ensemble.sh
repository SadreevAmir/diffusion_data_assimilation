#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-${MODE:-}}"
REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
SOURCE_EXPERIMENT="${SOURCE_EXPERIMENT:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "$MODE" != "validation_clean_checkpoint_deep_ensemble_sampling" ]]; then
  echo "[clean-deep-ensemble] untrusted mode: $MODE" >&2
  exit 2
fi
if [[ -z "$OUTPUT_DIR" || -z "$SOURCE_EXPERIMENT" ]]; then
  echo "[clean-deep-ensemble] OUTPUT_DIR and SOURCE_EXPERIMENT are required" >&2
  exit 2
fi

cd "$REPO_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}"
exec "$PYTHON_BIN" -m assim_lib.deep_ensemble \
  --output-dir "$OUTPUT_DIR" \
  --source-experiment "$SOURCE_EXPERIMENT"
