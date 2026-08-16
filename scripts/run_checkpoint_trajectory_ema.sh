#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-${MODE:-}}"
REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
SOURCE_EXPERIMENT="${SOURCE_EXPERIMENT:-}"
RUN_DIR="${RUN_DIR:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "$MODE" != "validation_checkpoint_trajectory_ema_sampling" ]]; then
  echo "[checkpoint-trajectory-ema] untrusted mode: $MODE" >&2
  exit 2
fi
if [[ -z "$OUTPUT_DIR" || -z "$SOURCE_EXPERIMENT" || -z "$RUN_DIR" ]]; then
  echo "[checkpoint-trajectory-ema] OUTPUT_DIR, SOURCE_EXPERIMENT and RUN_DIR are required" >&2
  exit 2
fi

cd "$REPO_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}"
exec "$PYTHON_BIN" -m assim_lib.snapshot_ensemble \
  --output-dir "$OUTPUT_DIR" \
  --source-experiment "$SOURCE_EXPERIMENT" \
  --run-dir "$RUN_DIR"
