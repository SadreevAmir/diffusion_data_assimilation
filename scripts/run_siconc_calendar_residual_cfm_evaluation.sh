#!/usr/bin/env bash
set -Eeuo pipefail

MODE="${1:-${MODE:-}}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
SOURCE_EXPERIMENT="${SOURCE_EXPERIMENT:?SOURCE_EXPERIMENT is required}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

if [[ "$MODE" != "validation_siconc_calendar_residual_cfm_sampling" ]]; then
  echo "[calendar-residual-cfm] untrusted mode: $MODE" >&2
  exit 2
fi
if [[ "$SOURCE_EXPERIMENT" != "siconc_calendar_residual_cfm_training_retry1" ]]; then
  echo "[calendar-residual-cfm] unexpected source experiment" >&2
  exit 2
fi
if [[ ! "$CUDA_DEVICE" =~ ^[0-9]+$ ]]; then
  echo "[calendar-residual-cfm] exactly one numeric CUDA device is required" >&2
  exit 2
fi

cd "$REPO_DIR"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export PYTHONDONTWRITEBYTECODE=1
exec "$PYTHON_BIN" -m assim_lib.calendar_residual_cfm_sampling \
  --output-dir "$OUTPUT_DIR" \
  --source-root "$RUN_DIR"
