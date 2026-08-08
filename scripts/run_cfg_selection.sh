#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
RUN_DIR="${1:-${RUN_DIR:-}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
CONFIG="${CONFIG:-config/experiments/select_cfg_balance_monthly_validation_then_test.json}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/evaluation/cfg_selection_${RUN_STAMP}}"

if [[ -z "$RUN_DIR" ]]; then
  echo "Usage: RUN_DIR=/path/to/run $0"
  echo "   or: $0 /path/to/run"
  exit 2
fi

cd "$REPO_DIR"

if [[ ! -f "$CONFIG" ]]; then
  echo "[cfg-selection] missing config: $CONFIG"
  exit 2
fi
if [[ ! -f "$RUN_DIR/metadata.json" ]]; then
  echo "[cfg-selection] missing run metadata: $RUN_DIR/metadata.json"
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

args=(
  --config "$CONFIG"
  --run-dir "$RUN_DIR"
  --output-dir "$OUTPUT_DIR"
)

append_value() {
  local env_name="$1"
  local flag="$2"
  local value="${!env_name:-}"
  if [[ -n "$value" ]]; then
    args+=("$flag" "$value")
  fi
}

append_value CHECKPOINT_NAME --checkpoint-name
append_value SELECTION_SCOPE --selection-scope
append_value SELECTION_REGION --selection-region
append_value SELECTION_METRIC --selection-metric
append_value DIRECTION --direction
append_value VALIDATION_ENSEMBLE_SIZE --validation-ensemble-size
append_value TEST_ENSEMBLE_SIZE --test-ensemble-size
append_value SAMPLE_BATCH_SIZE --sample-batch-size
append_value DEVICE --device

if [[ -n "${TRACK_WEIGHTS:-}" ]]; then
  read -r -a weight_values <<<"$TRACK_WEIGHTS"
  args+=(--track-weights "${weight_values[@]}")
fi
if [[ "${SAVE_TEST_ENSEMBLES:-}" == "true" ]]; then
  args+=(--save-test-ensembles)
elif [[ "${SAVE_TEST_ENSEMBLES:-}" == "false" ]]; then
  args+=(--no-save-test-ensembles)
fi

echo "[cfg-selection] run: $RUN_DIR"
echo "[cfg-selection] config: $CONFIG"
echo "[cfg-selection] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "[cfg-selection] output: $OUTPUT_DIR"

"$PYTHON_BIN" -m assim_lib.select_cfg_balance "${args[@]}"

echo "[cfg-selection] completed: $OUTPUT_DIR"
