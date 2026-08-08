#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
MODE="${1:-${MODE:-smoke_strict_full}}"
RUN_DIR="${2:-${RUN_DIR:-}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
CONFIG="${CONFIG:-config/experiments/experiment_m2m_flow_modes.json}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-auto}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/evaluation/${MODE}_${RUN_STAMP}}"

if [[ -z "$RUN_DIR" ]]; then
  echo "Usage: RUN_DIR=/path/to/run $0 [mode]"
  echo "   or: $0 mode /path/to/run"
  exit 2
fi

cd "$REPO_DIR"

if [[ ! -f "$CONFIG" ]]; then
  echo "[flow-experiment] missing config: $CONFIG"
  exit 2
fi
if [[ ! -f "$RUN_DIR/metadata.json" ]]; then
  echo "[flow-experiment] missing run metadata: $RUN_DIR/metadata.json"
  exit 2
fi
if [[ "$CHECKPOINT_NAME" != "auto" && ! -f "$RUN_DIR/$CHECKPOINT_NAME" ]]; then
  echo "[flow-experiment] missing checkpoint: $RUN_DIR/$CHECKPOINT_NAME"
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

args=(
  --config "$CONFIG"
  --mode "$MODE"
  --run-dir "$RUN_DIR"
  --checkpoint-name "$CHECKPOINT_NAME"
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

append_value FIELD_PROTOCOL --field-protocol
append_value CONDITIONING_MODE --conditioning-mode
append_value TRACK_WEIGHT --track-weight
append_value ENSEMBLE_SIZE --ensemble-size
append_value SAMPLE_BATCH_SIZE --sample-batch-size
append_value NUM_TIMESTEPS --num-timesteps
append_value METHOD --method
append_value RTOL --rtol
append_value ATOL --atol
append_value INFERENCE_PRECISION --inference-precision
append_value SEED --seed
append_value START_DATE --start-date
append_value END_DATE --end-date
append_value EXPECTED_NUM_CASES --expected-num-cases
append_value CASE_STRIDE_DAYS --case-stride-days

if [[ "${SAVE_ENSEMBLES:-}" == "true" ]]; then
  args+=(--save-ensembles)
elif [[ "${SAVE_ENSEMBLES:-}" == "false" ]]; then
  args+=(--no-save-ensembles)
fi

echo "[flow-experiment] mode: $MODE"
echo "[flow-experiment] run: $RUN_DIR"
echo "[flow-experiment] checkpoint policy: $CHECKPOINT_NAME"
echo "[flow-experiment] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "[flow-experiment] output: $OUTPUT_DIR"

"$PYTHON_BIN" -m assim_lib.compare_3dvar "${args[@]}"

echo "[flow-experiment] completed: $OUTPUT_DIR"
