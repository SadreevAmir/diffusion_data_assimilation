#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
RUN_DIR="${1:-${RUN_DIR:-}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
CONFIG="${CONFIG:-config/experiments/compare_m2m_3dvar_main_200d.json}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-auto}"
FIELD_PROTOCOL="${FIELD_PROTOCOL:-siconc-only}"
CONDITIONING_MODE="${CONDITIONING_MODE:-full}"
TRACK_WEIGHT="${TRACK_WEIGHT:-0.5}"
ENSEMBLE_SIZE="${ENSEMBLE_SIZE:-15}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-5}"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-25}"
METHOD="${METHOD:-dopri5}"
RTOL="${RTOL:-0.00001}"
ATOL="${ATOL:-0.000001}"
INFERENCE_PRECISION="${INFERENCE_PRECISION:-float32}"
SAVE_ENSEMBLES="${SAVE_ENSEMBLES:-false}"
START_DATE="${START_DATE:-}"
END_DATE="${END_DATE:-}"
EXPECTED_NUM_CASES="${EXPECTED_NUM_CASES:-}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/evaluation/3dvar_comparison_${FIELD_PROTOCOL}_${CONDITIONING_MODE}_${RUN_STAMP}}"

if [[ -z "$RUN_DIR" ]]; then
  echo "Usage: RUN_DIR=/path/to/run $0"
  echo "   or: $0 /path/to/run"
  exit 2
fi

cd "$REPO_DIR"

if [[ ! -f "$CONFIG" ]]; then
  echo "[3dvar-comparison] missing config: $CONFIG"
  exit 2
fi
if [[ "$CHECKPOINT_NAME" != "auto" && ! -f "$RUN_DIR/$CHECKPOINT_NAME" ]]; then
  echo "[3dvar-comparison] missing checkpoint: $RUN_DIR/$CHECKPOINT_NAME"
  exit 2
fi
if [[ ! -f "$RUN_DIR/metadata.json" ]]; then
  echo "[3dvar-comparison] missing checkpoint metadata: $RUN_DIR/metadata.json"
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

args=(
  --config "$CONFIG"
  --run-dir "$RUN_DIR"
  --checkpoint-name "$CHECKPOINT_NAME"
  --output-dir "$OUTPUT_DIR"
  --field-protocol "$FIELD_PROTOCOL"
  --conditioning-mode "$CONDITIONING_MODE"
  --ensemble-size "$ENSEMBLE_SIZE"
  --sample-batch-size "$SAMPLE_BATCH_SIZE"
  --num-timesteps "$NUM_TIMESTEPS"
  --method "$METHOD"
  --rtol "$RTOL"
  --atol "$ATOL"
  --inference-precision "$INFERENCE_PRECISION"
)

if [[ -n "$START_DATE" ]]; then
  args+=(--start-date "$START_DATE")
fi
if [[ -n "$END_DATE" ]]; then
  args+=(--end-date "$END_DATE")
fi
if [[ -n "$EXPECTED_NUM_CASES" ]]; then
  args+=(--expected-num-cases "$EXPECTED_NUM_CASES")
fi

if [[ "$CONDITIONING_MODE" == "independent-balance" ]]; then
  args+=(--track-weight "$TRACK_WEIGHT")
fi
if [[ "$SAVE_ENSEMBLES" == "true" ]]; then
  args+=(--save-ensembles)
else
  args+=(--no-save-ensembles)
fi

echo "[3dvar-comparison] repository: $REPO_DIR"
echo "[3dvar-comparison] run: $RUN_DIR"
echo "[3dvar-comparison] checkpoint: $CHECKPOINT_NAME"
echo "[3dvar-comparison] dates: ${START_DATE:-config default} .. ${END_DATE:-config default} (${EXPECTED_NUM_CASES:-config default} cases)"
echo "[3dvar-comparison] field protocol: $FIELD_PROTOCOL"
echo "[3dvar-comparison] conditioning: $CONDITIONING_MODE"
echo "[3dvar-comparison] ensemble: $ENSEMBLE_SIZE, sample batch: $SAMPLE_BATCH_SIZE"
echo "[3dvar-comparison] sampler: $METHOD, steps=$NUM_TIMESTEPS, precision=$INFERENCE_PRECISION"
echo "[3dvar-comparison] save ensembles: $SAVE_ENSEMBLES"
echo "[3dvar-comparison] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "[3dvar-comparison] output: $OUTPUT_DIR"

"$PYTHON_BIN" -m assim_lib.compare_3dvar "${args[@]}"

echo "[3dvar-comparison] completed: $OUTPUT_DIR"
