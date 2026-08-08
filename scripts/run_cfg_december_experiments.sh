#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
RUN_DIR="${1:-${RUN_DIR:-}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
YEAR="${YEAR:-2023}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-ema_best_model.pth}"
CONFIG="${CONFIG:-config/experiments/evaluate_m2m_concat_conditioning_diffusion_balanced_2f.json}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUN_DIR}/evaluation/cfg_december_${RUN_STAMP}}"

if [[ -z "$RUN_DIR" ]]; then
  echo "Usage: RUN_DIR=/path/to/run $0"
  echo "   or: $0 /path/to/run"
  exit 2
fi

cd "$REPO_DIR"

if [[ ! -f "$CONFIG" ]]; then
  echo "[cfg-december] missing experiment config: $CONFIG"
  exit 2
fi
if [[ ! -f "$RUN_DIR/$CHECKPOINT_NAME" ]]; then
  echo "[cfg-december] missing checkpoint: $RUN_DIR/$CHECKPOINT_NAME"
  exit 2
fi
if [[ ! -f "$RUN_DIR/metadata.json" ]]; then
  echo "[cfg-december] missing run metadata: $RUN_DIR/metadata.json"
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

echo "[cfg-december] repository: $REPO_DIR"
echo "[cfg-december] run: $RUN_DIR"
echo "[cfg-december] checkpoint: $CHECKPOINT_NAME"
echo "[cfg-december] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "[cfg-december] output root: $OUTPUT_ROOT"

echo "[cfg-december] starting short visual experiment"
"$PYTHON_BIN" -m assim_lib.cfg_experiments december \
  --config "$CONFIG" \
  --run-dir "$RUN_DIR" \
  --checkpoint-name "$CHECKPOINT_NAME" \
  --year "$YEAR" \
  --month 12 \
  --num-cases 2 \
  --ensemble-size 3 \
  --sample-batch-size 3 \
  --num-timesteps 8 \
  --method euler \
  --seed 1234 \
  --track-weights 0.0 0.25 0.5 0.75 1.0 \
  --track-source configured \
  --output-dir "$OUTPUT_ROOT/short_visual"

echo "[cfg-december] starting L40-calibrated synthetic-track sweep"
"$PYTHON_BIN" -m assim_lib.cfg_experiments track-sweep \
  --config "$CONFIG" \
  --run-dir "$RUN_DIR" \
  --checkpoint-name "$CHECKPOINT_NAME" \
  --year "$YEAR" \
  --month 12 \
  --ensemble-size 5 \
  --sample-batch-size 2 \
  --num-timesteps 25 \
  --method dopri5 \
  --min-tracks 1 \
  --max-tracks 10 \
  --track-weights 0.0 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1.0 \
  --target-hours 4.75 \
  --seed 1234 \
  --output-dir "$OUTPUT_ROOT/track_surface"

echo "[cfg-december] completed"
echo "[cfg-december] short visual: $OUTPUT_ROOT/short_visual"
echo "[cfg-december] quality surface: $OUTPUT_ROOT/track_surface"
