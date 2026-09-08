#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 || "$1" != "validation_two_stage_checkpoint_diagnostic" ]]; then
  echo "unexpected trusted mode" >&2
  exit 2
fi

: "${REPO_DIR:?REPO_DIR is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
: "${SOURCE_RESULT_DIR:?SOURCE_RESULT_DIR is required}"

if [[ ! -d "$REPO_DIR" || -L "$REPO_DIR" ]]; then
  echo "REPO_DIR must be an existing non-symlink directory" >&2
  exit 2
fi
if [[ ! "$OUTPUT_DIR" = /home/autoresearch_results/*/result ]]; then
  echo "OUTPUT_DIR is outside the trusted result namespace" >&2
  exit 2
fi
if [[ "$SOURCE_RESULT_DIR" != "/home/autoresearch_results/two_stage_native_learning_pilot_v1/result" ]]; then
  echo "SOURCE_RESULT_DIR differs from the frozen failed pilot" >&2
  exit 2
fi
if [[ "${CUDA_VISIBLE_DEVICES:-}" == *","* ]]; then
  echo "exactly one GPU must be visible" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
cd "$REPO_DIR"
export CLEARML_REQUIRE_ONLINE=1
unset CLEARML_OFFLINE_MODE
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

/opt/conda/bin/python -m unittest \
  tests.test_structured_joint_training.StructuredCodecTests.test_float64_physical_decode_preserves_representable_interior_tail

exec timeout --signal=TERM --kill-after=60 6600 \
  /opt/conda/bin/python -m assim_lib.two_stage_checkpoint_diagnostic \
  --source-dir "$SOURCE_RESULT_DIR" \
  --output-dir "$OUTPUT_DIR"
