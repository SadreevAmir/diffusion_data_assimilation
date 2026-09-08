#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 || "$1" != "validation_two_stage_native_learning_pilot" ]]; then
  echo "unexpected trusted mode" >&2
  exit 2
fi
: "${REPO_DIR:?REPO_DIR is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"

if [[ ! -d "$REPO_DIR" || -L "$REPO_DIR" ]]; then
  echo "REPO_DIR must be an existing non-symlink directory" >&2
  exit 2
fi
if [[ ! "$OUTPUT_DIR" = /home/autoresearch_results/*/result ]]; then
  echo "OUTPUT_DIR is outside the trusted result namespace" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
cd "$REPO_DIR"
export CLEARML_REQUIRE_ONLINE=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

/opt/conda/bin/python -m unittest \
  tests.test_two_stage_native_learning_pilot.TwoStageNativeLearningPilotTests

exec timeout --signal=TERM --kill-after=60 21000 \
  /opt/conda/bin/python -m assim_lib.two_stage_native_learning_pilot \
  --output-dir "$OUTPUT_DIR" \
  --contract-dir \
  /home/autoresearch_results/two_stage_native_speed_admission_shm48_v1/result/contracts
