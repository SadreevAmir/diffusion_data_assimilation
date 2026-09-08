#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 || "$1" != "validation_two_stage_native_speed_admission" ]]; then
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
  tests.test_structured_joint_training.StructuredConditioningTests.test_assimilation_pads_raw_lag0_before_applying_padded_track_mask \
  tests.test_structured_joint_training.StructuredConditioningTests.test_speed_admission_collate_excludes_non_training_payload \
  tests.test_structured_joint_training.StructuredConditioningTests.test_speed_admission_ipc_gate_fails_before_worker_iteration
exec /opt/conda/bin/python -m assim_lib.two_stage_native_speed_admission \
  --output-dir "$OUTPUT_DIR" \
  --reuse-contract-dir \
  /home/autoresearch_results/two_stage_native_speed_admission_shape_fix_v2/result/contracts
