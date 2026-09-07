#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'usage: %s NEW_OUTPUT_DIRECTORY\n' "$0" >&2
  exit 64
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
EXPERIMENT="${REPO_ROOT}/config/experiments/train_structured_joint_d0_d3_real_lagged_31e.json"
PROTOCOL="${REPO_ROOT}/config/admission/structured_joint_gpu_smoke_v1.json"

export CLEARML_OFFLINE_MODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export WANDB_DISABLED=true
export PYTHONPATH="${REPO_ROOT}"

exec python3 -m assim_lib.structured_joint_gpu_admission \
  --experiment "${EXPERIMENT}" \
  --protocol "${PROTOCOL}" \
  --output-dir "$1"
