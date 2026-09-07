#!/usr/bin/env bash
set -Eeuo pipefail

: "${REPO_DIR:?missing immutable publication worktree}"
: "${OUTPUT_DIR:?missing isolated admission result directory}"

EXPERIMENT="${REPO_DIR}/config/experiments/admit_structured_joint_gaussian_preconditioned_checkpointed.json"
PYTHON_BIN="/opt/conda/bin/python"
TIMEOUT_SECONDS=3540
KILL_GRACE_SECONDS=60

export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

test -f "${EXPERIMENT}" && test ! -L "${EXPERIMENT}"
test ! -e "${OUTPUT_DIR}" && test ! -L "${OUTPUT_DIR}"

cd "${REPO_DIR}"
exec timeout --signal=TERM --kill-after="${KILL_GRACE_SECONDS}s" \
    "${TIMEOUT_SECONDS}s" \
    "${PYTHON_BIN}" -m assim_lib.structured_gaussian_memory_admission \
    --experiment "${EXPERIMENT}" \
    --output-dir "${OUTPUT_DIR}"
