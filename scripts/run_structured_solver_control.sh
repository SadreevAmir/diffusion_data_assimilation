#!/usr/bin/env bash
set -Eeuo pipefail

: "${REPO_DIR:?missing immutable publication worktree}"
: "${RUN_DIR:?missing frozen structured training run}"
: "${OUTPUT_DIR:?missing isolated result directory}"
: "${REFERENCE_METADATA_SHA256:?missing frozen metadata SHA-256}"
: "${REFERENCE_EMA_SHA256:?missing frozen EMA SHA-256}"
: "${REFERENCE_RESUME_SHA256:?missing frozen resume SHA-256}"

EXPERIMENT="${REPO_DIR}/config/experiments/train_structured_joint_d0_d3_real_lagged_31e.json"
PYTHON_BIN="/opt/conda/bin/python"

export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export CLEARML_OFFLINE_MODE=1
export STRUCTURED_SOLVER_CONTROL_ATTEMPT="solver-control-$$"
OUTPUT_OWNED=false

write_failure_status() {
    local exit_code="$1"
    "${PYTHON_BIN}" - "${OUTPUT_DIR}" "${exit_code}" \
        "${STRUCTURED_SOLVER_CONTROL_ATTEMPT}" <<'PY'
import json
import os
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
path = output / "run_status.json"
owner = output / ".structured_solver_control_owner"
if not owner.is_file() or owner.read_text(encoding="utf-8").strip() != sys.argv[3]:
    raise SystemExit(0)
temporary = output / f".run_status.json.{os.getpid()}.tmp"
temporary.write_text(
    json.dumps(
        {
            "schema_version": "structured_solver_control_v1",
            "status": "failed",
            "exit_code": int(sys.argv[2]),
            "selection_permitted": False,
            "training_performed": False,
            "attempt_token": sys.argv[3],
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
os.replace(temporary, path)
PY
}

on_exit() {
    local exit_code="$?"
    trap - EXIT
    if [[ "${exit_code}" -ne 0 && "${OUTPUT_OWNED}" == true ]]; then
        write_failure_status "${exit_code}" || true
    fi
    exit "${exit_code}"
}
trap on_exit EXIT

test -f "${EXPERIMENT}"
test -f "${RUN_DIR}/metadata.json"
test -f "${RUN_DIR}/structured_recovery/epoch_0016/ema_state.pth"
if ! mkdir "${OUTPUT_DIR}"; then
    echo "solver-control output already exists: ${OUTPUT_DIR}" >&2
    exit 73
fi
OUTPUT_OWNED=true
printf '%s\n' "${STRUCTURED_SOLVER_CONTROL_ATTEMPT}" \
    > "${OUTPUT_DIR}/.structured_solver_control_owner"

cd "${REPO_DIR}"
"${PYTHON_BIN}" -m assim_lib.structured_solver_control \
    --experiment "${EXPERIMENT}" \
    --run-dir "${RUN_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --recovery-epoch 16 \
    --expected-metadata-sha256 "${REFERENCE_METADATA_SHA256}" \
    --expected-ema-sha256 "${REFERENCE_EMA_SHA256}" \
    --expected-resume-sha256 "${REFERENCE_RESUME_SHA256}"
