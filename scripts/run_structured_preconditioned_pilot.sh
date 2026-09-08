#!/usr/bin/env bash
set -Eeuo pipefail

: "${REPO_DIR:?missing immutable publication worktree}"
: "${OUTPUT_DIR:?missing isolated pilot result directory}"

EXPERIMENT="${REPO_DIR}/config/experiments/train_structured_joint_gaussian_preconditioned_checkpointed_pilot.json"
RAW_EXPERIMENT="${REPO_DIR}/config/experiments/train_structured_joint_d0_d3_real_lagged_31e.json"
PANEL="${REPO_DIR}/paper/STRUCTURED_PAIRED_PILOT_PANEL.json"
PROTOCOL="${REPO_DIR}/paper/STRUCTURED_GAUSSIAN_PRECONDITIONED_PILOT_PROTOCOL.json"
CONTRACT_VALIDATOR="${REPO_DIR}/paper/validate_structured_gaussian_pilot_contract.py"
PYTHON_BIN="/opt/conda/bin/python"
PILOT_TIMEOUT_SECONDS=86340
PILOT_KILL_GRACE_SECONDS=60

export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export CLEARML_REQUIRE_ONLINE=1
export STRUCTURED_PILOT_ATTEMPT="structured-pilot-$$"
OUTPUT_OWNED=false

write_failure_status() {
    local exit_code="$1"
    "${PYTHON_BIN}" - "${OUTPUT_DIR}" "${exit_code}" "${STRUCTURED_PILOT_ATTEMPT}" <<'PY'
import json
import os
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
owner = output / ".structured_pilot_owner"
if not owner.is_file() or owner.read_text(encoding="utf-8").strip() != sys.argv[3]:
    raise SystemExit(0)
path = output / "run_status.json"
temporary = output / f".run_status.json.{os.getpid()}.tmp"
temporary.write_text(json.dumps({
    "schema_version": "structured_gaussian_preconditioned_pilot_v1",
    "status": "failed", "exit_code": int(sys.argv[2]),
    "full_training_permitted": False, "test_2023_used": False,
    "attempt_token": sys.argv[3],
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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

# Establish an exact, attempt-owned terminal-status location before any scientific
# preflight.  Existing paths (including symlinks) are never adopted or modified.
if [[ -e "${OUTPUT_DIR}" || -L "${OUTPUT_DIR}" ]]; then
    echo "pilot output already exists: ${OUTPUT_DIR}" >&2
    exit 73
fi
if ! mkdir "${OUTPUT_DIR}"; then
    echo "could not create exact pilot output: ${OUTPUT_DIR}" >&2
    exit 73
fi
OUTPUT_OWNED=true
printf '%s\n' "${STRUCTURED_PILOT_ATTEMPT}" > "${OUTPUT_DIR}/.structured_pilot_owner"

: "${RAW_RUN_DIR:?missing frozen raw structured run}"
: "${REFINEMENT_RESULT_DIR:?missing frozen strict-FP32 refinement result}"
: "${RAW_METADATA_SHA256:?missing frozen raw metadata SHA-256}"
: "${RAW_EMA_SHA256:?missing frozen raw EMA SHA-256}"
: "${RAW_RESUME_SHA256:?missing frozen raw resume SHA-256}"
: "${REFINEMENT_SOLVER_CONTROL_SHA256:?missing refinement result SHA-256}"
: "${REFINEMENT_SOLVER_GATE_SHA256:?missing refinement gate SHA-256}"
: "${MEMORY_ADMISSION_RESULT:?missing passed memory admission result}"
: "${MEMORY_ADMISSION_RESULT_SHA256:?missing memory admission result SHA-256}"
export PYTHON_BIN EXPERIMENT RAW_EXPERIMENT RAW_RUN_DIR OUTPUT_DIR PANEL PROTOCOL
export RAW_METADATA_SHA256 RAW_EMA_SHA256 RAW_RESUME_SHA256
export REFINEMENT_RESULT_DIR REFINEMENT_SOLVER_CONTROL_SHA256 REFINEMENT_SOLVER_GATE_SHA256
export MEMORY_ADMISSION_RESULT MEMORY_ADMISSION_RESULT_SHA256

# A parent environment can silently force ClearML offline before Task.init().
# This pilot is publication-bound and must fail before its first optimizer step.
if [[ "${CLEARML_OFFLINE_MODE+x}" == x ]]; then
    echo "CLEARML_OFFLINE_MODE must be absent for the online pilot" >&2
    exit 78
fi

for path in "${EXPERIMENT}" "${RAW_EXPERIMENT}" "${PANEL}" "${PROTOCOL}" \
    "${CONTRACT_VALIDATOR}" "${MEMORY_ADMISSION_RESULT}" \
    "${RAW_RUN_DIR}/metadata.json" \
    "${RAW_RUN_DIR}/structured_recovery/epoch_0016/ema_state.pth" \
    "${RAW_RUN_DIR}/structured_recovery/epoch_0016/resume.json" \
    "${REFINEMENT_RESULT_DIR}/solver_control.json" \
    "${REFINEMENT_RESULT_DIR}/solver_gate.json"; do
    test -f "${path}" && test ! -L "${path}"
done

cd "${REPO_DIR}"
"${PYTHON_BIN}" "${CONTRACT_VALIDATOR}" \
    --memory-admission-result "${MEMORY_ADMISSION_RESULT}" \
    --memory-admission-sha256 "${MEMORY_ADMISSION_RESULT_SHA256}"

"${PYTHON_BIN}" - "${REFINEMENT_RESULT_DIR}" \
    "${REFINEMENT_SOLVER_CONTROL_SHA256}" "${REFINEMENT_SOLVER_GATE_SHA256}" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
result_path, gate_path = root / "solver_control.json", root / "solver_gate.json"
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
if sha(result_path) != sys.argv[2] or sha(gate_path) != sys.argv[3]:
    raise SystemExit("strict-FP32 refinement hash mismatch")
result, gate = json.loads(result_path.read_text()), json.loads(gate_path.read_text())
if result.get("status") != "converged" or gate.get("status") != "converged" or gate.get("pilot_permitted") is not True:
    raise SystemExit("strict-FP32 refinement does not permit pilot")
if gate.get("solver_control_sha256") != sys.argv[2]:
    raise SystemExit("refinement gate/result binding mismatch")
PY

timeout --signal=TERM --kill-after="${PILOT_KILL_GRACE_SECONDS}s" \
    "${PILOT_TIMEOUT_SECONDS}s" \
    bash -c 'set -Eeuo pipefail
        "${PYTHON_BIN}" -m assim_lib.structured_preconditioned_pilot train \
            --experiment "${EXPERIMENT}" --output-dir "${OUTPUT_DIR}"
        "${PYTHON_BIN}" -m assim_lib.structured_preconditioned_pilot evaluate \
            --experiment "${EXPERIMENT}" --raw-experiment "${RAW_EXPERIMENT}" \
            --raw-run-dir "${RAW_RUN_DIR}" \
            --candidate-run-dir "${OUTPUT_DIR}/training/seed1701" \
            --panel "${PANEL}" --protocol "${PROTOCOL}" \
            --output-dir "${OUTPUT_DIR}" \
            --raw-metadata-sha256 "${RAW_METADATA_SHA256}" \
            --raw-ema-sha256 "${RAW_EMA_SHA256}" \
            --raw-resume-sha256 "${RAW_RESUME_SHA256}" \
            --refinement-result "${REFINEMENT_RESULT_DIR}/solver_control.json" \
            --refinement-gate "${REFINEMENT_RESULT_DIR}/solver_gate.json" \
            --refinement-result-sha256 "${REFINEMENT_SOLVER_CONTROL_SHA256}" \
            --refinement-gate-sha256 "${REFINEMENT_SOLVER_GATE_SHA256}"'
