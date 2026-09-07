#!/usr/bin/env bash
set -Eeuo pipefail

WORKTREE="/workspace"
SMOKE_RESULT="/run/structured_joint_admission_result.json"
GPU_LOCK="/run/structured_joint_gpu.lock"
GPU_UUID="GPU-40ff1cbb-07cc-992e-23cb-8fe181972b76"
EXPERIMENT="config/experiments/train_structured_joint_d0_d3_real_lagged_31e.json"
LAUNCH_PAYLOAD_REL="paper/STRUCTURED_JOINT_FULL_LAUNCH_PAYLOAD_SHA256.json"
: "${STRUCTURED_LAUNCH_PAYLOAD_SHA:?missing frozen launch payload SHA-256}"

exec 9>"${GPU_LOCK}"
if ! flock -w 120 9; then
    echo "failed to acquire the shared GPU admission lock" >&2
    exit 75
fi

/opt/conda/bin/python -m assim_lib.structured_joint_full_launch_guard \
    --worktree "${WORKTREE}" \
    --smoke-result "${SMOKE_RESULT}" \
    --launch-payload-manifest "${LAUNCH_PAYLOAD_REL}" \
    --launch-payload-sha256 "${STRUCTURED_LAUNCH_PAYLOAD_SHA}" \
    --check-gpu

echo "STRUCTURED_JOINT_FULL_GUARD_PASSED"
exec /opt/conda/bin/python -m assim_lib.main --config "${EXPERIMENT}"
