#!/usr/bin/env bash
set -Eeuo pipefail

WORKTREE="/home/a.madreev/diffusion_data_assimilation/autoresearch_worktrees/structured_joint_forecast_v2_20260907_0255"
HOST_RESULTS="/home/a.madreev/diffusion_data_assimilation/autoresearch_results"
CONTAINER_RESULTS="/home/autoresearch_results"
ENV_FILE="/home/a.madreev/diffusion_data_assimilation/.env"
SMOKE_RESULT="${HOST_RESULTS}/structured_joint_gpu_admission_smoke/astra_ready_20260907_0710_retry6/result.json"
IMAGE_ID="sha256:6461c778a994df485f968da62fb9992a800f91ad58d455e0db20bd31d8ecc9e5"
GPU_UUID="GPU-40ff1cbb-07cc-992e-23cb-8fe181972b76"
CONTAINER_NAME="a.madreev_structured_joint_d0_d3_seed1701_gpu1"
OUTPUT_DIR="${HOST_RESULTS}/structured_joint_d0_d3_real_lagged_31e"
LOCK_PATH="${HOST_RESULTS}/structured_joint_gpu_admission_smoke/.${GPU_UUID}.host.lock"
LAUNCH_PAYLOAD_REL="paper/STRUCTURED_JOINT_FULL_LAUNCH_PAYLOAD_SHA256.json"
LAUNCH_PAYLOAD_SHA="d806c00bee1af5f431c90b942c43cf8283fd6f52f6516af85f8bc0a01c8e506b"
OWNER_LABEL="structured-joint-research"
EXPERIMENT_LABEL="structured-joint-d0-d3-seed1701"

docker_bounded() {
    local seconds="$1"
    shift
    timeout --signal=TERM --kill-after=5s "${seconds}s" docker "$@"
}

for required in "${WORKTREE}" "${HOST_RESULTS}" "${ENV_FILE}" "${SMOKE_RESULT}"; do
    if [[ ! -e "${required}" ]]; then
        echo "required launch input is missing: ${required}" >&2
        exit 2
    fi
done

mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/launches"
if [[ -L "${OUTPUT_DIR}" ]]; then
    echo "experiment output directory must not be a symlink: ${OUTPUT_DIR}" >&2
    exit 2
fi

exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
    echo "the shared smoke/full GPU lock is already held: ${LOCK_PATH}" >&2
    exit 3
fi

attempt_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
launch_dir="${OUTPUT_DIR}/launches/${attempt_id}"
mkdir -p "${launch_dir}"
record_path="${launch_dir}/launch_record.json"
cidfile="${launch_dir}/container.cid"
container_id=""
container_created="false"
container_started="false"
launch_committed="false"

write_record() {
    local status="$1"
    local detail="$2"
    local cid="${3:-}"
    RECORD_PATH="${record_path}" RECORD_STATUS="${status}" RECORD_DETAIL="${detail}" \
    RECORD_CID="${cid}" RECORD_ATTEMPT="${attempt_id}" RECORD_CONTAINER="${CONTAINER_NAME}" \
    RECORD_GPU="${GPU_UUID}" RECORD_IMAGE="${IMAGE_ID}" RECORD_WORKTREE="${WORKTREE}" \
    RECORD_LAUNCHER="${BASH_SOURCE[0]}" python3 - <<'PY'
import hashlib
import json
import os
import time
from pathlib import Path

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

destination = Path(os.environ["RECORD_PATH"])
launcher = Path(os.environ["RECORD_LAUNCHER"]).resolve()
payload = {
    "schema_version": "structured_joint_full_launch_record_v1",
    "attempt_id": os.environ["RECORD_ATTEMPT"],
    "updated_epoch_seconds": time.time(),
    "status": os.environ["RECORD_STATUS"],
    "detail": os.environ["RECORD_DETAIL"],
    "container_id": os.environ["RECORD_CID"] or None,
    "container_name": os.environ["RECORD_CONTAINER"],
    "gpu_uuid": os.environ["RECORD_GPU"],
    "image_id": os.environ["RECORD_IMAGE"],
    "worktree": os.environ["RECORD_WORKTREE"],
    "launcher_path": str(launcher),
    "launcher_sha256": sha256(launcher),
}
temporary = destination.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(destination)
PY
}

inspect_owned() {
    local cid="$1"
    local identity
    if ! identity="$(docker_bounded 30 inspect --format '{{.Id}}|{{.Name}}|{{index .Config.Labels "com.a-madreev.owner"}}|{{index .Config.Labels "com.a-madreev.experiment"}}' "${cid}")"; then
        return 1
    fi
    local actual_id actual_name actual_owner actual_experiment
    IFS='|' read -r actual_id actual_name actual_owner actual_experiment <<<"${identity}"
    [[ "${actual_id}" == "${cid}" ]] || return 1
    [[ "${actual_name}" == "/${CONTAINER_NAME}" ]] || return 1
    [[ "${actual_owner}" == "${OWNER_LABEL}" ]] || return 1
    [[ "${actual_experiment}" == "${EXPERIMENT_LABEL}" ]] || return 1
}

container_running() {
    local cid="$1"
    [[ "$(docker_bounded 30 inspect --format '{{.State.Running}}' "${cid}")" == "true" ]]
}

remove_stopped_owned() {
    local cid="$1"
    inspect_owned "${cid}" || return 1
    if container_running "${cid}"; then
        return 1
    fi
    docker_bounded 30 rm "${cid}" >/dev/null || return 1
    if docker_bounded 30 inspect "${cid}" >/dev/null 2>&1; then
        return 1
    fi
}

stop_and_remove_uncommitted() {
    local cid="$1"
    inspect_owned "${cid}" || return 1
    if container_running "${cid}"; then
        docker_bounded 75 stop --time 60 "${cid}" >/dev/null || return 1
    fi
    remove_stopped_owned "${cid}"
}

cleanup_on_exit() {
    local original_status="$?"
    trap - EXIT
    if [[ "${launch_committed}" == "false" && "${container_created}" == "true" && -n "${container_id}" ]]; then
        if stop_and_remove_uncommitted "${container_id}"; then
            write_record "cleaned_after_failed_launch" "exact owned uncommitted container stopped and removed" "${container_id}" || true
        else
            write_record "cleanup_failed" "manual inspection required for exact owned container" "${container_id}" || true
            echo "failed to clean exact owned uncommitted container: ${container_id}" >&2
            exit 91
        fi
    fi
    exit "${original_status}"
}
trap cleanup_on_exit EXIT

write_record "validating" "validating frozen identities and idle GPU"

python3 "${WORKTREE}/assim_lib/structured_joint_full_launch_guard.py" \
    --worktree "${WORKTREE}" \
    --smoke-result "${SMOKE_RESULT}" \
    --launch-payload-manifest "${LAUNCH_PAYLOAD_REL}" \
    --launch-payload-sha256 "${LAUNCH_PAYLOAD_SHA}" \
    --check-gpu >/dev/null

actual_image_id="$(docker_bounded 30 image inspect --format '{{.Id}}' "${IMAGE_ID}")"
if [[ "${actual_image_id}" != "${IMAGE_ID}" ]]; then
    write_record "failed" "local image identity differs"
    echo "local image identity differs: ${actual_image_id}" >&2
    exit 4
fi

if ! existing_output="$(docker_bounded 30 container ls -a --no-trunc --filter "name=^/${CONTAINER_NAME}$" --format '{{.ID}}')"; then
    write_record "failed" "could not enumerate pre-existing containers"
    exit 5
fi
mapfile -t existing_ids < <(printf '%s\n' "${existing_output}" | sed '/^$/d')
if (( ${#existing_ids[@]} > 1 )); then
    write_record "failed" "multiple exact-name containers reported"
    exit 5
fi
if (( ${#existing_ids[@]} == 1 )); then
    stale_id="${existing_ids[0]}"
    if ! inspect_owned "${stale_id}"; then
        write_record "failed" "exact-name container is not owned by this launcher" "${stale_id}"
        exit 5
    fi
    if container_running "${stale_id}"; then
        write_record "failed" "owned training container is already running" "${stale_id}"
        exit 5
    fi
    if ! remove_stopped_owned "${stale_id}"; then
        write_record "failed" "failed to remove exact owned stopped container" "${stale_id}"
        exit 5
    fi
    write_record "stale_container_removed" "exact owned stopped container removed before resume" "${stale_id}"
fi

host_uid="$(id -u)"
host_gid="$(id -g)"
create_status=0
create_output="$(docker_bounded 60 create \
    --cidfile "${cidfile}" \
    --name "${CONTAINER_NAME}" \
    --label "com.a-madreev.owner=${OWNER_LABEL}" \
    --label "com.a-madreev.experiment=${EXPERIMENT_LABEL}" \
    --restart no \
    --stop-timeout 60 \
    --gpus "device=${GPU_UUID}" \
    --cpus 12 \
    --pids-limit 2048 \
    --shm-size 32g \
    --read-only \
    --init \
    --security-opt no-new-privileges \
    --user "${host_uid}:${host_gid}" \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env OPENBLAS_NUM_THREADS=1 \
    --env NUMEXPR_NUM_THREADS=1 \
    --env HOME=/tmp/home \
    --env XDG_CACHE_HOME=/tmp/xdg-cache \
    --env MPLCONFIGDIR=/tmp/matplotlib \
    --env TORCH_HOME=/tmp/torch \
    --env "STRUCTURED_LAUNCH_PAYLOAD_SHA=${LAUNCH_PAYLOAD_SHA}" \
    --volume "${WORKTREE}:/workspace:ro" \
    --volume /mnt:/mnt:ro \
    --volume "${OUTPUT_DIR}:${CONTAINER_RESULTS}/structured_joint_d0_d3_real_lagged_31e:rw" \
    --volume "${ENV_FILE}:/home/.env:ro" \
    --volume "${SMOKE_RESULT}:/run/structured_joint_admission_result.json:ro" \
    --volume "${LOCK_PATH}:/run/structured_joint_gpu.lock:rw" \
    --tmpfs /tmp:rw,exec,nosuid,size=8589934592 \
    --workdir /workspace \
    --entrypoint /bin/bash \
    "${IMAGE_ID}" \
    /workspace/scripts/run_structured_joint_full_training_guarded.sh)" || create_status="$?"

if [[ -s "${cidfile}" ]]; then
    container_id="$(tr -d '[:space:]' < "${cidfile}")"
elif [[ "${create_output}" =~ ^[0-9a-f]{64}$ ]]; then
    container_id="${create_output}"
fi
if [[ -n "${container_id}" && ! "${container_id}" =~ ^[0-9a-f]{64}$ ]]; then
    container_id=""
fi
if [[ -z "${container_id}" ]]; then
    if ! recovered_output="$(docker_bounded 30 container ls -a --no-trunc --filter "name=^/${CONTAINER_NAME}$" --format '{{.ID}}')"; then
        write_record "cleanup_failed" "docker create outcome is uncertain and exact-name recovery failed"
        exit 91
    fi
    mapfile -t recovered_ids < <(printf '%s\n' "${recovered_output}" | sed '/^$/d')
    if (( ${#recovered_ids[@]} == 1 )); then
        container_id="${recovered_ids[0]}"
    elif (( ${#recovered_ids[@]} > 1 )); then
        write_record "cleanup_failed" "docker create outcome is uncertain and multiple exact-name containers were reported"
        exit 91
    fi
fi
if [[ -n "${container_id}" ]]; then
    if [[ "${container_id}" =~ ^[0-9a-f]{64}$ ]] && inspect_owned "${container_id}"; then
        container_created="true"
    else
        write_record "cleanup_failed" "docker create returned an unowned or invalid container identity" "${container_id}"
        exit 91
    fi
fi
if (( create_status != 0 )); then
    write_record "failed" "bounded docker create failed with status ${create_status}" "${container_id}"
    exit 6
fi
if [[ ! "${container_id}" =~ ^[0-9a-f]{64}$ || "${create_output}" != "${container_id}" ]]; then
    write_record "failed" "docker create returned inconsistent container identity" "${container_id}"
    exit 6
fi
if ! inspect_owned "${container_id}"; then
    write_record "failed" "created container ownership identity differs" "${container_id}"
    exit 7
fi

configured_gpu="$(docker_bounded 30 inspect --format '{{range .HostConfig.DeviceRequests}}{{range .DeviceIDs}}{{.}}{{end}}{{end}}' "${container_id}")"
configured_restart="$(docker_bounded 30 inspect --format '{{.HostConfig.RestartPolicy.Name}}:{{.HostConfig.RestartPolicy.MaximumRetryCount}}' "${container_id}")"
configured_read_only="$(docker_bounded 30 inspect --format '{{.HostConfig.ReadonlyRootfs}}' "${container_id}")"
configured_user="$(docker_bounded 30 inspect --format '{{.Config.User}}' "${container_id}")"
configured_mounts="$(docker_bounded 30 inspect --format '{{range .Mounts}}{{.Destination}}={{.RW}};{{end}}' "${container_id}")"
if [[ "${configured_gpu}" != "${GPU_UUID}" \
    || "${configured_restart}" != "no:0" \
    || "${configured_read_only}" != "true" \
    || "${configured_user}" != "${host_uid}:${host_gid}" ]]; then
    write_record "failed" "created container resource contract differs" "${container_id}"
    exit 7
fi
for mount_contract in "/workspace=false" "/mnt=false" "/home/.env=false" "/run/structured_joint_admission_result.json=false" "/run/structured_joint_gpu.lock=true" "${CONTAINER_RESULTS}/structured_joint_d0_d3_real_lagged_31e=true"; do
    if [[ "${configured_mounts}" != *"${mount_contract}"* ]]; then
        write_record "failed" "created container mount contract differs: ${mount_contract}" "${container_id}"
        exit 7
    fi
done

write_record "created" "exact owned container created and contract verified" "${container_id}"
start_status=0
docker_bounded 60 start "${container_id}" >/dev/null || start_status="$?"
if (( start_status != 0 )); then
    write_record "failed" "bounded docker start failed with status ${start_status}" "${container_id}"
    exit 8
fi
container_started="true"
if ! container_running "${container_id}"; then
    docker_bounded 30 logs --tail 80 "${container_id}" >&2 || true
    write_record "failed" "container did not enter running state" "${container_id}"
    exit 8
fi

# The inner process is blocked on the same inode. Releasing fd 9 transfers
# ownership to the guarded entrypoint, which validates identities and GPU again.
flock -u 9
guard_passed="false"
for _ in $(seq 1 30); do
    logs="$(docker_bounded 30 logs --tail 120 "${container_id}" 2>&1)" || {
        write_record "failed" "could not read bounded startup logs" "${container_id}"
        exit 9
    }
    if [[ "${logs}" == *"STRUCTURED_JOINT_FULL_GUARD_PASSED"* ]]; then
        guard_passed="true"
        break
    fi
    if ! container_running "${container_id}"; then
        printf '%s\n' "${logs}" >&2
        write_record "failed" "guarded entrypoint exited before admission" "${container_id}"
        exit 9
    fi
    sleep 1
done
if [[ "${guard_passed}" != "true" ]]; then
    write_record "failed" "guard admission marker was not observed within 30 seconds" "${container_id}"
    exit 9
fi
if ! container_running "${container_id}"; then
    write_record "failed" "container exited immediately after guard admission" "${container_id}"
    exit 9
fi

launch_committed="true"
write_record "running" "guard passed; detached training owns the shared GPU lock" "${container_id}"
trap - EXIT
echo "${container_id} ${CONTAINER_NAME} started on ${GPU_UUID}; launch_record=${record_path}"
