#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  printf 'usage: %s IMAGE_REFERENCE ATTEMPT_NAME\n' "$0" >&2
  exit 64
fi

IMAGE="$1"
ATTEMPT_NAME="$2"
GPU_UUID="GPU-40ff1cbb-07cc-992e-23cb-8fe181972b76"
EXPECTED_IMAGE_ID="sha256:6461c778a994df485f968da62fb9992a800f91ad58d455e0db20bd31d8ecc9e5"
SERVER_WORKTREE="/home/a.madreev/diffusion_data_assimilation/autoresearch_worktrees/structured_joint_forecast_v2_20260907_0255"
HOST_OUTPUT_ROOT="/home/a.madreev/diffusion_data_assimilation/autoresearch_results/structured_joint_gpu_admission_smoke"
CONTAINER_OUTPUT_ROOT="/home/autoresearch_results/structured_joint_gpu_admission_smoke"
HOST_LOCK="${HOST_OUTPUT_ROOT}/.${GPU_UUID}.host.lock"
TOTAL_WALL_SECONDS=1800
WATCHDOG_GRACE_SECONDS=30
WATCHDOG_SECONDS=$((TOTAL_WALL_SECONDS + WATCHDOG_GRACE_SECONDS))
STOP_GRACE_SECONDS=20
DOCKER_CONTROL_SECONDS=30
DOCKER_START_SECONDS=60
OWN_CONTAINER_ID=""
OWN_CONTAINER_NAME=""
CID_FILE=""
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

recover_owned_container_id() {
  local candidate
  [[ -z "${OWN_CONTAINER_ID}" && -n "${CID_FILE}" && -f "${CID_FILE}" ]] || return 0
  candidate="$(tr -d '[:space:]' <"${CID_FILE}")"
  if [[ ! "${candidate}" =~ ^[0-9a-f]{64}$ ]]; then
    printf 'owned smoke cidfile is invalid: %s\n' "${CID_FILE}" >&2
    return 1
  fi
  OWN_CONTAINER_ID="${candidate}"
}

inspect_owned_container() {
  local identity
  if ! identity="$(timeout --signal=TERM --kill-after=5s "${DOCKER_CONTROL_SECONDS}s" \
    docker container inspect --format '{{.Id}} {{.Name}}' "${OWN_CONTAINER_ID}")"; then
    return 1
  fi
  [[ "${identity}" == "${OWN_CONTAINER_ID} /${OWN_CONTAINER_NAME}" ]]
}


confirm_owned_container_removed() {
  local remaining
  if ! remaining="$(timeout --signal=TERM --kill-after=5s "${DOCKER_CONTROL_SECONDS}s" \
    docker container ls -aq --no-trunc --filter "id=${OWN_CONTAINER_ID}")"; then
    printf '%s\n' 'could not query Docker to confirm owned smoke container removal' >&2
    return 1
  fi
  if [[ -n "${remaining}" ]]; then
    printf 'owned smoke container still exists after cleanup: %s\n' "${OWN_CONTAINER_ID}" >&2
    return 1
  fi
}


remove_owned_container() {
  recover_owned_container_id || return 1
  [[ -n "${OWN_CONTAINER_ID}" ]] || return 0
  if inspect_owned_container; then
    timeout --signal=TERM --kill-after=5s "${DOCKER_CONTROL_SECONDS}s" \
      docker stop --time "${STOP_GRACE_SECONDS}" "${OWN_CONTAINER_ID}" \
      >/dev/null 2>&1 || true
    timeout --signal=TERM --kill-after=5s "${DOCKER_CONTROL_SECONDS}s" \
      docker rm -f "${OWN_CONTAINER_ID}" >/dev/null 2>&1 || true
  elif timeout --signal=TERM --kill-after=5s "${DOCKER_CONTROL_SECONDS}s" \
    docker container inspect "${OWN_CONTAINER_ID}" >/dev/null 2>&1; then
    printf 'refusing to touch container whose exact name/id ownership is not proven: %s\n' \
      "${OWN_CONTAINER_ID}" >&2
    return 1
  fi
  if ! confirm_owned_container_removed; then
    return 1
  fi
  OWN_CONTAINER_ID=""
  [[ -z "${CID_FILE}" ]] || rm -f "${CID_FILE}"
}


on_signal() {
  local exit_code="$1"
  trap - INT TERM
  remove_owned_container || exit 76
  exit "${exit_code}"
}


trap 'remove_owned_container' EXIT
trap 'on_signal 130' INT
trap 'on_signal 143' TERM

if ! ACTUAL_IMAGE_ID="$(timeout --signal=TERM --kill-after=5s \
  "${DOCKER_CONTROL_SECONDS}s" docker image inspect --format '{{.Id}}' "${IMAGE}")"; then
  printf 'could not inspect container image within %ss: %s\n' \
    "${DOCKER_CONTROL_SECONDS}" "${IMAGE}" >&2
  exit 67
fi
if [[ "${ACTUAL_IMAGE_ID}" != "${EXPECTED_IMAGE_ID}" ]]; then
  printf 'container image ID differs: expected %s, got %s\n' \
    "${EXPECTED_IMAGE_ID}" "${ACTUAL_IMAGE_ID}" >&2
  exit 65
fi
if [[ ! "${ATTEMPT_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  printf '%s\n' 'attempt name must be one safe basename' >&2
  exit 65
fi
if (( ${#ATTEMPT_NAME} > 64 )); then
  printf '%s\n' 'attempt name is too long for a bounded unique container name' >&2
  exit 65
fi
OWN_CONTAINER_NAME="structured-joint-gpu-smoke-${ATTEMPT_NAME}-$$-${RANDOM}"
if [[ ! -d "${SERVER_WORKTREE}" || -L "${SERVER_WORKTREE}" ]]; then
  printf 'server worktree is absent or unsafe: %s\n' "${SERVER_WORKTREE}" >&2
  exit 66
fi
mkdir -p "${HOST_OUTPUT_ROOT}"
if [[ -L "${HOST_OUTPUT_ROOT}" || ! -d "${HOST_OUTPUT_ROOT}" ]]; then
  printf 'host output root is unsafe: %s\n' "${HOST_OUTPUT_ROOT}" >&2
  exit 66
fi
CID_FILE="${HOST_OUTPUT_ROOT}/.${OWN_CONTAINER_NAME}.cid"
if [[ -e "${CID_FILE}" || -L "${CID_FILE}" ]]; then
  printf 'refusing an existing smoke cidfile: %s\n' "${CID_FILE}" >&2
  exit 66
fi

exec 9>"${HOST_LOCK}"
if ! flock -n 9; then
  printf '%s\n' 'the target GPU admission host lock is already held' >&2
  exit 75
fi

WATCHDOG_START_EPOCH="$(date +%s)"
set +e
RUN_OUTPUT="$(timeout --signal=TERM --kill-after=5s "${DOCKER_START_SECONDS}s" docker run -d \
  --pull=never \
  --name "${OWN_CONTAINER_NAME}" \
  --cidfile "${CID_FILE}" \
  --network none \
  --gpus "device=${GPU_UUID}" \
  --read-only \
  --security-opt no-new-privileges \
  --user "${HOST_UID}:${HOST_GID}" \
  --cpus 4 \
  --pids-limit 512 \
  --shm-size 1g \
  --tmpfs /tmp:rw,nosuid,nodev,noexec,size=2g \
  --env HOME=/tmp/home \
  --env XDG_CACHE_HOME=/tmp/xdg-cache \
  --env MPLCONFIGDIR=/tmp/matplotlib \
  --env TORCH_HOME=/tmp/torch \
  --env TRITON_CACHE_DIR=/tmp/triton \
  --env TMPDIR=/tmp \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env OMP_NUM_THREADS=1 \
  --env MKL_NUM_THREADS=1 \
  --env OPENBLAS_NUM_THREADS=1 \
  --env NUMEXPR_NUM_THREADS=1 \
  --entrypoint /bin/bash \
  --volume "${SERVER_WORKTREE}:/workspace:ro" \
  --volume "/mnt:/mnt:ro" \
  --volume "${HOST_OUTPUT_ROOT}:${CONTAINER_OUTPUT_ROOT}:rw" \
  --workdir /workspace \
  "${EXPECTED_IMAGE_ID}" \
  scripts/run_structured_joint_gpu_admission_smoke.sh \
  "${CONTAINER_OUTPUT_ROOT}/${ATTEMPT_NAME}")"
RUN_STATUS=$?
set -e

if [[ ${RUN_STATUS} -ne 0 ]]; then
  if ! remove_owned_container; then
    printf '%s\n' 'docker run failed and exact owned-container cleanup could not be confirmed' >&2
    exit 76
  fi
  exit "${RUN_STATUS}"
fi

OWN_CONTAINER_ID="$(tr -d '[:space:]' <<<"${RUN_OUTPUT}")"

if [[ ! "${OWN_CONTAINER_ID}" =~ ^[0-9a-f]{64}$ ]]; then
  printf 'docker returned an invalid owned container ID: %s\n' "${OWN_CONTAINER_ID}" >&2
  OWN_CONTAINER_ID=""
  remove_owned_container || true
  exit 76
fi
CID_CONTAINER_ID="$(tr -d '[:space:]' <"${CID_FILE}")"
if [[ "${CID_CONTAINER_ID}" != "${OWN_CONTAINER_ID}" ]]; then
  printf 'docker output/cidfile identity mismatch: %s versus %s\n' \
    "${OWN_CONTAINER_ID}" "${CID_CONTAINER_ID}" >&2
  exit 76
fi
if ! inspect_owned_container; then
  printf 'new smoke container identity/name verification failed: %s\n' "${OWN_CONTAINER_ID}" >&2
  exit 76
fi

WAIT_RESULT="$(mktemp "${HOST_OUTPUT_ROOT}/.wait.${OWN_CONTAINER_NAME}.XXXXXX")"
trap 'rm -f "${WAIT_RESULT}"; remove_owned_container' EXIT
WATCHDOG_ELAPSED_SECONDS=$(( $(date +%s) - WATCHDOG_START_EPOCH ))
WATCHDOG_REMAINING_SECONDS=$(( WATCHDOG_SECONDS - WATCHDOG_ELAPSED_SECONDS ))
if (( WATCHDOG_REMAINING_SECONDS <= 0 )); then
  printf 'GPU admission exhausted hard host deadline before docker wait; stopping only %s\n' \
    "${OWN_CONTAINER_ID}" >&2
  remove_owned_container
  rm -f "${WAIT_RESULT}"
  trap - EXIT
  exit 124
fi
set +e
timeout --signal=TERM --kill-after=5s "${WATCHDOG_REMAINING_SECONDS}s" \
  docker wait "${OWN_CONTAINER_ID}" >"${WAIT_RESULT}"
WAIT_STATUS=$?
set -e

if [[ ${WAIT_STATUS} -eq 124 ]]; then
  printf 'GPU admission exceeded hard host deadline %ss; stopping only %s\n' \
    "${WATCHDOG_SECONDS}" "${OWN_CONTAINER_ID}" >&2
  if ! inspect_owned_container; then
    printf '%s\n' 'cannot prove timed-out container ownership; refusing broad cleanup' >&2
    exit 76
  fi
  remove_owned_container
  rm -f "${WAIT_RESULT}"
  trap - EXIT
  exit 124
fi
if [[ ${WAIT_STATUS} -ne 0 ]]; then
  printf 'docker wait failed for owned smoke container with status %s\n' "${WAIT_STATUS}" >&2
  exit "${WAIT_STATUS}"
fi

CONTAINER_EXIT_CODE="$(tr -d '[:space:]' <"${WAIT_RESULT}")"
if [[ ! "${CONTAINER_EXIT_CODE}" =~ ^[0-9]+$ ]] || (( CONTAINER_EXIT_CODE > 255 )); then
  printf 'docker wait returned invalid container exit code: %s\n' "${CONTAINER_EXIT_CODE}" >&2
  exit 76
fi
if ! inspect_owned_container; then
  printf '%s\n' 'owned smoke container disappeared before verified removal' >&2
  exit 76
fi
timeout --signal=TERM --kill-after=5s "${DOCKER_CONTROL_SECONDS}s" \
  docker rm "${OWN_CONTAINER_ID}" >/dev/null
if ! confirm_owned_container_removed; then
  exit 76
fi
OWN_CONTAINER_ID=""
rm -f "${CID_FILE}"
rm -f "${WAIT_RESULT}"
trap - EXIT
exit "${CONTAINER_EXIT_CODE}"
