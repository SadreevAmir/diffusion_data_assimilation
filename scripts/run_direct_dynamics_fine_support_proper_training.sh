#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ID="${FINE_SUPPORT_TRAIN_RUN_ID:?FINE_SUPPORT_TRAIN_RUN_ID is required}"
ATTEMPT_ID="${FINE_SUPPORT_TRAIN_ATTEMPT_ID:?FINE_SUPPORT_TRAIN_ATTEMPT_ID is required}"
MODE="${FINE_SUPPORT_TRAIN_MODE:-fresh}"
for value in "$RUN_ID" "$ATTEMPT_ID"; do
  if [[ ! "$value" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ ]]; then
    echo "unsafe fine-support training identifier" >&2
    exit 2
  fi
done
if [[ "$MODE" != "fresh" && "$MODE" != "resume" ]]; then
  echo "FINE_SUPPORT_TRAIN_MODE must be fresh or resume" >&2
  exit 2
fi
CONFIG="${FINE_SUPPORT_TRAIN_CONFIG:-config/experiments/train_direct_dynamics_fine_support_proper_64_v1.json}"
EXPECTED_CONFIG="config/experiments/train_direct_dynamics_fine_support_proper_64_v1.json"
if [[ "$CONFIG" != "$EXPECTED_CONFIG" || ! -f "$CONFIG" ]]; then
  echo "unreviewed fine-support training config" >&2
  exit 2
fi

OUTPUT_ROOT="/home/autoresearch_results/direct_dynamics_cascade_v2/fine_support_proper_training"
LAUNCH_ROOT="/home/autoresearch_results/direct_dynamics_cascade_v2/fine_support_proper_training_launches"
OUTPUT="$OUTPUT_ROOT/$RUN_ID"
LAUNCH="$LAUNCH_ROOT/$ATTEMPT_ID"
mkdir -p "$LAUNCH_ROOT"
if [[ "$MODE" == "fresh" && ( -e "$OUTPUT" || -L "$OUTPUT" ) ]]; then
  echo "fresh mode refuses an existing training output" >&2
  exit 3
fi
if [[ "$MODE" == "resume" && ! -d "$OUTPUT" ]]; then
  echo "resume mode requires an existing training output" >&2
  exit 3
fi
remaining_budget_seconds() {
  python -c 'import json, math, sys, time; value=json.load(open(sys.argv[1]))["deadline_unix"]-time.time(); print(math.floor(value))' "$OUTPUT/status.json"
}
if [[ "$MODE" == "resume" ]]; then
  initial_remaining="$(remaining_budget_seconds)"
  if [[ ! "$initial_remaining" =~ ^[0-9]+$ || "$initial_remaining" -le 0 ]]; then
    echo "immutable training deadline is exhausted" >&2
    exit 3
  fi
fi
if ! mkdir -m 700 "$LAUNCH"; then
  echo "refusing to reuse fine-support training attempt" >&2
  exit 3
fi
printf '{"status":"gpu_admission","run_id":"%s","attempt_id":"%s","mode":"%s","code_commit":"%s"}\n' \
  "$RUN_ID" "$ATTEMPT_ID" "$MODE" "$(git rev-parse HEAD)" > "$LAUNCH/status.json"
finish() {
  code=$?
  trap - EXIT
  printf '{"run_id":"%s","attempt_id":"%s","mode":"%s","controller_exit_code":%d,"finished_at":"%s","code_commit":"%s"}\n' \
    "$RUN_ID" "$ATTEMPT_ID" "$MODE" "$code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
    > "$LAUNCH/exit.json"
  if [[ "$code" -eq 0 ]]; then
    printf '{"status":"complete","run_id":"%s","attempt_id":"%s","mode":"%s","code_commit":"%s"}\n' \
      "$RUN_ID" "$ATTEMPT_ID" "$MODE" "$(git rev-parse HEAD)" > "$LAUNCH/status.json"
  else
    printf '{"status":"failed","run_id":"%s","attempt_id":"%s","mode":"%s","exit_code":%d,"code_commit":"%s"}\n' \
      "$RUN_ID" "$ATTEMPT_ID" "$MODE" "$code" "$(git rev-parse HEAD)" > "$LAUNCH/status.json"
  fi
  exit "$code"
}
trap finish EXIT

exec 9>"/home/autoresearch_results/direct_dynamics_all_hours_v1/launches/.gpu_job.lock"
if ! flock -n 9; then
  echo "another project launch owns the GPU lock" >&2
  exit 4
fi
GPU_UUID="$(scripts/require_single_gpu_uuid.sh)"
readonly GPU_UUID

read_gpu() {
  local observation
  observation="$(nvidia-smi --id=0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')" || return 1
  [[ "$observation" =~ ^[0-9]+,[0-9]+$ ]] || return 1
  GPU_MEMORY="${observation%%,*}"
  GPU_UTILIZATION="${observation##*,}"
}
read_gpu || exit 5
initial_memory="$GPU_MEMORY"
if [[ "$GPU_MEMORY" -gt 1024 ]]; then
  for sample in {1..11}; do
    read_gpu || exit 6
    if [[ "$GPU_UTILIZATION" -ge 5 ]]; then
      echo "GPU busy: utilization=${GPU_UTILIZATION}%" >&2
      exit 7
    fi
    [[ "$sample" -eq 11 ]] || sleep 30
  done
fi
read_gpu || exit 8
if [[ "$initial_memory" -le 1024 && "$GPU_MEMORY" -gt 1024 ]]; then
  for sample in {1..11}; do
    read_gpu || exit 9
    if [[ "$GPU_UTILIZATION" -ge 5 ]]; then
      echo "GPU became busy: utilization=${GPU_UTILIZATION}%" >&2
      exit 10
    fi
    [[ "$sample" -eq 11 ]] || sleep 30
  done
  read_gpu || exit 11
fi
if [[ "$GPU_MEMORY" -gt 1024 && "$GPU_UTILIZATION" -ge 5 ]]; then
  echo "GPU busy at final observation" >&2
  exit 12
fi

export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export CLEARML_REQUIRE_ONLINE=1
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6
resume_arg=()
if [[ "$MODE" == "resume" ]]; then
  resume_arg=(--resume)
fi
term_seconds=1740
kill_after_seconds=60
if [[ "$MODE" == "resume" ]]; then
  remaining="$(remaining_budget_seconds)"
  if [[ ! "$remaining" =~ ^[0-9]+$ || "$remaining" -le 2 ]]; then
    echo "immutable training deadline expired during GPU admission" >&2
    exit 13
  fi
  kill_after_seconds=$(( remaining / 4 ))
  if [[ "$kill_after_seconds" -gt 60 ]]; then
    kill_after_seconds=60
  elif [[ "$kill_after_seconds" -lt 1 ]]; then
    kill_after_seconds=1
  fi
  term_seconds=$(( remaining - kill_after_seconds ))
fi
timeout --foreground --signal=TERM --kill-after="${kill_after_seconds}s" "${term_seconds}s" \
  python -m assim_lib.direct_dynamics_fine_support_proper_training \
  --config "$CONFIG" --output "$OUTPUT" "${resume_arg[@]}"
