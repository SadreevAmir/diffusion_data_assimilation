#!/usr/bin/env bash
set -euo pipefail

# Run this script inside the server container, from the repository checkout.
# It stops the current assim_lib training process, pulls the requested branch,
# recomputes M2M normalization statistics, and starts the diffusion-balanced run
# detached with nohup.

BRANCH="${BRANCH:-conditioned-model}"
REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-config/experiments/train_m2m_concat_conditioning_diffusion_balanced_2f.json}"
METHOD_CONFIG="${METHOD_CONFIG:-config/methods/concat_conditioning_diffusion_balanced_2f.json}"
DATA_CONFIG="${DATA_CONFIG:-config/data/m2m_2f_1y.json}"
STATS_OUTPUT="${STATS_OUTPUT:-outputs/stats/m2m_2f_1y_train_stats.json}"
STATS_WORKERS="${STATS_WORKERS:-8}"
LOG_PATH="${LOG_PATH:-logs/train_diffusion_balanced_background_mask.log}"

cd "$REPO_DIR"

echo "[restart] repo: $REPO_DIR"
echo "[restart] branch: $BRANCH"
echo "[restart] stopping current assim_lib training processes"

mapfile -t PIDS < <(ps -eo pid,args | awk '/[p]ython[0-9.]* -m assim_lib\.main/ {print $1}')
if ((${#PIDS[@]} > 0)); then
  printf '[restart] sending TERM to PIDs: %s\n' "${PIDS[*]}"
  kill -TERM "${PIDS[@]}" || true
  sleep 10

  mapfile -t LIVE_PIDS < <(ps -eo pid,args | awk '/[p]ython[0-9.]* -m assim_lib\.main/ {print $1}')
  if ((${#LIVE_PIDS[@]} > 0)); then
    printf '[restart] sending KILL to remaining PIDs: %s\n' "${LIVE_PIDS[*]}"
    kill -KILL "${LIVE_PIDS[@]}" || true
  fi
else
  echo "[restart] no running assim_lib.main process found"
fi

echo "[restart] fetching and resetting to origin/$BRANCH"
git fetch origin "$BRANCH"
git reset --hard "origin/$BRANCH"

echo "[restart] HEAD: $(git rev-parse --short HEAD)"
echo "[restart] verifying configs"
grep -n '"training_objective": "diffusion"' "$METHOD_CONFIG"
grep -n '"in_channels": 13' "$METHOD_CONFIG"
grep -n '"model_config": "../methods/concat_conditioning_diffusion_balanced_2f.json"' "$EXPERIMENT_CONFIG"

echo "[restart] computing M2M statistics"
mkdir -p "$(dirname "$STATS_OUTPUT")"
python3 -m assim_lib.m2m_stats \
  --config "$DATA_CONFIG" \
  --split train \
  --hour-mode auto \
  --workers "$STATS_WORKERS" \
  --output "$STATS_OUTPUT" \
  --update-config

echo "[restart] verifying updated stats"
grep -n '"means"\|"stds"' "$DATA_CONFIG"

echo "[restart] starting training"
mkdir -p "$(dirname "$LOG_PATH")"
nohup python3 -m assim_lib.main \
  --config "$EXPERIMENT_CONFIG" \
  > "$LOG_PATH" 2>&1 &
TRAIN_PID=$!

echo "[restart] started PID: $TRAIN_PID"
echo "[restart] log: $LOG_PATH"
echo "[restart] follow with:"
echo "tail -f $LOG_PATH"
