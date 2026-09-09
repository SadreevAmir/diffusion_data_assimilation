#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "usage: $0 RUN_ID" >&2
  exit 2
fi

run_id="$1"
result_root="/home/autoresearch_results/direct_dynamics_all_hours_v1/censored_joint_frozen_sample_audit"
launch_root="/home/autoresearch_results/direct_dynamics_all_hours_v1/censored_joint_frozen_sample_audit_launches"
output_dir="$result_root/$run_id"
status_dir="$launch_root/$run_id"
status_file="$status_dir/status.json"

if [[ -e "$output_dir" || -e "$status_dir" ]]; then
  echo "refusing to reuse frozen audit run id: $run_id" >&2
  exit 2
fi
mkdir -p "$result_root" "$launch_root" "$status_dir"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6
export TORCH_NUM_THREADS=6
export MPLBACKEND=Agg

started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '{"status":"running","run_id":"%s","started_at":"%s","cpu_only":true}\n' \
  "$run_id" "$started_at" > "$status_file"

set +e
timeout --signal=TERM --kill-after=120s 7080s python -m assim_lib.censored_joint_frozen_sample_audit \
  --baseline /home/autoresearch_results/direct_dynamics_all_hours_v1/censored_joint_heldout/heldout_energy_118d003_20260909T193003Z/diagnostics/update_1024/validation/samples.pt \
  --candidate /home/autoresearch_results/direct_dynamics_all_hours_v1/censored_joint_multiscale/multiscale_547942d_20260909T205619Z/diagnostics/update_1024/validation/samples.pt \
  --historical /home/autoresearch_results/direct_dynamics_all_hours_v1/evaluations/paired_val_e6_e8_20260909_v2/ema_epoch6_visual_samples.pt \
  --output-dir "$output_dir"
exit_code=$?
set -e

finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [[ $exit_code -eq 0 ]]; then
  printf '{"status":"finished","run_id":"%s","exit_code":0,"finished_at":"%s","cpu_only":true}\n' \
    "$run_id" "$finished_at" > "$status_file"
else
  printf '{"status":"failed","run_id":"%s","exit_code":%d,"finished_at":"%s","cpu_only":true}\n' \
    "$run_id" "$exit_code" "$finished_at" > "$status_file"
fi
exit "$exit_code"
