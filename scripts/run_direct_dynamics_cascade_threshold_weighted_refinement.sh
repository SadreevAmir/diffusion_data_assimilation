#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROPER_REFINEMENT_RUN_ID="${THRESHOLD_REFINEMENT_RUN_ID:?THRESHOLD_REFINEMENT_RUN_ID is required}"
export PROPER_REFINEMENT_MODE="${THRESHOLD_REFINEMENT_MODE:-admission}"
export PROPER_REFINEMENT_CONFIG="config/experiments/train_direct_dynamics_cascade_threshold_weighted_refinement_v1.json"
export PROPER_REFINEMENT_OUTPUT_GROUP="threshold_weighted_refinement"
exec "$ROOT/scripts/run_direct_dynamics_cascade_coarse_proper_refinement.sh"
