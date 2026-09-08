#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CLEARML_REQUIRE_ONLINE=1
exec python -m assim_lib.direct_dynamics_training \
  --config config/experiments/train_direct_dynamics_all_hours_v1.json
