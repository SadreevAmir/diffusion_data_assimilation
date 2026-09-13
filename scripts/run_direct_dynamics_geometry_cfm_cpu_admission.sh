#!/usr/bin/env bash
set -euo pipefail

output="${1:-docs/research/geometry_weighted_cfm_cpu_admission.json}"
python -m assim_lib.direct_dynamics_geometry_cfm --output "$output"
