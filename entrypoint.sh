#!/usr/bin/env bash

set -euo pipefail

if (($# > 0)); then
  exec "$@"
fi

exec python -m assim_lib.main \
  --config /home/config/experiments/train_m2m_concat_conditioning_diffusion_balanced_2f.json
