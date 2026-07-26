#!/usr/bin/env bash

set -euo pipefail

exec python -m assim_lib.main \
  --config /home/config/experiments/smoke_concat_conditioning_2f.json
