#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-config/experiments/occurrence_intensity_e1_sentinel.json}"
OUTPUT_DIR="${2:-/tmp/occurrence_intensity_e1_sentinel}"
python - "$CONFIG_PATH" "$OUTPUT_DIR" <<'PY'
import json
import sys
from assim_lib.occurrence_intensity_e1 import run_engineering_sentinel

print(json.dumps(run_engineering_sentinel(sys.argv[1], sys.argv[2]), sort_keys=True))
PY
