#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-config/experiments/occurrence_intensity_e1_sentinel.json}"
python - "$CONFIG_PATH" <<'PY'
import json
import sys
from assim_lib.occurrence_intensity_e1 import controller_request

request = controller_request(sys.argv[1])
print(json.dumps(request, sort_keys=True))
if not request["launch_authorized"]:
    raise SystemExit("independent admission PASS is required before server execution")
PY
