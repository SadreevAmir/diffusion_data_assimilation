#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-config/experiments/occurrence_intensity_e1_sentinel.json}"
OUTPUT_DIR="${2:-/tmp/occurrence_intensity_e1_sentinel}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "Neither python3 nor python is available; set PYTHON_BIN explicitly." >&2
    exit 127
  fi
fi

"$PYTHON_BIN" - "$CONFIG_PATH" "$OUTPUT_DIR" <<'PY'
import json
import sys
from assim_lib.occurrence_intensity_e1 import run_engineering_sentinel

print(json.dumps(run_engineering_sentinel(sys.argv[1], sys.argv[2]), sort_keys=True))
PY

"$PYTHON_BIN" paper/validate_occurrence_intensity_e1_compact_result.py \
  "$CONFIG_PATH" \
  "$OUTPUT_DIR/run_status.json" \
  "$OUTPUT_DIR/artifact_manifest.json"
