#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-/tmp/occurrence_intensity_e3_sentinel}"
MAPPING_RECORD="${2:-}"
CONFIG="config/experiments/occurrence_intensity_e3_sentinel.json"
CONTRACT="paper/NEXT_OCCURRENCE_INTENSITY_E3_CONTRACT.md"
STATUS="${OUTPUT_DIR}/run_status.json"
MANIFEST="${OUTPUT_DIR}/artifact_manifest.json"

# Fail before partial evidence is produced when the independent environment is
# not capable of executing the unchanged publication-side correctness tests.
python3 -c 'import torch'
python3 test/test_occurrence_intensity_e1.py
python3 test/test_occurrence_intensity_e2.py
python3 test/test_occurrence_intensity_e3.py

scripts/run_occurrence_intensity_e3_sentinel.sh "$CONFIG" "$OUTPUT_DIR"
python3 paper/validate_occurrence_intensity_e3_compact_result.py \
  "$CONFIG" "$STATUS" "$MANIFEST"

if [[ -n "$MAPPING_RECORD" ]]; then
  python3 paper/validate_occurrence_intensity_e3_mapping_record.py \
    "$MAPPING_RECORD" "$CONTRACT" "$CONFIG" "$STATUS" "$MANIFEST"
else
  printf '%s\n' 'E3_MAPPING_RECORD_PENDING_CONTROLLER_IDENTITIES'
fi
