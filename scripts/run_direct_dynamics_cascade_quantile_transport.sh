#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
RUN_ID="${CASCADE_QUANTILE_RUN_ID:?CASCADE_QUANTILE_RUN_ID is required}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ ]]; then exit 2; fi
OUTPUT="/home/autoresearch_results/direct_dynamics_cascade_v2/quantile_transport/$RUN_ID"
if [[ -e "$OUTPUT" || -L "$OUTPUT" ]]; then exit 3; fi
export CUDA_VISIBLE_DEVICES="" CLEARML_REQUIRE_ONLINE=1
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6
set +e
timeout --signal=TERM --kill-after=30s 3570s python -m assim_lib.direct_dynamics_cascade_quantile_transport --config config/experiments/audit_direct_dynamics_cascade_quantile_transport_v1.json --output "$OUTPUT"
STATUS=$?; set -e; mkdir -p "$OUTPUT"
python - "$OUTPUT/launcher_exit_status.json" "$STATUS" <<'PY'
import json, os, sys
path, status = sys.argv[1], int(sys.argv[2]); temporary = f"{path}.{os.getpid()}.incomplete"
with open(temporary, "w", encoding="utf-8") as handle: json.dump({"status": status}, handle, indent=2, allow_nan=False)
os.replace(temporary, path)
PY
exit "$STATUS"
