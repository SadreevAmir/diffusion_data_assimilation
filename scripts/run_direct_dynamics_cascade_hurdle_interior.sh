#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
RUN_ID="${CASCADE_HURDLE_RUN_ID:?CASCADE_HURDLE_RUN_ID is required}"; [[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ ]] || exit 2
OUTPUT="/home/autoresearch_results/direct_dynamics_cascade_v2/hurdle_interior/$RUN_ID"; [[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || exit 3
export CUDA_VISIBLE_DEVICES="" CLEARML_REQUIRE_ONLINE=1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6
set +e; timeout --signal=TERM --kill-after=30s 3570s python -m assim_lib.direct_dynamics_cascade_hurdle_interior --config config/experiments/audit_direct_dynamics_cascade_hurdle_interior_v1.json --output "$OUTPUT"; STATUS=$?; set -e; mkdir -p "$OUTPUT"
python - "$OUTPUT/launcher_exit_status.json" "$STATUS" <<'PY'
import json, os, sys
path, status=sys.argv[1],int(sys.argv[2]); temporary=f"{path}.{os.getpid()}.incomplete"
with open(temporary,"w",encoding="utf-8") as handle: json.dump({"status":status},handle,indent=2,allow_nan=False)
os.replace(temporary,path)
PY
exit "$STATUS"
