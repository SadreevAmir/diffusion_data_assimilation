#!/usr/bin/env bash
set -euo pipefail
SMOKE_ROOT="$(mktemp -d)"
trap 'rm -rf -- "$SMOKE_ROOT"' EXIT
mkdir -p "$SMOKE_ROOT/bin"
SMOKE_LOG="$SMOKE_ROOT/python.log"
export SMOKE_LOG
printf '%s\n' '#!/usr/bin/env bash' 'printf "%s|%s|%s|%s\n" "$CUDA_VISIBLE_DEVICES" "$CLEARML_REQUIRE_ONLINE" "$OMP_NUM_THREADS" "$*" > "$SMOKE_LOG"' > "$SMOKE_ROOT/bin/python"
chmod +x "$SMOKE_ROOT/bin/python"
PATH="$SMOKE_ROOT/bin:$PATH" \
SIT_SUPPORT_DECODER_OUTPUT="$SMOKE_ROOT/status.json" \
scripts/run_direct_dynamics_sit_support_decoder_audit.sh
grep -F '|1|6|-m assim_lib.direct_dynamics_sit_support_decoder_audit' "$SMOKE_LOG"
