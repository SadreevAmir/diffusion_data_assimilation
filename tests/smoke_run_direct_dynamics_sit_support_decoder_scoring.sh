#!/usr/bin/env bash
set -euo pipefail

TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TEMP_DIR"' EXIT
SMOKE_LOG="$TEMP_DIR/invocation.log"
cat > "$TEMP_DIR/python" <<'EOF'
#!/usr/bin/env bash
printf '%s|%s|%s|%s|%s|%s|%s\n' "$CUDA_VISIBLE_DEVICES" "$OMP_NUM_THREADS" "$MKL_NUM_THREADS" "$OPENBLAS_NUM_THREADS" "$NUMEXPR_NUM_THREADS" "$CLEARML_REQUIRE_ONLINE" "$*" > "${SMOKE_LOG:?}"
EOF
chmod +x "$TEMP_DIR/python"
PATH="$TEMP_DIR:$PATH" SMOKE_LOG="$SMOKE_LOG" \
  SIT_SUPPORT_DECODER_SCORING_OUTPUT="$TEMP_DIR/status.json" \
  scripts/run_direct_dynamics_sit_support_decoder_scoring.sh
grep -F -- '|6|6|6|6|1|-m assim_lib.direct_dynamics_sit_support_decoder_scoring' "$SMOKE_LOG"
