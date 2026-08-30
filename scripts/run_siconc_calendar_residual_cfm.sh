#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
CANONICAL_CONFIG="$REPO_DIR/config/experiments/train_siconc_calendar_residual_cfm.json"
CONFIG="$CANONICAL_CONFIG"
RUNTIME_CONFIG="$OUTPUT_DIR/runtime_training_config.json"
TRAINING_DIR="$OUTPUT_DIR/training"
STATUS_PATH="$OUTPUT_DIR/run_status.json"

if [[ ! "$CUDA_DEVICE" =~ ^[0-9]+$ ]]; then
  echo "[calendar-residual-cfm] exactly one numeric CUDA device is required" >&2
  exit 2
fi
if [[ ! -f "$CANONICAL_CONFIG" ]]; then
  echo "[calendar-residual-cfm] missing canonical config" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

write_status() {
  local status="$1"
  local detail="${2:-}"
  "$PYTHON_BIN" - "$STATUS_PATH" "$status" "$detail" <<'PY'
import json
import os
import pathlib
import sys
from datetime import datetime, timezone

path = pathlib.Path(sys.argv[1])
payload = {
    "status": sys.argv[2],
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "experiment_id": "siconc_calendar_residual_cfm_training_v1",
    "detail": sys.argv[3],
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
with temporary.open("rb") as handle:
    os.fsync(handle.fileno())
temporary.replace(path)
PY
}

completed=false
on_exit() {
  local code=$?
  if [[ "$completed" != true ]]; then
    write_status "failed" "training wrapper exited with code $code"
  fi
  exit "$code"
}
trap on_exit EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

write_status "running" "building immutable runtime config"

"$PYTHON_BIN" - "$CANONICAL_CONFIG" "$RUNTIME_CONFIG" "$REPO_DIR" "$OUTPUT_DIR" <<'PY'
import json
import os
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
destination = pathlib.Path(sys.argv[2])
repository = pathlib.Path(sys.argv[3])
output = pathlib.Path(sys.argv[4])
payload = json.loads(source.read_text(encoding="utf-8"))
payload["data_config"] = str(repository / "config/data/m2m_siconc_calendar_residual_1y.json")
payload["model_config"] = str(repository / "config/methods/siconc_calendar_residual_cfm.json")
payload["training"] = {
    **payload.get("training", {}),
    "base_output_dir": str(output),
    "run_name": "training",
    "clearml_enabled": False,
}
temporary = destination.with_suffix(destination.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
with temporary.open("rb") as handle:
    os.fsync(handle.fileno())
temporary.replace(destination)
PY

cd "$REPO_DIR"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export PYTHONDONTWRITEBYTECODE=1
write_status "running" "training"
"$PYTHON_BIN" -m assim_lib.main --config "$RUNTIME_CONFIG"

test -f "$TRAINING_DIR/metadata.json"
test -f "$TRAINING_DIR/metrics.json"
test -f "$TRAINING_DIR/ema_last_model.pth"

"$PYTHON_BIN" - "$OUTPUT_DIR" "$REPO_DIR" <<'PY'
import hashlib
import json
import os
import pathlib
import subprocess
import sys
from datetime import datetime, timezone

output = pathlib.Path(sys.argv[1])
repository = pathlib.Path(sys.argv[2])
training = output / "training"
source_metadata = json.loads((training / "metadata.json").read_text(encoding="utf-8"))
metrics = json.loads((training / "metrics.json").read_text(encoding="utf-8"))
commit = subprocess.check_output(
    ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
).strip()
payload = {
    "experiment_id": "siconc_calendar_residual_cfm_training_v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "publication_commit": commit,
    "training_run_dir": str(training),
    "checkpoint": "ema_last_model.pth",
    "checkpoint_sha256": hashlib.sha256((training / "ema_last_model.pth").read_bytes()).hexdigest(),
    "dataset_provenance": source_metadata.get("dataset_provenance", {}),
    "training_config": source_metadata.get("training_config", {}),
    "num_history_entries": len(metrics) if isinstance(metrics, list) else None,
    "test_data_used": False,
}
path = output / "metadata.json"
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
with temporary.open("rb") as handle:
    os.fsync(handle.fileno())
temporary.replace(path)
PY

completed=true
write_status "completed" "training and checkpoint provenance completed"
trap - EXIT
