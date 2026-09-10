#!/usr/bin/env bash
set -euo pipefail

if ! inventory="$(nvidia-smi --id=0 --query-gpu=uuid --format=csv,noheader)"; then
  echo "GPU admission failed: UUID inventory command failed" >&2
  exit 5
fi
count=0
gpu_uuid=""
while IFS= read -r row || [[ -n "$row" ]]; do
  [[ -z "$row" ]] && continue
  if [[ ! "$row" =~ ^GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$ ]]; then
    echo "GPU admission failed: expected exactly one valid GPU UUID" >&2
    exit 5
  fi
  count=$((count + 1))
  gpu_uuid="$row"
done <<< "$inventory"
if [[ "$count" -ne 1 ]]; then
  echo "GPU admission failed: expected exactly one valid GPU UUID" >&2
  exit 5
fi
printf '%s\n' "$gpu_uuid"
