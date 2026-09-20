#!/bin/bash
# P1-confirm (EXPERIMENT_SPECIFICATIONS §P1.2/P1.3): frozen config = pilot config with 20k steps, arms K0/K1/Kfresh/Krho05/Krho09,
# NEW seed blocks 10..14, checkpoints 0/500/2000/5000/10000/20000. Frozen 2026-09-20 09:00 before generating any confirmation data.
set -e
cd /Users/amir/sciml/diffusion_data_assimilation
export P1_STAGE=confirm
PY=.venv/bin/python; S=docs/research/invariance_audit_v1/p1_pilot.py
for b in 10 11 12 13 14; do
  $PY $S make-data --block $b
  for arm in K0 K1 Kfresh Krho05 Krho09; do
    $PY $S train --block $b --arm $arm
    $PY $S sample-evaluate --block $b --arm $arm
    $PY $S audit-v --block $b --arm $arm
    $PY $S diag-final --block $b --arm $arm
  done
  $PY $S baselines --block $b
done
$PY $S report --blocks 10 11 12 13 14
echo CONFIRM_DONE
