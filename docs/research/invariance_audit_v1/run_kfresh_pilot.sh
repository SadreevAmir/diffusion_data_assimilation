#!/bin/bash
# Review-3 §6 diagnostic pilot (development blocks 0,1; 20k steps = confirm length): Kfresh (fresh key per example) vs
# KfreshBatch (one fresh key per batch) vs K0, paired data/batch/z0/t streams and identical init; same inference rule.
set -e; cd /Users/amir/sciml/diffusion_data_assimilation; export P1_STAGE=confirm; PY=.venv/bin/python; S=docs/research/invariance_audit_v1/p1_pilot.py
until grep -q "written\|Traceback" autoresearch_results/invariance_audit_v1/confirm_decomposition_v2_run.log; do sleep 60; done
for b in 0 1; do
  [ -f autoresearch_results/invariance_audit_v1/p1_confirm/block$b/train_data.npz ] || $PY $S make-data --block $b
  for arm in K0 Kfresh KfreshBatch; do
    $PY $S train --block $b --arm $arm; $PY $S sample-evaluate --block $b --arm $arm; $PY $S diag-final --block $b --arm $arm
  done
done
echo KFRESH_PILOT_DONE
