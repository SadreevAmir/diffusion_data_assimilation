# Frozen locked-MC-dropout wrapper recovery contract

Status: implementation-ready, fail closed. This document freezes an operational
recovery step; it does not record a scientific result and does not authorize new
sampling.

## Purpose

Recover the already completed internal outputs of
`locked_mc_dropout_p010_sampling_valid` after its wrapper-only external-case
schema failure. The recovery must run on server CPU and must not read, alter or
recompute model outputs outside the server result root.

## Accepted input

The finalizer accepts exactly one source identifier:
`source_experiment=locked_mc_dropout_p010_sampling_valid`. It must bind that
identifier to the audited failed wrapper, the recorded launch, the publication
commit used by that launch and its external `cases_file=cases.json`. Any missing
or conflicting binding is a hard failure.

No fold, seed, checkpoint, dropout, solver, path, threshold, repair or sampling
parameter is runtime-configurable.

## Required validation

Before copying anything, the finalizer must verify:

- exactly 40 expected cases and exactly ten members for every case;
- the recorded final EMA checkpoint and unchanged conditioning metadata;
- finite outputs and exact recorded sample hashes for all 400 members;
- the unit-latent seed schedule and stored initial-noise hashes;
- dropout probability `0.1`, the exact fourteen admitted layers and the fixed
  mask seed rule `271828000 + 140*case + 14*member + layer`;
- presence and internal consistency of all metadata needed for the dependent
  gate to validate 5,600 frozen masks.

The finalizer must fail on an absent, duplicate, substituted, non-finite or
hash-mismatched member; an unexpected case; metadata disagreement; an
unapproved publication commit; or any attempted path outside the server result
root. It must not fill, repair or regenerate a missing artifact.

## Output and dependency boundary

On success, copy the validated artifacts only within the server result root and
emit a compact recovery summary containing the exact source, recovered
experiment id, publication commit, launch identity, case/member counts and
aggregate pass/fail counts for every check above. Retrieval remains
`summary_only`.

The scientific gate must use the recovered experiment id as its sole
`source_experiment`. Worker completion, the failed wrapper and this recovery
summary are not scientific evidence. Only the dependent
`validation_locked_mc_dropout_gate` compact payload may determine
`gate.overall_eligible` under
`LOCKED_MC_DROPOUT_RESULT_RECONCILIATION.md`.

## Frozen decision rule

If recovery fails, preserve the server outputs unchanged and report the exact
failed invariant; do not launch GPU sampling. If recovery succeeds, immediately
run the dependent unchanged gate. No scientific or publication claim is
admissible between these two stages.
