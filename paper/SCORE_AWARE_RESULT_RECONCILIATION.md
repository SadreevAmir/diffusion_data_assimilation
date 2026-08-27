# Score-aware raw-scenario result reconciliation

Status: PRE_RESULT_NO_TRUSTED_MODE

This file is the fail-closed publication handoff for the frozen mechanism in
`NEXT_SCORE_AWARE_RAW_REWEIGHTING_CONTRACT.md`. It is not evidence that a
trusted runner exists, does not authorize a proposal, and contains no scientific
result. Replace the pre-result status only after one literal controller-visible
mode has passed the combined v2 admission and returned its exact admitted
four-file compact directory.

## Authoritative input and admission

The only admissible evidence directory contains exactly
`case_selection.json`, `aggregate_selection.json`,
`paired_uncertainty.json` and `gate_decision.json`. Before reconciliation, run
`validate_score_aware_raw_reweighting_admission.py` against that directory and
the controller-visible admission record. Require `decision_bearing=True`, exact
runner, contract and reference identities, semantic parity, cross-file parity,
and a final byte-exact `compact_directory_sha256` match. A missing file, extra
file, changed byte, unknown mode, failed invariant or failed admission leaves
this document in the pre-result state and changes no manuscript claim.

Record these immutable values verbatim when evidence arrives:

- trusted mode: `PENDING`
- experiment_id: `PENDING`
- admission record identity: `PENDING`
- `compact_directory_sha256`: `PENDING`
- candidate identifier: `PENDING`
- completed cases and ensemble size: `PENDING`
- mandatory family Booleans: `PENDING`
- `overall_eligible`: `PENDING`

No aggregate metric may be used to infer a missing family Boolean or
`overall_eligible`. The gate record is authoritative for the decision; the
other three files must reconcile its identities, counts and reported metrics.

## Mutually exclusive scientific branches

Apply exactly one branch after successful admission.

### Positive branch

This branch is legal only when all five mandatory families and
`overall_eligible` are literally `true`. Record raw and candidate
`analysis_fair_crps`, `analysis_crps`, ensemble-mean RMSE, paired date intervals,
the non-gating block sensitivity, randomized-rank and attainable-order coverage
diagnostics, boundary errors, spatial/physical diagnostics, source-copy and mask
invariants, effective sample size, and operational counts. The claim is limited
to leakage-safe score-aware reweighting of existing learned-joint scenarios on
the development envelope. It does not imply independent generalization.

### Negative branch

This branch is mandatory when any mandatory family or `overall_eligible` is
literally `false`. Name every failed family and report the same compact metric
set without compensating one failure by gains elsewhere. Interpret the failure
by the frozen categories: proper-score failure rejects useful score-aware
selection; reliability failure rejects distributional repair; copy, mask or
operational failure invalidates execution; spatial/physical failure shows that
changed scenario frequencies damage the forecast distribution. Do not tune the
loss, ridge, predictors, weights, systematic selection, folds or thresholds.

## Atomic publication update map

After admission, update all of the following in one worktree change; a partial
update is not publication-ready.

1. `PAPER_DRAFT.md`: replace the pre-result text in `Final frozen development
   fallback` with the exact experiment identity, gate decision, family matrix,
   principal effect sizes and bounded interpretation. Add the new claim ID to
   `Claim-ledger traceability` and `Empirical evidence traceability`.
2. `CLAIM_LEDGER.md`: append the next contiguous claim row. The positive wording
   requires the positive branch above; otherwise record a rejected mechanism
   claim, its failed families and the no-retuning interpretation.
3. `PUBLICATION_READINESS.md`: append the admission identity, directory digest,
   decision and atomic reconciliation audit. A negative result cannot close the
   eligible-calibration row. A positive result may change that row only after
   the checker verifies every mandatory family; unrelated baseline rows remain
   unchanged.
4. `REPRODUCIBILITY.md`: replace the pending handoff with the exact trusted mode,
   experiment identity, admission command inputs, compact filenames, digest and
   immutable decision. Preserve `summary_only`; do not retrieve raw fields.
5. This file: replace every `PENDING`, set exactly one of
   `RECONCILED_POSITIVE` or `RECONCILED_NEGATIVE`, and delete no frozen
   requirement.

## Fail-closed consistency rules

- Pre-result status requires every immutable value above to remain `PENDING`.
- A reconciled status requires no `PENDING`, successful combined admission and
  one exact compact-directory digest.
- `RECONCILED_POSITIVE` requires all family Booleans and `overall_eligible=true`.
- `RECONCILED_NEGATIVE` requires at least one false family and
  `overall_eligible=false`.
- The manuscript, claim ledger, readiness audit, reproducibility handoff and
  this file must name the same experiment, candidate and decision.
- Neither branch closes conformal, probabilistic-DA or deterministic-comparison
  evidence rows. Readiness is derived from the full blocker matrix, never from
  this result alone.

