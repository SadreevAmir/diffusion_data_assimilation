# Latent-temperature compact-result reconciliation

Status: RECONCILED_NEGATIVE

The admissible compact payload from `latent_temperature_1p30_gate_retry1`
records `gate.overall_eligible=false`. Raw/candidate `analysis_fair_crps` is
`0.0584905850`/`0.0631177443`, with paired delta `0.0046271592`, date-bootstrap
95% interval `[0.0014355657,0.0075956683]`, and separately labelled four-case-
block sensitivity interval `[-0.0005197500,0.0082640843]`. Raw/candidate
`analysis_crps` is `0.0621082810`/`0.0684128432`. Reliability passes; proper-
score, boundary and spatial/physical families fail. The frozen construction is
therefore a negative mechanism result and no second temperature is admissible.

## Admissible evidence unit

Reconcile only the compact summary produced by
`latent_temperature_1p30_gate_retry1` from
`latent_temperature_1p30_sampling_retry1`. The retry sampling artifact is an
integrity-checked, server-side recovery of
`latent_temperature_1p30_sampling_valid`; it is not a second sampling run or a
different scientific candidate. Job completion, recovery aggregates, the
failed wrapper artifact, or a different gate id are not substitutes. Raw
ensembles remain server-side.

Before admitting either a positive or negative claim, require all of the
following in the same compact payload:

- exactly 40 completed cases and ten finite members per case;
- `gate.overall_eligible` and an explicit decision for every mandatory
  proper-score, finite-ensemble reliability, boundary, spatial/physical and
  operational family;
- raw and candidate `analysis_fair_crps`, their relative change, paired-date
  interval and the separately labelled four-case-block sensitivity interval;
- raw and candidate `analysis_crps` with its tolerance decision;
- checkpoint identity, exact latent scale `1.30`, base-seed schedule, base and
  scaled latent-hash checks, unchanged solver and conditioning metadata, and
  zero missing, substituted or non-finite members.
- recovery provenance tying all 40x10 sample hashes to
  `latent_temperature_1p30_sampling_valid`, with no GPU recomputation or sample
  mutation.

Any missing field, non-finite value, identifier mismatch, incomplete case,
failed provenance check or disagreement between a family decision and its
reported metric fails reconciliation. Do not infer a missing decision from an
aggregate mean and do not combine fields from retries or separate payloads.

## Completed atomic publication update

The complete payload passed the checks above. The required instruction was to
update all four files in one change; the following four publication files now
represent that reconciled evidence state:

1. `PAPER_DRAFT.md`: add the frozen mechanism, raw/candidate proper scores,
   paired-date interval, mandatory-family decisions and the no-compensation
   conclusion.
2. `CLAIM_LEDGER.md`: add one claim row whose status follows
   `gate.overall_eligible`, with the compact payload as evidence and the
   one-checkpoint development-period limitation.
3. `REPRODUCIBILITY.md`: record the failed wrapper id, recovery id and retry
   gate id, the frozen scale and seed schedule, case/member completeness,
   provenance checks and compact reconciliation anchors.
4. `PUBLICATION_READINESS.md`: update the scientific blocker honestly while
   retaining every independent-evaluation, baseline, authorship and release
   blocker that the result does not resolve.

Run `python3 paper/check_publication_artifacts.py` after the atomic update. A
passing local integrity audit checks consistency only; it does not override the
trusted gate.

## Decision branches

Positive admission requires all of these simultaneously:

- `gate.overall_eligible=true`;
- candidate `analysis_fair_crps` is at least 3% lower than raw and the upper
  endpoint of the paired-date 95% interval for candidate minus raw is below
  zero;
- `analysis_crps` satisfies its frozen tolerance;
- every mandatory family and every operational provenance check passes.

Only then may the manuscript call the frozen 1.30 construction an eligible
validation-set calibration. The claim remains limited to the fixed checkpoint
and development envelope; it does not establish independent generalization.

If any requirement fails, record the method as a negative mechanism result,
name every failed family and the decision-bearing metrics, and do not change
the temperature. The next mechanistically distinct route is the already
predeclared locked-MC-dropout pair; its exact reviewed sampling and dependent
gate contracts must be proposed unchanged. No dropout probability, layer set,
mask refresh rule, seed, checkpoint, solver or member schedule may be selected
after inspecting this result.
