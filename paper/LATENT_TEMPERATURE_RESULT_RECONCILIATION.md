# Latent-temperature compact-result reconciliation

Status: pre-result, fail closed. This checklist records no scientific outcome.

## Admissible evidence unit

Reconcile only the compact summary produced by
`latent_temperature_1p30_gate_valid` from
`latent_temperature_1p30_sampling_valid`. Job completion, sampling aggregates,
or a different gate id are not substitutes. Raw ensembles remain server-side.

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

Any missing field, non-finite value, identifier mismatch, incomplete case,
failed provenance check or disagreement between a family decision and its
reported metric fails reconciliation. Do not infer a missing decision from an
aggregate mean and do not combine fields from retries or separate payloads.

## Atomic publication update

After the payload passes the checks above, update all four files in one change:

1. `PAPER_DRAFT.md`: add the frozen mechanism, raw/candidate proper scores,
   paired-date interval, mandatory-family decisions and the no-compensation
   conclusion.
2. `CLAIM_LEDGER.md`: add one claim row whose status follows
   `gate.overall_eligible`, with the compact payload as evidence and the
   one-checkpoint development-period limitation.
3. `REPRODUCIBILITY.md`: record both experiment ids, the frozen scale and seed
   schedule, case/member completeness, provenance checks and compact
   reconciliation anchors.
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

