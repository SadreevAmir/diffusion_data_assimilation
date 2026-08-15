# Publication readiness audit

Audit date: 2026-08-15

Publication status: READY_FOR_HUMAN_REVIEW

Required scientific blockers: none

## Scientific readiness decision

The narrow validation-mechanism paper is internally auditable at the aggregate
case-mean level. The supplied compact-table contract rules out the previously
planned paired bootstrap because it lacks raw case-level fields; the manuscript
now states that limitation instead of treating an unavailable display as a
scientific blocker. Aggregate evidence is sufficient for the deliberately
narrow mechanism claim, and no additional experiment is required for human
review.

## Audit findings

- The abstract, results, evidence table, claim ledger, research plan and handoff
  agree on the selected fold scales and aggregate calibration metrics.
- `paper/REPRODUCIBILITY.md` freezes the audited compact input contract,
  expected aggregate reconciliation values and a numeric provenance-failure
  threshold.
- The claim is limited to one checkpoint, one seed, 40 development dates and
  ten members; cross-fitting is not described as independent generalization.
- The clipping caveat is explicit: the bounded RMSE change is not presented as
  intrinsic improvement of the pre-clipping ensemble center.
- The manuscript does not claim deterministic-method superiority, complete
  tail calibration, conditional calibration or fieldwise coverage.
- The manuscript contains an aggregate evidence table, scholarly references,
  data/code availability language and ethics/competing-interests language.
- No additional CPU or GPU experiment is needed for the present narrow claim.

## Honest limitations

- Raw and corrected case-level fields are not jointly available in the compact
  contract, so paired uncertainty intervals and improved-case counts are not
  reported or reconstructed.
- The result is a mechanism study on a reused development period, not an
  independent temporal generalization estimate.
- Evidence covers one legacy checkpoint, one sampling seed and a ten-member
  ensemble; conditional, spatial and fieldwise reliability remain untested.
- The nominal interval diagnostics are descriptive for the finite ensemble;
  residual upper-tail undercoverage remains.

## External-only actions before submission

- Authors must select the target venue/template and supply names, affiliations,
  funding, final competing-interest declarations, acknowledgements and the
  approved code/data release location and license.
- Human reviewers must decide whether the narrow validation-mechanism scope is
  appropriate for that venue and perform final copy-editing and
  reference-format checks.
- Any broader claim requires genuinely independent evidence and a new audited
  scientific decision; it is not implied by this readiness status.
