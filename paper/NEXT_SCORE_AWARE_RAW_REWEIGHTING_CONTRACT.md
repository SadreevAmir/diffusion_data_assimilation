# Frozen contingent method contract: score-aware raw-scenario reweighting

Status: DESIGN_FROZEN_TERTIARY_NO_RUNNER

## Activation and scientific question

Activate this development-only mechanism only after both the frozen
whole-field rank-coherent anomaly transport and the frozen analog-rank raw-member
reweighting have completed and failed the unchanged no-compensation gate.  It
must not delay either earlier test.  The question is distinct from rank-bin
calibration: can a leakage-safe estimate of scenario-wise proper-score risk move
probability mass toward useful *existing* raw scenarios without altering their
geometry?

The learned-joint ten-member ensemble is the baseline.  A positive result may
support a calibration claim only when every mandatory family and
`overall_eligible` pass.  A negative result closes score-aware selection among
the existing scenarios; it does not authorize a new smoothing value, feature
set, loss, temperature or external evaluation.

## Frozen folds, predictors and targets

- Use the established forty ordered development cases, five contiguous
  eight-case holdouts and the non-circular three-case purge.
- Identify the cases by their exact ISO dates from `2022-01-01` at five-day
  stride.  Each per-case compact record carries the complete ordered
  `training_case_ids` retained for its fold.  Admission independently derives
  the contiguous holdout and excludes that holdout plus up to three neighboring
  cases on either side without wrapping; any identifier, order, fold or retained
  membership mismatch fails closed.
- For every raw member compute exactly seven forecast-only member descriptors:
  spatially weighted ice area, extent at `0.15`, spatial mean, spatial standard
  deviation, semivariograms at lags 1 and 4, and absolute displacement of its
  weighted area from the raw ensemble-mean area.  Append the six frozen
  forecast-only case descriptors used by the analog contracts.  No truth enters
  these predictors.
- Standardize each descriptor on retained training cases only.  A non-finite or
  zero training standard deviation is an operational failure.
- For each retained training member, define the fixed target as its spatially
  weighted mean absolute error against that date's truth plus `0.25` times its
  spatially weighted squared error.  These coefficients and spatial weights are
  constants, not selectable hyperparameters.
- Fit one ridge linear model per fold to the member targets using an unpenalized
  intercept and the thirteen standardized predictors.  The intercept is the
  training-target mean.  Center the training targets and compute predictor
  coefficients as `V diag(s/(s^2+1.0)) U^T y_centered`, using the deterministic
  SVD of the standardized training design.  Set factors for singular values at
  or below `1e-12 * s_max` to zero.  The ridge coefficient is exactly `1.0`.
  Any non-finite coefficient or prediction fails operationally.

## Frozen probability construction

For a held-out case, predict risk for its ten raw members.  Subtract the minimum
predicted risk and set

`w_m = exp(-r_m) / sum_k exp(-r_k)`.

There is no fitted temperature.  Generate ten deterministic probability
positions `u_j=(j+0.5)/10`.  Apply systematic inverse-CDF selection to member
indices ordered by `(predicted_risk, raw_member_index)`.  Copy the selected raw
field at every position.  Repetition is allowed; interpolation, clipping,
projection, perturbation and recentering are forbidden.

Record all predicted risks, normalized weights, selected source indices,
multiplicities, unique-member count and effective sample size
`1 / sum_m (n_m/10)^2`.  Every output must be bitwise equal to its recorded raw
source and preserve its zero, positive-ice, established-ice, exact-one and
`>=0.999` masks exactly.

## Falsifiable prediction and decision

The predeclared prediction is at least a 3% improvement in
`analysis_fair_crps`, with the paired-date interval excluding zero in the
improving direction, while ordinary `analysis_crps` worsens by no more than 1%.
Absolute randomized-rank reliability and every attainable-order coverage
criterion must pass; no coverage error may worsen by more than `0.02`.  Every
unchanged boundary, spatial/physical and operational criterion must also pass.
The non-overlapping four-case-block interval is temporal sensitivity only.

- Proper-score failure rejects training-only score-aware scenario selection as
  useful calibration.
- Reliability failure means lower predicted scenario risk does not repair the
  ensemble probability distribution.
- A source-copy or member-mask failure is an implementation/provenance failure.
- A mean-field spatial/physical failure is scientific evidence that changed
  scenario frequencies damage the forecast distribution.
- Any held-out truth access, fold mismatch, non-finite value, weight-sum error
  above `1e-12`, missing probability position or output other than ten members
  is an operational failure.

Selection requires `overall_eligible=true` and every mandatory family true.
No coefficient, predictor, loss weight, ridge, cutoff, probability rule, member
ordering, fold, purge, threshold or gate may change after observing a result.

## Execution boundary and compact evidence

Construction and analysis must eventually run on `server_cpu` from
`source_experiment=joint_full_condition_validation_2022`, over the exact
forty-case, ten-member development envelope with `summary_only`.  A future
reviewed runner must accept only `source_experiment` and return the full gate,
raw/candidate aggregates, paired uncertainty, fold membership, fitted
coefficients, per-case risks and weights, source multiplicities, unique-member
counts, effective sample sizes, copy/mask invariants and operational counts.
No raw member, candidate or truth field is retrieved.

The four compact files are exactly `case_selection.json`,
`aggregate_selection.json`, `paired_uncertainty.json` and
`gate_decision.json`.  Copy/mask invariant counts are carried in the aggregate.
Before admission, the directory must pass
`validate_score_aware_compact_outputs.py`: it independently recomputes weights,
source multiplicities, unique-member counts and ESS from every case's recorded
risks; derives the exact case-to-fold and purged training membership; reconciles
all case totals with the aggregate; binds the operational
family to bitwise-copy and mask invariants; and requires `overall_eligible` to
equal the conjunction of all five no-compensation families.  Extra files,
schema drift or any cross-file disagreement fail closed.

The controller-visible admission record uses
`score-aware-raw-reweighting-admission-v2` and contains one
`compact_directory_sha256`.  That digest frames each lexicographically ordered
filename, filename length, payload length and exact payload bytes, so it binds
both the four-file membership and the filename-to-content mapping.  Combined
admission validates the runner semantics and directory parity in one process,
then recomputes the directory digest; a substitution before or during admission
fails closed even when the replacement JSON is semantically valid.

There is intentionally no experiment proposal or invented mode identifier.
Admission requires a literal trusted mode plus independent parity and fail-closed
checks.  This document freezes the mechanism before either prerequisite result
is known.
