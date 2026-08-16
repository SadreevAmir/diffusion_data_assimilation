# Frozen next calibration contract: purged coherent member-intercept expansion

## Scientific role

The best completed marginal calibration substantially improves fair CRPS,
spread--skill and the rank histogram, but fails the boundary and member/local
spatial safeguards because it changes ensemble anomalies independently at every
pixel.  This candidate tests a different hypothesis: the missing dispersion can
be supplied by a low-dimensional, spatially coherent member mode rather than by
pixelwise anomaly inflation or reconstructed members.

For each held-out case the method adds one spatially constant intercept to each
existing member and then performs a bounded, pixelwise mean-preserving
projection.  Before projection a constant intercept leaves every member's
spatial increments exactly unchanged; only adjustments forced by the physical
bounds can affect them.  A full pass would therefore show that coherent
large-scale concentration uncertainty is sufficient.  Failure closes this
fixed low-dimensional mechanism and must not trigger a denser amplitude grid or
a different member score on these dates.

This contract is frozen before the result of the checkpoint-trajectory/EMA
ensemble is known.

## Data envelope and folds

- Input is only the server-side ten-member output of
  `joint_full_condition_validation_2022` on the exact forty ordered dates from
  `2022-01-01` through `2022-07-15` at five-day stride.
- Use five contiguous eight-date holdouts.  For every holdout remove the three
  chronologically nearest cases on either side from training, without circular
  wrapping.  The resulting training counts must be `29, 26, 26, 26, 29`.
- Held-out truth is used only for final scoring.  It cannot affect the member
  score, projection, amplitude selection, numerical tolerances or fallbacks.
- No test-2023 data, raw array or member field leaves the server.

## Permutation-equivariant coherent member score

For raw member fields `x_i(p)`, fixed equal valid-cell weights `w(p)=1` and
pixelwise ensemble mean `m(p)`, compute

```
d_i = sum_p w(p) [x_i(p) - m(p)] / sum_p w(p)
r   = sqrt(mean_i d_i^2)
q_i = d_i / r
```

The arithmetic mean of the ten `q_i` values must be zero within `1e-14`; remove
only its floating-point residual before use.  If `r <= 1e-12`, set every `q_i`
to zero and record `degenerate_member_score=true`.  There is no member sorting,
tie randomization, member permutation, local score or truth-dependent score.
The construction is equivariant to permutation of the input members and keeps
their identities.

## Frozen transformation

The only candidate amplitudes are concentration offsets

```
A = {0.0, 0.01, 0.02, 0.04, 0.08, 0.16}.
```

For amplitude `a`, form the unbounded proposal

```
v_i(p) = x_i(p) + a q_i.
```

At each valid pixel project `v(:,p)` onto the closed bounded simplex

```
argmin_z sum_i [z_i - v_i(p)]^2
subject to 0 <= z_i <= 1 and sum_i z_i = sum_i x_i(p).
```

Use a deterministic active-set solution in float64.  Apply at most two residual
corrections over free components and fail if the final pixelwise ensemble-mean
error exceeds `1e-12`.  Pixels whose raw mean is exactly zero or one remain
exactly all-zero or all-one.  Reject non-finite or out-of-range source values.
Do not clip before projection, rescale individual members, smooth, localize,
rank-shuffle, recenter after projection or add stochastic jitter.

Record the maximum mean residual, fractions at both bounds, fraction of values
changed by the bound projection, member-score RMS and every selected amplitude.

## Leakage-safe amplitude selection

For each outer fold, evaluate every amplitude on retained training dates.  An
amplitude is feasible only when all of the following training summaries hold
against raw:

1. fair CRPS improves by at least `3%`, ordinary CRPS worsens by at most `0.5%`;
2. randomized-rank discrepancy improves by at least `20%`, and both member-range
   and inner-order attainable-coverage absolute errors decrease;
3. presence and established-ice Brier scores worsen by at most `1%`, exact-zero
   and exact-one mass errors do not increase;
4. every member and local variogram diagnostic at lags `1, 2, 4` worsens by at
   most `1%`, and the energy score worsens by at most `1%`;
5. every field, projection and mean-preservation check is valid.

Select the feasible amplitude with the lowest training fair CRPS, breaking an
exact `1e-12` tie toward the smaller amplitude.  If none is feasible, select
`0.0` and record `no_feasible_training_amplitude=true`; this is a mechanism
failure, not permission to relax the rules.  Apply the selected amplitude once
to the eight held-out dates.  Never pool held-out scores to choose a common
amplitude.

## Unchanged full decision gate

The forty held-out candidates must pass the same no-compensation gate used for
the completed calibration audits:

- **Proper scores:** fair CRPS improves at least `3%`; its paired-date 95% CI and
  the fixed non-overlapping four-date-block sensitivity interval are below
  zero; ordinary CRPS is no worse by more than `1%`; energy score is no worse by
  more than `2%`.
- **Finite-ensemble reliability:** rank discrepancy improves at least `20%`;
  member-range and inner-order attainable-coverage errors both decrease.
- **Boundary:** both Brier scores are no worse by more than `1%`; exact-zero and
  exact-one mass errors do not increase; all-zero and all-one raw pixels remain
  forced boundary pixels.
- **Spatial/physical:** pixelwise ensemble mean is preserved within `1e-12`, so
  mean RMSE, IIEE, edge, area, extent and mean-field variograms equal raw within
  the established tolerance; every member and local variogram diagnostic at
  lags `1, 2, 4` is no worse by more than `2%`.
- **Operational:** exactly forty cases and ten finite members per case complete;
  all folds, amplitudes and projections are recorded; no undeclared fallback is
  used.

Eligibility requires every family true and `gate.overall_eligible=true`.
Improvement in one family cannot compensate for failure in another.

Use 20,000 shared paired-date bootstrap resamples with seed `20220815`.  Report
the fixed ten non-overlapping consecutive four-date clusters with seed
`20220816` as temporal sensitivity.  The same resample index matrices must be
used for every reported metric.

## Trusted execution and compact result

The future reviewed server-CPU mode accepts exactly one runtime parameter,
`source_experiment=joint_full_condition_validation_2022`.  Amplitudes, folds,
purge, weights, scores, seeds, thresholds and projection tolerances are code
constants.  The runner must pin and verify the source experiment identity,
ordered date set, ensemble size, valid mask/weights, source sample manifest and
its own hash.

Return only `run_status.json`, `aggregate_case_mean_metrics.json`,
`metadata.json` and `per_case_metrics.csv`.  Compact metadata includes fold
membership, all training candidate summaries, selected amplitudes, degeneracy
counts, projection diagnostics, complete family booleans and provenance.  It
must state `test_data_used=false` and `raw_arrays_copied=false`.

Implementation and review are autonomous engineering work.  This document
freezes the scientific method before implementation or launch; it does not
claim a result.
