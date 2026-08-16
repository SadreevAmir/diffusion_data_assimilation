# Frozen calibration contract: slack-limited coherent member expansion

## Scientific role

The completed open-logit calibration improved proper scores and marginal
reliability but damaged member-scale spatial structure and upper-bound
behaviour.  The completed coherent-intercept audit preserved the mean and most
spatial diagnostics, but every non-zero amplitude created additional exact-one
mass through the bounded-simplex projection; the leakage-safe selector therefore
returned zero in every fold.  This contract tests the specific remaining
hypothesis: a coherent member mode can supply useful dispersion when its local
amplitude is limited by the existing physical slack, without clipping or
projection.

This document is frozen before the checkpoint-trajectory/EMA result and before
this method is executed.  A failure closes this exact slack-limited mechanism;
it is not permission to add amplitudes or relax the joint gate on these dates.

## Data envelope and folds

- Input is only the server-side ten-member result
  `joint_full_condition_validation_2022` on the exact forty ordered dates from
  `2022-01-01` through `2022-07-15` at five-day stride.
- Use five contiguous eight-date outer holdouts.  Purge the three nearest cases
  on each side without circular wrapping, giving training counts
  `29, 26, 26, 26, 29`.
- Held-out truth is used only for final scoring.  No test-2023 data, raw array or
  member field may leave the server.

## Coherent member score

For raw member fields `x_i(p)` and pixelwise ensemble mean `m(p)`, compute the
same permutation-equivariant global score as in the preceding coherent audit:

```
d_i = mean_p [x_i(p) - m(p)]
r   = sqrt(mean_i d_i^2)
q_i = d_i / r.
```

Remove only the floating-point mean of the ten `q_i` values.  Require its final
absolute mean to be at most `1e-14`.  If `r <= 1e-12`, use all-zero scores and
record the degeneracy.  The score cannot use truth, local ranks, member sorting
or held-out information.

## Frozen slack-limited transformation

The only global amplitude candidates are

```
A = {0.0, 0.01, 0.02, 0.04, 0.08, 0.16}.
```

For every valid pixel define its non-negative safe amplitude

```
h(p) = min_i {
  (1 - x_i(p)) / q_i,  if q_i > 0,
  x_i(p) / (-q_i),     if q_i < 0,
  +infinity,           if q_i = 0
}.
```

For candidate amplitude `a`, use

```
s_a(p) = min(a, (1 - 1e-12) h(p)),
z_i(p) = x_i(p) + s_a(p) q_i.
```

The same scalar `s_a(p)` multiplies every member score at a pixel.  Because the
scores sum to zero, the pixelwise ensemble mean is preserved.  Because the
scale is strictly inside the available slack, the transform cannot create a
new exact zero or one.  Pixels blocked by an outward-moving boundary member
receive `s_a(p)=0`.  Do not clip, project, smooth, recenter, permute, rank-shuffle
or add stochastic jitter.  Fail if any output is non-finite/outside `[0,1]`, if
the maximum mean residual exceeds `1e-12`, or if a new exact-bound value is
created.  Record requested/effective amplitude, blocked-pixel fraction,
slack-limited fraction, score RMS and boundary counts.

## Leakage-safe selection

Within each retained training fold evaluate the six amplitudes.  Feasibility
requires all of the following relative to raw:

1. fair CRPS improves at least `3%`; ordinary CRPS worsens at most `0.5%`;
2. rank discrepancy improves at least `20%`; member-range and inner-order
   attainable-coverage errors both decrease;
3. presence and established-ice Brier scores worsen at most `1%`; exact-zero
   and exact-one mass errors do not increase; no new exact bound is created;
4. energy score and every member/local variogram diagnostic at lags `1,2,4`
   worsen at most `1%`;
5. all finite, identity, bound and mean-preservation checks pass.

Choose the feasible amplitude with minimum training fair CRPS, with an exact
`1e-12` tie broken toward the smaller amplitude.  If none is feasible, choose
zero and record `no_feasible_training_amplitude=true`.  Apply the selected
amplitude once to the corresponding eight held-out cases.

## Unchanged held-out gate

Eligibility requires all families and `gate.overall_eligible=true`:

- proper scores: fair CRPS improves at least `3%`, paired-date and fixed
  four-case-block 95% intervals are below zero, ordinary CRPS is no worse by
  more than `1%`, and energy score by more than `2%`;
- finite-ensemble reliability: rank discrepancy improves at least `20%`, and
  both attainable-coverage errors decrease;
- boundary: both Brier scores are no worse by more than `1%`, exact-zero and
  exact-one mass errors do not increase, and no new exact bound is created;
- spatial/physical: the mean-field invariants equal raw within `1e-12`, and all
  member/local variogram diagnostics are no worse by more than `2%`;
- operational: exactly forty cases and ten finite members complete, all folds
  and amplitude decisions are recorded, and there is no undeclared fallback.

Use 20,000 shared paired-date resamples with seed `20220815` and the fixed ten
non-overlapping four-date blocks with seed `20220816` for every metric.

## Trusted execution and compact evidence

The reviewed server-CPU mode accepts only
`source_experiment=joint_full_condition_validation_2022`.  The runner pins the
source sample manifest and fixes all folds, amplitudes, tolerances, seeds and
thresholds in code.  It returns only `run_status.json`,
`aggregate_case_mean_metrics.json`, `metadata.json` and
`per_case_metrics.csv`, with `test_data_used=false` and
`raw_arrays_copied=false`.

