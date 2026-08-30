# Frozen next-baseline contract: topology-preserving stratified transport

## Scientific role and falsifiable hypothesis

This is the next mechanistically distinct calibration method after the rejected
global-spread, hurdle-isotonic/ECC-Q, projected-spread, open-logit and
ZOIB-EMOS/ECC-Q routes. Those results isolate two coupled failure modes:
marginal reconstruction changes memberwise ice-event geometry, while unrestricted
spread expansion creates or removes boundary atoms. The proposed transport acts
on the existing ten fields without ECC, preserves every member's event masks and
preserves the ensemble mean at every pixel.

The falsifiable hypothesis is that useful underdispersion remains *within* the
raw physical regimes. A leakage-safe expansion restricted to those regimes will
improve fair CRPS by at least 3% while exact invariants prevent the previously
observed boundary and mean-field failures. Failure of proper scores rejects this
within-regime hypothesis; failure of member/local spatial diagnostics shows that
even topology-preserving amplitude transport damages scenario structure.

## Server-side inputs and development envelope

- The only scientific input is the server-side ten-member ensemble and verifying
  field from `joint_full_condition_validation_2022` for the same 40 ordered
  development cases. No ensemble or pixel-level artifact is copied locally.
- The runner additionally receives the immutable source manifest, ordered case
  identifiers, grid-cell weights/mask and source runner identity already attached
  to that experiment. It must reject any identity mismatch or missing case.
- Use five contiguous eight-case holdouts. For holdout `k`, purge the three
  chronologically nearest cases on each side from its training set (one-sided at
  endpoints, never wrapping). All scale selection uses only the remaining dates.
- Pixels contribute within a date, but every training date has total weight one.
  Held-out dates never affect candidate feasibility, scale selection, tie rules,
  tolerances or standardization (none is required).

## Exact transport and invariants

For raw concentration `x[i,p]` of member `i` at pixel `p`, assign one immutable
stratum using exact float64 comparisons:

1. `Z`: `x == 0`;
2. `L`: `0 < x <= 0.15`;
3. `H`: `0.15 < x < 1`;
4. `O`: `x == 1`.

Reject source values outside `[0,1]` or non-finite values. Members in `Z` and `O`
are unchanged. Independently for `L` and `H`, let `I` be the member indices in
that stratum, `g` their raw mean and `lambda >= 1`. Form
`v_i = g + lambda * (x_i-g)` and compute the unique Euclidean projection

```
y_I = argmin_z sum_i (z_i-v_i)^2
      subject to sum_i z_i = sum_i x_i and lower <= z_i <= upper
```

with `(lower, upper)=(nextafter(0,+inf),0.15)` for `L` and
`(nextafter(0.15,+inf),nextafter(1,-inf))` for `H`. Use deterministic bisection
of the projection multiplier for at most 80 iterations, stopping when the
absolute sum residual is at most `1e-13`; then apply a deterministic residual
correction to the lowest original member indices with available slack. Reject a
pixel if final sum error exceeds `5e-13` or any value crosses its closed/open
bound. A stratum with fewer than two members is unchanged.

This construction must verify, not merely assume, all of the following for every
held-out value: exact `Z` and `O` masks unchanged; `siconc > 0` and
`siconc > 0.15` masks unchanged member by member; member identity unchanged; no
permutation; pixelwise ensemble-mean error at most `5e-13`; values finite and in
`[0,1]`. There is no clipping, ECC, smoothing, location/regime fitting, random
jitter or post-result repair.

## Leakage-safe selection of transport strength

The candidate set is frozen to
`lambda in {1.0,1.25,1.5,2.0,2.75,4.0}`. These log-spaced strengths distinguish
no transport, mild, moderate and aggressive within-stratum expansion; they are
not a local refinement of a previously observed optimum. For each fold, apply
each value to training dates and retain it only if all conditions hold versus
raw training summaries:

- case-mean fair CRPS improves by at least 3%; ordinary CRPS worsens by no more
  than 0.5%;
- member-range and inner-order attainable-coverage absolute errors both decrease
  and randomized-rank discrepancy decreases by at least 20%;
- every member and local variogram diagnostic at lags 1, 2 and 4 worsens by no
  more than 1%;
- all exact invariants above pass.

Among retained values choose minimum case-mean fair CRPS, then the smaller
`lambda` on an exact tie within `1e-12`. If none is retained, freeze
`lambda=1.0` for that holdout and record `no_feasible_training_scale=true`; this
is an informative mechanism failure, not permission to relax a condition. Apply
the selected value once to the eight held-out dates. Do not pool held-out
results to refit or select a common value.

Record fold/holdout/purge membership, every training candidate summary,
feasibility flags, selected values, projection residual maxima, invariant counts,
source manifest identity and runner hash. There are no runtime tuning parameters.

## Frozen joint no-compensation gate

Eligibility requires every family independently:

1. **Proper scores:** held-out case-mean fair CRPS improves by at least 3% over
   raw; ordinary CRPS is no worse by more than 1%; the paired date-bootstrap 95%
   interval for candidate-minus-raw fair CRPS lies strictly below zero.
2. **Finite-ensemble reliability:** randomized-rank discrepancy decreases at
   least 20%; member-range and inner-order attainable-coverage absolute errors
   each decrease; no additional frozen central-coverage error worsens by more
   than `0.02`.
3. **Boundary behaviour:** exact-zero and exact-one mass errors do not increase;
   presence and established-ice Brier scores are each no worse by more than 1%.
   In addition, all four memberwise masks declared above must be bitwise equal to
   raw. Any mismatch fails the family regardless of aggregate metrics.
4. **Spatial/physical preservation:** analysis-mean RMSE, mean IIEE, edge
   disagreement, area error, extent error and every mean-field variogram value
   must match raw within numerical tolerance (`5e-13` fieldwise mean and
   `1e-12` aggregate). Every member and local variogram diagnostic at lags 1, 2
   and 4 must be no worse than raw by more than 2%. Report energy score as a
   diagnostic, not compensation.
5. **Operational validity:** exactly 40 held-out cases and ten members complete;
   all metrics are finite; every fold has a recorded selection; all projections
   and exact invariants pass; no undeclared fallback occurs.

Use 20,000 paired date-bootstrap resamples with seed `20220815`. Also report the
fixed ten non-overlapping consecutive four-case block bootstrap interval with
seed `20220816`; it is a temporal sensitivity, not an acceptance gate. Report
raw/candidate aggregate and paired summaries for fair and ordinary CRPS,
ensemble-mean RMSE, energy score, rank histogram, attainable coverage, exact
boundary masses, both Brier scores, IIEE, area, extent, edge, and mean/member/
local variograms.

## Interpretation frozen before execution

- **Full pass:** supports the narrow claim that event-topology-preserving
  within-regime transport can calibrate a finite bounded ensemble without
  sacrificing its physical event geometry or analysis mean.
- **No feasible scale in at least one fold or proper-score failure:** rejects
  sufficient within-regime underdispersion under the frozen transport family;
  do not densify the scale set.
- **Reliability failure with proper-score pass:** amplitude expansion improves
  scoring but cannot repair discrete ten-member reliability without changing
  event topology.
- **Member/local spatial failure:** preserving masks and pixel means is
  insufficient to preserve memberwise dependence; reject the transport rather
  than weakening the spatial gate.
- **Invariant or projection failure:** operationally invalid runner; fix only a
  demonstrable implementation error and rerun the identical contract.

## Minimal trusted runner interface and compact return

The reviewed server CPU runner must expose exactly one runtime parameter:
`source_experiment=joint_full_condition_validation_2022`.
The controller must assign an implemented mode identifier during runner review;
this document does not invent one. The runner performs fitting, transport,
scoring and uncertainty server-side and accepts no scale, fold, seed, threshold,
path or optimizer override.

Default retrieval is `summary_only`. Return exactly the controller's four fixed
compact artifacts: `run_status.json`, `aggregate_case_mean_metrics.json`,
`metadata.json` and `per_case_metrics.csv`. The per-case file contains only the
40 raw and 40 held-out candidate metric rows; it contains no pixel fields or
ensemble arrays. The JSON artifacts contain schema version, immutable
input/runner identities, fold selections and feasibility, invariant
maxima/counts, aggregate metrics, paired intervals, family booleans,
`no_compensation_across_families=true` and `overall_eligible`. No raw ensemble or
pixel-level artifact is retrieved. Any later paper figure must be derived from
these compact outputs or use a separately reviewed exact artifact request.

## Execution status

The frozen design has been executed by the reviewed trusted server CPU runner.
The compact gate reports an inactive held-out correction: all three
finite-ensemble reliability criteria, both fair-CRPS criteria and all-fold
training feasibility fail, while the exact boundary, mask, mean and spatial
invariants pass. The scale set, strata, folds and thresholds remain frozen; this
route must not be rerun or densified. The earlier ZOIB-EMOS/ECC-Q result remains
an immutable negative baseline in `REPRODUCIBILITY.md`, `CLAIM_LEDGER.md` and
the manuscript.
