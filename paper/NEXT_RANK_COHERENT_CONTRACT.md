# Frozen next-method contract: purged rank-targeted coherent anomaly transport

Status: DESIGN_FROZEN_NO_RUNNER

## Claim role and hypothesis

This is a development-only mechanism test. It asks whether underdispersion can
be repaired by transporting complete, spatially coherent member-anomaly fields
between analogous forecasts, instead of calibrating pixels or marginal
quantiles independently. The raw learned-joint ensemble is the primary
baseline; the completed coherent-offset and analog-residual constructions are
negative mechanism controls.

The falsifiable prediction is that the candidate improves case-mean fair CRPS
by at least 3% relative to raw, passes the absolute randomized-rank uniformity
criterion and all attainable-order coverage criteria, while every unchanged
boundary, deterministic, member-spatial and operational criterion passes. A
proper-score gain cannot compensate for rank or spatial failure.

## Leakage-safe folds and forecast-only analog selection

- Use exactly the existing 40 ordered development cases, split into five
  contiguous eight-case holdouts with a non-circular three-case purge.
- Compute the six forecast-only state features already frozen for the completed
  analog-residual experiment: weighted raw-mean ice area, extent at `0.15`,
  spatial mean, spatial standard deviation, and raw-mean semivariograms at lags
  1 and 4.
- Standardize features using retained training cases only. Reject a fold if a
  population standard deviation is zero or non-finite.
- For each held-out case choose the ten nearest retained training cases by
  squared Euclidean distance, with earlier ordered case index breaking ties.
  Held-out truth is unavailable to feature construction and neighbor choice.

## Coherent anomaly library and fixed rank targets

For every selected training case, retain its ten complete raw member-anomaly
fields `a[k,j] = x[k,j] - mean_j(x[k,j])`. Do not clip, rescale, localize,
rotate, recenter by pixel, or shuffle values within a field.

On retained training cases only, compute the normalized truth rank of the
spatially weighted case-mean concentration among the ten raw member case means,
using the existing randomized-tie seed contract. Sort the ten selected analog
dates by that scalar rank and take exactly one anomaly field from each date:
the member whose spatially weighted case-mean anomaly has the same order
statistic `j=0,...,9`. Pair those ten fields with the held-out raw member order
statistics of spatially weighted case-mean anomaly. All sorting ties use member
index, then ordered case index. This fixed stratification targets the missing
case-level rank dispersion while moving complete anomaly fields.

Construct provisional held-out member `j` as `m + a[k_j,j_j]`, where `m` is
the held-out raw ensemble mean. Apply one common scalar `alpha` to all ten
borrowed anomaly fields. Select `alpha` from exactly
`{0.0,0.5,0.75,1.0,1.25}` on retained training cases by minimum fair CRPS,
subject to every training-only boundary and member-semivariogram tolerance
passing. Ties choose the smaller value. If no positive value is feasible,
select `0.0` and record `no_positive_feasible_alpha=true`.

Apply the existing exact capped-simplex projection independently at each pixel
to preserve the held-out raw ensemble mean and bounds. The projection is a
known possible failure source, not a harmless implementation detail. Record
its changed-member fraction, lower/upper cap masses, maximum mean error, and
member-semivariogram distortion before and after projection.

## Frozen decision and failure interpretation

Apply the full unchanged no-compensation gate out of fold on all 40 cases,
including paired date uncertainty and the fixed non-overlapping four-case block
sensitivity. Selection is positive only when `overall_eligible=true` and every
mandatory family passes.

- Rank or attainable-coverage failure rejects case-rank stratification as an
  adequate reliability mechanism.
- Proper-score failure rejects useful transported diversity even if ranks look
  flatter.
- Pre-projection spatial failure rejects transferability of complete anomaly
  fields; post-projection-only spatial or boundary failure attributes the
  rejection to bounded mean preservation.
- Selection of `alpha=0.0` in any fold rejects feasible signal amplitude under
  the frozen training constraints.
- Any leakage, incomplete fold, non-finite value, projection mean error above
  `1e-10`, or tie-rule mismatch is an operational failure.

No feature, purge, neighbor count, rank functional, scale set, feasibility
threshold, projection, seed or tie rule may change after a result. A negative
result closes this exact mechanism and does not authorize another external
evaluation.

## Execution and compact artifacts

All construction and analysis must run on `server_cpu` from
`source_experiment=joint_full_condition_validation_2022`. The future trusted
runner must expose that sole parameter and the exact 40-case, ten-member
envelope; the artifact policy is `summary_only`. It must return the complete
gate, aggregate raw/candidate metrics, paired uncertainty, fold selections and
alphas, rank-target balance, projection diagnostics, member-spatial deltas and
operational counts. No raw member, truth or anomaly field is retrieved.

This contract intentionally has no experiment proposal: no currently
implemented trusted mode implements it. Freezing the contract is the active
autonomous research step; runner review is ordinary engineering work, not an
external scientific blocker.

The dependency-free review oracle `rank_coherent_reference.py` makes the exact
non-circular purge, date/member tie ordering, complete-field pairing, frozen
alpha selection, one-parameter interface and capped-simplex mean invariant
executable on synthetic inputs. It reads no project data and is not an
experiment entry point; a trusted server implementation and independent review
remain required before this method can be proposed.
