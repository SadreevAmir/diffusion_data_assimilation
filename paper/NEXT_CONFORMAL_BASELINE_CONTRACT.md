# Frozen block-conformal sea-ice-area baseline contract

Status: `FROZEN_NOT_EXECUTABLE`

## Publication role

This contract freezes the fastest missing minimum-tier comparison without
claiming that a trusted runner or result exists. It is a field-functional
interval baseline, not an ensemble postprocessor and not a candidate for the
joint no-compensation calibration gate. Its sole claim is whether a
leakage-safe split-conformal correction makes the learned-joint ensemble's
date-level sea-ice-area interval useful. The split-conformal method family and
its finite-sample marginal-coverage role follow Lei et al. (2018),
doi:10.1080/01621459.2017.1307116; that source does not validate this temporal
purge, area functional or project-specific decision threshold.

## Inputs and immutable envelope

- Reuse only the forty server-side cases and ten raw members from
  `joint_full_condition_validation_2022`.
- Use the existing truth, cell-area weights and established-ice definition
  already sealed for that source. No raw member or truth artifact is retrieved.
- Use five contiguous eight-date scoring blocks. For each block, remove the
  block and the three nearest cases on each temporal side from calibration;
  the purge is non-circular.
- No month, regime, track-density or outcome-dependent subdivision is allowed.

## Frozen statistic and correction

For every date and member, compute total established-ice area with the existing
cell-area weights. The raw interval is the minimum and maximum of the ten member
areas. For a calibration date with truth area `y`, lower endpoint `L` and upper
endpoint `U`, define the nonconformity score exactly as

`s = max(L - y, y - U, 0)`.

Within each purged training set of size `n`, sort scores in ascending order and
take `q` at one-indexed rank `ceil((n + 1) * 0.90)`, capped at `n`. The held-out
conformal interval is exactly `[L - q, U + q]`. Do not clip area endpoints,
studentize scores, interpolate quantiles, tune nominal coverage or alter the
raw center. Ties use the ordinary inclusive `<=` coverage rule.

## Falsifiable prediction and decision

The pre-result prediction is that raw member-range area coverage is too low and
the correction will achieve held-out coverage of at least `0.85`, while its
mean interval width is no more than `1.50` times the raw mean width. Both
conditions are required for `CONFORMAL_USEFUL`; otherwise the result is
`CONFORMAL_NEGATIVE`. Report coverage counts out of forty, mean and median
widths, width ratio, all five training sizes and quantiles, and paired per-date
width changes. A circular four-date-block bootstrap interval may be reported
only as temporal sensitivity and cannot change the decision.

Success closes the conformal minimum-tier row as a useful field-functional
baseline but does not establish calibrated members, spatial reliability or
overall eligibility. Failure closes the exact mechanism as a negative baseline
and is interpreted as excessive width inflation or inadequate temporally
blocked coverage, not as evidence against all conformal methods.

## Execution boundary

All computation and analysis must run on the server with `summary_only`
retrieval. The compact return is limited to one JSON decision summary and one
CSV containing the forty date-level raw/conformal endpoints, truth area and
fold identifier. No currently admitted trusted mode implements this contract;
therefore it must not be proposed under an invented mode identifier. Admission
requires a separately reviewed runner using this literal contract.
