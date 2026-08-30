# Frozen contingent contract: leave-one-observation-out scenario likelihood reweighting

Status: `FROZEN_PRE_REVIEW`. This document does not name or authorize an
executor mode. It freezes the next mechanistically distinct calibration test
after the score-aware raw-scenario route and the casewise safety selector have
returned trusted negative gates. No result from either route may change this
contract.

## Claim and mechanism

The learned-joint ensemble may contain useful, physically coherent scenarios
with misallocated probability mass. The candidate therefore changes only the
empirical probabilities of the ten existing raw members. It never edits,
clips, transports or interpolates a field. Unlike score-aware reweighting, its
member score uses only observation-space predictive consistency; unlike the
casewise selector, it produces one weighted distribution rather than choosing
between two complete ensembles.

For each held-out case and observed track location, reconstruct all ten member
values using the otherwise unchanged conditioning context with that single
observation masked.  Member `m` receives the component log likelihood of the
withheld value under a Gaussian centred on member `m`'s reconstruction.  The
common training-fold bandwidth is defined below.  This is deliberately not the
log density of the equally weighted ten-component mixture: that density would
be identical for every member and could not identify scenario probabilities.

For withheld locations `j=1,...,J`, reconstructed values `x_mj`, observations
`y_j` and training-fold bandwidth `h`, freeze

```text
ell_m = sum_j [-0.5 * ((y_j - x_mj) / h)^2 - log(h) - 0.5*log(2*pi)]
w_m   = exp(ell_m - max_k ell_k) / sum_k exp(ell_k - max_l ell_l).
```

The candidate is the weighted empirical distribution on the ten unchanged raw
fields.  No resampling is used for scoring or diagnostics.  This is a
diagnostic reallocation of probability among raw scenarios, not a new
assimilation claim.

## Leakage and frozen computation

- Use five contiguous eight-case holdouts with a non-circular three-case purge.
- All kernel widths are estimated from training folds only as the median
  absolute leave-one-observation-out innovation, multiplied by the fixed
  normal-consistency factor `1.4826`; use the single training-fold pooled value.
- A zero or non-finite width, a missing masked-observation reconstruction, or
  any non-finite weight makes the case operationally invalid; there is no
  fallback width.
- The withheld verifying field, its summaries and candidate scores are never
  inputs to the weights. Track coordinates, observed values and the original
  conditioning masks are the only held-out-case inputs.
- Ties in member log score receive equal probability. No temperature, weight
  floor, effective-sample-size constraint, bandwidth grid or post-hoc blend is
  allowed.
- Raw member arrays and their exact-zero, presence, established-ice,
  exact-one and `>=0.999` masks must be byte-identical between source and
  candidate artifacts. Report the ten probabilities and effective sample size
  per case in compact form; retrieve no raw arrays.

## Weighted score and reliability definitions

All candidate metrics consume the weights directly.  For member values `x_m`,
truth `y`, normalized non-negative weights `w_m`, and
`s2=sum_m(w_m^2)`, the frozen finite-ensemble fair CRPS is

```text
sum_m w_m * abs(x_m-y)
- sum_m sum_n w_m*w_n*abs(x_m-x_n) / (2*(1-s2)).
```

This reduces exactly to the existing equal-weight fair CRPS at `w_m=0.1`.
`s2 >= 1` is operational failure.  Ordinary weighted CRPS uses the same first
term and denominator `2` rather than `2*(1-s2)` in the pairwise term.

Finite-ensemble reliability is evaluated from the weighted randomized PIT,
not by pretending that unequal weights are ten exchangeable equal-mass ranks.
At each verified scalar, let `W_lt` be the total weight of members strictly
below truth and `W_eq` the total weight exactly equal to truth.  Freeze
`u = W_lt + U*W_eq`, where `U` uses the unchanged common gate seed and indexing
contract.  The common absolute-uniformity, centering, range-coverage and
inner-coverage thresholds are applied to these `u` values with their existing
date-balanced aggregation.  The raw baseline is evaluated by the identical
formula with `w_m=0.1`.  Missing tie randomization, equal-rank substitution or
member resampling is an implementation failure.

## Falsifiable prediction and decision

Baseline is the equally weighted `joint_full_condition_validation_2022`
ensemble. The primary objective is `analysis_fair_crps`. Success requires all
of the following on the exact 40-case, ten-member development envelope:

1. date-mean fair CRPS improves by at least `3%`, its paired-date 95% interval
   excludes zero in the improving direction, and ordinary CRPS worsens by no
   more than `1%`;
2. the absolute randomized-rank uniformity, centering, range-coverage and
   inner-coverage criteria all pass under the unchanged finite-ensemble gate;
3. every boundary and spatial/physical criterion in the common
   no-compensation gate passes; byte-identical fields make any disagreement an
   implementation failure rather than a tolerable metric change;
4. all 40 cases are valid, all weights sum to one within `1e-12`, every weight
   is finite and non-negative, source hashes and masks match, and median
   effective sample size is at least `3.0` with no case below `2.0`.

The candidate is eligible only if every family and `overall_eligible` are
true. A proper-score failure rejects observation consistency as a useful
scenario-probability signal. A reliability failure means the likelihood
concentrates mass without repairing finite-ensemble ranks. A boundary or
spatial mismatch invalidates the promised copy-only implementation. An
effective-sample-size failure rejects unstable probability collapse. There is
no bandwidth, temperature, floor, mask, fold or threshold revision after the
result.

## Publication role and admission boundary

A positive result supports only a leakage-safe development claim that
observation-consistency reallocation improves probabilities among fixed joint
scenarios. A negative result closes this mechanism as an informative ablation
between scenario support and scenario probability. In either branch report
paired date uncertainty and non-overlapping four-case block sensitivity, the
full family matrix, per-case effective-sample-size summaries and exact source
identity.

Before execution, independent review must bind a literal controller-visible
mode, publication commit, trusted-runner identity, compact schemas and a
decision-bearing validator with `deviations=[]`. Until then the contract remains
pre-review engineering work and must not appear in an experiment proposal.
