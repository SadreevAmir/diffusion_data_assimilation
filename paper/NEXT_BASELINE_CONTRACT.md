# Frozen next-baseline contract: ZOIB-EMOS with ECC-Q

## Scientific role and hypothesis

This is the next strong, mechanistically distinct calibration baseline. It is a
zero/one-inflated Beta ensemble-model-output-statistics (ZOIB-EMOS) distribution
with rank-preserving ECC-Q reconstruction. Unlike the rejected hurdle-isotonic
construction, it estimates a low-dimensional parametric conditional
distribution; unlike projected spread and open-logit transforms, it models both
boundary probabilities explicitly instead of moving existing members.

The falsifiable hypothesis is that a leakage-safe ZOIB-EMOS marginal model can
repair boundary masses and finite-ensemble reliability while ECC-Q preserves
enough of the raw empirical copula to pass the already frozen joint
no-compensation gate. Success requires every gate family below to pass. A
proper-score gain alone is not success.

## Data envelope and folds

- Input is the server-side ten-member output of
  `joint_full_condition_validation_2022`; no raw ensemble is copied locally.
- Evaluate the same 40 ordered development cases and no other split.
- Use five contiguous eight-case holdout blocks. For holdout block `k`, remove
  it and the three chronologically nearest cases on each side before fitting.
  At the endpoints, remove the available one-sided neighbours; do not wrap.
- Every transformation, standardization statistic and fitted coefficient for a
  held-out case uses training cases only. Pixels are observations for fitting,
  but each date has total weight one, so dates rather than pixels determine the
  empirical objective. Uncertainty is computed only from paired case summaries.
- Fit one pooled model over the domain per fold. No month, location, regime,
  track-density or distance-to-track interactions are allowed in this baseline.

## Frozen predictors and distribution

For every pixel, compute from the raw ten-member ensemble: mean `m`, standard
deviation `s` with divisor ten, exact-zero fraction `z`, and exact-one fraction
`o`. Define `clip_logit(x) = logit(min(max(x, 1e-4), 1-1e-4))` and
`ls = log(s + 1e-4)`. Standardize `clip_logit(m)` and `ls` using weighted
training-fold mean and standard deviation; a standard deviation below `1e-8`
is replaced by one. Do not standardize `z` or `o`.

The mixture has masses `p0` at zero and `p1` at one and a Beta interior with
mass `pi = 1-p0-p1`. Boundary logits are a three-class softmax with the interior
class as reference:

```
eta0 = a0 + a1 * standardized_clip_logit_m + a2 * standardized_ls + a3 * z
eta1 = b0 + b1 * standardized_clip_logit_m + b2 * standardized_ls + b3 * o
(p0, pi, p1) = softmax(eta0, 0, eta1)
```

The interior Beta parameters are

```
mu = sigmoid(c0 + c1 * standardized_clip_logit_m + c2 * standardized_ls)
kappa = 2 + softplus(d0 + d1 * standardized_ls)
alpha = mu * kappa
beta = (1 - mu) * kappa
```

This is exactly 13 fitted coefficients. There is no case-specific, pixelwise or
post-result parameter selection.

## Frozen fitting contract

- Minimize date-balanced negative log likelihood of the mixed distribution plus
  `1e-4 * sum(theta**2)` over all 13 coefficients. For targets strictly inside
  `(0,1)`, use the Beta density; exact endpoints use their corresponding point
  masses. The ridge coefficient is fixed and not searched.
- Initialize all coefficients to zero except `c1=1`, `d0=log(expm1(18))`.
- Use float64 L-BFGS with maximum 500 iterations, gradient infinity norm
  tolerance `1e-8`, relative objective tolerance `1e-10`, and no random start.
- A fold is operationally invalid if optimization is non-finite, does not
  converge, or any fitted coefficient has absolute value above 50. There is no
  fallback optimizer, alternate ridge value or refit after examining scores.
- Record fold membership, purged cases, training standardizers, coefficients,
  convergence status, iteration count, final objective and runner hash.

## Frozen ten-member reconstruction

For held-out pixels, evaluate mixture quantiles at `q_j=(j-0.5)/10`,
`j=1,...,10`. Quantiles in the lower point mass are exactly zero and quantiles
above `1-p1` are exactly one; interior quantiles use the deterministic float64
inverse Beta CDF. Assign the ten sorted quantiles using the raw ensemble member
rank order (ECC-Q). Break raw ties by increasing original member index. Report
the number of strict order violations, which must be zero. No clipping, jitter,
mean restoration, spatial smoothing or permutation search is permitted.

## Joint no-compensation stop/go gate

The candidate is eligible only when all families pass independently:

1. **Proper scores:** fair CRPS improves by at least 3% versus raw; ordinary
   CRPS is no worse than raw by more than 1%; paired date-bootstrap 95% interval
   for the fair-CRPS delta excludes zero in the improving direction.
2. **Finite-ensemble reliability:** randomized-rank discrepancy falls at least
   20%; member-range and inner-order attainable-coverage absolute errors each
   decrease, allowing at most `0.02` worsening for any additional frozen central
   coverage diagnostic.
3. **Boundary behaviour:** absolute errors of exact-zero and exact-one member
   masses do not increase; Brier scores for presence and established ice are
   each no worse than raw by more than 1%.
4. **Spatial/physical preservation:** mean IIEE and edge disagreement are each
   no worse than raw by more than 2%; absolute area and extent errors satisfy
   the frozen relative/near-zero tolerance in `RESEARCH_PLAN.md`; all reported
   member and local variogram diagnostics at lags 1, 2 and 4 are no worse than
   raw by more than 2%.
5. **Operational validity:** all 40 cases and all metrics are finite, every fold
   converges under the single optimizer contract, and ECC-Q has zero strict
   rank-order violations.

Report fair and ordinary CRPS, energy score, rank histogram, both attainable
coverage diagnostics, boundary masses and Brier scores, IIEE, area, extent,
edge disagreement, and mean/member/local variogram diagnostics. Use 20,000
paired date bootstrap resamples with seed `20220815` and the fixed ten
non-overlapping consecutive four-case clusters with seed `20220816`. The block
interval is a temporal sensitivity, not an acceptance gate.

## Interpretation frozen before result

- **Full pass:** supports the paper claim that explicit boundary-mixture
  distributional regression plus empirical-copula reconstruction can jointly
  improve a finite bounded ensemble on the development envelope.
- **Boundary pass, spatial failure:** rejects ECC-Q as sufficient preservation
  of the useful conditional spatial copula after parametric marginal repair.
- **Proper-score/reliability pass, boundary failure:** rejects the fixed ZOIB
  regression specification as adequate boundary calibration despite explicit
  atoms; do not tune ridge, predictors or mixture links post hoc.
- **Spatial pass, score failure:** the method is a boundary-aware negative
  baseline, not a calibrated-ensemble method.
- **Optimization failure:** the fixed low-dimensional baseline is operationally
  invalid. Do not silently substitute another optimizer or regularizer.

## Executor boundary

No currently reviewed trusted-executor mode implements this contract. The next
controller change, outside this worktree's allowed scope, must add and review a
server-CPU runner with `source_experiment=joint_full_condition_validation_2022`
as its only runtime parameter. Until that mode exists, this document is a design
freeze and must not be represented as an executable experiment proposal.
