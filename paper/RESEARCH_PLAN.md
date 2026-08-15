# Research plan: reliable finite generative ensembles

## Research question

Can a finite ensemble from a conditional diffusion/flow data-assimilation model
be calibrated for bounded, zero-inflated spatial fields while preserving useful
joint scenarios and sparse-observation skill?

## Frozen method and completed decision

The paper method is five-fold, deterministic date-stratified cross-fitted global
spread calibration of the learned-joint ensemble. Within each fold, choose one
scale from `1.0:0.1:4.0` on the other 32 cases by minimum case-mean fair CRPS and
apply it to eight held-out cases. The member transform is
`mean + scale * (member - mean)`; clip only for bounded scoring. Selected scales
are `2.6, 2.7, 2.8, 2.8, 2.6`. This method is frozen.

Stop/go criteria were met: fair CRPS improves by at least 3% (observed 4.79%),
ordinary CRPS is at most 0.061526 (observed 0.061101), 90% coverage is within
0.85–0.94 (observed 0.878775), all selected scales are interior, and pre-clipping
mean-invariance error is below `1e-10` (observed `4.16e-16`). No further method
search is justified by the current paper claim.

## Correctness status

- Complete for the frozen decision: fair and ordinary CRPS are both reported;
  the affine-logit family is a documented negative baseline; coverage is called
  a finite-ensemble pointwise diagnostic; and pre-clipping center invariance is
  verified numerically.
- Not claimed: temporal-block uncertainty intervals, conditional calibration by
  regime, randomized-rank uniformity after correction, or fieldwise coverage.
  These are manuscript limitations, not reasons to reopen method selection.
- Required before any broader confirmatory claim: repeat the frozen transform
  without retuning on a genuinely independent period and add date/block-level
  uncertainty estimates. Pixels must never be treated as independent replicates.

## Deferred alternatives, not current method development

Boundary-aware conditional, case-adaptive, hybrid-sampling and affine
mean-and-spread methods remain future work. With 40 cases they add estimation
flexibility, confound the clean mechanism attribution, or require new sampling
without repairing a failed criterion.

1. Calibrate occurrence probabilities for exact open water and ice threshold
   events using regularized isotonic/Beta or hurdle components.
2. Calibrate the conditional distribution in the interior `(0,1)` with a
   monotone low-dimensional map.
3. Condition with hierarchical shrinkage on month, distance to tracks,
   observation density and ice regime; avoid independent per-month fitting.
4. Reconstruct ensemble members using raw member ranks/ECC-like coupling so
   calibrated marginals do not arbitrarily replace the empirical copula.
5. Add block-conformal correction for explicitly chosen pointwise or
   field-functional coverage targets.
6. Select a Pareto-feasible method under predeclared criteria:
   fair CRPS, calibration error, sharpness, variogram distortion and physical
   boundary preservation.

## Required baselines

- raw ensemble;
- physical-space bias/spread scaling;
- naive affine-logit scaling as a documented failure mode;
- zero/one-inflated Beta or EMOS-like SIC postprocessing;
- isotonic/quantile mapping;
- conformal intervals;
- ECC-Q/ECC-T or rank-preserving reconstruction;
- deterministic background and 3D-Var;
- probabilistic DA baseline such as EnKF/LETKF when a fair implementation is
  available.

## Evaluation

Marginal and finite-ensemble:

- fair and empirical CRPS;
- randomized rank histograms with tie handling;
- attainable order-statistic coverage and interval width;
- Brier score/reliability for `siconc > 0` and `siconc > 0.15`;
- corrected spread-skill;
- calibration by month, ice regime, track density and distance to track.

Multivariate and physical:

- energy and patchwise variogram scores;
- ice area, extent and edge distributions;
- IIEE and edge displacement;
- spatial correlations and radial power spectra;
- exact-zero/exact-one masses;
- connected components and multiscale Fraction Skill Score.

## Claim and data discipline

- Development remains on temporally blocked 2022 validation data.
- Sequential adaptation to the same dates is recorded and never reported as a
  final generalization estimate.
- Cross-fitting separates scale selection and scoring for each case, but does
  not turn the reused development period into an independent temporal test.
- Report fair CRPS `0.0584905850 → 0.0556896736`, ordinary CRPS
  `0.0621082810 → 0.0611013421`, spread-skill `0.724066 → 1.061484`, and all
  four coverage diagnostics. State residual 95% undercoverage.
- Report the bounded RMSE change only with the clipping caveat; the transformation
  is mean-preserving before clipping.
- Freeze CFG, calibrator family, grids, metrics, ensemble size and random seeds
  before opening test-2023.
- Publication checkpoint is retrained on the corrected disjoint split, ideally
  with at least three training seeds.
- The final 200-day test is run once after explicit approval.

## Publication tiers

Minimum strong domain/SciML paper:

- clean checkpoint, frozen 200-day test, exact 3D-Var comparison;
- scientifically correct finite-ensemble diagnostics;
- strong SIC calibration baselines;
- spatial/physical preservation and uncertainty intervals.

Top-ML main-track package:

- a general calibrator rather than sea-ice-specific tuning;
- formal preservation or coverage result;
- another public structured task and another generative backbone;
- ensemble-size and OOD robustness;
- downstream utility of the calibrated analysis ensemble.
