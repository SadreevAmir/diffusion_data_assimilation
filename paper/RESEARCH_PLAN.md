# Research plan: reliable finite generative ensembles

## Research question

Can a finite ensemble from a conditional diffusion/flow data-assimilation model
be calibrated for bounded, zero-inflated spatial fields while preserving useful
joint scenarios and sparse-observation skill?

## Immediate correctness phase

1. Add fair finite-ensemble CRPS alongside empirical CRPS.
2. Add finite-member spread-skill correction and document the ideal target for
   each ensemble size.
3. Replace literal nominal-coverage interpretation for small ensembles with:
   randomized ranks, order-statistic coverage with its attainable target, and
   continuous-distribution/conformal coverage when available.
4. Report diagnostics separately for exact zero, marginal ice
   `0 < siconc < 0.15`, and established ice `siconc >= 0.15`.
5. Add paired temporal-block bootstrap intervals. Pixels are never treated as
   independent replicates.
6. Reclassify the existing affine-logit family as a negative baseline until its
   post-transform reliability and boundary mass are measured.

## Method-development phase

Working method: boundary-aware rank-preserving calibration.

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

## Data discipline

- Development remains on temporally blocked 2022 validation data.
- Sequential adaptation to the same dates is recorded and never reported as a
  final generalization estimate.
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

