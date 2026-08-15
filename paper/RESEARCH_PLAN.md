# Research plan: reliable finite generative ensembles

## Research question

Can a finite ensemble from a conditional diffusion/flow data-assimilation model
be calibrated for bounded, zero-inflated spatial fields while preserving useful
joint scenarios and sparse-observation skill?

## Frozen mechanism result; broad calibration claim rejected

The current reference method is five-fold, deterministic date-stratified cross-fitted global
spread calibration of the learned-joint ensemble. Within each fold, choose one
scale from `1.0:0.1:4.0` on the other 32 cases by minimum case-mean fair CRPS and
apply it to eight held-out cases. The member transform is
`mean + scale * (member - mean)`; clip only for bounded scoring. Selected scales
are `2.6, 2.7, 2.8, 2.8, 2.6`. Its transform, folds and grid are frozen as a
reference baseline. The completed joint audit rejects it as the paper's
well-calibrated ensemble method: proper scores improve, but mandatory boundary
and spatial/physical preservation criteria fail.

The earlier narrow score/coverage gate was met: fair CRPS improves by at least 3% (observed 4.79%),
ordinary CRPS is at most 0.061526 (observed 0.061101), 90% coverage is within
0.85–0.94 (observed 0.878775), all selected scales are interior, and pre-clipping
mean-invariance error is below `1e-10` (observed `4.16e-16`). That gate diagnoses
global underdispersion but is insufficient for bounded, zero-inflated SIC, so
the joint gate below was applied as predeclared and failed.

The failure is mechanistically informative. Established-ice Brier score worsens
by 2.15% (`0.0569726` to `0.0581999`) against a 1% tolerance; exact-one member
mass moves from `0.0090419` to `0.1648321` while truth exact-one mass is zero;
mean IIEE worsens by 5.50%, edge disagreement by 4.41%, and absolute ice-extent
error by 10.05%. Global anomaly inflation therefore repairs a marginal
underdispersion signal while introducing clipping-related boundary atoms and
degrading physical/spatial diagnostics. No family averaging is used to excuse
these failures.

A fixed boundary-aware comparator has also been completed under the same joint
gate. Five contiguous eight-date holdouts with a three-case purge fit a hurdle
isotonic bounded distribution and reconstruct ten members with ECC-Q from the
raw rank template. It substantially improves exact-zero and exact-one mass
errors and randomized-rank discrepancy, but fair CRPS worsens by `0.0270909`
and mean IIEE by `0.103785`, with paired date intervals excluding zero; edge
disagreement worsens by `0.270714`. This rejects the specific hypothesis that
the fixed marginal hurdle fit plus ECC-Q can preserve useful conditional
spatial structure. It counts as a strong negative mechanism baseline, not as an
eligible calibrated ensemble and not as exhaustive coverage of the required
baseline families.

A targeted exact capped-simplex projection has now tested whether the frozen
spread signal can be retained while preserving the raw ensemble mean exactly.
It passes proper scores, all mean-field invariants and operational validity:
fair CRPS improves from `0.0584906` to `0.0563001`, with both paired date and
four-case-block intervals excluding zero, while maximum mean error is
`3.87e-16`. It nevertheless fails boundary, finite-ensemble reliability and
member-spatial safety. Upper-cap mass reaches `0.167438`, established-ice Brier
worsens, inner-order error does not improve and local/member variogram
tolerances fail. This rejects spread-only correction even under exact mean
preservation and localizes the unresolved mechanism beyond mean-field shift.

## Predeclared calibration stop/go gate

All candidates must be evaluated out of fold on the same 40 date-level cases
and compared pairwise with the raw ensemble and frozen global scaling. A method
is eligible for the paper only if every mandatory family passes; no weighted
average may compensate for a failed family.

1. **Proper scores:** case-mean fair CRPS improves by at least 3% over raw and
   ordinary CRPS is no worse than raw by more than 1%. Report paired date-level
   and contiguous four-case block 95% intervals; the fair-CRPS interval must not
   cross zero for a positive calibration claim.
2. **Finite-ensemble reliability:** randomized-rank histogram with randomized
   tie handling must reduce a predeclared discrepancy from discrete uniformity
   by at least 20% versus raw. Report attainable order-statistic coverage and
   width for `M=10`; each reported central coverage diagnostic must move closer
   to its finite-ensemble target, with none worsening by more than 0.02.
3. **Boundary behaviour:** report exact-zero and exact-one member mass and Brier
   scores for `siconc > 0` and `siconc > 0.15`. Neither Brier score may worsen by
   more than 1%, and the absolute error in each boundary mass may not increase
   versus raw. A method that removes an observed boundary atom fails.
4. **Spatial and physical preservation:** paired case-level IIEE plus ice area,
   extent and an edge metric are mandatory. Mean IIEE and edge error may worsen
   by at most 2%; absolute mean area and extent bias may worsen by at most 2% of
   their raw absolute bias (or by at most 0.1 percentage point of domain area
   when raw bias is near zero). At least one patchwise variogram or spectral
   discrepancy must not worsen by more than 2%.
5. **Operational validity:** all 40 cases must be finite, fold selection must use
   only the other folds, member identities or raw ranks must be preserved for
   rank-reconstruction methods, and all thresholds and tie randomization seeds
   must be fixed before reading candidate summaries.

Failure of any mandatory family rejects a broad "well-calibrated ensemble"
claim. A method may remain a named mechanism or negative baseline when its
failure isolates a scientifically useful tradeoff.

## Correctness status

- Complete for the frozen decision: fair and ordinary CRPS are both reported;
  the affine-logit family is a documented negative baseline; coverage is called
  a finite-ensemble pointwise diagnostic; and pre-clipping center invariance is
  verified numerically.
- Now available for the narrow mechanism claim: paired date-bootstrap intervals
  and a circular four-date-block interval. The latter is reported honestly as a
  post-hoc temporal sensitivity, not as a predeclared acceptance gate. The
  fair-CRPS interval excludes zero, the ordinary-CRPS interval does not, and the
  IIEE interval supports degradation; none rescues the failed joint gate.
- Not claimed: conditional calibration by regime, randomized-rank uniformity
  after an accepted correction, or fieldwise coverage. These are manuscript
  limitations, not reasons to reopen selection of the rejected global-spread
  reference.
- Required before any broader confirmatory claim: repeat the fully frozen
  procedure without retuning on a genuinely independent period. Pixels must
  never be treated as independent replicates.

## Deferred alternatives, not current method development

Additional boundary-aware conditional, case-adaptive, hybrid-sampling and
affine mean-and-spread methods are not selected by the current evidence. With
40 cases they add estimation flexibility, confound the clean mechanism
attribution, or require new sampling without repairing a failed criterion. The
completed fixed purged hurdle-isotonic/ECC-Q construction is retained below as
an evaluated negative instance of items 1, 2 and 4, not as evidence that every
method in those families fails.

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
- zero/one-inflated Beta or EMOS-like SIC postprocessing (the frozen
  ZOIB-EMOS/ECC-Q construction is completed and rejected by the joint gate);
- isotonic/quantile mapping (one fixed hurdle-isotonic construction completed
  and rejected by the joint gate);
- conformal intervals;
- ECC-Q/ECC-T or rank-preserving reconstruction (ECC-Q completed within both
  rejected hurdle-isotonic and ZOIB-EMOS constructions; broader dependence-
  reconstruction coverage remains missing);
- deterministic background and 3D-Var;
- probabilistic DA baseline such as EnKF/LETKF when a fair implementation is
  available.

The completed baseline was design-frozen in `NEXT_BASELINE_CONTRACT.md`: a pooled,
date-balanced ZOIB-EMOS distribution with purged contiguous cross-fitting and
ECC-Q reconstruction. Its optimizer, predictors, folds, seeds, reconstruction
and joint no-compensation interpretation were fixed before execution. The
reviewed run is complete and rejected by the joint gate: boundary masses and
randomized ranks improve, but proper-score, inner-order, boundary and spatial
families fail.

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
