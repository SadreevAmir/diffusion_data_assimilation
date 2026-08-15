# Claim ledger

This ledger separates immutable evidence from hypotheses and planned results.
Every manuscript claim must point to a row here before it is strengthened.

Last audited: 2026-08-15.

## Evidence status

| ID | Claim | Status | Evidence | Limitation / required upgrade |
|---|---|---|---|---|
| C1 | The conditional model supports independent post-training guidance from dense background and sparse tracks while assigning zero coefficient to the joint branch. | Verified in code | `assim_lib/trainer.py`, `assim_lib/sampler.py` | The scientific advantage over learned joint conditioning is not yet established. |
| C2 | The strict comparison provides only `siconc` background and track information, uses real SRAL footprint geometry for `d,d-1,d-2`, and hides `sithic`. | Verified in code and run metadata | `assim_lib/data.py`, `assim_lib/compare_3dvar.py`; validation metadata | This is a model-to-model/OSSE-like experiment: observation values come from M2M truth, not measured SRAL SIC. |
| C3 | On the same 40 validation-2022 dates, changing track/background CFG from 0.5/0.5 to 0.625/0.375 improves full ensemble-mean RMSE from 0.21291 to 0.20366 and empirical ensemble CRPS from 0.07070 to 0.06609. | Verified preliminary validation evidence | Saved immutable validation artifacts, ensemble size 10 | The weight was adapted on these validation dates. This is not an independent generalization estimate. |
| C4 | At CFG 0.625, the ensemble-mean analysis improves full-field RMSE over the background from 0.25603 to 0.20366 and withheld next-track RMSE from 0.19046 to 0.12384. | Verified preliminary validation evidence | Saved 40-date validation artifact | Requires temporal-block uncertainty intervals, clean retraining seeds and locked test confirmation. |
| C5 | The raw finite ensemble exhibits nonuniform randomized ranks and insufficient dispersion. | Supported preliminary evidence | M=10 saved ensembles; extreme ranks and spread diagnostics | Naive nominal coverage and raw spread-skill targets are finite-ensemble biased and must not be compared directly with 0.9 and 1.0. |
| C6 | Blocked affine-logit postprocessing improves held-out empirical CRPS and ensemble-mean RMSE. | Verified for the searched validation protocol | CRPS 0.06609 → 0.06092; RMSE 0.20366 → 0.19428 | Repeated grid adaptation used the same 40 dates; final performance is meta-adapted. It is not a locked estimate. |
| C7 | The affine-logit result is a calibrated ensemble. | Rejected | Out-of-fold transforms change the nominal 90% interval diagnostic from 0.646 to 0.367 and member-range coverage from 0.722 to 0.396; truth is exactly zero on 53.6% of evaluated pixels, raw members on 24.6%, and calibrated members on 0% | Finite-M targets still require correction, but the direction and zero-mass failure are unambiguous. Call this score-optimized postprocessing or a negative baseline. |
| C8 | Finite-ensemble-aware, boundary-aware calibration can improve reliability without destroying spatial dependence. | Active hypothesis, unverified | Motivated by the affine-logit failure mode and classical ensemble postprocessing; joint stop/go gate is predeclared in `RESEARCH_PLAN.md` | Requires an allowlisted existing-ensemble analysis with strong baselines, randomized ranks, boundary diagnostics and paired spatial/physical metrics. |
| C9 | The frozen global-spread method outperforms 3D-Var on the independent evaluation period. | Unknown and outside current scope | No admissible final artifact | Do not infer deterministic-method superiority from the present validation calibration result. |
| C10 | The method generalizes beyond sea ice and one flow checkpoint. | Unknown | No evidence | Add clean retraining seeds, another generative backbone and at least one public structured benchmark. |
| C11 | The learned-joint ten-member ensemble is globally underdispersed on the 40-date validation set, and a mean-preserving global anomaly scale removes most of its proper-score deficit. | Supported validation mechanism claim | Five-fold date-stratified cross-fit; scales 2.6, 2.7, 2.8, 2.8, 2.6; fair CRPS 0.0584905850 → 0.0556896736 (4.79%); spread-skill 0.724066 → 1.061484 | One checkpoint, one seed and one 40-date development period. “Principally global” is supported; universal or conditional calibration is not. |
| C12 | The frozen spread correction preserves the ensemble center before clipping. | Verified numerically and algebraically | `x_bar + s(x_k-x_bar)`; maximum reported pre-clipping invariance error 4.16e-16 | Bounded scoring clips members; bounded RMSE 0.1903366 → 0.1876254 must not be called an intrinsic center-skill gain. |
| C13 | The frozen correction improves the reported pointwise coverage diagnostics. | Verified validation evidence | 50%: 0.208196 → 0.596277; 80%: 0.471026 → 0.856292; 90%: 0.505084 → 0.878775; 95%: 0.525005 → 0.891464 | These are small-ensemble pointwise diagnostics, not guaranteed continuous, casewise or fieldwise coverage; residual upper-tail undercoverage remains. |
| C14 | The frozen correction improves ordinary ensemble CRPS. | Verified validation evidence | 0.0621082810 → 0.0611013421 | Ordinary CRPS improves less than fair CRPS and was not the selection objective; report both. |
| C15 | The available compact corrected-case table supports paired uncertainty against the raw ensemble. | Rejected by artifact contract | The 40-row contract lists corrected metrics but no `raw_*` fields | Aggregate raw and corrected means cannot identify paired case differences; no paired interval or improved-case count is reported. |
| C16 | The frozen spread correction preserves spatial and physical sea-ice structure. | Unknown; required for minimum publication tier | Raw learned-joint IIEE is available, but the calibrated compact result reports no paired IIEE, edge, area/extent, variogram or spectrum metric | Obtain paired case/block uncertainty for predeclared physical metrics from an allowlisted server analysis; do not infer preservation from marginal CRPS. |
| C17 | The current baseline set is sufficient for a strong domain/SciML paper. | Rejected by readiness audit | Only the raw ensemble and failed affine-logit postprocessor are evaluated | Add at least one boundary-aware distributional baseline and one rank-preserving or conformal baseline under the same blocked protocol. |
| C18 | Passing the earlier fair-CRPS and pointwise-coverage gate establishes a well-calibrated SIC ensemble. | Rejected by revised scientific objective | The global scaling improves fair CRPS and reported coverage, but no randomized ranks, boundary-mass preservation or calibrated spatial/physical metrics are available | Treat global scaling as a frozen reference baseline; accept a paper method only if every family in the joint stop/go gate passes. |

## Mandatory language discipline

- Say **model-to-model sparse-observation assimilation experiment**, not real-observation assimilation.
- Say **empirical ensemble CRPS** until fair finite-ensemble CRPS is reported.
- Say **nominal quantile interval diagnostic** for the current M=10 coverage values.
- Say **score-optimized affine postprocessing**, not calibration, for the current logit result.
- Say **cross-fitted global spread calibration on the validation set**, not independently validated calibration.
- Attribute the fair-CRPS and coverage gains to dispersion correction; do not attribute the bounded RMSE change to an unclipped mean shift.
- Never state superiority to 3D-Var or test-year performance before the locked run.
