# Claim ledger

This ledger separates immutable evidence from hypotheses and planned results.
Every manuscript claim must point to a row here before it is strengthened.

Last audited: 2026-08-13.

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
| C8 | Finite-ensemble-aware, boundary-aware calibration can improve reliability without destroying spatial dependence. | Central hypothesis, unverified | Motivated by current failure mode and classical ensemble postprocessing | Requires a new method, frozen nested protocol, strong baselines and spatial diagnostics. |
| C9 | The proposed method outperforms 3D-Var on test-2023. | Unknown and locked | No admissible final artifact | Freeze the complete method first; open the 200-day test once after explicit approval. |
| C10 | The method generalizes beyond sea ice and one flow checkpoint. | Unknown | No evidence | Add clean retraining seeds, another generative backbone and at least one public structured benchmark. |

## Mandatory language discipline

- Say **model-to-model sparse-observation assimilation experiment**, not real-observation assimilation.
- Say **empirical ensemble CRPS** until fair finite-ensemble CRPS is reported.
- Say **nominal quantile interval diagnostic** for the current M=10 coverage values.
- Say **score-optimized affine postprocessing**, not calibration, for the current logit result.
- Never state superiority to 3D-Var or test-year performance before the locked run.
