# Minimum-tier comparison evidence audit

Status: `INCOMPLETE_FAIL_CLOSED`

This inventory maps every baseline family required by `RESEARCH_PLAN.md` to a
concrete manuscript presentation and to the compact publication source that can
support it. `MISSING` is evidence status, not a planned result: no numerical or
comparative claim may be inferred for that row.

| Required comparison | Manuscript presentation | Compact publication source | Evidence status |
|---|---|---|---|
| Raw ensemble | Section 8 frozen-mechanism tables | `REPRODUCIBILITY.md` | PRESENT |
| Physical-space bias/spread scaling | Section 8 frozen-mechanism and paired-diagnostic tables; Figure 1 | `REPRODUCIBILITY.md` | PRESENT_NEGATIVE |
| Naive affine-logit scaling | Section 6 affine-logit paragraph | `CLAIM_LEDGER.md` | PRESENT_NEGATIVE |
| Zero/one-inflated Beta or EMOS-like SIC postprocessing | Section 8 ZOIB-EMOS/ECC-Q table | `REPRODUCIBILITY.md` | PRESENT_NEGATIVE |
| Isotonic/quantile mapping | Section 8 purged hurdle-IDR/ECC-Q table | `REPRODUCIBILITY.md` | PRESENT_NEGATIVE |
| Conformal intervals | Section 5 baseline inventory | `RESEARCH_PLAN.md` | MISSING |
| ECC-Q/ECC-T or rank-preserving reconstruction | Section 8 hurdle-IDR/ECC-Q and ZOIB-EMOS/ECC-Q tables | `REPRODUCIBILITY.md` | PRESENT_PARTIAL_NEGATIVE |
| Deterministic background and 3D-Var | Section 5 deterministic-comparison paragraph | `CLAIM_LEDGER.md` | PRESENT_DEVELOPMENT_ONLY |
| Probabilistic DA baseline such as EnKF/LETKF | Section 5 baseline inventory | `RESEARCH_PLAN.md` | MISSING |

The comparison tier therefore remains incomplete for exactly two baseline
families: conformal intervals and a probabilistic DA comparator. The available
deterministic comparison is development-only and cannot be promoted into an
independent comparison. These gaps are distinct from the missing eligible
calibration candidate.

The exact next conformal comparison is now pre-result frozen in
`NEXT_CONFORMAL_BASELINE_CONTRACT.md`: a purged, five-fold split-conformal
member-range interval for date-level sea-ice area with immutable coverage and
width criteria. Its status remains `MISSING` until a separately reviewed
trusted runner executes the literal contract; the document is not numerical
evidence and no unimplemented mode may be proposed for it.

The exact probabilistic DA comparison is likewise pre-result frozen in
`NEXT_PROBABILISTIC_DA_COMPARISON_CONTRACT.md`: a ten-member LETKF with
hash-identical observations, five contiguous purged holdouts, a fixed
localization/inflation set, training-only fair-CRPS selection and joint
fair-CRPS/rank/RMSE usefulness criteria. Its status remains `MISSING` until a
separately reviewed trusted runner executes the literal contract. Invalid
common-information or leakage checks cannot close the row.
