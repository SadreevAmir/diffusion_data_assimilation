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
| Naive affine-logit scaling | Section 6 affine-logit paragraph | `REPRODUCIBILITY.md` | PRESENT_NEGATIVE |
| Zero/one-inflated Beta or EMOS-like SIC postprocessing | Section 8 ZOIB-EMOS/ECC-Q table | `REPRODUCIBILITY.md` | PRESENT_NEGATIVE |
| Isotonic/quantile mapping | Section 8 purged hurdle-IDR/ECC-Q table | `REPRODUCIBILITY.md` | PRESENT_NEGATIVE |
| Conformal intervals | Section 5 baseline inventory | `RESEARCH_PLAN.md` | MISSING |
| ECC-Q/ECC-T or rank-preserving reconstruction | Section 8 hurdle-IDR/ECC-Q and ZOIB-EMOS/ECC-Q tables | `REPRODUCIBILITY.md` | PRESENT_PARTIAL_NEGATIVE |
| Deterministic background and 3D-Var | Section 5 deterministic-comparison paragraph | `REPRODUCIBILITY.md` | PRESENT_DEVELOPMENT_ONLY |
| Probabilistic DA baseline such as EnKF/LETKF | Section 5 baseline inventory | `RESEARCH_PLAN.md` | MISSING |

The comparison tier therefore remains incomplete for exactly two baseline
families: conformal intervals and a probabilistic DA comparator. The available
deterministic comparison is development-only and cannot be promoted into an
independent comparison. These gaps are distinct from the missing eligible
calibration candidate.
