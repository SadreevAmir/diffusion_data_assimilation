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

Claim-level audit: the two `MISSING` rows are the only absent baseline-family
results. `PRESENT_DEVELOPMENT_ONLY` is deliberately not absence: it records an
existing deterministic comparison whose evidence strength is insufficient for
the main comparison table. Therefore the manuscript conclusion may say that an
independent deterministic comparison remains required, but must not count it as
a third missing family. Baseline-row closure remains independent of every
learned-joint no-compensation gate decision.

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

## Completed no-compensation gate cross-check

The completed compact decisions admitted on the publication surfaces were
cross-checked against the claim ledger and reproducibility handoff on
2026-08-27. No admitted candidate has `overall_eligible=true`. This is a
decision inventory, not a new numerical analysis: it neither reconstructs
server-only case data nor changes any comparison row above.

| Mechanism family | Decision-bearing publication record | Joint-gate result |
|---|---|---|
| Global anomaly scaling | `CLAIM_LEDGER.md` C8, C18--C20 | `overall_eligible=false` |
| Purged hurdle-IDR/ECC-Q | `CLAIM_LEDGER.md` C21--C22 | `overall_eligible=false` |
| Mean-preserving projected spread | `CLAIM_LEDGER.md` C23--C25 | `overall_eligible=false` |
| Mean-preserving open-logit desaturation | `CLAIM_LEDGER.md` C26--C27 | `overall_eligible=false` |
| ZOIB-EMOS/ECC-Q | `CLAIM_LEDGER.md` C28--C29 | `overall_eligible=false` |
| Topology-preserving stratified transport | `CLAIM_LEDGER.md` C30 | `overall_eligible=false` |
| Purged analog-residual dressing | `CLAIM_LEDGER.md` C31 | `overall_eligible=false` |
| Frozen guidance mixture | `CLAIM_LEDGER.md` C32 | `overall_eligible=false` |
| Coherent and slack-limited member offsets | `CLAIM_LEDGER.md` C33--C34 | `overall_eligible=false` |
| Frozen latent temperature | `CLAIM_LEDGER.md` C35 | `overall_eligible=false` |
| Locked MC dropout | `CLAIM_LEDGER.md` C38 | `overall_eligible=false` |
| Cross-fitted iid calendar global-bias mixture | `CLAIM_LEDGER.md` C37 | `overall_eligible=false` |

The independent primary in C39 is also negative, but it is not counted as an
additional development calibration candidate and does not repair a missing
minimum-tier row. The superseded primary recovery is excluded from this audit.
Consequently the eligible-calibration blocker remains open.  The current
decision-bearing line is the running calendar-residual CFM training followed by
its frozen final-EMA sampling and full no-compensation gate.  Early training
diagnostics are not selection evidence, and no parallel calibration route is
admissible while that line is active.  A scientifically valid negative compact
verdict may activate at most the first matching branch in
`RESIDUAL_CFM_FOLLOWUP_DECISION_CONTRACT.md`; the older score-aware
raw-scenario reweighting material remains historical pre-result provenance and
is not the current next action.  This conclusion preserves every existing
evidence row and forbids rerunning or retuning a rejected mechanism merely to
create activity.

## Decision-bearing evidence guard

The rows below are normative and repeated verbatim on every publication surface
that can close these routes. `NONE` means that no compact evidence record is
admitted; while it remains present, only pre-result or development-only wording
is allowed.

| Route | Evidence status | Compact evidence record SHA-256 | Allowed presentation |
|---|---|---|---|
| `conformal` | `MISSING` | `NONE` | `PRE_RESULT_ONLY` |
| `probabilistic_da` | `MISSING` | `NONE` | `PRE_RESULT_ONLY` |
| `independent_deterministic` | `PRESENT_DEVELOPMENT_ONLY` | `NONE` | `DEVELOPMENT_ONLY` |
