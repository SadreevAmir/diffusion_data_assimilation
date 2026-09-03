# Calendar-residual CFM compact-result reconciliation

Status: RECONCILED_NEGATIVE_ROUTER_FAIL_CLOSED

The trusted compact result for `siconc_calendar_residual_cfm_gate_valid`
completed all 40 cases and records `overall_eligible=false`. The final-EMA
candidate passes the proper-and-mean-skill family, with
`analysis_fair_crps=0.07794889197556103` below
`background_mean_absolute_error=0.10635058634102776` and
`analysis_mean_rmse=0.20251928397420751` below
`background_mean_rmse=0.25420465006832627`. This partial skill result cannot
compensate for failed coverage, randomized-rank, truth-relative-event and
spatial-distribution families.

The decision-bearing absolute diagnostics include
`rank_tv_to_uniform=0.19860927053290894`,
`rank_max_bin_deviation=0.1971123742306176`, and
`rank_normalized_mean_rank=0.6108636429056068`, whose four-date-block interval
is `[0.5731363558093633, 0.6466001051154471]`. Member-range attainable coverage
has absolute error `0.09998322451855779`. All three pooled member-variogram
safeguards fail, and the truth-relative event family fails at the fixed event
levels. Raw members were not read for this reconciliation.

## Frozen follow-up router

The first-match router in `RESIDUAL_CFM_FOLLOWUP_DECISION_CONTRACT.md` is
applied fail closed. Branches 1 and 4 require every local/member variogram
safeguard to pass, which contradicts the compact family decisions. Branch 3
requires every absolute finite-ensemble reliability decision to pass and the
proper-score family to fail; the observed result has the opposite proper-score
decision and failed reliability. Branch 2 requires a predeclared concentration
fraction for rank-TV excess or paired fair-CRPS degradation and spatial-family
pass outside the implicated stratum. Those decision fields are not emitted by
the admitted aggregate or per-case compact schema, so branch 2 is not proven
and cannot be selected post hoc.

Therefore no natural follow-up is authorized from this result. This is not an
eligible calibration and does not change the `MISSING_ELIGIBLE_RESULT` guard.
It is a negative natural-generative mechanism result: improved mean and proper
scores coexist with severe upper-rank accumulation, insufficient attainable
range, event-frequency error and member-spatial mismatch. No conditioning,
loss, checkpoint, member count or routing threshold may be tuned after this
result.
