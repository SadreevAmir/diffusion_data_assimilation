# Residual-CFM: frozen diagnostic follow-up decision contract

Status: PRE_RESULT_ROUTER_FROZEN

## Purpose and evidence boundary

This contract is frozen before the compact evaluation of
`siconc_calendar_residual_cfm_training_retry1`. It does not select, authorize or
launch another model. Its only purpose is to make the permitted response to a
negative baseline result falsifiable: at most one natural conditional-generative
follow-up may be selected, and only from the dominant failure signature reported
by the unchanged trusted gate. Early training loss, scheduler state and relative
improvement without absolute gate passage are not selection evidence.

If `gate.overall_eligible=true`, no follow-up is selected. The candidate advances
to publication reconciliation. If the execution is operationally invalid, no
scientific branch is selected; the exact same frozen evaluation is repaired or
recovered without changing the model.

## Required compact diagnostic inputs

The router consumes only the completed gate summary and its already frozen
per-date compact metrics. It requires family Booleans for proper score,
finite-ensemble reliability, boundary, spatial/physical and operational validity;
the 11-bin equal-date randomized-rank histogram; attainable range and inner
coverage; truth-referenced event diagnostics at `q={0,.15,.90,.95,.99}`; paired
per-date score changes; casewise calendar month, ice-area and established-ice
fraction; area/extent error; and mean/member/local variogram summaries. It never
reads raw members and introduces no new validation metric, threshold or fitted
weight.

## Deterministic routing order

Only the first satisfied branch is admissible. A branch must satisfy every item
listed for it; otherwise the router continues downward. The selected branch is a
model-family contract to be implemented and independently reviewed later, not a
parameter grid.

1. **Natural hurdle conditional generative law.** Select only when the boundary
   family fails and the failure is localized to occurrence or boundary-event
   probabilities: at least one truth-referenced Brier diagnostic at
   `q={0,.15,.90,.95,.99}` fails, while the interior-only rank TV is at most
   `0.10` and every local/member variogram safeguard passes. The follow-up must
   jointly learn occurrence/boundary atoms and a continuous interior residual;
   it must retain iid generative members and may not be a post-hoc transform.

2. **Regime-aware residual flow.** Select only when branch 1 is false and the
   absolute reliability or proper-score family fails with a pre-existing regime
   concentration: the worst calendar-quarter or established-ice-fraction quartile
   contributes at least `50%` of the total absolute per-date rank-TV excess or
   paired fair-CRPS degradation, despite containing at most `35%` of dates, and
   the boundary and local/member spatial families pass outside that stratum. The
   follow-up may add seasonal/ice-regime conditioning or train-only balanced
   weights, but no weight may be optimized on these evaluation dates.

3. **Proper-score-aware fine-tuning.** Select only when branches 1--2 are false,
   every absolute finite-ensemble reliability decision passes, boundary and
   spatial/physical families pass, but the proper-score family fails with paired
   fair-CRPS degradation on at least `24/40` dates. The follow-up must construct
   its small differentiable ensemble CRPS/rank surrogate using train-only
   cross-fitting or held-out training years and must preserve iid members. No
   coefficient may be selected from evaluation scores.

4. **Hierarchical low-frequency plus local residual flow.** Select only when
   branches 1--3 are false, attainable range or inner coverage fails on the low
   side, the absolute area-error or normalized mean-rank diagnostic fails, and
   every local/member variogram safeguard passes. The follow-up must jointly
   model a global low-frequency ice-area residual latent and a conditional local
   residual field; uniform output-space spread inflation is prohibited.

If no branch is satisfied, no new model is selected from this result. The paper
records the frozen residual-CFM negative result and its unresolved failure
signature. A mixed failure is not permission to choose the most convenient
model, combine branches, weaken thresholds or run a sweep.

## Common stop/go and publication role

Exactly one selected follow-up receives one frozen full evaluation. It succeeds
only with literal `gate.overall_eligible=true` under the same absolute
no-compensation families. Failure closes that diagnosed mechanism; it does not
trigger a second natural follow-up. A positive result supports the narrow claim
that the diagnosed structural extension repairs absolute calibration while
preserving predictive and physical utility. A negative result remains a named
mechanism test and limits the paper's calibration claim.

