# Frozen contingent method contract: purged casewise safety selector

Status: DESIGN_FROZEN_AFTER_SCORE_AWARE_ROUTE

## Scientific question and activation

Activate this development-only mechanism only if the frozen score-aware
raw-scenario route is completed and fails the unchanged no-compensation gate.
It asks a distinct question: is the failure of mean-preserving projected spread
confined to forecast-identifiable regimes, so that a leakage-safe case-level
policy can retain its proper-score gain without accepting its boundary,
reliability or member-spatial failures?  The learned-joint raw ensemble remains
the baseline.  This contract does not authorize execution until an independently
reviewed literal trusted mode exists.

## Immutable inputs and folds

- Reuse only the completed raw and mean-preserving projected-spread compact
  server artifacts; never download fields and never resample.
- Use the established forty ordered development cases, five contiguous
  eight-case holdouts and a non-circular three-case purge.
- For each case compute exactly the six forecast-only descriptors already frozen
  for analog selection: raw ensemble-mean ice area, extent at `0.15`, domain
  mean, domain standard deviation, semivariogram at lag 1 and semivariogram at
  lag 4. Standardize on retained training cases only; zero or non-finite
  training standard deviation is an operational failure.
- The action is binary: copy all ten raw members or all ten projected-spread
  members for the held-out case. Mixing members or pixels across actions is
  forbidden.

## Frozen training objective and policy

For every retained training case and action, form a scalar loss from already
reported case-level diagnostics:

`L = fair_crps + 4*max(0, rank_abs_error-rank_limit) + 4*max(0, inner_coverage_error-inner_limit) + 4*max(0, boundary_error-boundary_limit) + 2*max(0, iiee_ratio-1.02) + 2*max(0, edge_ratio-1.02) + 2*max(0, variogram_ratio-1.02)`.

Here every `limit` is the unchanged threshold from the joint gate and ratios use
the raw case diagnostic as denominator with the gate's existing near-zero
convention. Missing case-level components fail operationally. Fit one ridge
linear loss model per action and fold from the six standardized descriptors,
with an unpenalized intercept, ridge coefficient exactly `1.0`, deterministic
SVD, and relative singular-value cutoff `1e-12`. Select projected spread only
when its predicted loss is at least `0.002` lower than predicted raw loss;
otherwise select raw. The margin is fixed as a conservative fraction of the
observed aggregate fair-CRPS scale, not tuned by folds. Ties select raw.

Record fold membership, standardization statistics, both coefficient vectors,
both predicted losses, selected action, action margin and exact source hashes.
No tree, interaction, alternate penalty, threshold search, feasibility veto or
post-result subgroup rule is permitted.

## Falsifiable prediction, stop/go and publication role

Prediction: the selector chooses projected spread for at least one held-out case
and raw for at least one held-out case, improves `analysis_fair_crps` by at least
3% versus raw with the paired-date 95% interval excluding zero in the improving
direction, worsens `analysis_crps` by no more than 1%, and passes every unchanged
finite-ensemble reliability, boundary, spatial/physical and operational gate.
The four-case-block interval is temporal sensitivity only.

Success requires `overall_eligible=true` and all five mandatory families true.
If only one action is selected for all cases, the policy is a mechanistic null
or a replay of the rejected projected-spread candidate and cannot support the
conditional-safety claim. Proper-score failure rejects forecast-identifiable
case gating as useful calibration. Reliability or boundary failure shows that
case selection cannot isolate the projected-spread defect. Spatial failure
shows that whole-case switching does not protect member geometry. Source-hash,
fold, finiteness or missing-diagnostic failure is operational, not scientific.

A positive result supports the paper claim that a predeclared forecast-regime
policy can conditionally deploy an otherwise unsafe spread correction. A
negative result closes this binary safety-selection mechanism; no margin,
descriptor, coefficient, fold, loss weight or source candidate may be tuned
afterward.

## Execution and compact evidence boundary

Any future execution is one `server_cpu`, `summary_only` full run over the exact
forty-case, ten-member development envelope, with sole runtime parameter
`source_experiment=joint_full_condition_validation_2022`. Expected wall time is
at most one hour because it reuses compact server artifacts. The result must
contain exactly four compact files: `case_selection.json`,
`aggregate_selection.json`, `paired_uncertainty.json` and `gate_decision.json`.
They must expose every recorded policy quantity, raw/projected source identities,
case counts, action counts, paired uncertainty and the full unchanged gate. No
raw member, candidate or truth field is retrieved.

This document freezes a contingent scientific mechanism, not a mode name or a
proposal. Independent runner review, exact artifact identities and an atomic
fail-closed admission remain mandatory before execution.
