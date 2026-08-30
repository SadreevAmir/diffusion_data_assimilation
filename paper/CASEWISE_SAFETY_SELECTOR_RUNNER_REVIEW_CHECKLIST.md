# Casewise safety-selector trusted-runner review checklist

Status: `LOCAL_REVIEW_HANDOFF_READY_NO_TRUSTED_MODE`

This outcome-agnostic checklist binds an independent trusted implementation to
`NEXT_CASEWISE_SAFETY_SELECTOR_CONTRACT.md`. It does not name or authorize an
executor mode. Admission is `NO_GO` unless one controller-visible review record
satisfies every item below without deviation.

## Interface and immutable identities

- [ ] The mode accepts exactly
  `source_experiment=joint_full_condition_validation_2022`; additional or
  substituted parameters are rejected.
- [ ] Execution is `server_cpu`, `summary_only`, forty ordered cases and ten
  members. Raw members, projected candidates and truth remain server-side.
- [ ] One admission identity binds the reviewed publication commit and exact
  SHA-256 identities of the trusted runner, frozen contract, independent
  reference and compact validator. A proposal may consume only the literal
  `reviewed_mode` emitted by that same admission operation.

## Leakage-safe policy parity

- [ ] The five contiguous eight-case holdouts and non-circular three-case purge
  exactly match the frozen ordered split; descriptor standardization and both
  action-specific fits use retained training cases only.
- [ ] The six forecast-only descriptors, scalar loss terms and weights,
  unpenalized intercept, ridge coefficient `1.0`, deterministic SVD and relative
  cutoff `1e-12` match `casewise_safety_selector_reference.py` exactly.
- [ ] The trusted runner matches the independent reference for purge-edge,
  exact-margin-tie, non-degenerate-action and zero-variance fixtures. Held-out
  truth cannot enter descriptors, fitting, predicted losses or action choice.
- [ ] Each held-out action copies all ten members from exactly one recorded
  source. Projected spread is selected only when its predicted loss is at least
  `0.002` lower; ties select raw. Pixelwise or memberwise action mixing fails.

## Compact evidence and scientific decision

- [ ] The output directory contains exactly `case_selection.json`,
  `aggregate_selection.json`, `paired_uncertainty.json` and
  `gate_decision.json`, with no field arrays or additional files.
- [ ] `validate_casewise_safety_selector_compact_outputs.py` independently
  reconstructs fold and purge membership, margin decisions, cross-case source
  identities, action counts, aggregate parity, paired uncertainty, operational
  validity and the five-family conjunction defining `overall_eligible`.
- [ ] Success requires both actions to be selected, at least 3% improvement in
  `analysis_fair_crps` with the paired-date 95% interval wholly improving,
  `analysis_crps` degradation no larger than 1%, and every reliability,
  boundary, spatial/physical and operational criterion to pass.
- [ ] The four-case-block interval is temporal sensitivity only. No descriptor,
  loss weight, ridge setting, action margin, fold, source candidate or gate
  threshold changes after seeing the result.

## Independent negative-path review

- [ ] Trusted tests reject envelope, case order, fold, purge, descriptor,
  standardization, loss, ridge, SVD, margin, tie-breaking and whole-case-copy
  drift, as well as non-finite or zero-variance training inputs.
- [ ] Trusted tests reject source-hash disagreement, action/count mismatch,
  aggregate or paired-uncertainty disagreement, non-Boolean family flags, false
  eligibility, field substitution and every extra compact field or file.
- [ ] One synthetic dry run contains no project-data metric, passes semantic
  parity against the independent reference and produces the exact reviewed
  four-file schema. The review records the exact command, success sentinel,
  `decision_bearing_validation=PASS` and `deviations=[]`.

## Atomic admission and proposal boundary

Independent review must register exactly one literal executor-visible mode and
bind its runner, contract, reference and compact validator in one fail-closed
admission identity. Registration alone, an unbound review JSON or a local
semantic `GO` does not authorize execution. The proposal must use the literal
mode emitted by admission and the sole frozen source parameter; it must not
reconstruct or infer the mode name later.

Only that atomic `admission=GO` authorizes one full `server_cpu`,
`summary_only` proposal over the frozen envelope. Scientific success still
requires `overall_eligible=true`, all five families true and a non-degenerate
policy. A negative result closes this exact conditional-safety mechanism
without tuning; a single-action result is reported as the frozen mechanistic
null interpretation.
