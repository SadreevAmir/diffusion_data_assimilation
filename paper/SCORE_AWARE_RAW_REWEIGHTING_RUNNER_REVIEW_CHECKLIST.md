# Score-aware raw-scenario reweighting trusted-runner review checklist

Status: `LOCAL_REVIEW_HANDOFF_READY_NO_TRUSTED_MODE`

This outcome-agnostic checklist binds an independent trusted implementation to
`NEXT_SCORE_AWARE_RAW_REWEIGHTING_CONTRACT.md`.  It neither names nor authorizes
a production mode.  Admission is `NO_GO` unless one controller-visible record
satisfies every item below with no deviation.

## Interface and immutable identities

- [ ] The mode accepts exactly
  `source_experiment=joint_full_condition_validation_2022`; every additional or
  substituted parameter is rejected.
- [ ] Execution is `server_cpu`, `summary_only`, forty ordered cases and ten
  members.  Raw members, candidate fields and truth remain server-side.
- [ ] The record uses schema
  `score-aware-raw-reweighting-admission-v2` and binds the reviewed publication
  commit plus exact SHA-256 identities for the trusted runner, frozen contract,
  independent reference and complete compact directory.

## Leakage-safe score model and selection parity

- [ ] Five contiguous eight-case holdouts and the non-circular three-case purge
  match the frozen ordered split; all standardization and fitting use retained
  training rows only.
- [ ] The thirteen predictors, weighted absolute-plus-squared-error target,
  unpenalized intercept, ridge coefficient `1.0`, deterministic SVD and relative
  cutoff exactly match `score_aware_raw_reweighting_reference.py`.
- [ ] Held-out risks use no held-out truth.  Softmax risk weights, midpoint
  inverse-CDF positions, risk/member-index ordering and selected source indices
  match the independent reference for null, repeated-risk and extreme-risk
  fixtures.
- [ ] Every candidate member is a bitwise copy of its recorded raw source.
  Multiplicities, unique-member count, effective sample size and all five frozen
  member-mask invariants recompute exactly.

## Compact evidence and scientific decision

- [ ] The output directory contains exactly `case_selection.json`,
  `aggregate_selection.json`, `paired_uncertainty.json` and
  `gate_decision.json`; no field arrays or extra files are present.
- [ ] `validate_score_aware_compact_outputs.py` independently reconstructs the
  weights, selections, fold membership, purged training identities, aggregate
  totals, operational invariants and the conjunction defining
  `overall_eligible`.
- [ ] The primary prediction is at least 3% improvement in
  `analysis_fair_crps` with the paired-date interval wholly improving, while
  `analysis_crps` worsens by no more than 1%.  Absolute rank uniformity and all
  attainable coverage criteria pass, no coverage error worsens by more than
  `0.02`, and every boundary, spatial/physical and operational criterion passes.
- [ ] The fixed four-case-block interval is reported as temporal sensitivity
  only.  No family compensation, post-result threshold change or activity
  cutoff is introduced.

## Independent negative-path review

- [ ] Trusted tests reject envelope, order, fold, purge, predictor, target,
  ridge, SVD, softmax, inverse-CDF, tie-breaking or source-copy drift.
- [ ] Trusted tests reject non-finite inputs, incomplete cases or members,
  invalid weights, mask mismatch, aggregate/paired disagreement, false family
  eligibility, compact-file substitution and any extra compact file.
- [ ] One synthetic dry run contains no project-data metric, passes semantic
  parity against the independent reference and produces the exact reviewed
  four-file directory digest.
- [ ] The exact review command and success sentinel are recorded.  The reviewer
  records `decision_bearing_validation=PASS` and `deviations=[]`.

## Atomic admission and proposal boundary

The independent reviewer supplies exactly one JSON record with the v2 schema
required by `validate_score_aware_raw_reweighting_admission.py`.  The combined
admission command must validate the record, trusted runner and complete compact
directory in one process and emit the literal `reviewed_mode` together with the
record, publication commit and four verified artifact identities.  The proposal
must consume that emitted payload atomically; rereading an unbound review file
is not equivalent.

Only an emitted `admission=GO` for a literal executor-visible mode authorizes
one full `server_cpu`, `summary_only` proposal with the sole frozen source
parameter.  Scientific success still requires `overall_eligible=true`.  A
negative result closes this exact score-aware probability-mass mechanism without
retuning; a null-action result is reported under the interpretation frozen in
the contract.
