# Coverage/occurrence calibration result reconciliation

Status: PRE_RESULT_COMPACT_PAYLOAD_PENDING

Controller-visible inspection on 2026-08-29 found no local compact JSON whose
filename or path contains either exact experiment identity. The controller
reports both jobs as `summary_ready`, but that scheduler fact is deliberately
not admitted as a family decision. Consequently no gate Boolean, effect size or
scientific outcome has been copied to a publication surface.

Exact blocker: the publication worktree lacks the two immutable compact gate
records consumed by `validate_coverage_occurrence_admission.py`. The minimum
recovery is evidence delivery, not recomputation: materialize the existing
`gate_decision.json` bytes from each completed server result root into the
controller-visible compact-artifact channel. Do not rerun either forty-case
calculation. Once both files are visible, pass their actual paths together to
the command below; the validator will bind their bytes by SHA-256 and fail
closed on any schema, identity or conjunction defect.

This handoff covers the completed trusted jobs
`joint_coverage_occurrence_threshold_valid` and
`joint_coverage_occurrence_joint_rank_valid`. Their completion status is not a
scientific result. Reconciliation requires both controller-visible compact
payloads in the same audit; neither result may be interpreted alone while the
other payload is missing.

## Exact identities

The first payload must identify mechanism
`coverage_spread_occurrence_calibration`, variant
`purged_trace_ice_threshold`, 40 completed cases and ten members. The second
must identify mechanism `joint_coverage_occurrence_rank_calibration`, variant
`purged_joint_scale_threshold_minimax`, the same completed envelope and ten
members. Both must name source `joint_full_condition_validation_2022` and prove
training-fold-only selection under their frozen grids.

For each payload copy, without inference, these Boolean decisions:

- `proper_score`
- `reliability`
- `boundary`
- `spatial_physical`
- `operational`
- `overall_eligible`

Reject a payload with a missing or non-Boolean decision, or when
`overall_eligible` is not the literal conjunction of the five families. Never
infer a family decision from an aggregate metric or job completion status.

## Atomic decision procedure

Before changing any publication surface, run the fail-closed joint admission:

```bash
python paper/validate_coverage_occurrence_admission.py \
  --threshold /server/compact/threshold/gate_decision.json \
  --joint-rank /server/compact/joint_rank/gate_decision.json
```

The two paths are placeholders for controller-visible compact files, not local
result locations. The command emits `admission=GO` only after both exact
identities, source, 40-by-10 envelope, Boolean family decisions and the literal
family conjunction have passed. Its SHA-256 values bind the admitted bytes used
by the subsequent atomic publication update.

1. Admit both compact payloads and bind each to its exact experiment identity
   and immutable digest.
2. Verify counts, source, folds, purge, candidate identity, frozen selection
   grid and all operational invariants independently for each payload.
3. Copy the five family decisions and `overall_eligible` from each authoritative
   gate record into one comparison table.
4. If either candidate has all five families and `overall_eligible=true`, update
   `PAPER_DRAFT.md`, `CLAIM_LEDGER.md`, `REPRODUCIBILITY.md` and
   `PUBLICATION_READINESS.md` atomically with its exact effect sizes and bounded
   development-envelope claim.
5. If both candidates have `overall_eligible=false`, name every failed family,
   close both frozen occurrence mechanisms without tuning, and only then allow
   consideration of the separately frozen score-aware contract. That later
   mechanism still requires a literal reviewed mode; this handoff does not
   invent or authorize one.

Relative rank improvement, attainable-coverage improvement or proper-score gain
cannot compensate for any failed family. The threshold lists, scale grid,
folds, purge, objectives and gate thresholds remain frozen after either result.

## Required publication update

Successful admission changes five files in one worktree transaction:
`PAPER_DRAFT.md`, `CLAIM_LEDGER.md`, `REPRODUCIBILITY.md`,
`PUBLICATION_READINESS.md` and this handoff. Until then, retain the status above
and record no family outcome or effect size. The absence of the compact payloads
is an evidence gap, not evidence of failure and not authorization to repeat the
completed computations.
