# E2 trusted-executor mapping review

Status: PREDECLARED_DEPENDENT_MAPPING_NO_IMPLEMENTED_MODE

This is an outcome-agnostic review surface for mapping the frozen scientific
contract in `NEXT_OCCURRENCE_INTENSITY_E2_CONTRACT.md` to a future trusted
executor. It does not implement a runner, invent a mode, authorize a launch or
relax the dependency on a literal clean E1 admission response.

## Admission dependency

- [ ] The independently returned E1 response has exactly the accepted shape
  `{"decision":"PASS","failed_checks":[],"failed_tests":[]}`.
- [ ] `validate_occurrence_intensity_e1_admission.py` accepts that response
  against the unchanged request and reports `launch_authorized=false`.
- [ ] The reviewed E1 implementation, tests, config and runner identities are
  the same identities named by `OCCURRENCE_INTENSITY_E1_ADMISSION_REQUEST.json`.
- [ ] No E1 scientific outcome is inferred from the engineering sentinel.

Any failed or missing item above is `NO_GO`. A literal E1 `REJECT` routes only
to repair of the named invariant and repetition of the unchanged admission.

## Interface and immutable identities

- [ ] The controller exposes one literal reviewed E2 mode; the proposal copies
  that identifier rather than constructing a new name.
- [ ] The mode accepts exactly one upstream source identity and rejects every
  unknown runtime field.
- [ ] The runner binds the reviewed publication commit, E2 contract digest,
  runner digest, ordered training inventory, ordered held-out case identities
  and the accepted E1 identity before computation.
- [ ] E1 and E2 share preprocessing, optimizer, update budget, checkpoint rule,
  initialization seeds and the ten-member sampling schedule.
- [ ] Retrieval is `summary_only`; raw members, observations, truths and
  trajectory fields remain inside the server result root.

## Trajectory and likelihood mapping

- [ ] The generated state contains exactly `b_t`, `b_{t-1}` and `b_{t-2}` in
  that order; E1 lag-specific exogenous backgrounds remain conditioning inputs
  and are not misrepresented as a generated trajectory.
- [ ] Every innovation is computed as `y_{t-k} - b_{t-k}` under its matching
  finite mask; broadcasting `b_t` across lags fails closed.
- [ ] The reviewed adapter is parity-tested against
  `assim_lib.occurrence_intensity_e2.apply_operators_at_own_time`, including
  distinct operators at each lag and NaN-safe masked observations.
- [ ] Each lag preserves the accepted E1 eight-field ordering, physical
  occurrence atom, bounded conditional intensity, geometry/value provenance,
  masks and both orientation landmarks bit-for-bit.
- [ ] Masked values are exactly zero and masked leakage is at most `1e-7`.
- [ ] Real tracks are the only primary training tracks; synthetic or
  truth-derived tracks, hard clipping, epsilon clipping and outcome-time
  selection are rejected.
- [ ] Lag count, channels, loss weights, seeds, budget, thresholds, checkpoint
  and case subset are immutable runner constants, not proposal parameters.

## Paired compact evidence

- [ ] The compact result records exact E1/E2 identities, inventory and code
  digests, ordered case count, all invariant decisions and literal gate result.
- [ ] Casewise observed-footprint innovation RMSE is reported separately for
  all three lags and its primary aggregate gives equal weight to cases.
- [ ] Casewise fair CRPS and off-track anomaly energy use the frozen E1
  definitions; off-track uses the complement of the same two-pixel dilation.
- [ ] Aggregate deltas reconcile exactly with their casewise vectors and use
  E2 minus E1 orientation throughout.
- [ ] Ordinary CRPS, absolute randomized-rank adequacy, attainable coverage,
  truth-relative boundary events and unchanged spatial/physical diagnostics
  are present as diagnostics and cannot retune the method.
- [ ] Missing vectors, non-finite values, identity drift or lag mismatch yield
  `E2_INVALID`, never a scientific negative.

## Literal conjunctive decision

- [ ] `TEMPORAL_MECHANISM_USEFUL` requires every invariant plus innovation-RMSE
  reduction of at least 10%, fair-CRPS degradation of at most 2%, off-track
  anomaly-energy increase of at most 5%, and all operational checks.
- [ ] Any finite threshold failure yields `TEMPORAL_MECHANISM_NEGATIVE`; no rank,
  coverage or secondary gain compensates for it.
- [ ] Routing is literal: useful E2 becomes the temporal base; negative E2
  retains E1; invalid E2 repairs only the named implementation/evidence defect.

## Required controller-visible review record

The mapping is `GO` only when an independent record contains every field below
with no nulls, placeholders, omissions or deviations. Digests are lowercase
SHA-256 values and `publication_commit` is the exact reviewed 40-hex commit.

```json
{
  "e1_admission_decision": "PASS",
  "reviewed_mode": "<literal implemented trusted mode>",
  "publication_commit": "<40-hex reviewed commit>",
  "runner_sha256": "<64-hex digest>",
  "contract_sha256": "<64-hex digest>",
  "synthetic_result_sha256": "<64-hex digest>",
  "test_command": "<exact command>",
  "test_sentinel": "<exact success sentinel>",
  "decision_bearing_validation": "PASS",
  "deviations": []
}
```

Every unchecked, failed, waived or not-applicable item is `NO_GO`. Even `GO`
only admits one frozen proposal under the separately exposed mode; scientific
success still requires the complete conjunctive E2 decision above.
