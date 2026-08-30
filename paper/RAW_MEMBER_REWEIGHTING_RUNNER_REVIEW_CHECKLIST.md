# Raw-member reweighting trusted-runner review checklist

Status: LOCAL_ADAPTER_READY_NO_TRUSTED_MODE

This outcome-agnostic checklist binds a future trusted implementation to
`NEXT_RAW_MEMBER_REWEIGHTING_CONTRACT.md`. It does not authorize a mode name or
an experiment. A proposal is admissible only after every item is independently
recorded as `PASS` in one controller-visible review with no deviations.

## Interface and provenance

- [ ] The mode accepts exactly
  `source_experiment=joint_full_condition_validation_2022` and rejects every
  other parameter.
- [ ] The runner accepts only the frozen 40-case, ten-member development
  envelope and `summary_only`; raw members, truth and candidate fields remain
  server-side.
- [ ] The result binds the reviewed publication commit, runner digest, frozen
  contract digest, source identity and ordered case identities before scoring.

## Leakage-safe construction parity

- [ ] Five contiguous eight-case holdouts and the non-circular three-case purge
  match the established ordered split for every held-out case.
- [ ] The six forecast-only features, training population standardization and
  ten-nearest-analog selection use retained training cases only. Distance ties
  prefer the earlier ordered case index.
- [ ] Training truth ranks use only retained dates and the existing sealed tie
  uniforms. The ten bin counts, fixed `+0.5` smoothing and denominator `15`
  match `raw_member_reweighting_reference.py` exactly.
- [ ] Systematic inverse-CDF positions are exactly `(j+0.5)/10`; held-out raw
  members are sorted by weighted case-mean concentration with member-index tie
  breaking, and the oracle's selected source indices match exactly.

## Identity and operational invariants

- [ ] Every output member is bitwise identical to its recorded held-out raw
  source member. No clipping, projection, interpolation or perturbation occurs.
- [ ] Exact-zero, positive-ice, established-ice, exact-one and `>=0.999` masks
  equal the recorded source masks for every output member.
- [ ] Each case reports ten source indices, ten multiplicities summing to ten,
  unique-member count and effective sample size; all recompute exactly from the
  source-index vector.
- [ ] Non-finite features, zero/non-finite training scale, invalid probability
  sum, missing rank position, source-copy mismatch or output size other than ten
  is an operational failure.

## Compact evidence and no-compensation gate

- [ ] Compact output contains the complete gate, raw/candidate aggregates,
  paired date and fixed four-case-block uncertainty, fold/analog provenance,
  rank-bin probabilities, source multiplicities, identity/mask invariants and
  operational counts, but no raw-member, candidate or truth fields.
- [ ] Candidate-minus-raw deltas reconcile to ordered case records. The primary
  prediction is at least 3% improvement in `analysis_fair_crps` with the paired
  date interval wholly improving; ordinary `analysis_crps` may worsen by at
  most 1%.
- [ ] Absolute rank uniformity passes, every attainable coverage diagnostic
  moves toward target, and none worsens by more than `0.02`.
- [ ] Proper-score, finite-ensemble reliability, boundary, spatial/physical and
  operational flags are each conjunctions of non-empty criterion maps;
  `overall_eligible` is their literal conjunction.

## Independent review evidence

- [ ] The unchanged oracle and focused tests pass at the reviewed publication
  commit.
- [ ] Trusted tests include positive synthetic parity and negative fixtures for
  envelope drift, held-out-truth leakage, fold/purge/tie drift, probability or
  inverse-CDF drift, incomplete output, non-finite input, source-copy/mask
  mismatch, aggregate/paired mismatch and false family eligibility.
- [ ] A synthetic server dry run emits no project-data metric and matches the
  frozen construction and compact schemas.
- [ ] The reviewer records the literal implemented mode, exact test command and
  sentinel, publication commit, runner digest, contract digest and synthetic
  compact digest.

## Controller-visible admission record

Every placeholder, null, missing field, non-`PASS` validation or non-empty
`deviations` list is `NO_GO`.

```json
{
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

`GO` authorizes one full `server_cpu`, `summary_only` proposal using only the
literal reviewed mode and sole frozen source parameter. Scientific success
still requires `overall_eligible=true`; a negative result closes this exact
probability-mass mechanism without retuning.

The local fail-closed handoff is executable as
`python -m paper.validate_raw_member_reweighting_review REVIEW_JSON RUNNER_PY SYNTHETIC_RESULT`.
It requires the exact record schema above, binds the runner, contract and
synthetic-result bytes, requires the parsed synthetic result to equal the
runner's deterministic outcome-agnostic dry-run record, checks the runner's
pure construction surface against `raw_member_reweighting_reference.py`,
detects substitution of the review record, runner, contract or synthetic result
during the combined admission operation, and emits `admission=GO` only with the
reviewed literal mode. It snapshots and rechecks those same artifact identities
around payload construction, so a post-validation substitution cannot be paired
with stale verified digests. This check does not implement, name or launch a trusted
mode; controller visibility and the independent review remain mandatory.

Admission semantic parity also checks the exact five contiguous holdouts,
non-circular three-case purge, training-population feature scaling, ten unique
analog selections and deterministic case-index ordering for an exact distance
tie. A reviewed runner with drift in any of these upstream construction steps
cannot reach `admission=GO`, even if its downstream rank-selection helpers still
match the oracle.

The successful JSON payload is self-contained: alongside `admission=GO` and
the literal `reviewed_mode`, it carries the SHA-256 of the exact review record,
the reviewed publication commit and the verified runner, contract and synthetic
result SHA-256 identities. A downstream proposal must consume those emitted
bindings atomically; rereading an unbound review file is not equivalent.

The publication worktree now also contains the outcome-agnostic executor
boundary `raw_member_reweighting_server_adapter.py`. It accepts the sole frozen
parameter and exact CPU/validation/envelope/summary contract, registers at most
one immutable literal mode from a self-contained `admission=GO` payload, and
permits construction only after resolving that mode from the same inventory.
Direct admission dictionaries and unregistered modes cannot reach construction.
It deliberately contains no literal
production mode, filesystem access, sealed-data loader or controller mutation.
Thus its focused tests establish adapter fail-closed behavior, but do not satisfy
the unchecked independent-review items above and do not make a proposal admissible.
