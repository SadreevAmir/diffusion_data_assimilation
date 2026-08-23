# Rank-coherent trusted-runner review checklist

Status: REVIEW_CONTRACT_READY_NO_IMPLEMENTED_MODE

This checklist is an outcome-agnostic admission test for a future trusted
server implementation of `NEXT_RANK_COHERENT_CONTRACT.md`. It is not permission
to invent a mode name, launch a job, or alter the frozen method. A proposal is
admissible only after an independent reviewer records every item below as
`PASS`, records the reviewed publication commit and runner digest, and the
controller exposes that exact reviewed mode.

## 1. Interface and provenance

- [ ] The mode accepts exactly one parameter,
  `source_experiment=joint_full_condition_validation_2022`, and rejects every
  unknown parameter.
- [ ] The runner accepts only the exact `valid`, 40-case, ten-member,
  stride-five development envelope and `summary_only` retrieval.
- [ ] Inputs remain inside the server result root; no raw member, truth or
  anomaly field is copied into compact artifacts.
- [ ] The result records publication commit, runner digest, source identity,
  ordered case identities and the frozen contract digest before computation.

## 2. Leakage and deterministic selection

- [ ] Five ordered contiguous eight-case holdouts and a non-circular three-case
  purge exactly match `rank_coherent_reference.py` for every held-out index.
- [ ] All six forecast-only features, their standardization and every
  feasibility decision use retained training cases only; zero or non-finite
  population standard deviation fails the fold.
- [ ] Exactly ten complete training dates are selected by squared standardized
  feature distance; deterministic distance ties prefer the earlier ordered
  case index.
- [ ] Normalized truth ranks are computed only for retained training dates with
  the existing randomized-tie seed contract. Date, member and order-statistic
  ties match the frozen rules byte-for-byte.

## 3. Candidate construction

- [ ] Borrowed objects are complete member-anomaly fields. The implementation
  performs no clipping, rescaling, localization, rotation, pixel shuffle or
  per-pixel recentering before the bounded projection.
- [ ] Each of the ten fixed rank targets is used exactly once per held-out case;
  compact counts are therefore exactly 40 for every target.
- [ ] One common `alpha` is selected per fold from exactly
  `{0.0,0.5,0.75,1.0,1.25}` by retained-training fair CRPS subject to all
  frozen boundary and member-semivariogram feasibility checks; ties choose the
  smaller value.
- [ ] `no_positive_feasible_alpha` is true if and only if `selected_alpha` is
  `0.0`; there is no fallback scale or post-result tuning path.
- [ ] The exact capped-simplex projection preserves the raw pixelwise ensemble
  mean within `1e-10`, enforces `[0,1]`, and fails closed on incompatible shape
  or any non-finite input or output.

## 4. Compact evidence and scientific gate

- [ ] The compact payload contains only the complete gate, aggregate
  raw/candidate metrics, paired uncertainty, five fold selections, rank-target
  balance, projection diagnostics, member-spatial deltas and operational
  counts required by the frozen contract.
- [ ] Aggregate `analysis_fair_crps`, `analysis_crps` and
  `analysis_mean_rmse` deltas equal candidate minus raw; paired date and fixed
  non-overlapping four-case-block records reconcile to those same means and
  have finite ordered interval endpoints.
- [ ] Projection diagnostics have the exact schema enforced by the oracle and
  report pre/post member-semivariogram distortion, cap masses, changed-member
  fraction and maximum mean error.
- [ ] Every member-spatial decision follows its non-negative frozen tolerance,
  and the spatial family criteria map one-to-one to those records.
- [ ] Each mandatory family flag is the conjunction of a non-empty boolean
  criterion map; `overall_eligible` is exactly the conjunction of all five
  mandatory family flags. No family can compensate for another.

## 5. Required review evidence

- [ ] `python3 paper/rank_coherent_reference.py` passes unchanged at the
  reviewed publication commit.
- [ ] Runner unit tests include positive synthetic parity with the oracle plus
  negative fixtures for envelope drift, leakage, tie drift, incomplete folds,
  ragged/non-finite fields, invalid alpha, projection invariant failure,
  aggregate/paired mismatch, false spatial pass and inconsistent family gate.
- [ ] A server dry run on synthetic fixtures produces no project-data metrics
  and matches the local oracle's candidate, fold, diagnostic and gate schemas.
- [ ] An independent reviewer records no deviations, the exact reviewed mode,
  publication commit, runner digest, test command and test sentinel in the
  controller-visible implementation review.

## Controller-visible admission record

The implementation review is complete only when one controller-visible record
contains every field below. String placeholders, nulls, omitted fields and a
non-empty `deviations` list are `NO_GO`. Digests are lowercase SHA-256 values;
`publication_commit` is the exact reviewed 40-hex commit; and
`decision_bearing_validation` must be the literal success record from
`validate_result_directory(path, decision_bearing=True)`.

```json
{
  "reviewed_mode": "<implemented trusted mode>",
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

The proposal must copy `reviewed_mode` literally and may use only the sole
frozen `source_experiment` parameter. The publication commit, runner and
contract digests are evidence for admission, never experiment parameters.

## Admission decision

Any unchecked, failed, waived or not-applicable item is a hard `NO_GO` for an
experiment proposal. `GO` means only that one full `server_cpu`, `summary_only`
development proposal may use the separately reviewed implemented mode with the
sole frozen source parameter. Scientific success still requires
`overall_eligible=true`; a negative result closes this exact mechanism.
