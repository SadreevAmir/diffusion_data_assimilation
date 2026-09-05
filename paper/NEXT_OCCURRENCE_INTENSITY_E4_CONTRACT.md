# Frozen E4 contract: structured analog uncertainty source

Status: EXECUTABLE_PUBLICATION_ORACLE_NO_EXECUTOR_MODE

## Publication-side executable handoff

`assim_lib/occurrence_intensity_e4.py` and the immutable
`config/experiments/occurrence_intensity_e4_sentinel.json` now implement the
eight-case engineering oracle. `scripts/run_occurrence_intensity_e4_sentinel.sh`
exercises the six forecast-only features, training-only standardization,
chronological tie resolution and complete-field addition. It emits only
`run_status.json` and `artifact_manifest.json`; it never authorizes a launch or
claims a scientific gate result. `test/test_occurrence_intensity_e4.py` freezes
the selection and no-clipping semantics for independent repetition.

## Purpose and admissible base

E4 tests whether complete historical anomaly fields add coherent uncertainty
that marginal occurrence/intensity calibration cannot provide. It may enter
trusted-executor review only after E1 has passed every engineering invariant.
Its base is selected by the already frozen conjunctive gate: accepted E2, then
accepted E3 if useful; otherwise the last valid accepted predecessor. An invalid
predecessor must be repaired before routing and cannot count as a negative result.

No controller-visible trusted-executor mode currently implements this contract.
This document freezes the science; it does not authorize execution or invent a
mode identifier.

## Falsifiable hypothesis and paper role

Training-fold-only retrieval of complete residual fields should increase
attainable inner coverage by at least `0.05` and reduce absolute randomized-rank
total variation by at least `10%` relative to the accepted base while preserving
memberwise spatial structure. E4 is the paper's coherent-uncertainty-source
mechanism test, not a post-hoc ensemble repair.

## Immutable intervention

- Use the same contiguous purged folds, held-out cases, preprocessing,
  checkpoint, conditioning, ten-member schedule and seeds as the accepted base.
- Compute only six forecast-only state features: domain mean, domain standard
  deviation, ice extent, established-ice area, meridional centroid and zonal
  centroid. Standardize each feature on the training fold only.
- For each held-out forecast and member, select the ten nearest complete
  training residual fields by Euclidean feature distance. Resolve equal
  distances by chronological training index. Neither held-out truth nor any
  outcome metric may enter retrieval or ordering.
- Preserve each selected residual as one complete field and add it to the
  corresponding accepted-base member. Apply the base's frozen physical inverse
  exactly once. Pixelwise residual resampling, spatial smoothing, clipping,
  scale fitting and outcome-time selection are prohibited.
- Neighbor count, features, distance, folds, purge, thresholds, seeds, member
  assignment, base routing and case subset are not runtime parameters.

## Evaluation and compact evidence

Evaluation is server-side on the identical held-out cases. Retrieval is
`summary_only`. The compact result must bind code, input, base and fold digests;
analog identities and distances; casewise metric vectors; paired aggregate
differences; boundary masses; spatial diagnostics; and the literal decision.
Raw members and residual fields remain server-side.

Primary metrics are absolute date-balanced randomized-rank total variation and
attainable inner coverage. Safety families are case-mean fair and ordinary
CRPS; truth-relative exact-zero, occurrence, established-ice, exact-one and
near-one errors; memberwise variogram error; IIEE; edge disagreement; area and
extent error; off-track anomaly energy; and operational integrity.

## Conjunctive stop/go decision

E4 is `STRUCTURED_ANALOG_USEFUL` if and only if:

1. all E1 and base provenance, finiteness, range, mask and orientation
   invariants pass, with no held-out information in retrieval;
2. attainable inner coverage improves by at least `0.05`;
3. absolute randomized-rank total variation decreases by at least `10%`;
4. fair CRPS worsens by no more than `2%`, and ordinary CRPS worsens by no more
   than `1%`;
5. every truth-relative boundary-event absolute error worsens by no more than
   `1%`; and memberwise variogram error, IIEE, edge disagreement, area/extent
   error and off-track anomaly energy each worsen by no more than `5%`; and
6. every unchanged operational check passes.

A finite contract-valid result missing any scientific threshold is
`STRUCTURED_ANALOG_NEGATIVE`; do not tune neighbors, features, distances or
scales. Missing identities/vectors, leakage, a changed case inventory,
non-finite output or invariant failure is `E4_INVALID` and permits repair only
of the named defect. No proper-score improvement compensates for reliability,
boundary, spatial or operational failure.

## Frozen routing

- `STRUCTURED_ANALOG_USEFUL`: retain E4 as the candidate, then apply the full
  unchanged publication no-compensation gate and reconcile manuscript,
  claim-ledger, figures and reproducibility to that exact result.
- `STRUCTURED_ANALOG_NEGATIVE`: close this mechanism without tuning and retain
  the last accepted predecessor; the calibration mission remains open unless
  that predecessor already passes the full gate.
- `E4_INVALID`: repair only the named implementation or evidence defect and
  repeat this identical contract.
