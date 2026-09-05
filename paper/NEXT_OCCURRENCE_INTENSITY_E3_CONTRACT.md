# Frozen E3 contract: atom-aware conditional anamorphosis

Status: EXECUTABLE_PUBLICATION_SIDE_PENDING_INDEPENDENT_ADMISSION

## Purpose and dependency

E3 tests boundary calibration independently of the E2 temporal intervention.
It may enter executor review only after E1 passes every engineering invariant in
`PROVENANCE_AWARE_OCCURRENCE_INTENSITY_CONTRACT.md`.  E3 uses the accepted E1
model if E2 is negative and the accepted E2 model if E2 is useful.  An invalid
E2 does not authorize E3 until the named E2 defect is repaired.

No controller-visible trusted-executor mode currently implements E3.  The
publication-side correctness oracle is implemented by
`assim_lib/occurrence_intensity_e3.py`, its literal frozen configuration by
`config/experiments/occurrence_intensity_e3_sentinel.json`, and its local
entrypoint by `scripts/run_occurrence_intensity_e3_sentinel.sh`.  They exercise
the training-only mid-rank law, exact atom preservation, inverse support and
compact evidence, but neither authorize a launch nor supply a trusted mode.

## Falsifiable hypothesis and paper role

The occurrence atom and the bounded positive intensity have different sampling
laws.  Preserving exact boundary atoms while applying a reversible conditional
CDF transform only to `0 < SIC < 1` should reduce absolute rank-histogram total
variation by at least 15% and improve attainable inner coverage by at least
0.05, without degrading fair CRPS, truth-relative boundary events, or spatial
structure.  E3 is the paper's boundary-calibration mechanism test, not a generic
postprocessor comparison.

## Immutable scientific intervention

- The physical occurrence atom is exactly `A=1{SIC>0}`.  Exact zero remains an
  atom and is never replaced by an epsilon.  Exact one remains a separate upper
  atom.  Established ice remains the derived event `SIC>0.15`.
- Conditional on `0<SIC<1`, one generalized-inverse transform is fit on training
  data only, separately within each already frozen contiguous purged fold.  The
  empirical CDF uses mid-ranks with deterministic index order for ties; its
  piecewise-linear inverse uses the same ordered training values.
- Sampling first draws the frozen occurrence and upper-atom decisions, then
  maps the frozen interior latent quantiles through that training-fold inverse.
  Boundary decisions and member identity are unchanged by the interior map.
- The accepted temporal base, training inventory, held-out identities,
  preprocessing, optimizer, update budget, checkpoint rule, seeds and
  ten-member schedule are unchanged.  E3 changes only the conditional interior
  representation and its exact inverse.
- Hard clipping, epsilon clipping, outcome-time selection, interpolation across
  an atom, alternative plotting positions, and runtime choices of transform,
  folds, purge, thresholds, seeds, checkpoint or case subset are prohibited.
  A later reviewed runner may expose only an upstream source identity.

## Evaluation and compact evidence

Evaluation is server-side on the identical held-out cases used by its accepted
base.  The compact summary must bind the base and E3 identities, inventory and
code digests, case count, exact atom counts before and after the transform,
round-trip errors for every small-positive value, casewise metric vectors,
aggregate paired differences and the literal gate decision.  Retrieval is
`summary_only`; raw members remain server-side.

Primary reliability metrics are date-balanced randomized-rank total variation
from discrete uniformity and attainable inner coverage.  The unchanged safety
families are case-mean fair CRPS, truth-relative exact-zero, occurrence,
established-ice, exact-one and near-one errors, memberwise variogram error, IIEE,
edge disagreement, area/extent error, off-track anomaly energy and operational
integrity.  Secondary diagnostics cannot be used to retune the transform.

## Stop/go decision

E3 is `ATOM_AWARE_MECHANISM_USEFUL` if and only if all conditions hold:

1. every E1 finiteness, range, mask, provenance and orientation invariant is
   exact, every base zero and one atom is preserved memberwise, every
   `0<SIC<=0.15` round-trips with absolute error at most `1e-7`, and no clipping
   occurs;
2. absolute randomized-rank total variation is at least 15% lower than the
   accepted base;
3. attainable inner coverage is at least 0.05 higher than the accepted base;
4. case-mean fair CRPS is no more than 2% worse than the accepted base;
5. no truth-relative boundary-event absolute error is worse by more than 1%,
   and memberwise variogram error, IIEE, edge disagreement, area/extent error
   and off-track anomaly energy are each no more than 5% worse; and
6. every unchanged operational check passes.

Any finite, contract-valid result missing one or more scientific thresholds is
`ATOM_AWARE_MECHANISM_NEGATIVE`.  A missing vector or identity, changed atom,
lost small-positive value, clipping, non-finite output or invariant failure is
`E3_INVALID`.  The decision is conjunctive: proper-score improvement cannot
compensate for rank, coverage, boundary or spatial failure.

## Frozen routing after the result

- `ATOM_AWARE_MECHANISM_USEFUL`: retain E3 as the accepted base and evaluate the
  already predeclared structured analog E4 mechanism under a separate reviewed
  contract.
- `ATOM_AWARE_MECHANISM_NEGATIVE`: retain the accepted E1/E2 base, close this
  atom-aware transform without tuning, and evaluate E4 independently.
- `E3_INVALID`: repair only the named implementation or evidence defect and
  repeat the identical contract.

E4 selection remains based on the unchanged no-compensation gate, not on one
metric.  This routing is frozen before any E3 scientific summary exists.
