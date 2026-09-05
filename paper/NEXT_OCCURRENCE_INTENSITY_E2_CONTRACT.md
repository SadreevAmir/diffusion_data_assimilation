# Frozen E2 contract: likelihood-consistent fixed-lag trajectory

Status: PREDECLARED_DEPENDENT_NO_EXECUTOR_MODE

## Purpose and dependency

E2 is a mechanism test of temporal assimilation, not a second engineering
sentinel and not an age-channel ablation renamed as a smoother.  It is eligible
for executor review only after E1 passes every frozen engineering invariant in
`PROVENANCE_AWARE_OCCURRENCE_INTENSITY_CONTRACT.md`.  An E1 rejection blocks E2
and routes directly to repair of the reported E1 invariant; it does not permit
changing this contract.

No controller-visible trusted-executor mode currently implements E2.  This file
freezes the scientific contract but neither authorizes a launch nor supplies a
mode identifier.

## Falsifiable hypothesis and paper role

Relative to accepted E1, which uses lag-matched *exogenous background* fields
to construct finite conditioning innovations but generates only the current
state, E2 generates an explicit three-time state trajectory and evaluates each
likelihood operator at its own generated time.  This adds temporally coherent
assimilation information rather than merely adding age channels.  The paper
claim is supported only if E2 improves held-out observed-footprint innovation RMSE by at least 10%
while preserving proper-score and off-track safety constraints.  A failure is
publishable evidence that age metadata is not an adequate proxy for a
likelihood-consistent smoother and that this fixed-lag construction adds no
defensible temporal benefit.

## Immutable scientific intervention

- The generated state contains exactly the current state and the two preceding
  daily states, ordered current, one-day old, two-days old.  This is distinct
  from E1's lag-specific exogenous background fields used in conditioning.
- Each observation innovation is evaluated only against its matching
  background frame: `y_{t-k} - b_{t-k}` under the corresponding finite mask.
  Broadcasting the current background across lags is a hard failure.
- The eight provenance-aware fields per lag, physical occurrence atom
  `A=1{SIC>0}`, bounded conditional intensity, channel ordering, masks and
  orientation landmarks are bit-for-bit the accepted E1 definitions.
- The publication-side correctness oracle is
  `assim_lib.occurrence_intensity_e2.apply_operators_at_own_time`; it accepts an
  explicit `[B,3,C,H,W]` generated trajectory and fails closed on a broadcast
  current-state surrogate, operator-count drift, invalid shapes or non-finite
  observed values.  It is not a runner and does not authorize computation.
- E1 and E2 use the same frozen training inventory, held-out case identities,
  preprocessing, optimizer, update budget, checkpoint rule, initialisation
  seeds and ten-member sampling schedule.  The only scientific difference is
  generation of the explicit lagged state trajectory and evaluation of the
  matching observation operator at each generated time; E1 conditioning stays
  unchanged.
- Primary training uses real tracks only.  Synthetic tracks, truth-derived
  tracks, hard output clipping, epsilon clipping and outcome-time model or
  checkpoint selection are prohibited.
- There is no runtime choice of lag count, channel set, loss weight, seed,
  budget, threshold, checkpoint or case subset.  Any later trusted mode may
  expose only an upstream source identity; all scientific choices remain sealed
  in its reviewed runner.

## Evaluation and compact evidence

Evaluation is server-side and uses the same frozen held-out cases for E1 and E2.
All comparisons are paired by case.  The decision-bearing compact summary must
contain the exact E1 and E2 identities, inventory and code digests, case count,
all invariant checks, casewise metric vectors, aggregate differences and the
literal gate decision.  Raw members remain server-side; retrieval is
`summary_only` unless a later reviewed publication renderer names specific
compact panels.

The primary metric is observed-footprint innovation RMSE, computed separately
for each lag and then averaged with equal case weight.  The safety metrics are
case-mean fair CRPS and off-track anomaly energy, where off-track uses the
complement of the same frozen two-pixel dilation used by E1.  The summary also
reports ordinary CRPS, absolute randomized-rank adequacy, attainable coverage,
truth-relative boundary events and the unchanged spatial/physical diagnostics;
none may be used to retune E2.

Before any controller admission, the result interface is frozen by
`validate_occurrence_intensity_e2_compact_result.py`.  It requires exact
identities and SHA-256 digests, the ordered case inventory, exact lag order,
complete finite paired vectors, reconciled aggregates, all named invariants and
operational checks, and non-empty secondary diagnostic records.  It recomputes
the literal decision and rejects extra or missing fields.  This validator is a
publication-side schema oracle only: it neither implements E2 nor creates a
trusted mode.

## Stop/go decision

E2 is `TEMPORAL_MECHANISM_USEFUL` if and only if all conditions hold:

1. every E1 finiteness, range, exact-mask, provenance and orientation invariant
   remains exact, including masked leakage at most `1e-7` and zero hard clips;
2. the equally case-weighted observed-footprint innovation RMSE is at least 10%
   lower than E1;
3. case-mean fair CRPS is no more than 2% worse than E1;
4. off-track anomaly energy is no more than 5% higher than E1; and
5. the unchanged no-compensation operational checks pass.

Any failed condition yields `TEMPORAL_MECHANISM_NEGATIVE`.  Non-finite output,
identity mismatch, lag mismatch, missing casewise vectors or an invariant
failure yields `E2_INVALID`, not a scientific negative.  The decision is
conjunctive: no improvement in rank, coverage or another secondary diagnostic
can compensate for a failed condition.

## Frozen routing after the result

- `TEMPORAL_MECHANISM_USEFUL`: retain E2 as the temporal base and evaluate the
  already predeclared atom-aware E3 mechanism under a separately reviewed
  contract.
- `TEMPORAL_MECHANISM_NEGATIVE`: retain accepted E1 as the base, close the
  fixed-lag trajectory hypothesis without tuning, and evaluate E3 independently.
- `E2_INVALID`: repair only the named implementation or evidence defect and
  repeat the identical contract; do not change scientific choices.

E4 remains dependent on the best independently accepted result among E1--E3
under the unchanged no-compensation gate.  This routing is fixed before any E2
scientific summary exists.
