# Frozen contingent method contract: purged analog rank reweighting of raw members

Status: ACTIVATED_DESIGN_FROZEN_NO_RUNNER

## Activation and claim role

This development-only mechanism test is activated only if the frozen
whole-field rank-coherent anomaly-transport candidate is completed and rejected
by the unchanged no-compensation gate. It must not delay or replace that first
test. The candidate asks a different question: is the learned-joint ensemble's
reliability defect mainly a probability-mass error among already plausible raw
scenarios, rather than a need to synthesize or geometrically alter scenarios?

The activation condition is now met. The completed coherent anomaly-transport
routes and the subsequent frozen joint spread--occurrence construction all
report `overall_eligible=false`. The latter improves relative rank discrepancy
but still fails absolute rank adequacy and creates boundary and spatial losses.
This activates the exact contract below without changing any feature, analog,
smoothing, sampling or gate choice. It does not authorize an invented mode: a
literal reviewed server mode and independent parity remain required before a
proposal can be decision-bearing.

The raw learned-joint ensemble is the primary baseline. The completed coherent
offset, analog-residual and whole-field anomaly-transport results are mechanism
controls. The candidate may support a calibrated-ensemble claim only when the
full out-of-fold gate reports `overall_eligible=true`.

## Frozen leakage-safe construction

- Use exactly the established 40 ordered development cases, five contiguous
  eight-case holdouts and the non-circular three-case purge.
- For every case compute the same six forecast-only state features frozen for
  the analog-residual and whole-field anomaly-transport contracts: weighted raw
  ensemble-mean ice area, extent at `0.15`, spatial mean, spatial standard
  deviation, and raw-mean semivariograms at lags 1 and 4.
- Standardize with retained training cases only. A zero or non-finite training
  population standard deviation is an operational failure.
- For each held-out case select exactly the ten nearest retained training cases
  by squared Euclidean distance. Distance ties use earlier ordered case index.
- On each selected training case, compute the normalized randomized truth rank
  of the spatially weighted case-mean concentration among its ten raw member
  case means under the existing sealed tie uniforms. Map it to one of the ten
  attainable member-rank bins with `min(9, floor(10 * rank))`.
- Let `c[r]` be the count of selected analog dates assigned to bin `r`. Define
  fixed smoothed probabilities `p[r] = (c[r] + 0.5) / 15`. No bandwidth,
  smoothing constant, feature subset or neighbor count is selectable.
- Sort the held-out raw members by spatially weighted case-mean concentration;
  ties use member index. Produce ten deterministic rank positions by systematic
  inverse-CDF sampling from `p` at `u_j=(j+0.5)/10`, `j=0,...,9`. Copy the
  corresponding held-out raw member field at each position. Repeated raw members
  are allowed; no field value is changed, clipped, projected, interpolated or
  perturbed.

This construction changes only empirical probability mass across complete raw
scenarios. For every output member, record its source raw-member index. Verify
bitwise equality with that source field and exact preservation of its zero,
positive-ice, established-ice, exact-one and `>=0.999` masks. Record the number
of unique selected raw members, all ten source multiplicities and the effective
sample size `1 / sum_r (n_r/10)^2`.

## Falsifiable prediction and stop/go decision

The predeclared prediction is that analog-conditioned rank mass, without new
field geometry, improves case-mean `analysis_fair_crps` by at least 3% relative
to raw, makes the paired date interval exclude zero in the improving direction,
passes absolute randomized-rank uniformity, and moves every attainable-order
coverage diagnostic closer to its target with none worsening by more than
`0.02`. Every unchanged boundary, spatial/physical and operational criterion
must also pass; ordinary `analysis_crps` may worsen by at most 1%.

Apply the full unchanged no-compensation gate out of fold on all 40 cases and
report the fixed non-overlapping four-case-block interval as temporal
sensitivity only. Selection is positive only when `overall_eligible=true` and
all mandatory families pass.

- Rank or coverage failure rejects probability-mass misallocation among raw
  scenarios as an adequate reliability explanation.
- Proper-score failure means reweighting plausible scenarios is not useful even
  if the rank histogram becomes flatter.
- Boundary or member-spatial failure is an implementation/provenance failure,
  because every candidate field must be an exact raw-field copy. Mean-field
  spatial or physical failure instead shows that changed scenario frequencies
  damage the forecast distribution and rejects the method scientifically.
- Any held-out truth use, fold or tie mismatch, non-finite feature, source-copy
  mismatch, probability sum error above `1e-12`, missing rank position, or
  output other than ten members is an operational failure.

No probability smoothing, analog count, feature, distance, fold, purge, tie
uniform, rank functional, systematic-sampling position, threshold or gate may
change after any candidate summary is observed. A negative result closes this
exact mechanism and does not authorize another independent evaluation.

## Execution and compact evidence contract

All construction and analysis must run on `server_cpu` from
`source_experiment=joint_full_condition_validation_2022`, using the exact
40-case, ten-member envelope and `summary_only`. The future trusted runner must
accept only `source_experiment` and return the full gate, aggregate raw/candidate
metrics, paired uncertainty, fold membership, analog indices, per-case bin
counts and probabilities, source multiplicities, unique-member counts,
effective sample sizes, exact-copy/mask invariants and operational counts. No
raw member, truth or candidate field is retrieved.

There is intentionally no experiment proposal or invented mode identifier.
This contingent contract is frozen before the first rank-coherent result; a
reviewed trusted mode and parity checks remain ordinary autonomous engineering
work if and only if the activation condition is met.

The dependency-free oracle `raw_member_reweighting_reference.py` makes the
ten-bin smoothing, endpoint handling, midpoint inverse-CDF selection, the final
rank-position-to-raw-member mapping with member-index tie breaking,
multiplicity accounting and effective-sample-size calculation executable
without reading project data. Its focused tests prove the identity result for
one analog rank in every bin, freeze the exact repeated-member selection for a
maximally concentrated analog library, verify a permuted held-out ordering with
equal case means, and reject incomplete, non-finite or out-of-range rank and
member-mean inputs. It is a construction oracle, not an experiment entry point
or decision-bearing runner.
