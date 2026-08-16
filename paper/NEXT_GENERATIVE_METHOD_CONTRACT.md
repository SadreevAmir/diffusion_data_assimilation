# Frozen next generative-method contract: guidance-mixture ensemble

## Scientific role and falsifiable hypothesis

The completed postprocessors show a repeated tradeoff: anomaly inflation can
improve fair CRPS but damages bounded spatial fields, whereas topology-preserving
transport cannot expose useful new dispersion. Purged analog residuals improve
finite-ensemble ranks and coverage but transfer large mean, boundary and spatial
errors. The next mechanism therefore creates diversity inside the conditional
generator, before any clipping or member reconstruction.

The candidate is a ten-member deterministic mixture over the five already
evaluated independent-CFG track weights. It tests whether guidance uncertainty,
rather than a post-hoc marginal transform, supplies useful bounded joint
scenarios. A pass requires the unchanged no-compensation gate. Failure rejects
this fixed guidance-mixture mechanism and must not trigger weight, allocation,
seed or clipping adjustment on these dates.

## Frozen construction

- Use the same learned checkpoint, conditioning tensors, ordered 40 cases and
  physical output transform as the existing independent-CFG validation runs.
- Set `cfg_mode=independent`. Use exactly two members at each track/background
  scale pair: `(0.5,0.5)`, `(0.625,0.375)`, `(0.75,0.25)`,
  `(0.875,0.125)`, and `(1.0,0.0)`. Member order is the listed scale order,
  with the two seeds in ascending order within each pair.
- Use the first two already frozen member seeds from the original validation
  sampling contract for every scale pair. Reuse the same two initial-noise
  tensors across all five scale pairs (common random numbers); do not search or
  regenerate seeds after scoring.
- Keep `start_mode`, `sample_start_noise_level`, `sample_target`, solver,
  timesteps, observation enforcement, normalization and masks identical to the
  source independent-CFG runs. Do not clip, recenter, rank-shuffle, rescale or
  postprocess members beyond the established physical output transform.
- The raw comparison remains the ten-member learned-joint ensemble from
  `joint_full_condition_validation_2022`. The candidate is not described as a
  calibrated distribution unless it passes the full gate.

## Frozen decision and reporting

Evaluate all 40 cases with the existing proper-score, finite-ensemble
reliability, boundary, spatial/physical and operational families. Success is
only `gate.overall_eligible=true`, including at least 3% lower
`analysis_fair_crps`, paired date CI excluding zero, ordinary CRPS no worse than
1%, and every other mandatory family passing without compensation. Report the
fixed four-case-block interval as temporal sensitivity, not as a predeclared
gate.

Operational validity additionally requires exactly two finite members from
each scale pair for every case, exact seed/noise pairing across the five pairs,
and no fallback member substitution. Report per-scale member counts and hashes
of the ordered seed identifiers. A proper-score failure means guidance mixture
does not provide useful conditional diversity; a reliability failure means it
does not repair finite-ensemble exchangeability; a boundary or spatial failure
means heterogeneous guidance changes physical regimes or joint geometry
unsafely. Any such failure closes the fixed mechanism.

## Execution and artifact contract

This contract is frozen before implementation. A reviewed GPU runner must have
only `source_experiment=joint_full_condition_validation_2022`; the scale pairs,
allocation, seeds, solver and full gate are constants, not runtime parameters.
All sampling and analysis run server-side. Retrieval is `summary_only`; the
compact return contains the complete gate, raw/candidate aggregates, paired
date and four-case-block uncertainty, per-scale member counts, seed-identifier
hashes and operational counts. No raw member or truth field is requested.

No trusted mode currently implements this contract. This is an autonomous
engineering task for the next controller cycle, not an external dependency and
not permission to substitute an existing mode or launch an experiment.
