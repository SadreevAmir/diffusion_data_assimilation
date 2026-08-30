# Frozen next generative-method contract: clean-checkpoint deep ensemble

Status: FROZEN_BUT_NOT_SELECTABLE_ON_MEASURED_WALL_CLOCK

This contract is retained as a reproducible scientific design, not as an active
fallback.  The subsequently measured training throughput was approximately 40
seconds per batch over 3,255 batches per epoch.  Under the frozen forty-epoch,
three-seed construction, training would therefore take months rather than the
earlier planning estimate.  It must not be proposed as a fast route to the
current calibration decision.  Reactivation requires a controller-attested
change in measured throughput that preserves the scientific contract; reducing
epochs, seeds, or training coverage after observing the calibration failures is
not permitted.

## Decision context and scientific role

The fixed independent-CFG guidance mixture is complete and rejected by the
unchanged no-compensation gate. Operational construction succeeded, but the
candidate failed all three proper-score criteria, two of three finite-ensemble
reliability criteria, and the established-ice and exact-one boundary criteria.
Changing its weights, member allocation, seeds or guidance scales after seeing
that result is prohibited.

The next mechanism moves uncertainty upstream of sampling and postprocessing:
it combines independently trained clean checkpoints. This tests whether the
remaining underdispersion is primarily epistemic/model uncertainty that cannot
be recovered by latent-noise sampling from one fitted checkpoint. It also
produces the clean multi-seed checkpoint evidence required by the minimum paper
tier, so a negative result remains publication-relevant.

## Falsifiable prediction and paper claim

Three independently initialized checkpoints trained with the same corrected,
disjoint training protocol will produce complementary conditional analyses.
Their pooled ten-member ensemble is predicted to pass the full proper-score,
finite-ensemble reliability, boundary, spatial/physical and operational gate
against the frozen learned-joint raw ensemble. In particular, fair CRPS must
improve by at least 3% with its paired-date interval excluding zero, ordinary
CRPS may worsen by at most 1%, randomized-rank discrepancy must fall by at
least 20%, and every boundary and spatial/physical safeguard must pass without
family compensation.

Success supports the narrow claim that clean multi-checkpoint epistemic
diversity supplies useful bounded joint scenarios where single-checkpoint
latent and guidance diversity do not. Failure rejects this fixed three-seed
deep-ensemble construction; it means that independent initialization under the
same training objective is insufficient. Failure must not trigger seed
selection, checkpoint weighting, member reallocation or post-hoc calibration
on the development dates.

## Frozen training contract

- Train exactly three checkpoints with training seeds `1701`, `1702`, and
  `1703`, in that order.
- Every seed uses the corrected disjoint split and the exact architecture,
  loss, optimizer, schedule, normalization, observation-mask policy, stopping
  epoch and checkpoint-selection rule of the clean publication training
  configuration. Only the initialization/data-order seed differs.
- The runner must resolve one controller-attested immutable publication
  training configuration. Configuration hashes must match across seeds after
  removing only the seed field. No fallback to a legacy checkpoint is allowed.
- A seed is admissible only when training finishes normally, the selected
  checkpoint is finite, and the frozen selection rule identifies exactly one
  checkpoint without looking at calibration-gate metrics.
- If any seed is missing or inadmissible, the experiment fails operationally;
  it must not proceed with two checkpoints or replace the seed.

The exact clean publication training configuration and stopping rule must be
implemented and independently reviewed in the trusted runner before admission.
They are not runtime parameters. Runner review may expose a mismatch with this
contract, but may not choose them after reading candidate scores.

## Frozen ten-member construction

For ordered case index `i` in `0..39`, allocate four members to checkpoint
`i mod 3` and three members to each other checkpoint. Thus the extra member is
balanced as `14/13/13` over the forty cases rather than permanently favoring
one seed. Use latent member seeds `2401`, `2402`, `2403` for every checkpoint;
the checkpoint receiving the fourth member additionally uses `2404`.

Reuse the same initial-noise tensor for a given latent seed across checkpoints
and cases only according to the existing deterministic case/seed derivation;
this is a common-random-number comparison, not a shared tensor across different
cases. Member order is checkpoint seed ascending and then latent seed ascending.

Use learned-joint conditioning and the exact frozen publication sampler. Keep
conditioning tensors, physical output transform, `start_mode`, start noise
level, target, solver, timesteps, observation enforcement, normalization and
masks identical across checkpoints. Do not clip beyond the established physical
output transform, recenter, rank-shuffle, rescale, weight checkpoints, or apply
postprocessing.

The reference plan oracle is `deep_ensemble_reference.py`. Admission must agree
with its forty case allocations, `14/13/13` extra-member accounting, ordered
member identifiers and fail-closed metadata checks.

The compact pre-score admission manifest has exactly the schema implemented by
`validate_admission_manifest`: schema version, source experiment, one shared
non-seed configuration hash, three ordered training-run records, and forty
ordered case records. Each training record proves normal completion, one finite
selected checkpoint and the same non-seed configuration hash. Each member
record carries its exact member identity, checkpoint and latent seeds,
checkpoint hash, initial-noise hash and finite flag. Unknown or missing fields,
duplicate checkpoint identities, reordered cases or members, a substituted
checkpoint, a non-finite member, or unequal initial-noise hashes for the same
case/latent seed across checkpoints fail before any score is evaluated.

## Frozen evaluation and decision

Evaluate the same ordered forty development cases and the complete existing
no-compensation gate. The comparison baseline remains the ten-member raw
learned-joint ensemble. Paired date uncertainty is mandatory; the fixed
four-case-block interval is reported only as temporal sensitivity.

Operational validity additionally requires:

- all three training seeds and all three checkpoint identities;
- identical non-seed training metadata hashes;
- exactly ten finite members per case and the frozen `4/3/3` rotation;
- exact checkpoint/member seed identities and common-noise hashes;
- no checkpoint, member or case substitution;
- all forty cases completed and every reported metric finite.

Only `gate.overall_eligible=true` permits selection. Any failed mandatory
family closes this construction without compensation.

## Execution and artifact contract

The reviewed trusted modes
`validation_clean_checkpoint_deep_ensemble_sampling` and
`validation_clean_checkpoint_deep_ensemble_gate` execute this contract. The
sampling mode has a one-parameter interface only:
`source_experiment=joint_full_condition_validation_2022`. Training seeds,
member seeds, allocation, configuration, sampler and gate are constants.

Training, sampling and analysis run server-side. Retrieval is `summary_only`.
The compact result must include the complete gate, raw/candidate aggregates,
paired date and four-case-block uncertainty, immutable training/configuration
hashes, selected-checkpoint identities, per-case checkpoint/member counts,
ordered seed hashes, common-noise checks and operational counts. No raw member,
conditioning or truth field is requested.

This document freezes the scientific contract and does not itself authorize or
launch an experiment.
