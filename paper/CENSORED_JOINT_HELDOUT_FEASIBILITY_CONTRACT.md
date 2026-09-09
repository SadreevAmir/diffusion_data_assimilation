# Censored-joint held-out feasibility pilot

Status: **design only; GPU launch is blocked until independent code review**.

## Scientific question

Can the unchanged persistence-centred censored joint generator trained with the
unbiased joint energy score learn transferable conditional SIC/SIT fields on
validation year 2022, while preserving physical atoms and realistic spatial
dependence in individual ensemble members?

This is a bounded feasibility experiment.  It cannot establish publication
quality, final calibration, or test-set performance.

## Frozen provenance and model law

- Reference implementation commit:
  `b3d534dca25484347565c3942b98c84759cd5c1e`.  Source configuration
  SHA-256 values are experiment
  `06c335a798d5c446c601616a73754d2d158f16c513d18d9ae426e9e27a9cd635`,
  data `6fb71cc079f17fc0888dd4b4c0d7f06dd57c217b0729403ee3916ac3f82903de`,
  and model `bbac44789470eb88e9e47b8b93a1b63e3da99e3ff4de154a6a4f3497caca1029`.
- Fresh UNet, image size `320x256`, 23 input channels, 6 output channels,
  block widths `[96,192,384,384]`, one layer per block, group norm with 32
  groups, down blocks `[Down,Down,AttnDown,Down]`, and up blocks
  `[Up,AttnUp,Up,Up]`.
- The output convolution is freshly initialized from seed 2014 with iid
  Gaussian weights of standard deviation `1e-3` and zero bias; its realized
  norm and SHA-256 of the step-zero checkpoint are recorded.
- Input: six independent Gaussian-noise fields, two coordinate grids, and the
  unchanged fifteen-channel dynamics conditioning.
- Output latent: normalized persistence trajectory plus one learned residual.
- Physical law inside both training and evaluation: clamp SIC to `[0, 1]` and
  clamp SIT below at zero.
- One forward pass at fixed model timestep 500; no ODE solve.
- Dropout, EMA, activation checkpointing, per-member MSE, antithetic noise,
  straight-through gradients, target-dependent masks, spatial smoothing, and
  auxiliary spatial losses are forbidden.
- Objective: unbiased energy score for the complete six-field valid-ocean
  vector after division by train-only SIC/SIT standard deviations, with two
  iid members per condition.
- The only physical normalization is the train-only pair
  `means=(0.19301218262209952, 0.18969311571248842)` and
  `stds=(0.36890927421384667, 0.4359687842884708)`, repeated over three leads.
  Dynamic forcing statistics must exactly equal the train-only values in the
  pinned data configuration.

## Training envelope

- Split: all 51,792 train cases from 2016--2021, all 24 archive slices.
- Deterministic shuffled order, recorded seed, no replacement within a loader
  pass, and no validation case in the loader.
- Condition batch 8; network batch 16 after the two-member expansion.
- Four workers, prefetch factor four, with the existing `/dev/shm` preflight
  before iterator construction.
- AdamW, learning rate `1e-4`, linear warmup for 32 updates, then constant.
- BF16 network execution and FP32 physical score.
- At most and normally exactly 1,024 optimizer updates, with diagnostic
  checkpoints at 0, 256, 512, and 1,024.  A clean controlled early failure may
  produce fewer updates but can never be interpreted scientifically.
- The launcher enforces one visible idle GPU, the common GPU lock, online
  ClearML established before optimizer construction, and timeout `7080s` plus
  `120s` TERM-to-KILL grace.  CPU thread limits are six intra-op and one
  inter-op.  Training and sampling may expose at most 16 simultaneous network
  inputs.  Scoring is performed casewise on CPU.  Checkpoint creation precedes
  its diagnostic.  Partial contracts, checkpoints, metrics, and a failure
  status are retained on controlled failure.  There is no automatic resume,
  retry, extension, or continuation.

## Frozen validation anchors

Build the validation dataset before training.  The frozen manifest is:

| dataset index | d0 | slice | d+3 | d+6 | d+9 |
| ---: | --- | ---: | --- | --- | --- |
| 120 | 2022-01-06 | 0 | 2022-01-09 | 2022-01-12 | 2022-01-15 |
| 846 | 2022-02-05 | 6 | 2022-02-08 | 2022-02-11 | 2022-02-14 |
| 1572 | 2022-03-07 | 12 | 2022-03-10 | 2022-03-13 | 2022-03-16 |
| 2298 | 2022-04-06 | 18 | 2022-04-09 | 2022-04-12 | 2022-04-15 |
| 3000 | 2022-05-06 | 0 | 2022-05-09 | 2022-05-12 | 2022-05-15 |
| 3726 | 2022-06-05 | 6 | 2022-06-08 | 2022-06-11 | 2022-06-14 |
| 4452 | 2022-07-05 | 12 | 2022-07-08 | 2022-07-11 | 2022-07-14 |
| 5178 | 2022-08-04 | 18 | 2022-08-07 | 2022-08-10 | 2022-08-13 |
| 5880 | 2022-09-03 | 0 | 2022-09-06 | 2022-09-09 | 2022-09-12 |
| 6606 | 2022-10-03 | 6 | 2022-10-06 | 2022-10-09 | 2022-10-12 |
| 7332 | 2022-11-02 | 12 | 2022-11-05 | 2022-11-08 | 2022-11-11 |
| 8058 | 2022-12-02 | 18 | 2022-12-05 | 2022-12-08 | 2022-12-11 |

Every target path basename must equal
`ocean+atmosphere_24_<manifest target date>.npy`.  Conditioning may contain
only the exact d0 SIC/SIT state, the valid-ocean mask, d0 dynamic forcing
values and availability masks at the same archive slice, and d0 calendar
features.  No d+3/d+6/d+9 field or metadata-derived target value may enter it.

The runner must fail unless:

1. every selected item reports split `valid` and target year 2022;
2. reported archive-slice indices equal the frozen cycle;
3. all twelve case IDs and target dates are unique;
4. every pair of closed date windows `[d0, d0+9 days]` is disjoint;
5. all target paths and conditioning tensors are finite and available.

The complete train calendar-pair inventory must additionally prove that every
train d0 and every d+3/d+6/d+9 target date lies inside 2016--2021.  The same
inventory check must keep every validation target date inside 2022.

The four historical train anchors are retained only as memorization controls
and must be reported separately from validation.  Aggregates must never mix
the two groups.

## Fixed sampling

- Eight iid diagnostic noises per condition, generated once on CPU from a
  recorded seed and reused byte-identically at every checkpoint.
- Generation is chunked by condition and member; changing chunks must not
  change the fixed noise tensors.
- Save raw physical individual members, uncensored normalized latents, truth,
  persistence, valid mask, case metadata, checkpoint SHA-256, and diagnostic
  noise SHA-256.
- ClearML must show truth, persistence, member 0, and member 1 for every
  validation anchor and lead.  Full tensors remain server-side.

## Required validation diagnostics

Let `M=8`.  Each valid pixel receives equal weight within its case and each of
the twelve cases receives equal weight in the aggregate.  For every output:

- `RMSE = sqrt(mean_case(MSE_case))` for both ensemble mean and persistence;
- `spread = sqrt(mean_case(mean_valid_pixel(unbiased_member_variance)))` and
  `spread/skill = spread/RMSE`;
- fair CRPS per pixel is
  `mean_m |X_m-y| - sum_(m!=n)|X_m-X_n|/[2M(M-1)]`, followed by within-case
  pixel averaging and equal case averaging;
- the CRPS ratio divides the two aggregated scores.  A zero persistence
  denominator is recorded as `null` with reason `zero_baseline_score`; no
  epsilon or case-ratio averaging is allowed.

- Ensemble-mean RMSE and persistence RMSE for every one of six outputs.
- Finite-ensemble fair CRPS for every output and its ratio to the deterministic
  persistence absolute error.
- Mean spread, RMSE, and spread/skill ratio without interpreting nonzero spread
  as success by itself.
- Tie-aware rank histogram uses exact fractional ties: if `L` members are
  strictly below truth and `E` members equal truth, add weight `1/(E+1)` to
  every bin `L...L+E`.  No random tie-breaking is used.
- Fractions of exact SIC zero, exact SIC one, and exact SIT zero in members and
  truth, by case and output.
- Mean absolute spatial increments at 1, 2, 4, and 8 pixels, separately for
  each individual member, ensemble mean, truth, and persistence, using only
  valid-ocean pixel pairs.
- Maps of ensemble SD and atom frequency for each validation anchor.
- Cross-lead dependence is computed separately for each validation case and
  field.  For each lead, subtract its eight-member spatial-mean ensemble mean
  from every member's valid-ocean spatial mean, then correlate the resulting
  eight member anomalies across d+3/d+6/d+9.  If either variance is zero, store
  `null` and a `zero_member_variance` reason rather than NaN.  Also report the
  truth and persistence spatial-mean temporal changes, without claiming a
  population correlation from twelve cases.
- Exact physical support counts and explicit NaN/Inf checks.
- Collapse is assessed only on heterogeneous case-output pairs, defined by a
  nonzero range of truth over valid-ocean pixels.  Byte-identical values for
  all eight members over all valid pixels of such a pair are an exact collapse
  failure.  A correctly predicted homogeneous boundary field may have zero
  spread and is reported separately, never counted as collapse.

## Fail-closed interpretation

The terminal scientific status is always
`complete_pending_independent_review`; the runner has no automatic scientific
PASS.  Missing artifacts, support/finite failure, or exact heterogeneous-case
collapse are recorded as hard failures.  Skill relative to persistence,
spatial texture, atom frequencies, ranks, and spread are reported without a
post-hoc numerical acceptance threshold and require independent joint review.
No single aggregate score may compensate for an unacceptable physical,
spatial, reliability, or provenance family.

Validation-2022 has already influenced development and is exploratory, not an
independent confirmation.  Even a favorable review only permits design review
for a substantive training run.  It does not itself permit one, and it does
not authorize test-2023 access.
