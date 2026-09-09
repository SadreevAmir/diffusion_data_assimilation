# Censored joint multiscale-score pilot contract

Status: CPU implementation gate only. A GPU launch remains blocked until independent
Astra review.

## Frozen scientific question

Can a loss-only change remove the repeated pits/streaks found in individual members
of heldout baseline `118d003`, without losing its calibration and skill? The data,
generator, censoring, architecture, initialization, optimizer, train order, model-noise
stream, diagnostic cases and diagnostic-noise stream remain identical to the baseline.

The candidate must load baseline run
`heldout_energy_118d003_20260909T193003Z/model_step_0000.pth` with SHA-256
`8a89a40bb335738124a26e823166bca85a8e288705a48ee30bfe1106337c1802`.
Recreating a merely similar initialization is not sufficient.

The generated physical law is unchanged:

1. predict a normalized joint residual for all six `(SIC, SIT) × (d+3,d+6,d+9)`
   channels from a full-field condition and iid Gaussian noise;
2. add normalized persistence;
3. denormalize with train-only field statistics;
4. censor each member physically: SIC to `[0,1]`, SIT to `[0,∞)`;
5. score the censored physical members. No STE, member MSE or post-hoc smoothing.

## Frozen proper objective

For a valid-ocean region `R`, transform a physical field by train-derived channel
standard deviations and divide Euclidean distances by `sqrt(6 * number_valid_R)`.
Use the unbiased ensemble energy-score estimator, jointly across all six channels.

The single candidate is

`L = 0.50 ES_global + 0.25 mean_k ES_patch8,k + 0.25 mean_k ES_patch16,k`.

For each condition and update, sample eight valid-ocean centres with replacement using
an RNG isolated from loader, network initialization and model noise. The 8×8 and 16×16
patches share the same centres. Coordinates outside the image and land/padding pixels
are excluded; there is no wrapping or reflection. Centre selection depends only on the
static valid mask. Patches are sliced from the already generated full field, never
generated independently.

Positive global weight retains identification of the full joint law; the patch terms
increase sensitivity to local spatial dependence. The weights and scales are fixed
before candidate validation and cannot be tuned on val-2022.

## Required CPU FP64 gate

- Manual two-member estimator equality and finite zero-distance backward.
- Invariance to values added outside the valid mask.
- Reproducible valid-only centre sampling from an independent RNG.
- A finite smooth joint law must beat a spatially scrambled law with identical
  pointwise six-channel marginals and must beat its degenerate ensemble mean.
- Monte Carlo patch-centre averaging must match the exhaustive valid-centre mean.

## Future bounded run (not yet authorized)

One GPU, at most 1024 updates, checkpoints `0/256/512/1024`, online ClearML, existing
IPC gate, CPU `6/1`, timeout `7080+120 s`, no retry/resume/extension. Reuse the exact
baseline step-zero checkpoint, train order, model-noise stream, and fixed diagnostic
noise. Save raw physical members and uncensored latents before evaluation, plus all
checkpoint/noise/patch identities and failure evidence. Test-2023 remains closed.

For evaluation, the runner preserves the baseline metrics, adds fixed 128 centres/case
for the candidate score, and reports boundary-event Brier scores. Directional
autocorrelation/spectrum and stride-phase shift-equivariance probes are a separate,
frozen CPU postprocess over the saved raw members/checkpoints. That postprocess must be
implemented, audited and completed for baseline and candidate before any scientific
verdict; a successful training exit alone cannot pass the review gates.

## Predeclared review gates

- Finite/support/provenance gates pass.
- For every output and lag 1/2/4/8, case-equal individual-member increment/truth is
  within `[0.80,1.35]`; zero denominators are reported separately.
- Individual panels contain no repeated pits/streaks and are not blurred.
- Versus baseline-1024, mean of six CRPS ratios is at most `1.01`; every CRPS and RMSE
  ratio is at most `1.05`.
- Mean absolute SSR error relative to `sqrt(8/9)` grows by at most `0.02`; every rank-TV
  is at most baseline plus `0.02`.
- Boundary-event Brier scores and SIC=0/1, SIT=0 frequencies are reported, not optimized
  after inspection.

A failure at 1024 is a failed/inconclusive candidate, not permission to extend budget.
