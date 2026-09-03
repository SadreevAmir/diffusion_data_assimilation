# Provenance-aware occurrence–intensity CFM: corrected representation contract

Status: P0_CORRECTED_INDEPENDENT_READMISSION_REQUIRED

## Scientific hypothesis

The rejected residual CFM confounded observation age and origin by merging
lagged real footprints and synthetic truth-derived tracks into one narrow mask.
It also learned an unbounded residual and relied on hard output clipping.  The
next mechanism makes both assumptions identifiable: each lag contributes six
separate fields (innovation, observed value, mask, age, real provenance and
synthetic provenance), while SIC is generated as occurrence plus bounded
conditional intensity.  No generated concentration is hard-clipped.

The earlier sentinel is withdrawn and must not be admitted or launched. Age
channels are only a cheap conditional ablation; they are not a substitute for a
likelihood-consistent fixed-lag trajectory smoother.

The primary protocol uses real tracks only. Synthetic tracks are prohibited in
primary training and sentinel selection; they may appear only in one separately
labelled augmentation ablation after the primary sentinel passes.

## Correctness invariants

- Lag order is exactly current, one-day old, two-days old; age fields are
  respectively 0, 1 and 2 on observed pixels and zero elsewhere.
- Innovation at day `t-k` is exactly `y_{t-k} - b_{t-k}` under its finite mask.
  Passing only current `b_t` for multiple lags is a shape error, not broadcast.
- Provenance is one-hot real/synthetic on every observed pixel and zero outside
  each lag mask. Values and innovations are zero outside their own mask.
- Occurrence is the physical atom `A=1{SIC>0}`. The representation preserves
  every `0<SIC<=0.15` exactly. The `0.15` threshold exists only as the derived
  established-ice diagnostic/event and never controls encode/decode.
- The auditable primitive stores bounded conditional intensity directly and is
  exactly invertible on `[0,1]`. It has no epsilon clip. A later anamorphosis may
  be admitted only with an explicit generalized inverse or distributional
  transform that preserves the boundary atoms.
- Any non-binary mask/provenance, negative age, non-finite target or target
  outside `[0,1]` fails before training.
- Every training run must fail closed unless `clearml.enabled=true`; its task
  records data/model hashes, protocol (`primary_real` or `synthetic_ablation`),
  seed, checkpoints, train/validation curves and sentinel artifacts.

## Withdrawn sentinel

The prior eight-case sentinel is not admissible and must not run. Its numerical
criteria below are retained only as historical design context pending independent
readmission of corrected code. It had proposed eight predeclared validation cases spanning
low/high ice area and sparse/dense real-track coverage, selected from training
metadata without reading their truth fields. Train only the short predeclared
budget; do not choose an epoch from sentinel rank metrics. Compare against the
completed residual CFM and the background on the same cases.

For every case, save one compact correctly oriented panel containing all ten
individual members, truth, background, each lag mask, ensemble mean and ensemble
standard deviation. The server report also includes per-case values and paired
aggregates for area bias, RMSE, fair CRPS, randomized-rank histogram, rank TV,
truth-above-all rate, attainable range/inner coverage, exact-zero/one masses and
a track-imprint statistic: mean absolute anomaly within a two-pixel dilation of
the observed tracks divided by the same quantity off-track.

Proceed to full training only if all conditions hold:

1. every member is finite and within `[0,1]`, with zero hard-clipped pixels;
2. no panel has a visually repeated narrow track imprint in at least two members,
   confirmed by two fixed orientation landmarks embedded in each panel;
3. absolute mean area bias is at most `0.06` and improves by at least 25% versus
   the rejected residual CFM;
4. truth-above-all rate is at most `0.20` and rank TV is at most `0.16`, with
   neither diagnostic worse than the rejected residual CFM;
5. median track-imprint ratio is at most `1.25` and at least 20% lower than the
   rejected residual CFM;
6. fair CRPS does not worsen by more than 2% and background RMSE is improved.

Failure of any condition stops this mechanism without tuning thresholds,
occurrence cutoff, channel definitions, lag count or short budget. A failure
dominated by occurrence/boundary calibration activates the predeclared
train-only seasonal atom-aware normal-score mechanism; persistent track imprint
or poor ranks instead supports the conclusion that a new clean real-track-only
retraining protocol is required.

## Execution boundary

No trusted executor mode currently names this method. Local code and tests are
not a scientific result and do not authorize training. Admission requires an
independent repeat audit and then an independently reviewed literal mode implementing this exact contract; runtime
parameters must not expose scientific choices. Retrieval defaults to
`summary_only`, with only the eight named sentinel panels additionally selected.
