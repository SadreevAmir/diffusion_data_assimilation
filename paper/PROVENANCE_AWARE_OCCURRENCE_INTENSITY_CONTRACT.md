# Provenance-aware occurrence–intensity CFM: corrected representation contract

Status: P0_IMPLEMENTED_INDEPENDENT_READMISSION_REQUESTED

## Scientific hypothesis

The rejected residual CFM confounded observation age and origin by merging
lagged real footprints and synthetic truth-derived tracks into one narrow mask.
It also learned an unbounded residual and relied on hard output clipping.  The
next mechanism makes both assumptions identifiable: each lag contributes eight
separate fields (innovation, observed value, mask, age, real/synthetic geometry
provenance and real/synthetic value provenance), while SIC is generated as
occurrence plus bounded
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
- Geometry provenance and value provenance are separate one-hot real/synthetic
  fields. They are zero outside each lag mask; values and innovations use
  `torch.where`, so NaN at missing pixels cannot leak through multiplication.
  Finiteness is required exactly on observed pixels.
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
- The occurrence CFM target is `Z=(A+U)/2`, `U~Uniform[0,1)`, giving disjoint
  continuous laws on `[0,.5)` and `[.5,1)` and exact-zero recovery at threshold
  `.5`. At zero, intensity uses an independent `Uniform[0,1]` auxiliary law and
  is ignored by decoding. Positive intensity remains the exact SIC value.
- Exact-one handling is frozen only after the training inventory: presence of
  exact ones selects `explicit_exact_one_atom`; absence selects
  `no_exact_one_atom`. Sentinel information cannot change this policy.

## E1 engineering sentinel

The new eight-case sentinel remains blocked pending independent admission. It is
an engineering check only: it may test finiteness, bounds, channel leakage,
orientation, provenance and visually inspect individual members, but it has no
rank gate and cannot select a checkpoint or support a calibration claim. The
historical numerical criteria below are withdrawn and retained only as design
context. The cases span
low/high ice area and sparse/dense real-track coverage, selected from training
metadata without reading their truth fields. Train only the short predeclared
budget; do not choose an epoch from sentinel rank metrics. Compare against the
completed residual CFM and the background on the same cases.

For every case, save one compact correctly oriented panel containing all ten
individual members, background, each lag mask, ensemble mean and ensemble
standard deviation. The server report includes engineering diagnostics only:
finiteness, range, masked-channel leakage, orientation landmarks, provenance
consistency, hard-clip count and a track-imprint statistic (mean absolute
anomaly within a two-pixel dilation of the observed tracks divided by the same
quantity off-track). It must not emit truth-conditioned rank, coverage,
proper-score or checkpoint-selection diagnostics.

The sentinel is technically accepted only if all conditions hold:

1. every member is finite and within `[0,1]`, with zero hard-clipped pixels;
2. no panel has a visually repeated narrow track imprint in at least two members,
   confirmed by two fixed orientation landmarks embedded in each panel;
3. masked-channel leakage is at most `1e-7`, and all geometry/value provenance
   fields agree with the frozen channel layout;
4. the track-imprint ratio is no worse than `1.05 * background` on every case.

Failure of any condition stops E1 without tuning thresholds, occurrence cutoff,
channel definitions, lag count or short budget. Passing is necessary only for a
separately reviewed full experiment; it is not evidence of calibration benefit
and does not itself authorize that experiment.

## Execution boundary

No trusted executor mode currently names this method. Local code and tests are
not a scientific result and do not authorize training. Admission requires an
independent repeat audit and then an independently reviewed literal mode implementing this exact contract; runtime
parameters must not expose scientific choices. Retrieval defaults to
`summary_only`, with only the eight named sentinel panels additionally selected.
