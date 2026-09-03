# Corrected occurrence–intensity programme: E1–E4

Status: PREDECLARED_NO_GPU_ADMISSION

All four experiments are separate mechanism tests. None is authorized for launch
until an independent repeat audit accepts the corrected representation and lag
semantics. The established-ice event is always the derived `SIC>0.15`; physical
occurrence is always `SIC>0`.

## E1 — finite-mask and orientation baseline

- Claim role: implementation control and falsification of data-layout artifacts.
- Baseline: background and the rejected residual model on the same predeclared
  cases.
- Intervention: corrected finite masks, lag-specific `y_{t-k}-b_{t-k}`, and two
  immutable orientation landmarks; no learned lag trajectory.
- Prediction: all outputs are finite/in-range, masked channels are exactly zero,
  landmarks agree for every panel, and the track-imprint ratio is no worse than
  background by more than 5%.
- Stop/go: stop on any non-finite value, orientation mismatch, masked leakage
  above `1e-7`, or track-imprint ratio above `1.05 * background`. Passing E1 is
  necessary but not evidence for assimilation benefit.
- Failure interpretation: tensor layout, mask semantics or rendering remains
  defective; E2–E4 are uninterpretable.

## E2 — likelihood-consistent fixed-lag trajectory

- Claim role: mechanism test of temporal assimilation rather than age metadata.
- Baseline: E1 with age channels only.
- Intervention: a trajectory state containing `b_t,...,b_{t-K}` and innovations
  evaluated against their matching background frames under the observation
  likelihood.
- Prediction: versus E1, median observed-footprint innovation RMSE improves at
  least 10%, fair CRPS does not worsen by more than 2%, and off-track anomaly
  energy does not rise by more than 5%.
- Stop/go: proceed only if all three thresholds pass and orientation/mask checks
  remain exact.
- Failure interpretation: lag-consistent smoothing adds no defensible temporal
  information or propagates local observations unphysically; age channels cannot
  be described as an equivalent smoother.

## E3 — atom-aware anamorphosis

- Claim role: boundary-calibration mechanism test.
- Baseline: exact atom plus bounded identity intensity from the corrected
  primitive.
- Intervention: a predeclared generalized-inverse/distributional transform with
  explicit masses at zero and one; no epsilon clipping.
- Prediction: all `0<SIC<=0.15` round-trip exactly, exact-zero/one event errors do
  not worsen, rank TV falls by at least 15%, and fair CRPS does not worsen by more
  than 2%.
- Stop/go: stop on any lost small-positive value, changed boundary atom, rank-TV
  gain below 15%, or compensated proper-score/boundary failure.
- Failure interpretation: anamorphosis does not repair boundary reliability;
  retain the exact identity representation.

## E4 — structured analog source

- Claim role: test whether coherent historical anomaly fields provide useful
  ensemble structure beyond pixelwise noise.
- Baseline: the best independently accepted result among E1–E3, selected by the
  unchanged no-compensation gate rather than a single metric.
- Intervention: training-only analog selection using forecast-state features and
  complete residual fields with deterministic ties; no outcome-time selection.
- Prediction: attainable inner coverage rises by at least 0.05, absolute rank TV
  falls by at least 10%, and memberwise variogram error does not worsen by more
  than 5%.
- Stop/go: require every threshold plus unchanged boundary, proper-score and
  operational gates; otherwise stop without tuning neighbors or distances.
- Failure interpretation: the analog source does not add useful coherent
  uncertainty and cannot support the paper's probabilistic-assimilation claim.

E1 precedes E2–E4 because it is a validity control. After E1, E2, E3 and E4 test
distinct temporal, marginal-boundary and spatial-source mechanisms; dependencies
must follow scientific baselines, not hardware availability.
