# Reliable Generative Data Assimilation: Finite-Ensemble Calibration under Sparse Spatial Observations

> Living preprint draft. Claims marked `[HYPOTHESIS]` or `[PLANNED]` are not
> results. See `CLAIM_LEDGER.md` for provenance and limitations.

## Abstract

Conditional generative models provide a computationally attractive route to
ensemble data assimilation, but a collection of plausible spatial samples is
not necessarily a reliable predictive ensemble. This distinction is especially
important in practice, where only a small number of members is generated and
the assimilated field can be bounded and zero-inflated. We study conditional
flow-based data assimilation for Arctic sea-ice concentration using a dense
model background and sparse observations under real satellite-track geometry.
On 40 validation dates, increasing observation-track guidance from 0.5 to
0.625 reduces ensemble-mean RMSE by 4.34% and empirical ensemble CRPS by 6.52%.
However, standard affine-logit postprocessing improves cross-validated,
adaptively reused development CRPS while severely degrading boundary
reliability because it removes exact
open-water mass. This exposes two frequently conflated effects: finite-ensemble
bias in common diagnostics and calibration failure in structured bounded
outputs. `[HYPOTHESIS]` We develop a boundary-aware, finite-ensemble-aware and
rank-preserving calibrator that separately treats occurrence, positive
intensity and spatial dependence. `[PLANNED]` A frozen out-of-year evaluation
will compare the calibrated generative ensemble with classical postprocessing
and 3D-Var under an identical model-to-model protocol. The broader goal is to
turn generative analysis samples into statistically reliable spatial scenarios
without sacrificing sharpness or physical structure.

## 1. Introduction

Diffusion models, flow matching and stochastic interpolants have made it
possible to sample high-dimensional conditional distributions at useful
spatial resolutions. In data assimilation, this offers a compelling alternative
to a single optimized analysis: the output can be an ensemble that represents
multiple states compatible with a physical background and sparse observations.

Yet generative fidelity and probabilistic reliability are different targets.
An ensemble mean may be accurate while the members are underdispersed; an
aggregate proper score may improve while interval coverage deteriorates; and a
pointwise marginal correction may destroy fronts, connectivity or dependence
between distant locations. These problems are hidden further when an expensive
generative model is evaluated with only 5–20 members. Nominal quantiles are then
poorly resolved, conventional ensemble CRPS contains a finite-sample component,
and uncorrected spread-skill ratios do not have an ideal value of one under the
same conventions.

Our motivating application is model-to-model assimilation of Arctic sea-ice
concentration. We condition a learned generator on a dense forecast-like
background and sparse concentration values sampled under real SRAL footprint
geometry. The setting combines several difficult properties: the field is
bounded in `[0,1]`, more than half of evaluated ocean pixels can be exactly zero,
the observation geometry changes daily, and scientifically relevant error is
concentrated near the ice edge.

This draft begins with an empirical finding. A stronger observation-guidance
weight improves validation RMSE, empirical CRPS and withheld-track skill. A
regularized affine-logit transformation further improves blocked-validation
CRPS and RMSE, but removes exact zero mass and sharply worsens coverage. Thus
score optimization alone does not solve the calibration problem.

### Contributions

The intended contributions are:

1. **Finite-ensemble-aware evaluation.** `[IN PROGRESS]` We distinguish
   empirical-distribution scores from fair ensemble estimators and derive or
   document attainable reliability targets for small ensembles.
2. **Boundary-aware structured calibration.** `[HYPOTHESIS]` We separately
   calibrate occurrence and positive intensity, then preserve/reconstruct the
   raw ensemble's rank dependence instead of applying an unconstrained field
   correction.
3. **A locked sparse-observation protocol.** We enforce temporally separated
   development and test years, exact conditioning-channel provenance, a frozen
   model-to-model comparison with 3D-Var, and date-block uncertainty estimates.
4. **A diagnostic study of generative assimilation.** We evaluate reliability
   by season, ice regime, track geometry and distance to observation, as well as
   spatial spectra, edge statistics and multivariate scores.

Only the protocol and preliminary validation evidence are complete at the time
of this draft. The method and locked test contributions remain hypotheses.

## 2. Conditional generative assimilation

Let `x` denote the target sea-ice state, `b` a dense background and `o` sparse
track observations with mask `m`. The trained conditional vector field supports
unconditional, background-only, observation-only and jointly conditioned
branches. At inference, independent guidance uses

\[
v = v_\varnothing
  + (1-w)(v_b-v_\varnothing)
  + w(v_o-v_\varnothing),
\]

with a zero coefficient for the joint branch. This gives a post-training control
over the balance between the dense background and sparse observations.

The strict comparison evaluates only sea-ice concentration. Thickness inputs
and masks are zeroed, although the legacy network retains two output channels.
Track conditioning uses footprints for the target date and two preceding days;
newer observations overwrite older values on overlap. A next-day footprint,
not used in conditioning, provides a sparse track-imitation diagnostic.

## 3. Why finite ensembles complicate calibration

For `M` generated members, the empirical ensemble CRPS and the fair score for
an underlying sampling distribution use different pairwise normalizations. The
latter removes the self-pair finite-sample bias and is required when ensemble
sizes differ or when dispersion is interpreted as a property of the underlying
generator.

Likewise, order-statistic intervals from a small exchangeable ensemble have
discrete attainable coverage. With `M=10`, even the member range has expected
coverage `9/11` under continuity and exchangeability. Interpolated 5% and 95%
sample quantiles therefore must not be interpreted as a literal 90% predictive
interval without a continuous predictive model or an explicit finite-ensemble
construction. Randomized ranks provide a more direct exchangeability check.

`[PLANNED]` We formalize the estimands used throughout the paper and report both
finite-ensemble diagnostics and distribution-level scores where identifiable.

## 4. Boundary-aware rank-preserving calibration

`[HYPOTHESIS]` A useful calibrator for sea-ice concentration should represent:

1. probability mass at exact open water and, where supported, exact compact ice;
2. a conditional distribution for values in the interior of the physical range;
3. dependence across pixels and ensemble members.

The proposed pipeline first learns a regularized occurrence calibrator for
events such as `x>0` and `x>0.15`. It then learns a monotone conditional map for
positive concentration. Parameters use hierarchical shrinkage across month,
distance to observations, track density and ice regime. Calibrated marginal
quantiles are assigned according to raw member ranks, retaining an empirical
copula in the spirit of ensemble copula coupling. A temporal-block conformal
layer is considered only for explicitly stated coverage targets.

The method is selected under a predeclared constrained objective: fair CRPS is
the primary score; calibration error and boundary mass are constraints;
variogram distortion, spatial spectra and physical edge statistics are
guardrails. A method that improves CRPS while violating reliability is rejected.

## 5. Experimental protocol

Training targets cover 2016–2021, development targets use 2022, and the locked
test consists of 200 consecutive target days in 2023. The present checkpoint is
a legacy final-epoch checkpoint and is used only for preliminary development.
Publication results require retraining with the corrected split and multiple
seeds.

The primary deterministic quantity is the generated ensemble mean, which is
compared with the single 3D-Var analysis. Probabilistic scores apply only to the
generative and probabilistic baselines. All uncertainty intervals resample
dates or temporal blocks, never individual pixels.

Baselines include raw ensembles, physical bias/spread scaling, naive logit
postprocessing, zero/one-inflated distributional regression, isotonic or
quantile mapping, conformal intervals and ECC-like reconstruction.

## 6. Preliminary validation results

On the same 40 validation dates with ten generated members, changing the
track/background weights from `0.5/0.5` to `0.625/0.375` changes full-field
ensemble-mean RMSE from `0.21291` to `0.20366`, empirical CRPS from `0.07070`
to `0.06609`, and IIEE from `0.08767` to `0.08537`. At `w=0.625`, the analysis
improves background RMSE from `0.25603` to `0.20366`; on the next withheld
track, RMSE improves from `0.19046` to `0.12384`.

The improvement is regime-dependent. Preliminary read-only analysis indicates
that CRPS improves predominantly in established ice and can worsen at exact
zero and in marginal ice. This motivates a boundary-aware method rather than a
single global transform.

A four-fold temporally blocked affine-logit search changes cross-validated
development empirical
CRPS from `0.06609` to `0.06092` and ensemble-mean RMSE from `0.20366` to
`0.19428`. This is not evidence of successful calibration. The transform clips
exact zeros before applying a logit and returns no exact zero after inversion.
Across the out-of-fold transforms, the current nominal 90% interval diagnostic
falls from `0.646` to `0.367`, while member-range coverage falls from `0.722`
to `0.396`. Exact zero occurs on `53.6%` of truth pixels and `24.6%` of raw
member pixels, but on no calibrated member pixels. Although the finite-ensemble
targets for these interval statistics require correction, the transformation's
boundary failure is unambiguous. Repeated refinement on the same 40 dates also
makes the final blocked estimate meta-adaptive rather than locked.

## 7. Limitations

The completed evidence uses one legacy checkpoint, one sampling seed, ten
members and 40 development dates. The observation values are M2M truth sampled
under real satellite geometry, not direct satellite SIC retrievals. No locked
test or exact numerical comparison with 3D-Var is reported yet. The proposed
calibrator and generalization beyond Arctic sea ice remain unverified.

## 8. Planned figures and tables

- Figure 1: conditioning and independent-guidance pipeline.
- Figure 2: finite-ensemble bias and attainable coverage versus ensemble size.
- Figure 3: exact-zero failure of naive logit postprocessing.
- Figure 4: raw and calibrated reliability conditional on ice regime and track distance.
- Figure 5: fair-CRPS/reliability/spatial-distortion Pareto frontier.
- Table 1: frozen background, 3D-Var and flow deterministic comparison.
- Table 2: probabilistic postprocessing baselines.
- Table 3: boundary, spatial and ensemble-size ablations.
- Table 4: out-of-distribution and second-task generalization.
