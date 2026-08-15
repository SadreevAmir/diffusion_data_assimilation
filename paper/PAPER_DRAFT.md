# Reliable Generative Data Assimilation: Finite-Ensemble Calibration under Sparse Spatial Observations

> Validation-mechanism draft, not a submission-ready paper. See
> `CLAIM_LEDGER.md` and `PUBLICATION_READINESS.md` for provenance, minimum-tier
> blockers and limitations. This document does not report an independent
> generalization result.

## Abstract

Conditional generative models provide a computationally attractive route to
ensemble data assimilation, but a collection of plausible spatial samples is
not necessarily a reliable predictive ensemble. This distinction is especially
important in practice, where only a small number of members is generated and
the assimilated field can be bounded and zero-inflated. We study conditional
flow-based data assimilation for Arctic sea-ice concentration using a dense
model background and sparse observations under real satellite-track geometry.
On 40 validation dates, a learned joint-conditioning ensemble improves the
background ensemble-mean RMSE but is strongly underdispersed. We therefore
freeze a deliberately simple postprocessing rule: in each of five deterministic
date-stratified folds, select one global multiplicative anomaly scale on the
other 32 dates by fair CRPS, then apply it to the eight held-out dates. Selected
scales are 2.6, 2.7, 2.8, 2.8 and 2.6. Cross-fitted fair CRPS decreases from
0.058491 to 0.055690 (4.79%), ordinary CRPS decreases from 0.062108 to 0.061101,
and spread-skill ratio changes from 0.7241 to 1.0615. The nominal 90% interval
diagnostic rises from 0.5051 to 0.8788. The transform preserves the ensemble
center before bounded-score clipping to numerical precision. These validation
results support a narrow diagnosis of predominantly global underdispersion. A
predeclared joint audit nevertheless rejects the correction as a well-calibrated
ensemble: clipping creates a large spurious exact-one atom, established-ice
Brier score worsens by 2.15%, and IIEE and edge disagreement worsen beyond their
2% tolerances. The result exposes a marginal-score/physical-structure tradeoff,
not an independent generalization claim or superiority to a deterministic
method.

## 1. Introduction

Diffusion models, flow matching and stochastic interpolants have made it
possible to sample high-dimensional conditional distributions at useful
spatial resolutions [1,2]. In data assimilation, this offers a compelling
alternative
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

### Contributions supported by the present evidence

The present validation study contributes:

1. **Finite-ensemble-aware evaluation.** We distinguish
   empirical-distribution scores from fair ensemble estimators and derive or
   document attainable reliability targets for small ensembles.
2. **A mean-preserving global spread correction.** We cross-fit a single
   multiplicative anomaly scale, preserving member ranks and the pre-clipping
   ensemble center, and report both ordinary and fair CRPS.
3. **An auditable sparse-observation protocol.** We record the conditioning
   channels, real footprint geometry, model-to-model observation values, frozen
   checkpoint and deterministic cross-fitting procedure.
4. **A documented negative baseline.** We show why score improvement from an
   affine-logit transform is insufficient when it destroys exact boundary mass
   and worsens interval diagnostics.

The method is frozen from validation evidence. Independent evaluation,
multi-seed training and broader generalization remain outside the present claim.

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
latter removes the self-pair finite-sample bias [3] and is required when ensemble
sizes differ or when dispersion is interpreted as a property of the underlying
generator.

Likewise, order-statistic intervals from a small exchangeable ensemble have
discrete attainable coverage. With `M=10`, even the member range has expected
coverage `9/11` under continuity and exchangeability. Interpolated 5% and 95%
sample quantiles therefore must not be interpreted as a literal 90% predictive
interval without a continuous predictive model or an explicit finite-ensemble
construction. Randomized ranks provide a more direct exchangeability check.

We report ordinary ensemble CRPS alongside fair CRPS, using the latter as the
scale-selection objective. Coverage remains a descriptive interval diagnostic
for the ten-member ensemble rather than a literal continuous-distribution
guarantee. CRPS is interpreted as a proper scoring rule for a predictive
distribution [4].

## 4. Cross-fitted global spread calibration

For members `x_k` and their pointwise ensemble mean `x_bar`, the frozen method
forms `x'_k = x_bar + s(x_k - x_bar)`. Values are clipped to `[0,1]` only when
bounded scores are computed. Five folds are assigned deterministically by date
strata. For each holdout fold, `s` minimizes daily case-mean fair CRPS on the
other four folds over the predeclared grid 1.0, 1.1, ..., 4.0. Each fold has 32
selection dates and eight held-out dates; every reported calibrated case is
therefore scored out of fold.

The selected scales (2.6, 2.7, 2.8, 2.8 and 2.6) are interior and stable across
folds. This supports the mechanism interpretation that the learned-joint
ensemble's proper-score deficit is largely a global spread error. Because
clipping can move the bounded ensemble mean, any post-clipping RMSE change is a
secondary numerical consequence, not evidence that the method improves the
unclipped center.

## 5. Experimental protocol

The present mechanism study uses 40 dates from the development split. The
checkpoint, case set, ensemble size, cross-fitting folds, scale grid and scoring
rules are fixed for the analysis reported here. The checkpoint is a legacy
final-epoch checkpoint, so the study is explicitly presented as a
validation-mechanism result rather than an independent generalization result.

The primary deterministic quantity is the generated ensemble mean, which is
compared with the single 3D-Var analysis. Probabilistic scores apply only to the
generative and probabilistic baselines. Any future uncertainty interval must
resample dates or contiguous temporal blocks, never individual pixels. No
uncertainty interval is available from the current aggregate contract.

The reported baselines are the raw ensemble and the score-optimized affine-logit
negative baseline. Other distributional and rank-preserving postprocessors were
not evaluated and are not implied by this paper's evidence. This baseline set is
insufficient for the minimum strong domain/SciML tier; a boundary-aware
distributional comparator and a rank-preserving or conformal comparator remain
scientific blockers rather than editorial tasks.

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

For the learned-joint ensemble, five-fold cross-fitted global spread calibration
selects scales `2.6, 2.7, 2.8, 2.8, 2.6`. Fair CRPS changes from `0.0584905850`
to `0.0556896736`, a `4.79%` reduction, while ordinary CRPS changes from
`0.0621082810` to `0.0611013421`. Spread-skill ratio changes from `0.724066` to
`1.061484`. Coverage diagnostics change from `0.208196` to `0.596277` (50%),
`0.471026` to `0.856292` (80%), `0.505084` to `0.878775` (90%), and `0.525005`
to `0.891464` (95%). The maximum pre-clipping mean-invariance error is
`4.16e-16`. Bounded ensemble-mean RMSE changes from `0.1903366` to `0.1876254`,
but clipping can cause this difference and it is not attributed to center-skill
improvement. The 95% diagnostic remains below its nominal level, so the result
does not establish complete tail calibration.

The subsequent fixed-contract joint audit applies the predeclared
no-compensation gate to all 40 cases. Although the proper-score family passes,
the boundary and spatial/physical families do not. Established-ice Brier score
increases from `0.0569726` to `0.0581999` (2.15%, versus a 1% tolerance).
Exact-one member mass increases from `0.0090419` to `0.1648321` despite zero
truth mass, consistent with clipping inflated anomalies at the upper bound.
Mean IIEE increases from `0.0796004` to `0.0839798` (5.50%), edge disagreement
from `0.0351291` to `0.0366784` (4.41%), and absolute ice-extent error from
`0.0461536` to `0.0507921` (10.05%). Global spread correction is therefore a
negative mechanism baseline: it repairs marginal underdispersion by proper
scores but is not a structure-preserving calibration method.

Figure 1 summarizes this aggregate mechanism result. The dashed references are
descriptive finite-ensemble targets, not confidence bounds; paired date/block
intervals remain unavailable from the present compact contract.

![Aggregate proper-score and reliability diagnostics before and after the
cross-fitted spread correction.](figures/calibration_summary.svg)

**Figure 1.** Aggregate validation diagnostics for the raw learned-joint
ensemble and cross-fitted global spread correction. Lower is better for both
CRPS rows. Coverage and spread-skill references show only the direction of the
reliability change; they do not establish continuous, casewise or fieldwise
coverage for a ten-member ensemble.

## 7. Limitations

The completed evidence uses one legacy checkpoint, one sampling seed, ten
members and 40 development dates. Scale selection and evaluation are separated
case-wise by cross-fitting, but both reuse the same limited development period;
the result is therefore not an independent temporal generalization estimate.
The observation values are M2M truth sampled under real satellite geometry, not
direct satellite SIC retrievals. Coverage is aggregated pointwise and does not
establish casewise or fieldwise coverage. Clipping complicates bounded mean
comparisons, and residual upper-tail undercoverage remains. No independent
comparison with 3D-Var is claimed. Generalization across checkpoints, seeds,
ensemble sizes, regions or observation systems remains unverified.

## 8. Evidence table for the frozen mechanism claim

| Diagnostic | Raw ensemble | Cross-fitted correction | Interpretation |
|---|---:|---:|---|
| Fair CRPS | 0.058491 | 0.055690 | 4.79% reduction; selection objective |
| Ordinary CRPS | 0.062108 | 0.061101 | Secondary proper-score improvement |
| Spread-skill ratio | 0.7241 | 1.0615 | Strong raw underdispersion is largely corrected |
| 50% interval diagnostic | 0.2082 | 0.5963 | Improved, descriptive for ten members |
| 80% interval diagnostic | 0.4710 | 0.8563 | Improved, descriptive for ten members |
| 90% interval diagnostic | 0.5051 | 0.8788 | Meets the predeclared acceptance interval |
| 95% interval diagnostic | 0.5250 | 0.8915 | Improved but residual upper-tail undercoverage remains |
| Pre-clipping center error | 0 | 4.16e-16 | Mean preservation holds to numerical precision |
| Established-ice Brier score | 0.056973 | 0.058200 | 2.15% worse; fails the 1% boundary tolerance |
| Exact-one member mass | 0.00904 | 0.16483 | Spurious upper-boundary atom; truth mass is 0 |
| Mean IIEE | 0.07960 | 0.08398 | 5.50% worse; fails the 2% preservation tolerance |
| Edge disagreement | 0.03513 | 0.03668 | 4.41% worse; fails the 2% preservation tolerance |
| Absolute ice-extent error | 0.04615 | 0.05079 | 10.05% worse; fails the preservation gate |
| Lag-1 analysis-mean variogram error | 0.000936 | 0.000896 | Improves, but cannot compensate for failed mandatory families |
| Lag-2 analysis-mean variogram error | 0.002168 | 0.002066 | Improves, but cannot compensate for failed mandatory families |

The earlier spread-only compact-table contract contains 40 corrected case rows
without corresponding raw case-level fields. The later joint-audit contract is
long-form and contains `target_date` and `method` for 160 rows, including raw
and corrected methods; it can support a trusted server-side paired analysis.
No such paired uncertainty summary is currently present in the publication
worktree, so intervals are neither reconstructed from aggregate means nor
reported. Conditional and independent-period studies are not presented as
contributions of this draft. The completed joint audit checks IIEE, ice
area/extent, edge geometry and variograms; its failed mandatory preservation
criteria prevent a submission-readiness claim even though the aggregate
marginal mechanism result is valid.

## Data and code availability

The manuscript reports only compact validation summaries. Raw ensembles remain
outside the publication worktree. The frozen transform, scale grid, fold sizes,
selected scales, scoring conventions and reconciliation values are documented
in `REPRODUCIBILITY.md`; release location and licensing require author approval.

## Ethics and competing interests

The study uses model fields and satellite-footprint geometry and involves no
human participants or personal data. Authors must supply the final funding and
competing-interest declarations before submission.

## References

1. Ho, J., Jain, A. & Abbeel, P. Denoising Diffusion Probabilistic Models.
   *Advances in Neural Information Processing Systems* **33** (2020).
2. Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M. & Le, M. Flow Matching
   for Generative Modeling. *International Conference on Learning
   Representations* (2023), arXiv:2210.02747.
3. Ferro, C. A. T. Fair scores for ensemble forecasts. *Quarterly Journal of
   the Royal Meteorological Society* **140**, 1917–1923 (2014).
   doi:10.1002/qj.2270.
4. Gneiting, T. & Raftery, A. E. Strictly Proper Scoring Rules, Prediction, and
   Estimation. *Journal of the American Statistical Association* **102**,
   359–378 (2007). doi:10.1198/016214506000001437.
