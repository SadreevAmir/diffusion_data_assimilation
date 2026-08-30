# Calendar-residual CFM: frozen evaluation contract

Status: PREPARED_NOT_EXECUTED

## Scientific question

The candidate is the unchanged final-EMA conditional flow trained for the
calendar-year residual. The experiment asks whether modelling the stochastic
field increment directly produces an absolutely calibrated predictive
distribution while retaining useful proper scores and physical field structure.
It is not a post-processing or member-reweighting experiment.

## Immutable sampling envelope

- Source training experiment: `siconc_calendar_residual_cfm_training_retry1`.
- Evaluation config: `config/experiments/evaluate_siconc_calendar_residual_cfm.json`.
- Use the final EMA checkpoint only; evaluation scores cannot select a checkpoint.
- Use `split=valid`, 40 targets at five-day stride, `ensemble_size=10`, seed 1234,
  and the frozen 25-step DOPRI5 solver (`rtol=1e-5`, `atol=1e-6`).
- Sample `residual`, add it to the exact calendar-year background, and clip only
  the resulting physical concentration to `[0,1]`.
- Do not tune CFG, latent temperature, solver, checkpoint, seed, member count,
  observation realization or clipping after inspecting results.

Sampling keeps ensembles server-side because the spatial gate needs member
fields. Retrieval is `summary_only`; final paper production may request only the
named rank-histogram and joint-gate figure payloads.

## Falsifiable prediction and decision

Primary prediction: the date-balanced randomized rank histogram satisfies the
pre-existing absolute uniformity, centering, range-coverage and inner-coverage
thresholds. This is necessary but not sufficient. `overall_eligible=true`
requires the unchanged proper-score, finite-ensemble reliability,
truth-referenced boundary, spatial/physical and operational families to all
pass without compensation.

Stop/go rule: proceed only when all five families are true and their conjunction
is literally `overall_eligible=true`. Any false family rejects the candidate as
the paper's calibrated predictive distribution. A reliability failure falsifies
the residual-distribution calibration claim; a proper-score failure means the
learned increment law is not predictively useful; a boundary or spatial failure
attributes rejection to physical-field reconstruction rather than rank quality.

## Compact outputs required from the trusted gate

The gate must emit aggregate metrics, per-date compact metrics, paired
uncertainty, and a gate decision. The aggregate includes randomized-rank bins
and absolute decisions, attainable range and inner coverage, `analysis_crps`,
`analysis_fair_crps`, `analysis_mean_rmse`, energy score, truth-referenced
boundary diagnostics, Brier scores, IIEE, edge, area, extent, and
mean/member/local variograms. The paired date bootstrap is inferential; the
non-overlapping four-date-block interval is temporal sensitivity only.
