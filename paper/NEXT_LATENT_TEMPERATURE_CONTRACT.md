# Frozen latent-temperature calibration contract

Status: frozen before implementation and before inspecting any candidate result.

## Scientific question

The existing ten-member learned-joint ensemble is underdispersed
(`corrected_spread_skill_ratio = 0.759407`). Previous output-space calibration
improved marginal reliability only by damaging boundary behaviour or member
spatial structure. This experiment changes the generative uncertainty upstream:
it samples the fixed final EMA model from a wider Gaussian latent distribution.

## Fixed sampling experiment

- mode: `validation_latent_temperature_sampling`
- experiment id: `latent_temperature_1p30_sampling_valid`
- source experiment: `joint_full_condition_validation_2022`
- split: validation only
- dates: exactly 40 cases from `2022-01-01` through `2022-07-15`, every five days
- checkpoint: fixed `ema_last_model.pth`, SHA-256
  `cd73cedc97a9f19d15c70ba31d28d3248a77dc6ef568edf8793326763810fdfe`
- checkpoint selection: fixed final epoch; neither 2023 nor a legacy best
  checkpoint is used
- ensemble size: 10
- base seed: 1234; member seed is
  `1234 + case_index * 10 + member_index`
- initial latent scale: exactly `1.30`
- operation: draw the usual standard-normal initial tensor, record its canonical
  float32 hash, multiply that tensor by `1.30` before the ODE solve, and record
  the scaled canonical float32 hash
- sampler: `dopri5`, 25 time points, `rtol=1e-5`, `atol=1e-6`, float32
- conditioning: unchanged learned-joint full conditioning, CFG disabled
- physical conversion and `[0, 1]` clipping: unchanged from the frozen baseline
- no training, checkpoint mixing, member fallback, rejection sampling, tuning, or
  use of test-2023

The value `1.30` is fixed from the pre-existing raw diagnostic
`1 / 0.759407 = 1.317`, rounded before candidate sampling. No second temperature
will be selected after seeing this result.

Raw arrays stay in the server result directory. Retrieval is restricted to the
four compact files `run_status.json`, `aggregate_case_mean_metrics.json`,
`per_case_metrics.csv`, and `metadata.json`.

## Fixed gate experiment

- mode: `validation_latent_temperature_gate`
- experiment id: `latent_temperature_1p30_gate_valid`
- sole parameter: `source_experiment=latent_temperature_1p30_sampling_valid`
- dependency: exactly `latent_temperature_1p30_sampling_valid`
- comparator: the frozen raw `joint_full_condition_validation_2022` ensemble
- evaluation: the existing full no-compensation ten-member gate, including
  ordinary and fair CRPS, paired date and four-case-block uncertainty,
  randomized-rank discrepancy, member-range and inner-order coverage, exact-zero
  and exact-one masses, presence and established-ice Brier scores, energy and
  local variogram scores, member and mean variograms at lags 1/2/4, RMSE, IIEE,
  edge, area, and extent errors
- operational admission additionally requires the exact checkpoint identity,
  `1.30` scale, seed schedule, ten finite members for all 40 dates, matching base
  noise hashes for the frozen seed schedule, distinct scaled hashes, no fallback,
  and no test data

Success requires `gate.overall_eligible=true`. In particular, fair CRPS must
improve by at least 3%, its paired-date interval must support improvement,
ordinary CRPS must not worsen by more than 1%, rank discrepancy must improve by
at least 20%, boundary diagnostics must not worsen beyond their existing
tolerances, and every spatial/physical and operational sub-gate must pass. No
family may compensate for failure in another family.

## Prediction and stopping rule

Prediction: a modest upstream increase in latent variance will widen coherent
model-generated alternatives and improve fair CRPS and finite-ensemble
reliability while preserving substantially more spatial structure than
output-space spread transforms.

If any mandatory family fails, the fixed `1.30` latent-temperature mechanism is
rejected. The result may be reported as a negative mechanistic ablation, but it
does not authorize post-result temperature tuning. The next experiment must use
a mechanistically distinct uncertainty source rather than an adjacent latent
scale.
