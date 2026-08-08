# Experiment handoff: flow matching vs 3D-Var

Last updated: 2026-08-08. This document is the operational source of truth for
continuing the experiment in a new Codex chat. Read it before running anything.

## Current status

The implementation of the comparison protocol is ready, but the real numerical
experiment described here has not yet been completed. A synthetic end-to-end
test of monthly selection passed, as did Python compilation, Ruff, JSON parsing,
shell syntax checks and `git diff --check`.

At the time this file was written, the repository was on branch
`conditioned-model` at base commit `7b3000c`. The comparison implementation and
documentation were still uncommitted. Always begin with:

```bash
cd /path/to/diffusion_data_assimilation
git status --short
git branch --show-current
git log -1 --oneline
```

Do not discard or overwrite the working tree. It contains the current
comparison pipeline.

## Server safety and execution rules

The user requires every server experiment to run inside Docker.

- Before starting or restarting any experiment on the server, ask the user for
  explicit permission.
- Read-only inspection is allowed, but do not stop, kill or replace containers
  or GPU processes.
- First look for an existing container containing the repository and trained
  run. Create another container only after explicit permission.
- Do not run training or inference directly on the host.
- Do not put passwords, VPN configuration, private host addresses or tokens in
  this repository.
- GPU 0 was intended for the additional experiment container, but its current
  state must be checked again. Never assume it is free and never kill another
  process to free it.

Safe discovery commands on the server host are:

```bash
docker ps --format 'table {{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}'
nvidia-smi
docker exec CONTAINER_NAME nvidia-smi
docker exec CONTAINER_NAME bash -lc 'cd /home && git status --short'
```

The Docker image uses `/home` as the repository working directory. The compose
file mounts the repository at `/home`, the data root read-only at `/mnt`, and
selects a GPU using `GPU_DEVICE_ID`.

## Scientific question

Compare the trained conditional flow/diffusion data-assimilation model with the
colleagues' model-to-model 3D-Var experiment under the same data and evaluation
conditions. The flow model generates an ensemble. The deterministic comparison
uses the ensemble-mean analysis; probabilistic ensemble metrics are additional
flow-only diagnostics.

The primary flow procedure uses a different preselected CFG balance for every
calendar month. Each balance is selected exclusively on the corresponding
month of validation year 2022 and is then frozen for the same month of test year
2023.

## Exact temporal protocol

The enforced split is:

```text
role        background/input dates       target dates
train       2015-01-01 .. 2020-12-31     2016-01-01 .. 2021-12-31
validation  2021-01-01 .. 2021-12-31     2022-01-01 .. 2022-12-31
test        2022-01-01 .. 2022-07-19     2023-01-01 .. 2023-07-19
```

The test contains exactly 200 consecutive target days. The input and target are
365 days apart. `assim_lib.data.validate_temporal_split_protocol` rejects a
configuration that claims this protocol but changes any boundary.

## Exact information available to both methods

The strict comparable protocol is `field_protocol=siconc-only`:

- target and scored field: `siconc`;
- target hour: 23;
- background: corresponding M2M forecast from the previous year;
- observations: M2M `siconc` values under real SRAL footprints;
- conditioning track dates: target day `d`, `d-1` and `d-2`;
- SRAL footprint transform: rule 1 (`sral_transform_index=1`);
- overlapping tracks: the newest available day wins;
- test observations never use synthetic tracks;
- `sithic` background values, observations and masks are explicitly zeroed in
  strict comparison mode;
- the network still outputs two fields, but only `siconc` is evaluated.

Hiding `sithic` is deliberate: the colleagues' 3D-Var experiment assimilates
only `siconc`. `field_protocol=native-two-field` is available only as a separate
flow ablation and must not be mixed into the strict comparison table.

The code verifies the `d,d-1,d-2` rule-1 SRAL union for every test case and
raises an error if a non-`siconc` condition leaks into strict mode.

## Model and checkpoint policy

Training configuration:

- task: `concat_conditioning_diffusion_balanced_50m_2f_1y`;
- objective: diffusion;
- output channels: 2 (`siconc`, `sithic`);
- 40 epochs, batch size 16, BF16 training;
- beta timestep sampler with parameters `[1.0, 1.5]`;
- EMA decay: 0.999;
- inference start: noise with level 1.0;
- default sampler: `dopri5`, 25 steps, `rtol=1e-5`, `atol=1e-6`;
- observations are not hard-enforced during sampling;
- observation and smoothness auxiliary loss weights are zero.

The expected trained run is below
`checkpoints/concat_conditioning/m2m_2f_diffusion_balanced_50m/`. Resolve the
exact run directory from its `metadata.json`; do not guess from the directory
name alone.

Always use `CHECKPOINT_NAME=auto`:

- a new run trained with the corrected disjoint split resolves to
  `ema_best_model.pth`, selected on validation-2022;
- the existing recognized legacy run used validation-2023 and therefore
  resolves to `ema_last_model.pth`;
- using a `best` checkpoint from that legacy run is rejected because it would
  leak test-year information;
- an unknown training split is rejected.

The current legacy final-epoch model is acceptable for the present preliminary
comparison. A future publication training should use the corrected split and
can safely use its validation-selected EMA best checkpoint.

## CFG balance

The experiment uses independent background/track CFG:

```text
v = v_none
    + (1 - w_track) * (v_background - v_none)
    + w_track       * (v_track - v_none)
```

The coefficient of the branch conditioned jointly on background and track is
exactly zero. Thus `background_weight = 1 - track_weight`.

The candidate track weights are:

```text
0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0
```

Monthly selection is mandatory in the default launcher:

- January weight: selected on January 2022, used on January 2023;
- February weight: selected on February 2022, used on February 2023;
- continue through July;
- validation uses every third day of the corresponding month;
- per-month validation case counts are `11, 10, 11, 10, 11, 10, 7`;
- validation selection uses 5 ensemble members;
- selection criterion is the minimum `full/analysis_mean_rmse`;
- all seven validation searches finish before the first test run starts;
- final test uses 15 members on all 200 daily cases.

The single global-weight selector remains available only as an explicit
baseline. Do not use it accidentally: `scripts/run_cfg_selection.sh` defaults
to the monthly configuration.

## Important repository files

### Core implementation

- `assim_lib/data.py`: dataset construction, exact split enforcement, real and
  synthetic track creation, and `d,d-1,d-2` observation assembly.
- `assim_lib/sampler.py`: model sampling and independent CFG formula.
- `assim_lib/compare_3dvar.py`: strict 3D-Var-compatible inference, protocol
  assertions, metrics, provenance and progress tracking.
- `assim_lib/select_cfg_balance.py`: global or monthly validation-only CFG
  selection followed by a frozen test run.
- `assim_lib/model_io.py`: safe automatic checkpoint resolution.
- `assim_lib/metrics.py`: ensemble CRPS and interval coverage.
- `assim_lib/cfg_experiments.py`: short December visual diagnostics and
  synthetic track-count sweeps. These are diagnostics, not the publication
  comparison.

### Data, model and experiment configs

- `config/data/m2m_2f_1y.json`: dataset paths, fields, normalization and exact
  train/validation/test split.
- `config/methods/concat_conditioning_diffusion_balanced_2f.json`: model,
  training and sampler parameters.
- `config/experiments/train_m2m_concat_conditioning_diffusion_balanced_2f.json`:
  future training entry point.
- `config/experiments/experiment_m2m_flow_modes.json`: reusable smoke,
  validation and test modes.
- `config/experiments/select_cfg_balance_monthly_validation_then_test.json`:
  default monthly balance selection and final test.
- `config/experiments/select_cfg_balance_validation_then_test.json`: optional
  single global balance baseline.
- `config/experiments/compare_m2m_3dvar_main_200d.json`: fixed learned-full-
  condition 200-day comparison.
- `config/experiments/compare_m2m_3dvar_dev_40d.json`: old short validation
  development preset.

### Launchers

- `scripts/run_flow_experiment.sh`: run one named mode.
- `scripts/run_cfg_selection.sh`: default complete monthly `validation -> test`
  workflow.
- `scripts/run_3dvar_comparison_200d.sh`: fixed-condition 3D-Var-compatible
  run.
- `scripts/run_cfg_december_experiments.sh`: short December visual diagnostics.

## Preflight inside the existing container

Run these read-only checks before asking permission to launch an experiment:

```bash
cd /home
nvidia-smi
git status --short

RUN_DIR=/path/to/the/trained/run
test -f "$RUN_DIR/metadata.json"
python3 -m json.tool "$RUN_DIR/metadata.json" >/dev/null
ls -lh "$RUN_DIR"/ema_last_model.pth "$RUN_DIR"/ema_best_model.pth 2>/dev/null || true

test -d /mnt/sciml/data_assimilation/da_arctic_2015-2024_v0.1
test -d /mnt/sciml/data_assimilation/sral_si/sral_new_format
test -f /mnt/sciml/data_assimilation/land_mask.npy
```

Inspect the stored split before deciding which checkpoint `auto` will use:

```bash
python3 - "$RUN_DIR/metadata.json" <<'PY'
import json
import sys

metadata = json.load(open(sys.argv[1]))
print(json.dumps(metadata.get("data_config", {}), indent=2))
PY
```

Do not launch if the run metadata or required checkpoint is missing.

## Required launch sequence

### 1. One-day smoke test

Ask the user for permission, then run inside the container:

```bash
cd /home

RUN_DIR=/path/to/the/trained/run \
CUDA_DEVICE=0 \
OUTPUT_DIR=/path/to/new/output/smoke_strict_full \
bash scripts/run_flow_experiment.sh smoke_strict_full
```

Success criteria:

- exit code zero;
- `run_status.json` says `completed`;
- one target date, `2023-01-01`;
- no protocol assertion or missing-data error;
- both `full` and `track_imitation` rows exist in `per_case_metrics.csv`;
- `metadata.json` records the resolved checkpoint and strict `siconc-only`
  protocol.

### 2. Complete monthly validation selection and frozen test

Ask the user again before this main launch. Run inside the container:

```bash
cd /home

RUN_DIR=/path/to/the/trained/run \
CUDA_DEVICE=0 \
OUTPUT_DIR=/path/to/new/output/cfg_selection_monthly \
bash scripts/run_cfg_selection.sh
```

No `CONFIG` override is needed: the launcher defaults to monthly selection.
The output directory must be new or empty so results cannot be mixed.

For a detachable run from an interactive shell inside the container:

```bash
cd /home
mkdir -p /path/to/new/output

nohup env \
  RUN_DIR=/path/to/the/trained/run \
  CUDA_DEVICE=0 \
  OUTPUT_DIR=/path/to/new/output/cfg_selection_monthly \
  bash scripts/run_cfg_selection.sh \
  > /path/to/new/output/cfg_selection_monthly.log 2>&1 &
```

Record the PID printed by the shell. Do not start a second copy if the first is
still running.

### 3. Optional fixed learned-full-condition baseline

This is useful for an ablation against the learned joint branch, but it is not
a replacement for the requested monthly CFG experiment. Ask permission first:

```bash
RUN_DIR=/path/to/the/trained/run \
CUDA_DEVICE=0 \
OUTPUT_DIR=/path/to/new/output/full_condition_main_200d \
bash scripts/run_3dvar_comparison_200d.sh
```

### 4. Optional global CFG baseline

The global balance is no longer the default. Request it explicitly:

```bash
RUN_DIR=/path/to/the/trained/run \
CUDA_DEVICE=0 \
CONFIG=config/experiments/select_cfg_balance_validation_then_test.json \
OUTPUT_DIR=/path/to/new/output/cfg_selection_global \
bash scripts/run_cfg_selection.sh
```

## Monitoring

The monthly selector writes its top-level state after every candidate weight:

```bash
watch -n 30 cat /path/to/new/output/cfg_selection_monthly/selection.json
tail -f /path/to/new/output/cfg_selection_monthly.log
```

Each active validation or test subrun also has `run_status.json` with completed
cases, elapsed time and ETA. Locate the latest without changing anything:

```bash
find /path/to/new/output/cfg_selection_monthly -name run_status.json -print
```

Normal phase order is:

```text
running_monthly_validation
running_monthly_test
completed
```

If `selection.json` says `failed`, read its `error` field and the log. Do not
delete a partial output and silently restart; preserve it and use a new output
directory after fixing the cause.

## Expected monthly output

```text
cfg_selection_monthly/
  selection.json
  monthly_selected_weights.csv
  validation_monthly/
    month_01/track_weight_*/...
    ...
    month_07/track_weight_*/...
  test_monthly/
    month_01/...
    ...
    month_07/...
  test_combined/
    per_case_metrics.csv
    aggregate_case_mean_metrics.csv
    aggregate_case_mean_metrics.json
```

Required final invariants:

- `selection.json.status == "completed"`;
- seven selected monthly weights exist;
- every selection comes from validation-2022;
- `test_combined` contains exactly 200 unique dates;
- each date has exactly one `full` and one `track_imitation` row;
- the first date is `2023-01-01`, the last is `2023-07-19`;
- combined metrics are recomputed over 200 daily cases, not averaged from seven
  monthly averages;
- metadata reports `dopri5`, 25 steps, ensemble size 15, hour 23,
  `siconc-only`, and the resolved checkpoint.

## Metrics and comparison table

Metrics are calculated separately over:

- `full`: all valid ocean pixels;
- `track_imitation`: same-day dense M2M truth restricted to the following day's
  rule-1 SRAL footprint, matching the colleagues' sparse validation idea.

The deterministic metrics are:

- MAE, RMSE, MSE and IIEE for the background;
- MAE, RMSE, MSE and IIEE for the ensemble-mean analysis;
- analysis-minus-background deltas;
- minimum, mean and maximum error across individual ensemble members.

The final deterministic Flow-vs-3D-Var point comparison must use
`analysis_mean_*`, because 3D-Var produces one analysis and Flow produces an
ensemble. Do not compare 3D-Var with the best Flow ensemble member.

Flow-only probabilistic diagnostics are:

- CRPS;
- ensemble spread;
- spread-skill ratio;
- 50%, 80% and 90% interval coverage.

Metrics are first computed per day and then averaged arithmetically over days,
matching the colleagues' aggregation. The edge threshold for IIEE is 0.15.

The colleagues' reference recorded in output metadata is the `LiubovAnt/3d_var`
model-to-model `main_200d` experiment, config
`config/assimilation/model2model/main_200d/var_100_main_200d.yaml`, data config
`config/data/model/var_200d.yaml`, and task
`3d_var_baseline_var_100_main_200d`.

## Ensemble calibration: designed but not implemented

The repository currently diagnoses calibration but does not transform the
ensemble. Do not describe the current results as calibrated.

The proposed future calibration is performed only after monthly CFG weights
are frozen. On validation-2022, fit bounded logit-space affine mean correction
and spread scaling:

```text
z_k      = logit(clip(x_k, eps, 1-eps))
z_mean   = mean_k(z_k)
z_cal_k  = a_month + b_month * z_mean + s_month * (z_k - z_mean)
x_cal_k  = sigmoid(z_cal_k)
```

Fit calibration parameters on validation only, primarily by daily-mean CRPS,
regularize monthly parameters toward global parameters, freeze them, and apply
them once to test-2023. Save both raw and calibrated results. Add randomized
rank histograms and Brier/reliability metrics for the event `siconc > 0.15`.

Calibration must use the same ensemble size as the test. This is a future task
and must not be silently added to the present uncalibrated comparison.

## Approximate compute budget

The monthly validation grid evaluates 70 date/weight cases for each of 9
weights with 5 ensemble members: 3,150 generated validation members. The final
test generates 200 x 15 = 3,000 members. Total planned generation is therefore
6,150 ensemble members, excluding small smoke checks.

This was designed to remain in the same approximate L40 budget as the earlier
five-hour experiment, but the real duration must be estimated from the first
completed subruns. Report the observed ETA rather than promising a fixed time.

## Validation already performed locally

The following checks passed before this handoff was written:

- Ruff over `assim_lib`;
- Python bytecode compilation;
- parsing of every JSON config;
- `bash -n` over every launcher;
- `git diff --check`;
- fake end-to-end monthly orchestration with month-dependent optimal weights;
- assertion that all 63 validation subprocesses precede all 7 test subprocesses;
- assertion that validation uses 5 members and test uses 15;
- assertion that the combined test contains exactly 200 dates and both regions.

These checks do not replace the real one-day integration test against mounted
M2M/SRAL data and the trained checkpoint.

## New-chat quick start

In a new chat, give Codex this instruction:

> Read `EXPERIMENT_HANDOFF.md` and inspect the current working tree. Preserve all
> existing changes. We only run on the server inside the existing Docker
> container. Do not kill or replace anything. Inspect the container, GPU,
> dataset mounts, run metadata and checkpoint read-only. Before every actual
> experiment launch, ask me for permission. Then run the one-day smoke test; if
> it passes and I approve, start the default monthly CFG validation-to-test
> workflow detached and report its measured progress and ETA.
