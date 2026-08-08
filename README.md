# Flow-matching data assimilation

Conditional flow matching for assimilating a dense sea-ice forecast and sparse
satellite-track observations. The model generates two physical fields:
sea-ice concentration (`siconc`) and sea-ice thickness (`sithic`).

For the exact 3D-Var comparison protocol, current implementation status and
server launch sequence, read [`EXPERIMENT_HANDOFF.md`](EXPERIMENT_HANDOFF.md).

The repository contains only the maintained training and inference pipeline:

```text
assim_lib/           data, training, sampling and evaluation code
config/data/         dataset configuration
config/methods/      model and sampler configuration
config/experiments/  training and evaluation entry points
scripts/             reproducible inference wrappers
```

Experiment files recursively merge their data and method configurations. Relative
paths are resolved from the experiment file location.

## Environment

Python 3.10 and CUDA 12.1 are used by the provided image. For a native environment,
install PyTorch, torchvision and xformers for the target CUDA platform first, then:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

The real-data pipeline expects these external paths:

```text
/mnt/sciml/data_assimilation/da_arctic_2015-2024_v0.1
/mnt/sciml/data_assimilation/sral_si/sral_new_format
/mnt/sciml/data_assimilation/land_mask.npy
```

ClearML is optional. Copy `.env.example` to `.env` and provide credentials only
when tracking is enabled; never commit `.env`.

## Docker

Build and start the configured training run:

```bash
GPU_DEVICE_ID=0 docker compose up --build
```

For an interactive container:

```bash
GPU_DEVICE_ID=0 docker compose run --rm --entrypoint bash flow_matching
```

The repository is mounted at `/home` and the external data root at `/mnt`.

## Normalization

Compute training-set statistics before a new training run:

```bash
python -m assim_lib.m2m_stats \
  --config config/data/m2m_2f_1y.json \
  --split train \
  --hour-mode auto \
  --output outputs/m2m_2f_1y_train_stats.json
```

Review the output before adding `--update-config`. A trained run stores the exact
normalization values in `metadata.json`; inference uses those checkpoint values.

## Temporal split

The data split follows the `3d_var` model-to-model suite and is enforced at
runtime:

```text
role        background/input dates       target dates
train       2015-01-01 .. 2020-12-31     2016-01-01 .. 2021-12-31
validation  2021-01-01 .. 2021-12-31     2022-01-01 .. 2022-12-31
test        2022-01-01 .. 2022-07-19     2023-01-01 .. 2023-07-19
```

The one-year input/target offset is intrinsic to the flow experiment. Thus the
train split uses the same raw 2015-2021 period that the colleague suite labels
as training, 2022 is used only for validation and hyperparameter selection, and
the 200 days in 2023 are used only once for the final test.

## Training

```bash
python -m assim_lib.main \
  --config config/experiments/train_m2m_concat_conditioning_diffusion_balanced_2f.json
```

Each run is written below
`checkpoints/concat_conditioning/m2m_2f_diffusion_balanced_50m/` and contains run
metadata, metric history, regular checkpoints and EMA checkpoints. Use
automatic checkpoint selection for inference: new runs resolve to
`ema_best_model.pth`, while the recognized legacy run resolves to
`ema_last_model.pth`.

Future runs use the disjoint split above. For the recognized legacy run, which
used 2023 as validation, the benchmark permits only `ema_last_model.pth`: its
fixed final-epoch weights were not selected by validation. A legacy `best`
checkpoint is rejected. The benchmark reads the stored `data_config` and makes
this choice automatically when `CHECKPOINT_NAME=auto`.

## Physical evaluation

```bash
RUN_DIR=checkpoints/concat_conditioning/m2m_2f_diffusion_balanced_50m/run_YYYYMMDD_HHMMSS

python -m assim_lib.evaluate \
  --config config/experiments/evaluate_m2m_concat_conditioning_diffusion_balanced_2f.json \
  --run-dir "$RUN_DIR" \
  --checkpoint-name auto \
  --no-clearml
```

Evaluation produces physical-unit aggregate and per-case metrics, metadata, and
optionally compressed ensemble arrays. CLI arguments override the evaluation JSON;
use `python -m assim_lib.evaluate --help` for the full interface.

## Background/track CFG inference

The controlled CFG experiments combine only the background-only and track-only
vector fields:

```text
v = w_background * v_background_only + w_track * v_track_only
w_background = 1 - w_track
```

The fully conditioned and unconditional coefficients are exactly zero.

Short visual December comparison:

These diagnostics use December 2022 from the validation split, never the 2023
test split.

```bash
python -m assim_lib.cfg_experiments december \
  --config config/experiments/evaluate_m2m_concat_conditioning_diffusion_balanced_2f.json \
  --run-dir "$RUN_DIR" \
  --output-dir "$RUN_DIR/evaluation/cfg_december_visual"
```

Synthetic track-count and balance sweep:

```bash
python -m assim_lib.cfg_experiments track-sweep \
  --config config/experiments/evaluate_m2m_concat_conditioning_diffusion_balanced_2f.json \
  --run-dir "$RUN_DIR" \
  --min-tracks 1 \
  --max-tracks 10 \
  --track-weights 0.0 0.25 0.5 0.75 1.0 \
  --output-dir "$RUN_DIR/evaluation/cfg_track_sweep"
```

For the complete L40-oriented December run:

```bash
RUN_DIR="$RUN_DIR" \
OUTPUT_ROOT="$RUN_DIR/evaluation/cfg_december_l40" \
CUDA_DEVICE=0 \
bash scripts/run_cfg_december_experiments.sh
```

All inference output directories must be new or empty. Check command-specific
options with:

```bash
python -m assim_lib.cfg_experiments december --help
python -m assim_lib.cfg_experiments track-sweep --help
```

## Exact 3D-Var comparison benchmark

`assim_lib.compare_3dvar` reproduces the information and evaluation protocol from
the `LiubovAnt/3d_var` model-to-model `main_200d` experiment. It is intentionally
separate from the December CFG diagnostics.

The primary comparison fixes all of the following:

```text
target dates          2023-01-01 through 2023-07-19 (200 consecutive days)
background dates      corresponding 2022 dates
target hour           23
assimilated field     siconc only
observation geometry  real rule-1 SRAL footprints
observation values    M2M siconc from each footprint's own day at hour 23
track history         target day, target-1 day, target-2 days
track merge           newest value wins at overlapping pixels
dense validation      same-day M2M siconc over the ocean
sparse validation     same-day truth on the next day's SRAL footprint
aggregation           arithmetic mean of per-day metrics
edge threshold        siconc > 0.15
```

In the default `siconc-only` protocol, both observation and background
information for `sithic` are hidden with zero values and zero condition masks.
This matches the information available to the single-field 3D-Var baseline.
The model still produces two output channels, but only `siconc` is scored.

The publication comparison defaults to automatic safe checkpoint selection:
new runs use `ema_best_model.pth`, selected exclusively by 2022 validation;
the recognized legacy run uses `ema_last_model.pth`, because its `best` was
selected on overlapping 2023 data. The 2023 test is never used for CFG-weight
selection.

Run a one-case integration check before starting the benchmark:

```bash
RUN_DIR=/path/to/run \
START_DATE=2023-01-01 \
END_DATE=2023-01-01 \
EXPECTED_NUM_CASES=1 \
ENSEMBLE_SIZE=3 \
OUTPUT_DIR=/path/to/new/output/pilot_1d \
bash scripts/run_3dvar_comparison_200d.sh
```

Tune one independent-CFG balance only on the separate 2022 development set.
For example, evaluate a small predeclared grid:

```bash
for TRACK_WEIGHT in 0.25 0.5 0.75; do
  RUN_DIR=/path/to/run \
  CONFIG=config/experiments/compare_m2m_3dvar_dev_40d.json \
  CONDITIONING_MODE=independent-balance \
  TRACK_WEIGHT="$TRACK_WEIGHT" \
  OUTPUT_DIR="/path/to/new/output/development_2022_w${TRACK_WEIGHT}" \
  bash scripts/run_3dvar_comparison_200d.sh
done
```

Here `TRACK_WEIGHT=w` implements
`v_none + (1-w)(v_background-v_none) + w(v_track-v_none)`: the coefficient of
the joint background-and-track branch is exactly zero. Freeze the selected
weight before opening the 2023 benchmark results.

Run the frozen 200-day comparison:

```bash
RUN_DIR=/path/to/run \
OUTPUT_DIR=/path/to/new/output/main_200d \
CUDA_DEVICE=0 \
bash scripts/run_3dvar_comparison_200d.sh
```

The primary run uses the learned full-conditioned branch. To evaluate the one
independent-CFG balance frozen on the 2022 development set:

```bash
RUN_DIR=/path/to/run \
CONDITIONING_MODE=independent-balance \
TRACK_WEIGHT=0.5 \
OUTPUT_DIR=/path/to/new/output/main_200d_cfg_050 \
bash scripts/run_3dvar_comparison_200d.sh
```

Never select the best `TRACK_WEIGHT`, sampler, checkpoint or seed from the 200
benchmark days. Run the frozen configuration once.

## Flow experiment modes

The preset config
`config/experiments/experiment_m2m_flow_modes.json` provides reproducible modes
that share the same dates, masks, metrics and automatic checkpoint policy:

```text
smoke_strict_full           one test day, 3 members, 8 Euler steps
validation_strict_full      40 validation days, learned joint condition
validation_background_only  40 validation days, background branch only
validation_balanced         40 validation days, independent CFG with w_track=0.5
validation_balance_selection 40 validation dates spanning January-July 2022
validation_track_only       40 validation days, track branch only
test_strict_full            official 200-day comparable flow result
test_strict_cfg             200-day strict result with a frozen CFG weight
test_native_two_field_full  200-day two-field ablation; not 3D-Var-comparable
```

Run any preset inside the container:

```bash
RUN_DIR=/path/to/run \
bash scripts/run_flow_experiment.sh smoke_strict_full

RUN_DIR=/path/to/run \
bash scripts/run_flow_experiment.sh validation_background_only

RUN_DIR=/path/to/run \
bash scripts/run_flow_experiment.sh test_strict_full
```

Tune weights only on validation by overriding the preset value:

```bash
for TRACK_WEIGHT in 0.25 0.5 0.75; do
  RUN_DIR=/path/to/run \
  TRACK_WEIGHT="$TRACK_WEIGHT" \
  OUTPUT_DIR="/path/to/output/validation_w${TRACK_WEIGHT}" \
  bash scripts/run_flow_experiment.sh validation_balanced
done
```

After freezing the selected value, run it once on test:

```bash
RUN_DIR=/path/to/run \
TRACK_WEIGHT=0.5 \
OUTPUT_DIR=/path/to/output/test_cfg_050 \
bash scripts/run_flow_experiment.sh test_strict_cfg
```

Environment variables such as `ENSEMBLE_SIZE`, `SAMPLE_BATCH_SIZE`,
`NUM_TIMESTEPS`, `METHOD`, `SAVE_ENSEMBLES`, `FIELD_PROTOCOL` and
`CONDITIONING_MODE` are explicit overrides; omitted values come from the preset.

## Automatic CFG-balance selection

`config/experiments/select_cfg_balance_validation_then_test.json` defines a
leakage-free two-stage experiment:

1. evaluate every candidate track weight on 40 validation dates sampled every
   five days across January-July 2022;
2. select one weight using the configured validation metric;
3. run exactly that weight once on the 200-day 2023 test set;
4. save every deterministic, member-wise and probabilistic test metric.

The default grid is `0.0, 0.125, ..., 1.0`. It minimizes
`full/analysis_mean_rmse`, the RMSE of the ensemble-mean field. Run it with:

```bash
RUN_DIR=/path/to/run \
CUDA_DEVICE=0 \
CONFIG=config/experiments/select_cfg_balance_validation_then_test.json \
OUTPUT_DIR=/path/to/output/cfg_selection \
bash scripts/run_cfg_selection.sh
```

Override the grid or selection criterion without changing the config:

```bash
RUN_DIR=/path/to/run \
TRACK_WEIGHTS="0.0 0.25 0.5 0.75 1.0" \
SELECTION_REGION=track_imitation \
SELECTION_METRIC=analysis_crps \
DIRECTION=minimize \
OUTPUT_DIR=/path/to/output/cfg_selection_crps \
bash scripts/run_cfg_selection.sh
```

Use `analysis_mean_rmse` for the primary deterministic comparison with 3D-Var,
or `analysis_crps` when selecting for probabilistic ensemble quality. Do not
change the criterion after inspecting test results.

The root output contains `selection.json`, updated after every validation
weight. `validation/track_weight_*/` contains the complete validation metrics;
`test_selected/` contains all metrics for the single selected test run.

### Month-specific CFG balance

Sea-ice errors and the relative value of tracks can be seasonal. The monthly
configuration therefore selects seven independent balances without looking at
test metrics: January is selected on January 2022 and used on January 2023,
February on February 2022/2023, and so on through 19 July. All seven validation
searches finish before the first 2023 test case is evaluated.

```bash
RUN_DIR=/path/to/run \
CUDA_DEVICE=0 \
OUTPUT_DIR=/path/to/output/cfg_selection_monthly \
bash scripts/run_cfg_selection.sh
```

Month-specific selection is the launcher's default. The global configuration
must be requested explicitly, as in the preceding section.

The validation grid uses every third day in each 2022 month and five ensemble
members, while the final daily 200-case test uses 15 members. This keeps the
selection affordable while distributing its dates across all seven seasons.
The selected weight for every month is recorded in `selection.json`.
The compact `monthly_selected_weights.csv` table contains the same choices and
their validation scores.
`test_monthly/month_*/` contains the individual monthly runs, and
`test_combined/aggregate_case_mean_metrics.csv` recomputes the final metrics
over all 200 daily cases (it does not average seven monthly averages).

Use the single globally selected weight above as the primary fixed-method
comparison with 3D-Var. Treat month-specific weights as a pre-declared seasonal
ablation; otherwise the extra seven hyperparameters make it a less direct
comparison.

For a non-comparable native multivariate ablation, expose both background and
track fields with `FIELD_PROTOCOL=native-two-field`. Keep this result separate
from the primary `siconc-only` table.

The output directory contains:

```text
per_case_metrics.csv             daily background/analysis metrics
aggregate_case_mean_metrics.csv  colleague-compatible daily means
aggregate_case_mean_metrics.json same aggregates as JSON
cases.json                       dates, three track days, file lists and mask hashes
metadata.json                    checkpoint and complete protocol provenance
run_status.json                  progress, elapsed time and ETA
samples/                         optional ensembles when SAVE_ENSEMBLES=true
```

Both full-ocean and `track_imitation` rows contain MAE, RMSE, MSE and IIEE for
the background and ensemble mean, plus `*_delta = analysis - background`. For
each metric they also contain `analysis_member_*_min`, `*_mean` and `*_max`:
the best, typical and worst individual ensemble-member errors. These member
summaries are separate from the metric of the ensemble-mean field, which is the
direct point-estimate comparison with deterministic 3D-Var. CRPS, spread and
interval coverage are saved as flow-only probabilistic diagnostics and should
not be presented as 3D-Var metrics.

Monitor a detached run with:

```bash
watch -n 30 cat /path/to/output/run_status.json
tail -f /path/to/log/file.log
```
