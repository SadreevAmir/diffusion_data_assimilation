# Flow-matching data assimilation

Conditional flow matching for assimilating a dense sea-ice forecast and sparse
satellite-track observations. The model generates two physical fields:
sea-ice concentration (`siconc`) and sea-ice thickness (`sithic`).

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

## Training

```bash
python -m assim_lib.main \
  --config config/experiments/train_m2m_concat_conditioning_diffusion_balanced_2f.json
```

Each run is written below
`checkpoints/concat_conditioning/m2m_2f_diffusion_balanced_50m/` and contains run
metadata, metric history, regular checkpoints and EMA checkpoints. Use
`ema_best_model.pth` for inference.

## Physical evaluation

```bash
RUN_DIR=checkpoints/concat_conditioning/m2m_2f_diffusion_balanced_50m/run_YYYYMMDD_HHMMSS

python -m assim_lib.evaluate \
  --config config/experiments/evaluate_m2m_concat_conditioning_diffusion_balanced_2f.json \
  --run-dir "$RUN_DIR" \
  --checkpoint-name ema_best_model.pth \
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
