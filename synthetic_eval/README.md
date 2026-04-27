# Synthetic Ensemble Evaluation

This package evaluates posterior/analysis ensembles for synthetic sea-ice data assimilation experiments.

The benchmark uses full `.npy` sea-ice fields as `x_true`, generates sparse synthetic observations
`y = H(x_true) + noise`, runs a replaceable ensemble runner, and scores the returned analysis ensemble
directly against the full state-space truth.

## Smoke Test

```bash
python -m synthetic_eval.cli \
  --input-dir /mnt/sciml/a.sadreev/sea_ice_data/valid \
  --output-dir /tmp/sea_ice_synth_eval_smoke \
  --runner dummy \
  --ensemble-size 4 \
  --mask-types random,block \
  --densities 0.05 \
  --noise-levels 0.0 \
  --max-timesteps 2
```

The dummy runner returns `x_true + synthetic noise`; it is only for testing the evaluation pipeline.

## Concat Model Runner

```bash
python -m synthetic_eval.cli \
  --input-dir /mnt/sciml/a.sadreev/sea_ice_data/valid \
  --output-dir /tmp/sea_ice_synth_eval_concat \
  --runner concat \
  --concat-run-dir checkpoints/concat/run_YYYYMMDD_HHMMSS \
  --checkpoint-name ema_best_model.pth \
  --ensemble-size 16 \
  --mask-types random,block,swath \
  --densities 0.01,0.05,0.10,0.25 \
  --noise-levels 0.0,0.02 \
  --num-timesteps 50 \
  --method euler
```

The concat runner preserves the existing sampler interface: it builds a normalized `mask` and
`observed = x_true * mask`, calls `concat.sampler.Sampler.sample_conditioned`, and denormalizes outputs.

## Daily-Year 365 x 15 Setup

For an hourly one-year dataset, use one ground-truth field per day and 15 ensemble members per field:

```bash
python -m synthetic_eval.cli \
  --input-dir /mnt/sciml/a.sadreev/sea_ice_data/valid \
  --output-dir /tmp/sea_ice_synth_eval_daily_365x15 \
  --runner concat \
  --concat-run-dir checkpoints/concat/run_YYYYMMDD_HHMMSS \
  --checkpoint-name ema_best_model.pth \
  --n-cases 365 \
  --case-selection daily \
  --daily-stride 24 \
  --ensemble-size 15 \
  --mask-types random,block \
  --densities 0.01,0.05,0.10,0.25 \
  --noise-levels 0.0 \
  --num-timesteps 50 \
  --method euler
```

This produces `365 * 15 = 5475` generated analysis samples for each mask/density/noise combination.

## First 30 Validation Samples x 15 Setup

For a smaller first-month-style run, use the first 30 files from the validation split and 15 ensemble
members per field:

```bash
python -m synthetic_eval.cli \
  --input-dir /mnt/sciml/a.sadreev/sea_ice_data/valid \
  --output-dir /tmp/sea_ice_synth_eval_val_first30x15 \
  --runner concat \
  --concat-run-dir checkpoints/concat/run_YYYYMMDD_HHMMSS \
  --checkpoint-name ema_best_model.pth \
  --n-cases 30 \
  --case-selection first \
  --ensemble-size 15 \
  --mask-types random,block \
  --densities 0.05 \
  --noise-levels 0.0 \
  --num-timesteps 50 \
  --method euler
```

This produces `30 * 15 = 450` generated analysis samples for each mask/density/noise combination.

## Outputs

- `metadata.json`: exact benchmark settings.
- `per_case_metrics.csv`: per-case contributions for later daily/weekly/block bootstrap.
- `aggregate_metrics.csv` and `aggregate_metrics.json`: aggregate metrics by variable, mask type, density and noise.
- `spread_skill_bins.csv`: binned spread-skill data.
- `arrays/rank_histograms.npz`: M+1 rank histogram counts.
- `arrays/*mean_error.npy` and `arrays/*crps.npy`: map diagnostics.
- `plots/`: rank histograms, coverage reliability, spread-skill, RMSE/CRPS vs density, and example panels.

## kNN Dataset Baseline By Day

To search nearest train fields for each December validation day:

```bash
python -m synthetic_eval.knn_baseline \
  --data-root /mnt/sciml/a.sadreev/sea_ice_data \
  --output-dir /mnt/sciml/a.sadreev/sea_ice_data/knn_baseline_december \
  --selection december \
  --n-days 31 \
  --k-neighbors 1 \
  --mask-type swath \
  --n-tracks-min 7 \
  --n-tracks-max 7
```

To run the same search for 30 consecutive days:

```bash
python -m synthetic_eval.knn_baseline \
  --data-root /mnt/sciml/a.sadreev/sea_ice_data \
  --output-dir /mnt/sciml/a.sadreev/sea_ice_data/knn_baseline_30days_2022-12-01 \
  --selection consecutive \
  --start-date 2022-12-01 \
  --n-days 30 \
  --k-neighbors 1 \
  --mask-type swath \
  --n-tracks-min 7 \
  --n-tracks-max 7
```

This writes `selected_references.csv`, `knn_neighbors.csv`, and `summary.json`.

## Metrics

Implemented metrics include ensemble-mean RMSE, ensemble CRPS, rank histograms with random tie-breaking,
central interval coverage at 50/80/90/95 percent, spread-skill ratio and binned diagnostics, energy score
on a configurable subsample, and simple ice-specific summaries for concentration/thickness fields.
