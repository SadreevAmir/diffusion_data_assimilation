# Flow-matching data assimilation

Research code for assimilating a dense model forecast and sparse satellite
observations with a conditional flow model. The maintained pipeline lives in
`assim_lib`; historical model prototypes have been removed.

## Repository layout

- `assim_lib/` — datasets, conditioning, training, sampling and physical evaluation.
- `config/data/` — dataset and normalization configuration.
- `config/methods/` — model, objective and sampler configuration.
- `config/experiments/` — runnable training and evaluation compositions.
- `tests/` — unit and smoke-level regression tests.
- `notebooks/` — physical evaluation and CFG steering dashboards.

Configuration paths in an experiment file are resolved relative to that file.
Experiment overrides are merged recursively into the referenced data and method
configuration.

## Setup

The pinned CUDA stack is documented at the top of `requirements.txt`. After
installing PyTorch, torchvision and xformers for the target platform:

```bash
python -m pip install -r requirements.txt
python -m pip install -e '.[dev]'
```

Copy `.env.example` to `.env` only when ClearML reporting is enabled. Credentials
must remain outside Git.

## Train

Run the fast CPU-compatible smoke experiment first:

```bash
python -m assim_lib.main \
  --config config/experiments/smoke_concat_conditioning_2f.json
```

Run the current two-field balanced flow-matching experiment:

```bash
python -m assim_lib.main \
  --config config/experiments/train_m2m_concat_conditioning_diffusion_balanced_2f.json
```

## Evaluate

```bash
python -m assim_lib.evaluate \
  --config config/experiments/evaluate_m2m_concat_conditioning_diffusion_balanced_2f.json \
  --run-dir checkpoints/concat_conditioning/m2m_2f_diffusion_balanced_50m/<run>
```

Evaluation writes physical-unit aggregate metrics, per-case metrics, metadata and
optionally full ensembles. Open the maintained notebook to inspect those outputs.

## Normalization statistics

```bash
python -m assim_lib.m2m_stats \
  --config config/data/m2m_2f_1y.json \
  --output outputs/m2m_2f_stats.json
```

Add `--update-config` only after reviewing the generated statistics.

## Tests

```bash
python -m unittest discover -s tests -v
ruff check assim_lib tests
```
