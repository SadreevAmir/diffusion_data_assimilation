# Data Assimilation Conditioning Directions

## Current Goal

Train a current-repo `concat_conditioning` diffusion or flow model for model-to-model sea-ice assimilation. The model receives a dense background forecast and sparse SRAL-track observations, then produces an analysis-like sea-ice state.

## Conditioning Options

### 1. Concat Conditioning

Feed the model the noisy state, coordinate grid, dense background, sparse observation values, and observation mask as concatenated input channels.

Status: current implementation.

Strengths: simple, stable, and easy to debug.

Risks: the network can ignore sparse observations unless sampling and losses strongly enforce them.

### 2. Hard Data Consistency

Force observed pixels to match the observation values after sampling or during every denoising step.

Status: final observed-value consistency is implemented for sampling.

Next improvement: enforce consistency after every sampler step, not only at the end.

Strengths: guarantees SRAL-track values are respected on-grid.

Risks: hard constraints can create local artifacts near tracks if the model has not learned smooth corrections.

### 3. Background-Initialized Sampling

Start sampling from a noised background field instead of pure Gaussian noise.

Status: implemented as `sample_start_mode = "background"`.

Strengths: much closer to data assimilation because the sampler corrects a forecast rather than generating the full state from scratch.

Risks: if the background is strongly biased, the analysis may stay too close to it.

### 4. Guidance-Based Conditioning

During sampling, add a gradient-based observation penalty such as `||H(x) - y||`, where `H` is the observation operator.

Status: not implemented for `concat_conditioning`.

Strengths: works with sparse tracks and can support arbitrary observation operators.

Risks: more expensive and needs guidance-scale tuning to avoid noisy or overfit samples.

### 5. Bridge Or Residual Model

Train the model to move from background to truth directly, for example with a bridge path `x_t = (1 - t) * truth + t * background` or by predicting an analysis increment.

Status: implemented as `training_objective = "bridge"` with `sample_start_mode = "bridge"` for the next experiment.

Strengths: naturally anchored to the forecast and more assimilation-like than pure noise-to-truth generation.

Risks: requires a training objective change and careful comparison against the current model.

### 6. ControlNet-Style Feature Conditioning

Inject background, observations, and masks into intermediate UNet features rather than only at the input.

Status: possible future architecture change.

Strengths: more expressive conditioning than input concatenation.

Risks: more parameters, more code, and more training complexity.

### 7. Observation Tokens Or Cross-Attention

Represent observations as sparse tokens containing location, value, time lag, uncertainty, and source, then condition the UNet with cross-attention.

Status: future direction for irregular or multi-source observations.

Strengths: well matched to real sparse altimetry and mixed observation types.

Risks: larger architecture change and needs robust batching of variable-length observations.

### 8. Temporal Conditioning

Include multiple background days, explicit 3-day observation windows, or prior analyses as conditioning channels.

Status: current SRAL loader uses a 3-day observation window but collapses it into one sparse condition tensor.

Strengths: can expose time-lag information that is currently hidden.

Risks: increases input size and may require explicit time-lag embeddings.

## Recommended Order

1. Keep concat conditioning but use EMA sampling, background start, more sampling steps, and hard observation consistency.
2. Add stepwise data consistency inside the sampler if final consistency is not enough.
3. Add validation metrics against the background baseline and monitor analysis skill in ClearML.
4. Run the background-to-truth bridge experiment and compare against the 200-step concat baseline.
5. If sparse SRAL tracks still have weak influence, add guidance-based conditioning or observation-token conditioning.
