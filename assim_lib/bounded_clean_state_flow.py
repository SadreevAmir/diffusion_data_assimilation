"""Bounded clean-state parameterization for direct SIC/SIT conditional flow.

The ODE state remains in the normalized six-channel SIC/SIT space.  The
network, however, predicts logits for the clean physical state.  SIC logits
are decoded with a sigmoid and SIT logits with a scaled softplus, after which
the physical prediction is normalized.  For the linear probability path

    z_t = (1 - t) x_0 + t eps,

the corresponding velocity reconstructed from a clean-state prediction is

    v(z_t, t) = (z_t - x_hat_0(z_t, t)) / t.

Sampling stops at a strictly positive epsilon and returns the final bounded
clean-state prediction.  This endpoint is part of the model definition; it is
not post-hoc clipping of an unconstrained sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .runtime import make_normalized_xy_grid


def _channel_values(
    values: tuple[float, ...] | list[float], reference: torch.Tensor, name: str
) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=reference.dtype, device=reference.device)
    if tensor.ndim != 1 or tensor.numel() != reference.shape[1]:
        raise ValueError(
            f"{name} must contain one value per channel ({reference.shape[1]}), "
            f"got shape {tuple(tensor.shape)}"
        )
    return tensor.view(1, -1, 1, 1)


def _expanded_valid_mask(valid_mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if valid_mask.ndim != 4:
        raise ValueError("valid_mask must have shape [B,C,H,W]")
    if valid_mask.shape[0] != reference.shape[0] or valid_mask.shape[-2:] != reference.shape[-2:]:
        raise ValueError("valid_mask batch and spatial dimensions must match the state")
    if valid_mask.shape[1] not in (1, reference.shape[1]):
        raise ValueError("valid_mask must contain one channel or one channel per state field")
    mask = valid_mask.to(device=reference.device)
    if not torch.all(torch.isfinite(mask)):
        raise ValueError("valid_mask must be finite")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("valid_mask must be binary")
    expanded = mask.expand_as(reference) > 0
    if not torch.any(expanded):
        raise ValueError("valid_mask must contain at least one active ocean point")
    return expanded


def _require_finite_active(
    value: torch.Tensor, valid: torch.Tensor, *, label: str
) -> None:
    invalid_count = int(((~torch.isfinite(value)) & valid).sum().item())
    if invalid_count:
        raise FloatingPointError(
            f"{label} contains {invalid_count} NaN/Inf values on active ocean points"
        )


def decode_bounded_sic_sit(logits: torch.Tensor, *, sit_scale: float = 1.0) -> torch.Tensor:
    """Decode interleaved SIC/SIT logits into a bounded physical state.

    In real arithmetic SIC lies in ``(0, 1)`` and SIT is positive for finite
    logits.  Floating-point saturation may produce exact boundary values, but
    those are not learned probability atoms.  This is a bounded continuous
    approximation and does not claim an exact atom law at SIC 0/1 or SIT 0.
    """
    if logits.ndim != 4 or logits.shape[1] <= 0 or logits.shape[1] % 2:
        raise ValueError("logits must have shape [B,2*T,H,W]")
    sit_scale = float(sit_scale)
    if not torch.isfinite(torch.tensor(sit_scale)) or sit_scale <= 0.0:
        raise ValueError("sit_scale must be finite and positive")
    physical = torch.empty_like(logits)
    physical[:, 0::2] = torch.sigmoid(logits[:, 0::2])
    physical[:, 1::2] = F.softplus(logits[:, 1::2]) * sit_scale
    return physical


def normalize_physical_state(
    physical: torch.Tensor,
    means: tuple[float, ...] | list[float],
    stds: tuple[float, ...] | list[float],
) -> torch.Tensor:
    channel_means = _channel_values(means, physical, "means")
    channel_stds = _channel_values(stds, physical, "stds")
    if not torch.all(torch.isfinite(channel_means)):
        raise ValueError("all normalization means must be finite")
    if not torch.all(torch.isfinite(channel_stds)) or not torch.all(channel_stds > 0):
        raise ValueError("all normalization stds must be finite and positive")
    return (physical - channel_means) / channel_stds


def bounded_clean_prediction(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    means: tuple[float, ...] | list[float],
    stds: tuple[float, ...] | list[float],
    sit_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return masked physical and normalized clean-state predictions."""
    valid = _expanded_valid_mask(valid_mask, logits)
    _require_finite_active(logits, valid, label="clean-state logits")
    # Inactive logits are neutralized before nonlinear decoding.  Masking only
    # after sigmoid/softplus would allow inactive NaN/Inf values to poison the
    # backward pass even when the scalar loss appears masked.
    safe_logits = torch.where(valid, logits, torch.zeros_like(logits))
    physical = decode_bounded_sic_sit(safe_logits, sit_scale=sit_scale)
    normalized = normalize_physical_state(physical, means, stds)
    _require_finite_active(physical, valid, label="decoded physical clean state")
    _require_finite_active(normalized, valid, label="normalized clean state")
    physical = torch.where(valid, physical, torch.zeros_like(physical))
    normalized = torch.where(valid, normalized, torch.zeros_like(normalized))
    return physical, normalized


def bounded_clean_state_loss(
    logits: torch.Tensor,
    normalized_truth: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    means: tuple[float, ...] | list[float],
    stds: tuple[float, ...] | list[float],
    sit_scale: float = 1.0,
) -> torch.Tensor:
    """Masked MSE on normalized clean states decoded from bounded logits."""
    if logits.shape != normalized_truth.shape:
        raise ValueError("logits and normalized_truth must have identical shapes")
    _, prediction = bounded_clean_prediction(
        logits,
        valid_mask,
        means=means,
        stds=stds,
        sit_scale=sit_scale,
    )
    valid = _expanded_valid_mask(valid_mask, logits)
    _require_finite_active(normalized_truth, valid, label="normalized clean-state truth")
    safe_truth = torch.where(valid, normalized_truth, torch.zeros_like(normalized_truth))
    loss = F.mse_loss(prediction[valid], safe_truth[valid])
    if not torch.isfinite(loss):
        raise FloatingPointError("bounded clean-state loss is NaN/Inf")
    return loss


def clean_prediction_velocity(
    state: torch.Tensor,
    normalized_clean: torch.Tensor,
    timesteps: torch.Tensor,
    *,
    minimum_time: float,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct ``(state-clean)/t`` and fail closed near the singular endpoint."""
    if state.shape != normalized_clean.shape:
        raise ValueError("state and normalized_clean must have identical shapes")
    time = torch.as_tensor(timesteps, dtype=state.dtype, device=state.device)
    if time.ndim == 0:
        time = time.expand(state.shape[0])
    if time.ndim != 1 or time.shape[0] != state.shape[0]:
        raise ValueError("timesteps must be scalar or have one value per batch item")
    minimum_time = float(minimum_time)
    if not 0.0 < minimum_time < 1.0:
        raise ValueError("minimum_time must lie strictly between zero and one")
    minimum = torch.as_tensor(minimum_time, dtype=state.dtype, device=state.device)
    tolerance = 8.0 * torch.finfo(state.dtype).eps * torch.maximum(
        torch.ones_like(time), time.abs()
    )
    if not torch.all(torch.isfinite(time)) or not torch.all(time >= minimum - tolerance):
        raise ValueError("velocity evaluation below minimum_time is forbidden")
    time = torch.maximum(time, minimum)
    time = time.view(-1, 1, 1, 1)
    velocity = (state - normalized_clean) / time
    valid = _expanded_valid_mask(valid_mask, state)
    _require_finite_active(state, valid, label="ODE state")
    _require_finite_active(normalized_clean, valid, label="ODE clean prediction")
    _require_finite_active(velocity, valid, label="reconstructed clean-state velocity")
    return torch.where(valid, velocity, torch.zeros_like(velocity))


@dataclass(frozen=True)
class BoundedSample:
    physical: torch.Tensor
    normalized: torch.Tensor
    terminal_ode_state: torch.Tensor


class BoundedCleanStateSampler:
    """Fixed-step RK4 sampler with an explicit bounded denoising endpoint."""

    def __init__(
        self,
        model,
        *,
        means: tuple[float, ...] | list[float],
        stds: tuple[float, ...] | list[float],
        sit_scale: float = 1.0,
        epsilon: float = 0.01,
    ) -> None:
        self.model = model
        self.means = tuple(float(value) for value in means)
        self.stds = tuple(float(value) for value in stds)
        self.sit_scale = float(sit_scale)
        self.epsilon = float(epsilon)
        if not 0.0 < self.epsilon < 1.0:
            raise ValueError("epsilon must lie strictly between zero and one")

    def _predict_clean(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        conditioning: torch.Tensor,
        valid_mask: torch.Tensor,
        grid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = _expanded_valid_mask(valid_mask, state)
        _require_finite_active(state, valid, label="sampler ODE state")
        if not torch.all(torch.isfinite(conditioning)):
            raise FloatingPointError("model conditioning contains NaN/Inf")
        if grid is None:
            grid = make_normalized_xy_grid(
                state.shape[-2], state.shape[-1], device=state.device, dtype=state.dtype
            ).expand(state.shape[0], -1, -1, -1)
        model_input = torch.cat((state, grid, conditioning), dim=1)
        output = self.model(model_input, time * 1000.0)
        if hasattr(output, "sample"):
            logits = output.sample
        elif isinstance(output, (tuple, list)):
            logits = output[0]
        else:
            logits = output
        logits = logits.to(dtype=state.dtype)
        if logits.shape != state.shape:
            raise ValueError("model output must match the ODE state shape")
        return bounded_clean_prediction(
            logits,
            valid_mask,
            means=self.means,
            stds=self.stds,
            sit_scale=self.sit_scale,
        )

    @torch.no_grad()
    def sample(
        self,
        initial_noise: torch.Tensor,
        conditioning: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        num_steps: int,
    ) -> BoundedSample:
        if initial_noise.ndim != 4:
            raise ValueError("initial_noise must have shape [B,C,H,W]")
        if (
            conditioning.ndim != 4
            or conditioning.shape[0] != initial_noise.shape[0]
            or conditioning.shape[-2:] != initial_noise.shape[-2:]
        ):
            raise ValueError("conditioning batch and spatial dimensions must match initial_noise")
        num_steps = int(num_steps)
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        valid = _expanded_valid_mask(valid_mask, initial_noise)
        _require_finite_active(initial_noise, valid, label="initial noise")
        if not torch.all(torch.isfinite(conditioning)):
            raise FloatingPointError("model conditioning contains NaN/Inf")
        was_training = bool(getattr(self.model, "training", False))
        if hasattr(self.model, "eval"):
            self.model.eval()
        try:
            state = torch.where(valid, initial_noise, torch.zeros_like(initial_noise))
            grid = make_normalized_xy_grid(
                state.shape[-2], state.shape[-1], device=state.device, dtype=state.dtype
            ).expand(state.shape[0], -1, -1, -1)
            # Shared nodes with explicit endpoints prevent the final RK4 stage
            # from crossing epsilon through accumulated Python-float error.
            nodes = torch.linspace(1.0, self.epsilon, num_steps + 1, dtype=torch.float64)
            nodes[0] = 1.0
            nodes[-1] = self.epsilon

            def velocity(value: torch.Tensor, scalar_time: float) -> torch.Tensor:
                time = torch.full(
                    (value.shape[0],), scalar_time, dtype=value.dtype, device=value.device
                )
                _, clean = self._predict_clean(
                    value, time, conditioning, valid_mask, grid=grid
                )
                return clean_prediction_velocity(
                    value,
                    clean,
                    time,
                    minimum_time=self.epsilon,
                    valid_mask=valid_mask,
                )

            for step in range(num_steps):
                time = float(nodes[step])
                next_time = float(nodes[step + 1])
                dt = next_time - time
                half_time = 0.5 * (time + next_time)
                k1 = velocity(state, time)
                k2 = velocity(state + 0.5 * dt * k1, half_time)
                k3 = velocity(state + 0.5 * dt * k2, half_time)
                k4 = velocity(state + dt * k3, next_time)
                state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                state = torch.where(valid, state, torch.zeros_like(state))
                _require_finite_active(state, valid, label="integrated ODE state")

            endpoint_time = torch.full(
                (state.shape[0],), self.epsilon, dtype=state.dtype, device=state.device
            )
            physical, normalized = self._predict_clean(
                state, endpoint_time, conditioning, valid_mask, grid=grid
            )
            _require_finite_active(state, valid, label="terminal ODE state")
            _require_finite_active(physical, valid, label="bounded physical endpoint")
            _require_finite_active(normalized, valid, label="normalized bounded endpoint")
            return BoundedSample(
                physical=physical,
                normalized=normalized,
                terminal_ode_state=state,
            )
        finally:
            if hasattr(self.model, "train"):
                self.model.train(was_training)


class FiniteMixtureOracle:
    """Exact posterior-mean denoiser for a finite clean-state toy law."""

    def __init__(self, atoms: torch.Tensor, weights: torch.Tensor) -> None:
        if atoms.ndim != 2 or atoms.shape[0] < 2:
            raise ValueError("atoms must have shape [K,D] with K >= 2")
        if weights.shape != (atoms.shape[0],):
            raise ValueError("weights must have one value per atom")
        if not torch.all(torch.isfinite(atoms)) or not torch.all(torch.isfinite(weights)):
            raise ValueError("oracle atoms and weights must be finite")
        if not torch.all(weights > 0):
            raise ValueError("oracle weights must be positive")
        self.atoms = atoms
        self.weights = weights / weights.sum()

    def denoise(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2 or state.shape[1] != self.atoms.shape[1]:
            raise ValueError("state must have shape [B,D]")
        time = torch.as_tensor(time, dtype=state.dtype, device=state.device)
        if time.ndim == 0:
            time = time.expand(state.shape[0])
        if time.shape != (state.shape[0],) or not torch.all(time > 0):
            raise ValueError("oracle time must be positive with one value per sample")
        atoms = self.atoms.to(device=state.device, dtype=state.dtype)
        weights = self.weights.to(device=state.device, dtype=state.dtype)
        residual = state[:, None, :] - (1.0 - time[:, None, None]) * atoms[None]
        log_likelihood = -0.5 * residual.square().sum(dim=-1) / time[:, None].square()
        posterior = torch.softmax(log_likelihood + torch.log(weights)[None], dim=1)
        return posterior @ atoms


class FiniteMixtureOracleLogitModel(torch.nn.Module):
    """Expose the exact toy denoiser through the production logit interface."""

    def __init__(
        self,
        oracle: FiniteMixtureOracle,
        *,
        means: tuple[float, float],
        stds: tuple[float, float],
        sit_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if oracle.atoms.shape[1] != 2:
            raise ValueError("bounded SIC/SIT oracle model requires two-dimensional atoms")
        self.oracle = oracle
        self.means = tuple(float(value) for value in means)
        self.stds = tuple(float(value) for value in stds)
        self.sit_scale = float(sit_scale)

    def forward(self, value: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        if tuple(value.shape[-2:]) != (1, 1):
            raise ValueError("finite-mixture production oracle expects a 1x1 spatial domain")
        state = value[:, :2, 0, 0]
        normalized_clean = self.oracle.denoise(
            state, timestep.to(dtype=state.dtype) / 1000.0
        )
        means = normalized_clean.new_tensor(self.means)
        stds = normalized_clean.new_tensor(self.stds)
        clean = normalized_clean * stds + means
        tiny = max(16.0 * torch.finfo(clean.dtype).eps, 1e-12)
        sic = clean[:, 0].clamp(min=tiny, max=1.0 - tiny)
        scaled_sit = (clean[:, 1] / self.sit_scale).clamp(min=tiny)
        sic_logits = torch.logit(sic)
        # Stable inverse softplus: y + log(1 - exp(-y)).
        sit_logits = scaled_sit + torch.log(-torch.expm1(-scaled_sit))
        return torch.stack((sic_logits, sit_logits), dim=1)[:, :, None, None]


def integrate_oracle_rk4(
    oracle: FiniteMixtureOracle,
    initial_noise: torch.Tensor,
    *,
    epsilon: float,
    num_steps: int,
    velocity_sign: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate the toy posterior flow and return endpoint prediction/state."""
    epsilon = float(epsilon)
    num_steps = int(num_steps)
    if not 0.0 < epsilon < 1.0 or num_steps <= 0:
        raise ValueError("epsilon and num_steps must define a positive integration interval")
    state = initial_noise.clone()
    dt = (epsilon - 1.0) / num_steps

    def velocity(value: torch.Tensor, scalar_time: float) -> torch.Tensor:
        time = torch.full((value.shape[0],), scalar_time, dtype=value.dtype, device=value.device)
        clean = oracle.denoise(value, time)
        return float(velocity_sign) * (value - clean) / time[:, None]

    for step in range(num_steps):
        time = 1.0 + step * dt
        half_time = time + 0.5 * dt
        k1 = velocity(state, time)
        k2 = velocity(state + 0.5 * dt * k1, half_time)
        k3 = velocity(state + 0.5 * dt * k2, half_time)
        k4 = velocity(state + dt * k3, time + dt)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    endpoint_time = torch.full(
        (state.shape[0],), epsilon, dtype=state.dtype, device=state.device
    )
    return oracle.denoise(state, endpoint_time), state


def finite_mixture_oracle_report(
    *, seed: int = 1701, sample_count: int = 4096, epsilon: float = 0.01, num_steps: int = 256
) -> dict[str, float | list[float]]:
    """Run the CPU mathematical gate used before any learned GPU pilot."""
    atoms = torch.tensor(
        [[0.0, 0.0], [1.0, 0.35], [0.3, 1.2], [0.9, 2.0]], dtype=torch.float64
    )
    weights = torch.tensor([0.30, 0.25, 0.25, 0.20], dtype=torch.float64)
    means = torch.tensor([0.19301218262209952, 0.18969311571248842], dtype=torch.float64)
    stds = torch.tensor([0.36890927421384667, 0.4359687842884708], dtype=torch.float64)
    normalized_atoms = (atoms - means) / stds
    oracle = FiniteMixtureOracle(normalized_atoms, weights)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    noise = torch.randn((int(sample_count), 2), generator=generator, dtype=torch.float64)
    model = FiniteMixtureOracleLogitModel(
        oracle,
        means=tuple(float(value) for value in means),
        stds=tuple(float(value) for value in stds),
    )
    conditioning = torch.zeros((noise.shape[0], 0, 1, 1), dtype=noise.dtype)
    valid = torch.ones((noise.shape[0], 1, 1, 1), dtype=noise.dtype)

    def production_sample(local_epsilon: float, local_steps: int) -> BoundedSample:
        return BoundedCleanStateSampler(
            model,
            means=tuple(float(value) for value in means),
            stds=tuple(float(value) for value in stds),
            epsilon=local_epsilon,
        ).sample(
            noise[:, :, None, None], conditioning, valid, num_steps=local_steps
        )

    production = production_sample(epsilon, num_steps)
    endpoint = production.physical[:, :, 0, 0]
    terminal_normalized = production.terminal_ode_state[:, :, 0, 0]
    terminal = terminal_normalized * stds + means
    refined = production_sample(epsilon, 2 * num_steps).physical[:, :, 0, 0]
    smaller_epsilon = production_sample(0.5 * epsilon, 2 * num_steps).physical[:, :, 0, 0]
    reference_normalized, _ = integrate_oracle_rk4(
        oracle, noise, epsilon=epsilon, num_steps=num_steps
    )
    reference = reference_normalized * stds + means
    distances = torch.cdist(endpoint, atoms)
    assignments = distances.argmin(dim=1)
    frequencies = torch.bincount(assignments, minlength=atoms.shape[0]).double() / endpoint.shape[0]
    target_mean = weights @ atoms
    target_mean_normalized = weights @ normalized_atoms
    target_centered = atoms - target_mean
    target_covariance = (target_centered.T * weights) @ target_centered
    sample_mean = endpoint.mean(dim=0)
    sample_covariance = torch.cov(endpoint.T)
    initial_distance = torch.linalg.vector_norm(noise - target_mean_normalized, dim=1).mean()
    first_time = torch.ones(noise.shape[0], dtype=noise.dtype)
    first_velocity = (noise - oracle.denoise(noise, first_time)) / first_time[:, None]
    reverse_probe = noise - 1e-3 * first_velocity
    reverse_distance = torch.linalg.vector_norm(
        reverse_probe - target_mean_normalized, dim=1
    ).mean()
    wrong_endpoint_normalized, _ = integrate_oracle_rk4(
        oracle, noise[:128], epsilon=epsilon, num_steps=max(num_steps // 4, 8), velocity_sign=-1.0
    )
    wrong_endpoint = wrong_endpoint_normalized * stds + means
    wrong_distances = torch.cdist(wrong_endpoint, atoms)
    wrong_assignments = wrong_distances.argmin(dim=1)
    wrong_frequencies = (
        torch.bincount(wrong_assignments, minlength=atoms.shape[0]).double()
        / wrong_endpoint.shape[0]
    )
    return {
        "support_min_sic": float(endpoint[:, 0].min()),
        "support_max_sic": float(endpoint[:, 0].max()),
        "support_min_sit": float(endpoint[:, 1].min()),
        "mean_linf_error": float((sample_mean - target_mean).abs().max()),
        "covariance_linf_error": float((sample_covariance - target_covariance).abs().max()),
        "component_frequency_linf_error": float((frequencies - weights).abs().max()),
        "mean_nearest_atom_distance": float(distances.min(dim=1).values.mean()),
        "doubled_steps_mean_abs_difference": float((endpoint - refined).abs().mean()),
        "halved_epsilon_mean_abs_difference": float((refined - smaller_epsilon).abs().mean()),
        "production_vs_reference_mean_abs_difference": float(
            (endpoint - reference).abs().mean()
        ),
        "reverse_probe_distance_ratio": float(reverse_distance / initial_distance),
        "wrong_sign_component_frequency_linf_error": float(
            (wrong_frequencies - weights).abs().max()
        ),
        "terminal_to_endpoint_mean_abs_difference": float((terminal - endpoint).abs().mean()),
        "endpoint_to_terminal_std_ratio": [
            float(value)
            for value in endpoint.std(dim=0, unbiased=True)
            / terminal.std(dim=0, unbiased=True).clamp(min=1e-12)
        ],
        "sample_std": [float(value) for value in endpoint.std(dim=0, unbiased=True)],
        "target_std": [float(value) for value in torch.sqrt(torch.diag(target_covariance))],
        "component_frequencies": [float(value) for value in frequencies],
        "target_weights": [float(value) for value in weights],
    }


def endpoint_variance_deficit_report(
    *, seed: int = 1701, sample_count: int = 8192, epsilon: float = 0.05
) -> dict[str, float | list[float]]:
    """Measure the unavoidable variance loss of a finite-epsilon posterior mean.

    A dense, narrow joint SIC/SIT law makes the effect visible without relying
    on widely separated atoms whose posterior assignments are already nearly
    deterministic.  This diagnostic is not a claim that the endpoint preserves
    variance; it quantifies the deficit that the learned pilot must audit.
    """
    coordinate = torch.linspace(0.1, 0.9, 41, dtype=torch.float64)
    atoms = torch.stack(
        (coordinate, 0.12 + 1.55 * coordinate + 0.08 * torch.sin(4.0 * coordinate)),
        dim=1,
    )
    weights = torch.exp(-0.5 * ((coordinate - 0.52) / 0.16).square())
    weights = weights / weights.sum()
    means = torch.tensor([0.19301218262209952, 0.18969311571248842], dtype=torch.float64)
    stds = torch.tensor([0.36890927421384667, 0.4359687842884708], dtype=torch.float64)
    normalized_atoms = (atoms - means) / stds
    oracle = FiniteMixtureOracle(normalized_atoms, weights)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    indices = torch.multinomial(
        weights, int(sample_count), replacement=True, generator=generator
    )
    clean = normalized_atoms[indices]
    noise = torch.randn(clean.shape, generator=generator, dtype=clean.dtype)
    state = (1.0 - float(epsilon)) * clean + float(epsilon) * noise
    time = torch.full((clean.shape[0],), float(epsilon), dtype=clean.dtype)
    endpoint = oracle.denoise(state, time)
    target_mean = weights @ normalized_atoms
    centered = normalized_atoms - target_mean
    target_covariance = (centered.T * weights) @ centered
    endpoint_covariance = torch.cov(endpoint.T)
    variance_ratio = torch.diag(endpoint_covariance) / torch.diag(target_covariance)
    return {
        "epsilon": float(epsilon),
        "variance_ratio": [float(value) for value in variance_ratio],
        "variance_deficit": [float(1.0 - value) for value in variance_ratio],
        "mean_abs_endpoint_correction": float((endpoint - state).abs().mean()),
    }
