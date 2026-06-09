from __future__ import annotations

import torch
from torchdiffeq import odeint

from utils import get_device, make_normalized_xy_grid
from .transforms import make_conditioned_model_input


_FIXED_STEP_METHODS = {"euler", "midpoint", "rk4", "heun3"}
_CFG_MODES = {"none", "independent", "background_delta"}


def _model_output(output):
    if hasattr(output, "sample"):
        return output.sample
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def _cfg_enabled(mode: str, background_scale: float, observation_scale: float) -> bool:
    return mode != "none" and (
        abs(float(background_scale) - 1.0) > 1e-12
        or abs(float(observation_scale) - 1.0) > 1e-12
        or mode == "independent"
    )


class Sampler:
    def __init__(self, model, metadata: dict | None = None):
        self.model = model
        self.metadata = dict(metadata or {})
        self.normalization_means = self.metadata.get("normalization_means")
        self.normalization_stds = self.metadata.get("normalization_stds")
        self._grids = {}

    def _grid(self, height: int, width: int, device, dtype: torch.dtype) -> torch.Tensor:
        key = (int(height), int(width), str(device), dtype)
        if key not in self._grids:
            self._grids[key] = make_normalized_xy_grid(height, width, device=device, dtype=dtype)
        return self._grids[key]

    @torch.no_grad()
    def sample_conditioned(
        self,
        background: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        water_mask: torch.Tensor,
        size: tuple[int, int],
        num_timesteps: int,
        background_mask: torch.Tensor | None = None,
        device=None,
        method: str = "euler",
        rtol: float = 1e-3,
        atol: float = 1e-4,
        start_mode: str = "noise",
        start_noise_level: float = 1.0,
        enforce_observations: bool = False,
        valid_mask: torch.Tensor | None = None,
        obs_guidance_scale: float = 0.0,
        obs_guidance_eps: float = 1e-8,
        initial_noise: torch.Tensor | None = None,
        sample_target: str = "state",
        memory_efficient_euler: bool = False,
        cfg_mode: str = "none",
        cfg_background_scale: float = 1.0,
        cfg_observation_scale: float = 1.0,
    ) -> torch.Tensor:
        device = device or get_device()
        background = background.to(device)
        if background_mask is None:
            background_mask = torch.ones_like(background)
        else:
            background_mask = background_mask.to(device)
        obs_values = obs_values.to(device)
        obs_mask = obs_mask.to(device)
        water_mask = water_mask.to(device)
        if valid_mask is not None:
            valid_mask = valid_mask.to(device)
        batch_size, channels = background.shape[:2]
        height, width = size
        num_timesteps = max(int(num_timesteps), 2)
        if sample_target not in ("state", "residual"):
            raise ValueError(f"Unknown sample_target={sample_target!r}; expected 'state' or 'residual'")
        cfg_mode = str(cfg_mode)
        if cfg_mode not in _CFG_MODES:
            raise ValueError(f"Unknown cfg_mode={cfg_mode!r}; expected one of {sorted(_CFG_MODES)}")
        cfg_background_scale = float(cfg_background_scale)
        cfg_observation_scale = float(cfg_observation_scale)
        use_cfg = _cfg_enabled(cfg_mode, cfg_background_scale, cfg_observation_scale)
        grid = self._grid(height, width, device, background.dtype).expand(batch_size, -1, -1, -1)
        if initial_noise is not None:
            expected_shape = (batch_size, channels, height, width)
            if tuple(initial_noise.shape) != expected_shape:
                raise ValueError(
                    f"initial_noise must have shape {expected_shape}, got {tuple(initial_noise.shape)}"
                )
            initial_noise = initial_noise.to(device=device, dtype=background.dtype)

        if start_mode == "noise":
            start_t = 1.0
            x0 = initial_noise if initial_noise is not None else torch.randn_like(background)
        elif start_mode == "background":
            start_t = min(max(float(start_noise_level), 0.001), 1.0)
            eps = initial_noise if initial_noise is not None else torch.randn_like(background)
            clean_start = torch.zeros_like(background) if sample_target == "residual" else background
            x0 = (1.0 - start_t) * clean_start + start_t * eps
        elif start_mode == "bridge":
            if sample_target == "residual":
                raise ValueError("sample_target='residual' is not compatible with start_mode='bridge'")
            start_t = 1.0
            x0 = background
        else:
            raise ValueError(f"Unknown start_mode={start_mode!r}; expected 'noise', 'background', or 'bridge'")

        timesteps = torch.linspace(start_t, 0.001, num_timesteps, device=device)

        def finalize(state_sample: torch.Tensor) -> torch.Tensor:
            analysis = background + state_sample if sample_target == "residual" else state_sample
            if enforce_observations:
                analysis = torch.where(obs_mask > 0, obs_values, analysis)
            if valid_mask is not None:
                analysis = torch.where(valid_mask > 0, analysis, background)
            return analysis

        if obs_guidance_scale > 0:
            if use_cfg:
                raise ValueError("obs_guidance_scale and CFG sampling cannot be enabled at the same time")
            state_sample = self._sample_guided_euler(
                x0=x0,
                timesteps=timesteps,
                grid=grid,
                background=background,
                background_mask=background_mask,
                obs_values=obs_values,
                obs_mask=obs_mask,
                water_mask=water_mask,
                obs_guidance_scale=float(obs_guidance_scale),
                obs_guidance_eps=float(obs_guidance_eps),
                sample_target=sample_target,
            )
            return finalize(state_sample)

        def model_velocity(
            state: torch.Tensor,
            model_grid: torch.Tensor,
            model_background: torch.Tensor,
            model_background_mask: torch.Tensor,
            model_obs_values: torch.Tensor,
            model_obs_mask: torch.Tensor,
            model_water_mask: torch.Tensor,
            t_tensor: torch.Tensor,
        ) -> torch.Tensor:
            model_input = make_conditioned_model_input(
                state,
                model_grid,
                model_background,
                model_background_mask,
                model_obs_values,
                model_obs_mask,
                model_water_mask,
            )
            return _model_output(self.model(model_input, t_tensor))

        def f(t, x):
            t_tensor = t.expand(batch_size) * 1000
            if not use_cfg:
                return model_velocity(
                    x,
                    grid,
                    background,
                    background_mask,
                    obs_values,
                    obs_mask,
                    water_mask,
                    t_tensor,
                )

            zeros_background = torch.zeros_like(background)
            zeros_obs_values = torch.zeros_like(obs_values)
            zeros_obs_mask = torch.zeros_like(obs_mask)
            zeros_background_mask = torch.zeros_like(background_mask)

            if cfg_mode == "independent":
                variant_background = torch.cat([zeros_background, background, zeros_background], dim=0)
                variant_background_mask = torch.cat(
                    [zeros_background_mask, background_mask, zeros_background_mask],
                    dim=0,
                )
                variant_obs_values = torch.cat([zeros_obs_values, zeros_obs_values, obs_values], dim=0)
                variant_obs_mask = torch.cat([zeros_obs_mask, zeros_obs_mask, obs_mask], dim=0)
            else:
                variant_background = torch.cat([zeros_background, background, background], dim=0)
                variant_background_mask = torch.cat(
                    [zeros_background_mask, background_mask, background_mask],
                    dim=0,
                )
                variant_obs_values = torch.cat([zeros_obs_values, zeros_obs_values, obs_values], dim=0)
                variant_obs_mask = torch.cat([zeros_obs_mask, zeros_obs_mask, obs_mask], dim=0)

            velocity = model_velocity(
                x.repeat(3, 1, 1, 1),
                grid.repeat(3, 1, 1, 1),
                variant_background,
                variant_background_mask,
                variant_obs_values,
                variant_obs_mask,
                water_mask.repeat(3, 1, 1, 1),
                t_tensor.repeat(3),
            )
            v_none, v_background, v_observation_or_both = velocity.chunk(3, dim=0)
            if cfg_mode == "independent":
                return (
                    v_none
                    + cfg_background_scale * (v_background - v_none)
                    + cfg_observation_scale * (v_observation_or_both - v_none)
                )
            return (
                v_none
                + cfg_background_scale * (v_background - v_none)
                + cfg_observation_scale * (v_observation_or_both - v_background)
            )

        if memory_efficient_euler and method == "euler":
            state = x0
            for index in range(timesteps.numel() - 1):
                state = state + (timesteps[index + 1] - timesteps[index]) * f(timesteps[index], state)
            return finalize(state)

        kwargs = {}
        if method in _FIXED_STEP_METHODS:
            time_span = abs(start_t - 0.001)
            kwargs["options"] = {"step_size": max(time_span / max(num_timesteps - 1, 1), 1e-6)}
        else:
            kwargs["rtol"] = rtol
            kwargs["atol"] = atol

        trajectory = odeint(f, x0, timesteps, method=method, **kwargs)
        return finalize(trajectory[-1])

    def _sample_guided_euler(
        self,
        x0: torch.Tensor,
        timesteps: torch.Tensor,
        grid: torch.Tensor,
        background: torch.Tensor,
        background_mask: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        water_mask: torch.Tensor,
        obs_guidance_scale: float,
        obs_guidance_eps: float,
        sample_target: str,
    ) -> torch.Tensor:
        batch_size = x0.shape[0]
        x = x0
        obs_den = obs_mask.reshape(batch_size, -1).sum(dim=1).clamp(min=1.0)

        for index in range(timesteps.numel() - 1):
            t = timesteps[index]
            dt = timesteps[index + 1] - t
            x_t = x.detach().requires_grad_(True)
            t_tensor = t.expand(batch_size) * 1000

            with torch.enable_grad():
                model_input = make_conditioned_model_input(
                    x_t,
                    grid,
                    background,
                    background_mask,
                    obs_values,
                    obs_mask,
                    water_mask,
                )
                v_pred = _model_output(self.model(model_input, t_tensor))
                x_hat = x_t - t * v_pred
                analysis_hat = background + x_hat if sample_target == "residual" else x_hat
                obs_error = (analysis_hat - obs_values) * obs_mask
                obs_loss = (obs_error.square().reshape(batch_size, -1).sum(dim=1) / obs_den).mean()
                grad = torch.autograd.grad(obs_loss, x_t)[0]

            grad = grad.detach()
            v_pred = v_pred.detach()
            grad_norm = grad.reshape(batch_size, -1).norm(dim=1).view(batch_size, 1, 1, 1)
            v_guided = v_pred + obs_guidance_scale * grad / (grad_norm + obs_guidance_eps)
            x = x_t.detach() + dt * v_guided

        return x
