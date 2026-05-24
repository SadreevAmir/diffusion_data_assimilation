from __future__ import annotations

import torch
from torchdiffeq import odeint

from utils import get_device, make_normalized_xy_grid
from .transforms import make_conditioned_model_input


_FIXED_STEP_METHODS = {"euler", "midpoint", "rk4", "heun3"}


def _model_output(output):
    if hasattr(output, "sample"):
        return output.sample
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


class Sampler:
    def __init__(self, model):
        self.model = model
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
        reconstruction_background: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = device or get_device()
        background = background.to(device)
        obs_values = obs_values.to(device)
        obs_mask = obs_mask.to(device)
        water_mask = water_mask.to(device)
        if valid_mask is not None:
            valid_mask = valid_mask.to(device)
        if reconstruction_background is None:
            reconstruction_background = background
        else:
            reconstruction_background = reconstruction_background.to(device=device, dtype=background.dtype)
        batch_size, channels = background.shape[:2]
        height, width = size
        num_timesteps = max(int(num_timesteps), 2)
        if sample_target not in ("state", "residual"):
            raise ValueError(f"Unknown sample_target={sample_target!r}; expected 'state' or 'residual'")
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
            analysis = reconstruction_background + state_sample if sample_target == "residual" else state_sample
            if enforce_observations:
                analysis = torch.where(obs_mask > 0, obs_values, analysis)
            if valid_mask is not None:
                fill_value = reconstruction_background if sample_target == "residual" else background
                analysis = torch.where(valid_mask > 0, analysis, fill_value)
            return analysis

        if obs_guidance_scale > 0:
            state_sample = self._sample_guided_euler(
                x0=x0,
                timesteps=timesteps,
                grid=grid,
                background=background,
                obs_values=obs_values,
                obs_mask=obs_mask,
                water_mask=water_mask,
                obs_guidance_scale=float(obs_guidance_scale),
                obs_guidance_eps=float(obs_guidance_eps),
                sample_target=sample_target,
                reconstruction_background=reconstruction_background,
            )
            return finalize(state_sample)

        def f(t, x):
            t_tensor = t.expand(batch_size) * 1000
            model_input = make_conditioned_model_input(x, grid, background, obs_values, obs_mask, water_mask)
            return _model_output(self.model(model_input, t_tensor))

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
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        water_mask: torch.Tensor,
        obs_guidance_scale: float,
        obs_guidance_eps: float,
        sample_target: str,
        reconstruction_background: torch.Tensor,
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
                    obs_values,
                    obs_mask,
                    water_mask,
                )
                v_pred = _model_output(self.model(model_input, t_tensor))
                x_hat = x_t - t * v_pred
                analysis_hat = reconstruction_background + x_hat if sample_target == "residual" else x_hat
                obs_error = (analysis_hat - obs_values) * obs_mask
                obs_loss = (obs_error.square().reshape(batch_size, -1).sum(dim=1) / obs_den).mean()
                grad = torch.autograd.grad(obs_loss, x_t)[0]

            grad = grad.detach()
            v_pred = v_pred.detach()
            grad_norm = grad.reshape(batch_size, -1).norm(dim=1).view(batch_size, 1, 1, 1)
            v_guided = v_pred + obs_guidance_scale * grad / (grad_norm + obs_guidance_eps)
            x = x_t.detach() + dt * v_guided

        return x
