from __future__ import annotations

import torch
from torchdiffeq import odeint

from utils import get_device, make_normalized_xy_grid


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

    @torch.no_grad()
    def sample_conditioned(
        self,
        background: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
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
    ) -> torch.Tensor:
        device = device or get_device()
        background = background.to(device)
        obs_values = obs_values.to(device)
        obs_mask = obs_mask.to(device)
        if valid_mask is not None:
            valid_mask = valid_mask.to(device)
        batch_size, channels = background.shape[:2]
        height, width = size
        num_timesteps = max(int(num_timesteps), 2)
        grid = make_normalized_xy_grid(height, width).to(device).expand(batch_size, -1, -1, -1)

        if start_mode == "noise":
            start_t = 1.0
            x0 = torch.randn((batch_size, channels, height, width), device=device)
        elif start_mode == "background":
            start_t = min(max(float(start_noise_level), 0.001), 1.0)
            eps = torch.randn_like(background)
            x0 = (1.0 - start_t) * background + start_t * eps
        elif start_mode == "bridge":
            start_t = 1.0
            x0 = background
        else:
            raise ValueError(f"Unknown start_mode={start_mode!r}; expected 'noise', 'background', or 'bridge'")

        timesteps = torch.linspace(start_t, 0.001, num_timesteps, device=device)

        if obs_guidance_scale > 0:
            sample = self._sample_guided_euler(
                x0=x0,
                timesteps=timesteps,
                grid=grid,
                background=background,
                obs_values=obs_values,
                obs_mask=obs_mask,
                obs_guidance_scale=float(obs_guidance_scale),
                obs_guidance_eps=float(obs_guidance_eps),
            )
            if enforce_observations:
                sample = torch.where(obs_mask > 0, obs_values, sample)
            if valid_mask is not None:
                sample = torch.where(valid_mask > 0, sample, background)
            return sample

        def f(t, x):
            t_tensor = t.expand(batch_size) * 1000
            model_input = torch.cat([x, grid, background, obs_values, obs_mask], dim=1)
            return _model_output(self.model(model_input, t_tensor))

        kwargs = {}
        if method in _FIXED_STEP_METHODS:
            time_span = abs(start_t - 0.001)
            kwargs["options"] = {"step_size": max(time_span / max(num_timesteps - 1, 1), 1e-6)}
        else:
            kwargs["rtol"] = rtol
            kwargs["atol"] = atol

        trajectory = odeint(f, x0, timesteps, method=method, **kwargs)
        sample = trajectory[-1]
        if enforce_observations:
            sample = torch.where(obs_mask > 0, obs_values, sample)
        if valid_mask is not None:
            sample = torch.where(valid_mask > 0, sample, background)
        return sample

    def _sample_guided_euler(
        self,
        x0: torch.Tensor,
        timesteps: torch.Tensor,
        grid: torch.Tensor,
        background: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_guidance_scale: float,
        obs_guidance_eps: float,
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
                model_input = torch.cat([x_t, grid, background, obs_values, obs_mask], dim=1)
                v_pred = _model_output(self.model(model_input, t_tensor))
                x_hat = x_t - t * v_pred
                obs_error = (x_hat - obs_values) * obs_mask
                obs_loss = (obs_error.square().reshape(batch_size, -1).sum(dim=1) / obs_den).mean()
                grad = torch.autograd.grad(obs_loss, x_t)[0]

            grad = grad.detach()
            v_pred = v_pred.detach()
            grad_norm = grad.reshape(batch_size, -1).norm(dim=1).view(batch_size, 1, 1, 1)
            v_guided = v_pred + obs_guidance_scale * grad / (grad_norm + obs_guidance_eps)
            x = x_t.detach() + dt * v_guided

        return x
