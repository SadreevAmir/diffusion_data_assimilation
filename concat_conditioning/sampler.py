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
    ) -> torch.Tensor:
        device = device or get_device()
        background = background.to(device)
        obs_values = obs_values.to(device)
        obs_mask = obs_mask.to(device)
        batch_size, channels = background.shape[:2]
        height, width = size
        grid = make_normalized_xy_grid(height, width).to(device).expand(batch_size, -1, -1, -1)
        timesteps = torch.linspace(1.0, 0.001, num_timesteps, device=device)

        x0 = torch.randn((batch_size, channels, height, width), device=device)

        def f(t, x):
            t_tensor = t.expand(batch_size) * 1000
            model_input = torch.cat([x, grid, background, obs_values, obs_mask], dim=1)
            return _model_output(self.model(model_input, t_tensor))

        kwargs = {}
        if method in _FIXED_STEP_METHODS:
            kwargs["options"] = {"step_size": 1.0 / num_timesteps}
        else:
            kwargs["rtol"] = rtol
            kwargs["atol"] = atol

        trajectory = odeint(f, x0, timesteps, method=method, **kwargs)
        return trajectory[-1]
