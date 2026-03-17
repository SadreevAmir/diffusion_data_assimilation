import torch
from torchdiffeq import odeint
from utils import make_normalized_xy_grid, get_device

_FIXED_STEP_METHODS = {'euler', 'midpoint', 'rk4', 'heun3'}


class ControlNetSampler:
    def __init__(self, model):
        self.model = model

    def _forward(
        self,
        x: torch.Tensor,
        t_tensor: torch.Tensor,
        grid: torch.Tensor,
        mask: torch.Tensor,
        observed: torch.Tensor,
    ) -> torch.Tensor:
        unet_input = torch.cat([x, grid], dim=1)
        controlnet_cond = torch.cat([mask, observed], dim=1)
        return self.model(unet_input, controlnet_cond, t_tensor)

    @torch.no_grad()
    def sample_conditioned(
        self,
        mask: torch.Tensor,
        observed: torch.Tensor,
        size: tuple,
        num_timesteps: int,
        device=None,
        method: str = 'euler',
        rtol: float = 1e-3,
        atol: float = 1e-4,
    ) -> torch.Tensor:
        batch_size = observed.shape[0]
        device = device or get_device()
        mask = mask.to(device)
        observed = observed.to(device)
        H, W = size
        grid = make_normalized_xy_grid(H, W).to(device).expand(batch_size, -1, -1, -1)
        timesteps = torch.linspace(1.0, 0.001, num_timesteps, device=device)

        x0 = torch.randn((batch_size, 2, H, W), device=device)

        def f(t, x):
            t_tensor = t.expand(batch_size) * 1000
            return self._forward(x, t_tensor, grid, mask, observed)

        kwargs = {}
        if method in _FIXED_STEP_METHODS:
            kwargs['options'] = {'step_size': 1.0 / num_timesteps}
        else:
            kwargs['rtol'] = rtol
            kwargs['atol'] = atol

        trajectory: torch.Tensor = odeint(f, x0, timesteps, method=method, **kwargs)  # type: ignore[assignment]
        return trajectory[-1]

