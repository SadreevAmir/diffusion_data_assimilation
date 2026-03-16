import torch
from torchdiffeq import odeint
from utils import make_normalized_xy_grid, get_device

_FIXED_STEP_METHODS = {'euler', 'midpoint', 'rk4', 'heun3'}


class Sampler:
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def sample_conditioned(
        self,
        mask,
        observed,
        size,
        num_timesteps,
        device=None,
        method='euler',
        rtol=1e-3,
        atol=1e-4,
    ):
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
            model_input = torch.cat([x, grid, mask, observed], dim=1)
            return -self.model(model_input, t_tensor).sample

        kwargs = {}
        if method in _FIXED_STEP_METHODS:
            kwargs['options'] = {'step_size': 1.0 / num_timesteps}
        else:
            kwargs['rtol'] = rtol
            kwargs['atol'] = atol

        trajectory = odeint(f, x0, timesteps, method=method, **kwargs)
        return trajectory[-1]
