import torch
from utils import make_normalized_xy_grid, add_noise, get_device

_IMAGE_SHAPE = (2, 320, 256)


class Sampler:
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def sample_no_condition(self, size, num_timesteps, batch_size, device=None):
        device = device or get_device()
        H, W = size
        grid = make_normalized_xy_grid(H, W).to(device).expand(batch_size, -1, -1, -1)
        timesteps = torch.linspace(1.0, 0.001, num_timesteps, device=device)
        dt = 1.0 / num_timesteps

        x = torch.randn((batch_size, 2, H, W), device=device)

        for t in timesteps:
            t_tensor = torch.full((batch_size,), t.item() * 1000, device=device)
            v_t = self.model(torch.cat([x, grid], dim=1), t_tensor).sample
            x = x - dt * v_t

        return x

    def _guided_step(self, x_t, t, y, mask, grid, dt, guidance_scale=1.0, eps=1e-8):
        batch_size = x_t.shape[0]
        device = x_t.device

        x_t = x_t.detach().requires_grad_(True)
        t_tensor = torch.full((batch_size,), t * 1000, device=device)

        with torch.enable_grad():
            v_t = self.model(torch.cat([x_t, grid], dim=1), t_tensor).sample
            x_hat_1 = x_t - t * v_t
            loss = ((x_hat_1 * mask - y) ** 2).view(batch_size, -1).mean(dim=1).mean()
            grad = torch.autograd.grad(loss, x_t)[0]

        v_t = v_t.detach()
        grad = grad.detach()

        grad_norm = torch.norm(grad.view(batch_size, -1), dim=1).view(batch_size, 1, 1, 1)
        grad_normalized = grad / (grad_norm + eps)
        v_guided = v_t + guidance_scale * grad_normalized

        with torch.no_grad():
            x_next = x_t.detach() + v_guided * dt

        return x_next

    @torch.no_grad()
    def sample_with_condition(self, y, mask, size, num_timesteps, guidance_scale=1.0, device=None, batch_size=12):
        device = device or get_device()
        H, W = size
        grid = make_normalized_xy_grid(H, W).to(device).expand(batch_size, -1, -1, -1)
        timesteps = torch.linspace(1.0, 0.001, num_timesteps, device=device)
        dt = -1.0 / num_timesteps

        y    = y.to(device).expand(batch_size, -1, -1, -1).contiguous()
        mask = mask.to(device).expand(batch_size, -1, -1, -1).contiguous()
        x    = torch.randn((batch_size, 2, H, W), device=device)

        for t in timesteps:
            x = self._guided_step(x_t=x, t=t.item(), y=y, mask=mask, grid=grid, dt=dt,
                                  guidance_scale=guidance_scale)
        return x

    @torch.no_grad()
    def correct_with_condition(self, background, y, mask, size, num_timesteps,
                               guidance_scale=1.0, device=None, batch_size=12, noise_val=0.5):
        device = device or get_device()
        H, W = size
        grid = make_normalized_xy_grid(H, W).to(device).expand(batch_size, -1, -1, -1)
        timesteps = torch.linspace(noise_val, 0.001, num_timesteps, device=device)
        dt = -noise_val / num_timesteps

        y          = y.to(device).expand(batch_size, -1, -1, -1).contiguous()
        mask       = mask.to(device).expand(batch_size, -1, -1, -1).contiguous()
        background = background.to(device)

        x, _ = add_noise(background, torch.tensor([noise_val], device=device))

        for t in timesteps:
            x = self._guided_step(x_t=x, t=t.item(), y=y, mask=mask, grid=grid, dt=dt,
                                  guidance_scale=guidance_scale)
        return x
