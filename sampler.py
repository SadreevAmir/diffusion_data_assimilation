import torch
from utils import make_normalized_xy_grid, add_noise, get_device

# (channels, H, W) — соответствует архитектуре UNet2DModel
_IMAGE_SHAPE = (2, 320, 256)


class Sampler:
    def __init__(self, model, noise_func=None):
        self.model = model
        self.noise_func = noise_func

    @torch.no_grad()
    def sample_no_condition(self, num_timesteps, batch_size, device=None):
        device = device or get_device()
        grid = make_normalized_xy_grid().expand(batch_size, -1, -1, -1).to(device)
        timesteps = torch.linspace(1.0, 0.001, num_timesteps, device=device)
        dt = 1.0 / num_timesteps

        x = torch.randn((batch_size, *_IMAGE_SHAPE), device=device)

        for t in timesteps:
            t_tensor = torch.full((batch_size,), t * 1000, device=device)
            v_t = self.model(torch.cat([x, grid], dim=1), t_tensor).sample
            x = x - dt * v_t

        return x

    def guided_flow_matching_step(
        self,
        x_t,
        t,
        y,
        mask,
        grid,
        dt,
        guidance_scale=1.0,
        eps=1e-8
    ):
        batch_size = x_t.shape[0]
        device = x_t.device

        x_t = x_t.detach().requires_grad_(True)
        t_tensor = torch.full((batch_size,), t * 1000, device=device)

        with torch.enable_grad():
            out = self.model(torch.cat([x_t, grid], dim=1), t_tensor)
            v_t = out.sample

            x_hat_1 = x_t - t * v_t

            loss_per_sample = ((x_hat_1 * mask - y) ** 2).view(batch_size, -1).mean(dim=1)
            loss = loss_per_sample.mean()

            grad = torch.autograd.grad(loss, x_t)[0]

        v_t = v_t.detach()
        grad = grad.detach()

        grad_flat = grad.view(batch_size, -1)
        grad_norm = torch.norm(grad_flat, dim=1).view(batch_size, 1, 1, 1)
        grad_normalized = grad / (grad_norm + eps)

        v_guided = v_t + guidance_scale * grad_normalized

        with torch.no_grad():
            x_next = x_t.detach() + v_guided * dt

        return x_next

    @torch.no_grad()
    def sample_with_condition(self, y, mask, num_timesteps, guidance_scale=1.0, device=None, batch_size=12):
        device = device or get_device()
        timesteps = torch.linspace(1.0, 0.0, num_timesteps, device=device)
        dt = -1.0 / num_timesteps
        grid = make_normalized_xy_grid().expand(batch_size, -1, -1, -1).to(device)

        y = y.to(device).expand(batch_size, -1, -1, -1).contiguous()
        mask = mask.to(device).expand(batch_size, -1, -1, -1).contiguous()

        x = torch.randn((batch_size, *_IMAGE_SHAPE), device=device)

        for t in timesteps:
            x = self.guided_flow_matching_step(
                x_t=x, t=t, y=y, mask=mask, grid=grid, dt=dt,
                guidance_scale=guidance_scale,
            )

        return x

    @torch.no_grad()
    def correct_with_condition(self, background, y, mask, num_timesteps, guidance_scale=1.0, device=None, batch_size=12, noise_val=0.5):
        device = device or get_device()
        timesteps = torch.linspace(noise_val, 0.0, num_timesteps, device=device)
        dt = -noise_val / num_timesteps
        grid = make_normalized_xy_grid().expand(batch_size, -1, -1, -1).to(device)

        y = y.to(device).expand(batch_size, -1, -1, -1).contiguous()
        mask = mask.to(device).expand(batch_size, -1, -1, -1).contiguous()
        background = background.to(device)

        x, _ = add_noise(background, torch.tensor([noise_val], device=device))
        x = x.to(device)

        for t in timesteps:
            x = self.guided_flow_matching_step(
                x_t=x, t=t, y=y, mask=mask, grid=grid, dt=dt,
                guidance_scale=guidance_scale,
            )

        return x
