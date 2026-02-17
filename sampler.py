"""
sampler.py

Адаптирован под VideoDiffusionModel:
  - нет grid (координаты внутри модели)
  - нет .sample (модель возвращает тензор напрямую)
  - поддерживает T кадров
  - модель принимает timesteps в [0, 1], не [0, 1000]
"""

import torch
from utils import add_noise


class Sampler:
    def __init__(self, model, noise_func=None):
        self.model = model
        self.noise_func = noise_func

    @torch.no_grad()
    def sample_no_condition(self, num_timesteps, batch_size, num_frames=1, device='cuda'):
        """
        Unconditional sampling через flow matching.

        Возвращает (batch_size, 2, num_frames, 320, 256) если num_frames > 1
                или (batch_size, 2, 320, 256)             если num_frames == 1
        """
        timesteps = torch.linspace(1.0, 0.001, num_timesteps, device=device)
        dt = 1.0 / num_timesteps

        x = torch.randn((batch_size, 2, num_frames, 320, 256), device=device)

        for t in timesteps:
            t_tensor = torch.full((batch_size,), t, device=device)   # [0,1] — не *1000
            v_t = self.model(x, t_tensor)                             # (B, 2, T, H, W)
            x = x - dt * v_t

        # Убираем временное измерение если T=1
        if num_frames == 1:
            x = x.squeeze(2)   # (B, 2, H, W)

        return x

    def guided_flow_matching_step(
        self,
        x_t,
        t,
        y,
        mask,
        dt,
        guidance_scale=1.0,
        eps=1e-8,
    ):
        """
        Один шаг guided flow matching с градиентным guidance.

        x_t  : (B, 2, T, H, W) или (B, 2, H, W)
        y    : наблюдения той же формы
        mask : маска наблюдений той же формы
        """
        batch_size = x_t.shape[0]
        device = x_t.device

        # Добавляем T=1 если нужно
        squeeze_out = False
        if x_t.ndim == 4:
            x_t = x_t.unsqueeze(2)
            y = y.unsqueeze(2)
            mask = mask.unsqueeze(2)
            squeeze_out = True

        x_t = x_t.detach().requires_grad_(True)
        t_tensor = torch.full((batch_size,), t, device=device)

        with torch.enable_grad():
            v_t = self.model(x_t, t_tensor)               # (B, 2, T, H, W)
            x_hat_1 = x_t - t * v_t                       # предсказание x_0

            loss_per_sample = ((x_hat_1 * mask - y) ** 2).view(batch_size, -1).mean(dim=1)
            loss = loss_per_sample.mean()
            grad = torch.autograd.grad(loss, x_t)[0]

        v_t = v_t.detach()
        grad = grad.detach()

        grad_flat = grad.view(batch_size, -1)
        grad_norm = torch.norm(grad_flat, dim=1).view(batch_size, 1, 1, 1, 1)
        grad_normalized = grad / (grad_norm + eps)

        v_guided = v_t + guidance_scale * grad_normalized

        with torch.no_grad():
            x_next = x_t.detach() + v_guided * dt

        if squeeze_out:
            x_next = x_next.squeeze(2)

        return x_next

    @torch.no_grad()
    def sample_with_condition(
        self, y, mask, num_timesteps, guidance_scale=1.0,
        device='cuda', batch_size=12, num_frames=1,
    ):
        timesteps = torch.linspace(1.0, 0.0, num_timesteps, device=device)
        dt = -1.0 / num_timesteps

        y = y.to(device).expand(batch_size, -1, -1, -1).contiguous()
        mask = mask.to(device).expand(batch_size, -1, -1, -1).contiguous()

        x = torch.randn((batch_size, 2, 320, 256), device=device)

        for t in timesteps:
            x = self.guided_flow_matching_step(
                x_t=x, t=t, y=y, mask=mask,
                dt=dt, guidance_scale=guidance_scale,
            )

        return x

    @torch.no_grad()
    def correct_with_condition(
        self, background, y, mask, num_timesteps,
        guidance_scale=1.0, device='cuda', batch_size=12, noise_val=0.5,
    ):
        timesteps = torch.linspace(noise_val, 0.0, num_timesteps, device=device)
        dt = -noise_val / num_timesteps

        y = y.to(device).expand(batch_size, -1, -1, -1).contiguous()
        mask = mask.to(device).expand(batch_size, -1, -1, -1).contiguous()
        background = background.to(device)

        x, _ = add_noise(background, torch.tensor([noise_val], device=device))
        x = x.to(device)

        for t in timesteps:
            x = self.guided_flow_matching_step(
                x_t=x, t=t, y=y, mask=mask,
                dt=dt, guidance_scale=guidance_scale,
            )

        return x

    @torch.no_grad()
    def concentration_to_thickness(
        self, y, mask, num_timesteps, guidance_scale=1.0,
        device='cuda', batch_size=12,
    ):
        timesteps = torch.linspace(1.0, 0.0, num_timesteps, device=device)
        dt = -1.0 / num_timesteps

        y = y.to(device).expand(batch_size, -1, -1, -1).contiguous()
        mask = mask.to(device).expand(batch_size, -1, -1, -1).contiguous()

        x = torch.randn((batch_size, 2, 320, 256), device=device)

        for t in timesteps:
            x = self.guided_flow_matching_step(
                x_t=x, t=t, y=y, mask=mask,
                dt=dt, guidance_scale=guidance_scale,
            )

        return x