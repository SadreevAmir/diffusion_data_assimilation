import copy
import torch
import torch.nn as nn
from diffusers.models.unets.unet_2d import UNet2DModel


class SeaIceControlNet(nn.Module):
   
    def __init__(self, unet: UNet2DModel, conditioning_channels: int = 2):
        super().__init__()

        self.cond_conv = nn.Conv2d(
            conditioning_channels, unet.config.block_out_channels[0],
            kernel_size=3, padding=1,
        )
        nn.init.zeros_(self.cond_conv.weight)
        nn.init.zeros_(self.cond_conv.bias)

        self.conv_in = copy.deepcopy(unet.conv_in)
        self.time_proj = copy.deepcopy(unet.time_proj)
        self.time_embedding = copy.deepcopy(unet.time_embedding)
        self.down_blocks = copy.deepcopy(unet.down_blocks)
        self.mid_block = copy.deepcopy(unet.mid_block)

        # Zero convs для каждого уровня skip connections
        # Структура down_block_res_samples:
        #   - 1 тензор от conv_in: block_out_channels[0]
        #   - для каждого блока с downsample: layers_per_block + 1 тензоров
        #   - для последнего блока (без downsample): layers_per_block тензоров
        down_channels = [unet.config.block_out_channels[0]]
        for i, out_ch in enumerate(unet.config.block_out_channels):
            n = unet.config.layers_per_block
            has_downsample = (i < len(unet.config.block_out_channels) - 1)
            down_channels.extend([out_ch] * (n + (1 if has_downsample else 0)))

        mid_channels = unet.config.block_out_channels[-1]

        self.zero_convs = nn.ModuleList([
            self._zero_conv(ch) for ch in down_channels
        ])
        self.mid_zero_conv = self._zero_conv(mid_channels)

    @staticmethod
    def _zero_conv(channels: int) -> nn.Conv2d:
        conv = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(conv.weight)
        nn.init.zeros_(conv.bias)
        return conv

    def _get_emb(self, timestep: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=sample.device)
        elif timestep.ndim == 0:
            timestep = timestep[None].to(sample.device)
        timestep = timestep * torch.ones(sample.shape[0], dtype=timestep.dtype, device=timestep.device)
        t_emb = self.time_proj(timestep).to(dtype=self.conv_in.weight.dtype)
        return self.time_embedding(t_emb)

    def forward(
        self,
        noisy_sample: torch.Tensor,
        conditioning: torch.Tensor,
        timestep: torch.Tensor,
    ):
        """
        Args:
            noisy_sample:  (B, 4, H, W) — тот же вход что у UNet (noisy + grid)
            conditioning:  (B, 3, H, W) — mask + observed values
            timestep:      (B,) — масштабированный timestep (0..1000)

        Returns:
            down_residuals: list[Tensor] matching down_block_res_samples
            mid_residual:   Tensor matching mid_block output
        """
        emb = self._get_emb(timestep, noisy_sample)

        # Encoder: UNet вход + conditioning через zero conv
        sample = self.conv_in(noisy_sample) + self.cond_conv(conditioning)

        down_block_res_samples = (sample,)
        for block in self.down_blocks:
            sample, res_samples = block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        if self.mid_block is not None:
            sample = self.mid_block(sample, emb)

        down_residuals = [zc(s) for zc, s in zip(self.zero_convs, down_block_res_samples)]
        mid_residual = self.mid_zero_conv(sample)

        return down_residuals, mid_residual


class ControlledUNet(nn.Module):
    """
    Замороженный UNet2DModel + обучаемый SeaIceControlNet.

    ControlNet инжектирует conditioning через skip connections UNet.
    Только параметры controlnet обучаются.

    Args:
        unet:       претренированный UNet2DModel (замораживается)
        controlnet: SeaIceControlNet (обучается)
    """

    def __init__(self, unet: UNet2DModel, controlnet: SeaIceControlNet):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        self.unet.requires_grad_(False)

    def _get_emb(self, timestep: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        unet = self.unet
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=sample.device)
        elif timestep.ndim == 0:
            timestep = timestep[None].to(sample.device)
        timestep = timestep * torch.ones(sample.shape[0], dtype=timestep.dtype, device=timestep.device)
        t_emb = unet.time_proj(timestep).to(dtype=unet.dtype)
        return unet.time_embedding(t_emb)

    def forward(
        self,
        noisy_sample: torch.Tensor,
        conditioning: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            noisy_sample:  (B, 4, H, W) — noisy image + XY grid
            conditioning:  (B, 3, H, W) — mask + observed values
            timestep:      (B,) — масштабированный timestep (0..1000)

        Returns:
            predicted velocity: (B, 2, H, W)
        """
        unet = self.unet
        emb = self._get_emb(timestep, noisy_sample)

        # UNet encoder
        sample = unet.conv_in(noisy_sample)
        down_block_res_samples = (sample,)
        for block in unet.down_blocks:
            sample, res_samples = block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        if unet.mid_block is not None:
            sample = unet.mid_block(sample, emb)

        # ControlNet residuals
        cn_down, cn_mid = self.controlnet(noisy_sample, conditioning, timestep)

        # Inject residuals into skip connections
        down_block_res_samples = tuple(d + r for d, r in zip(down_block_res_samples, cn_down))
        sample = sample + cn_mid

        # UNet decoder
        for block in unet.up_blocks:
            res_samples = down_block_res_samples[-len(block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(block.resnets)]
            sample = block(sample, res_samples, emb)

        # Post-process
        sample = unet.conv_norm_out(sample)
        sample = unet.conv_act(sample)
        sample = unet.conv_out(sample)

        return sample
