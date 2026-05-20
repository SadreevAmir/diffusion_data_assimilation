from __future__ import annotations

import os

import torch
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.training_utils import EMAModel

from .sampler import Sampler
from .trainer import TrainingConfig


def build_unet(config: TrainingConfig) -> UNet2DModel:
    return UNet2DModel(
        sample_size=config.image_size,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        layers_per_block=config.layers_per_block,
        block_out_channels=config.block_out_channels,
        down_block_types=config.down_block_types,
        up_block_types=config.up_block_types,
        norm_num_groups=config.norm_num_groups,
    )


def load_sampler(run_dir: str, checkpoint_name: str, model_config: dict, device=None) -> Sampler:
    config = TrainingConfig.from_dict(model_config)
    model = build_unet(config)
    checkpoint_path = os.path.join(run_dir, checkpoint_name)
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "shadow_params" in state:
        ema_model = EMAModel(model.parameters())
        ema_model.load_state_dict(state)
        ema_model.copy_to(model.parameters())
    else:
        model.load_state_dict(state)
    model.eval()
    if device is not None:
        model.to(device)
    return Sampler(model)
