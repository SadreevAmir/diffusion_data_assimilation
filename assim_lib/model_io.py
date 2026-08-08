from __future__ import annotations

import json
import os

import torch
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.training_utils import EMAModel

from .config import TrainingConfig
from .sampler import Sampler


def build_unet(config: TrainingConfig) -> UNet2DModel:
    return UNet2DModel(
        sample_size=config.image_size,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        layers_per_block=config.layers_per_block,
        block_out_channels=config.block_out_channels,
        down_block_types=config.down_block_types,
        up_block_types=config.up_block_types,
        dropout=config.dropout,
        norm_num_groups=config.norm_num_groups,
    )


def load_run_metadata(run_dir: str) -> dict:
    metadata_path = os.path.join(run_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        return {}
    with open(metadata_path) as handle:
        return json.load(handle)


def resolve_checkpoint_name(run_dir: str, checkpoint_name: str) -> str:
    """Choose a validation-safe checkpoint for new and recognized legacy runs."""
    if checkpoint_name != "auto":
        return checkpoint_name
    metadata = load_run_metadata(run_dir)
    data_config = metadata.get("data_config", {})
    if data_config.get("split_protocol") == "3dvar_main_200d":
        return "ema_best_model.pth"
    return "ema_last_model.pth"


def load_sampler(run_dir: str, checkpoint_name: str, model_config: dict, device=None) -> Sampler:
    checkpoint_name = resolve_checkpoint_name(run_dir, checkpoint_name)
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
    metadata = load_run_metadata(run_dir)
    metadata["resolved_checkpoint_name"] = checkpoint_name
    return Sampler(model, metadata=metadata)
