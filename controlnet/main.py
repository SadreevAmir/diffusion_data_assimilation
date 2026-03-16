import logging
import os
import numpy as np
import torch
from functools import partial
from .trainer import ControlNetTrainingConfig, ControlNetTrainer
from utils import NpyImageDataset, MixedSatelliteTrackDataset, channel_normalize, add_noise
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.controlnet import ControlNetModel
from diffusers.optimization import get_cosine_schedule_with_warmup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

assert torch.cuda.is_available(), "CUDA is not available"
torch.backends.cudnn.benchmark = True


def main():
    config = ControlNetTrainingConfig()

    transform = partial(
        channel_normalize,
        channel_mean=config.channel_mean,
        channel_std=config.channel_std,
    )

    dataset_train = NpyImageDataset(
        folder=config.data_dir_train,
        transform=transform,
        preload=False,
        mmap_mode='r',
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers_train,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )

    dataset_valid = NpyImageDataset(
        folder=config.data_dir_valid,
        transform=transform,
        preload=False,
        mmap_mode='r',
    )
    valid_dataloader = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers_val,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    valid_mask_path = os.path.join(os.path.dirname(config.data_dir_train), "mask_padding.npy")
    valid_mask = np.load(valid_mask_path).astype(np.float32)

    dataset_satellite_mask = MixedSatelliteTrackDataset(
        folder=config.data_dir_satellite_mask,
        image_size=config.image_size,
        valid_mask=valid_mask,
        npy_fraction=0.5,
        generate_fraction=0.3,
        empty_fraction=0.2,
        n_tracks_range=config.satellite_n_tracks_range,
        mmap_mode='r',
    )
    satellite_mask_dataloader = torch.utils.data.DataLoader(
        dataset_satellite_mask,
        batch_size=config.train_batch_size,
        shuffle=False,
        num_workers=config.num_workers_val,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # UNet processes: noisy ice (2ch) + XY grid (2ch) = 4 channels
    unet = UNet2DConditionModel(
        sample_size=config.image_size,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512, 512),
        down_block_types=(
            "DownBlock2D", "DownBlock2D", "DownBlock2D",
            "AttnDownBlock2D", "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D", "AttnUpBlock2D", "UpBlock2D",
            "UpBlock2D", "UpBlock2D",
        ),
        cross_attention_dim=None,
    )

    # ControlNet processes: satellite mask (1ch) + observed values (2ch) = 3 channels
    # Initialized from UNet encoder weights
    controlnet = ControlNetModel.from_unet(
        unet,
        conditioning_channels=config.controlnet_conditioning_channels,
    )

    optimizer = torch.optim.AdamW(
        controlnet.parameters(),
        lr=config.learning_rate,
    )

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=len(train_dataloader) * config.num_epochs,
    )

    trainer = ControlNetTrainer(
        config=config,
        unet=unet,
        controlnet=controlnet,
        optimizer=optimizer,
        data_loader_train=train_dataloader,
        data_loader_val=valid_dataloader,
        data_loader_satellite_mask=satellite_mask_dataloader,
        lr_scheduler=lr_scheduler,
        add_noise_func=add_noise,
    )

    trainer.train_loop()


if __name__ == "__main__":
    main()
