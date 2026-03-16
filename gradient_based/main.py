import logging
import torch
from functools import partial
from .trainer import TrainingConfig, UNetTrainer
from utils import NpyImageDataset, channel_normalize, add_noise
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

assert torch.cuda.is_available(), "CUDA is not available"
torch.backends.cudnn.benchmark = True


def main():
    config = TrainingConfig()

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

    model = UNet2DModel(
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
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=len(train_dataloader) * config.num_epochs,
    )

    trainer = UNetTrainer(
        config=config,
        model=model,
        optimizer=optimizer,
        data_loader_train=train_dataloader,
        data_loader_val=valid_dataloader,
        lr_scheduler=lr_scheduler,
        add_noise_func=add_noise,
    )

    trainer.train_loop()


if __name__ == "__main__":
    main()
