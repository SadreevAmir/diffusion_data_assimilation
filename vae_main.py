import logging
import os
import torch
from functools import partial
from vae import VAE
from vae_trainer import VAETrainingConfig, VAETrainer
from utils import NpyImageDataset, channel_normalize
from diffusers.optimization import get_cosine_schedule_with_warmup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

assert torch.cuda.is_available(), "CUDA is not available"
torch.backends.cudnn.benchmark = True


def main():
    config = VAETrainingConfig()

    transform = partial(
        channel_normalize,
        channel_mean=config.channel_mean,
        channel_std=config.channel_std,
    )

    dataset_train = NpyImageDataset(folder=config.data_dir_train, transform=transform, mmap_mode='r')
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

    dataset_valid = NpyImageDataset(folder=config.data_dir_valid, transform=transform, mmap_mode='r')
    valid_dataloader = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers_val,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    model = VAE(
        in_channels=2,
        latent_channels=8,
        base_channels=64,
        scale_factor=8,
        max_channels=512,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=len(train_dataloader) * config.num_epochs,
    )

    trainer = VAETrainer(
        config=config,
        model=model,
        optimizer=optimizer,
        data_loader_train=train_dataloader,
        data_loader_val=valid_dataloader,
        lr_scheduler=lr_scheduler,
    )

    trainer.train_loop()


if __name__ == "__main__":
    main()
