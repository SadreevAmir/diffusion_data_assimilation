import torch
import sys
import os
from trainer import TrainingConfig, UNetTrainer
from utils import NpyImageDataset, channel_normalize, add_noise
from diffusers import UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup

def main():
    
    channel_mean = [0.1382167, 0.1816227]
    channel_std = [0.32978467, 0.51380478]
    
    config = TrainingConfig()
    dataset_train = NpyImageDataset(
        folder="/mnt/sciml/a.sadreev/sea_ice_data/train",
        transform=lambda x: channel_normalize(x, channel_mean, channel_std),
        preload=False,
        mmap_mode='r',
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=config.train_batch_size, 
        shuffle=True, 
        num_workers=6,
        pin_memory=True
    )
    
    dataset_valid = NpyImageDataset(
        folder="/mnt/sciml/a.sadreev/sea_ice_data/valid",
        transform=lambda x: channel_normalize(x, channel_mean, channel_std),
        preload=False,
        mmap_mode='r',
    )
    
    valid_dataloader = torch.utils.data.DataLoader(
        dataset_valid, 
        batch_size=config.eval_batch_size,
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    model = UNet2DModel(
        sample_size=(320, 256),
        in_channels=4,
        out_channels=2,
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
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )
    
    trainer = UNetTrainer(
        config=config,
        model=model, 
        optimizer=optimizer, 
        data_loader_train=train_dataloader, 
        data_loader_val=valid_dataloader, 
        lr_scheduler=lr_scheduler, 
        add_noise_func=add_noise, 
        save_dir='checkpoints'
    )
    
    trainer.train_loop()

if __name__ == "__main__":
    main()