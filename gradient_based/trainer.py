import platform
import torch
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
from dataclasses import dataclass, asdict, field
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers.training_utils import EMAModel
from huggingface_hub import upload_folder, create_repo
from datetime import datetime

from checkpointing import save_training_checkpoint
from utils import make_normalized_xy_grid, channel_denormalize
from .sampler import Sampler

torch.set_float32_matmul_precision('high')

logger = get_logger(__name__)


def _default_mixed_precision() -> str:
    return 'bf16' if torch.cuda.is_available() else 'no'


def _default_data_dir() -> str:
    if platform.system() == 'Darwin':
        return '/Users/amir/sciml/sea_ice_data'
    return '/mnt/sciml/a.sadreev/sea_ice_data'


def _load_channel_stats(data_dir: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    with open(os.path.join(data_dir, "stats.json")) as f:
        stats = json.load(f)
    return tuple(stats["mean"]), tuple(stats["std"])


@dataclass
class TrainingConfig:
    data_dir_train: str = _default_data_dir() + "/train"
    data_dir_valid: str = _default_data_dir() + "/valid"

    channel_mean: tuple = field(default_factory=tuple)
    channel_std: tuple = field(default_factory=tuple)

    image_size: tuple = (320, 256)
    in_channels: int = 4
    out_channels: int = 2
    layers_per_block: int = 2
    block_out_channels: tuple = (64, 128, 256, 512, 512)
    down_block_types: tuple = (
        "DownBlock2D", "DownBlock2D", "DownBlock2D",
        "AttnDownBlock2D", "DownBlock2D",
    )
    up_block_types: tuple = (
        "UpBlock2D", "AttnUpBlock2D", "UpBlock2D",
        "UpBlock2D", "UpBlock2D",
    )

    num_workers_train: int = 6
    num_workers_val: int = 4

    train_batch_size: int = 24
    eval_batch_size: int = 1
    num_epochs: int = 15
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    mixed_precision: str = field(default_factory=_default_mixed_precision)
    seed: int = 0

    sample_every_n_epochs: int = 1
    num_sample_timesteps: int = 50

    push_to_hub: bool = True
    hub_model_id: str = 'amirsadreev/sea_ice_diffusion'
    base_output_dir: str = 'checkpoints/gradient_based'
    resume_from_checkpoint: str = ""

    def __post_init__(self):
        if not self.channel_mean or not self.channel_std:
            self.channel_mean, self.channel_std = _load_channel_stats(self.data_dir_train)


class UNetTrainer:
    def __init__(self, config: TrainingConfig, model, optimizer, data_loader_train,
                 data_loader_val, lr_scheduler, add_noise_func):

        self.config = config
        self.run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(config.base_output_dir, self.run_name)

        self.add_noise = add_noise_func
        self.best_val_loss = float('inf')
        self.val_history = []

        logging_dir = os.path.join(self.output_dir, "logs")
        self.accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="tensorboard",
            project_config=ProjectConfiguration(
                project_dir=self.output_dir,
                logging_dir=logging_dir,
            ),
        )

        if self.accelerator.is_main_process:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.output_dir, "config.json"), "w") as f:
                json.dump({k: str(v) if isinstance(v, tuple) else v for k, v in asdict(config).items()}, f, indent=4)
            if config.push_to_hub:
                create_repo(repo_id=config.hub_model_id, exist_ok=True)

        loggable_config = {k: str(v) if isinstance(v, tuple) else v for k, v in asdict(config).items()}
        self.accelerator.init_trackers("diffusion_training", config=loggable_config)

        if (self.accelerator.device.type == 'cuda'
                and hasattr(model, "enable_xformers_memory_efficient_attention")):
            model.enable_xformers_memory_efficient_attention()

        (self.model,
         self.optimizer,
         self.train_dataloader,
         self.val_dataloader,
         self.lr_scheduler) = self.accelerator.prepare(
            model, optimizer, data_loader_train, data_loader_val, lr_scheduler
        )

        self.unwrapped_model = self.accelerator.unwrap_model(self.model)
        self.ema_model = EMAModel(self.unwrapped_model.parameters(), decay=0.999)
        self.ema_model.to(self.accelerator.device)

        H, W = config.image_size
        self._grid = make_normalized_xy_grid(H, W).to(self.accelerator.device)

        if self.accelerator.is_main_process:
            self.save_model_custom("initial_model.pth", epoch=-1, global_step=0)

    def save_model_custom(self, name="last_model.pth", epoch: int | None = None, global_step: int | None = None):
        os.makedirs(self.output_dir, exist_ok=True)
        unwrapped = self.accelerator.unwrap_model(self.model)
        save_training_checkpoint(
            os.path.join(self.output_dir, name),
            model=unwrapped,
            config=self.config,
            run_name=self.run_name,
            model_state_dict=unwrapped.state_dict(),
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            epoch=epoch,
            global_step=global_step,
            best_val_loss=self.best_val_loss,
        )
        save_training_checkpoint(
            os.path.join(self.output_dir, f"ema_{name}"),
            model=unwrapped,
            config=self.config,
            run_name=self.run_name,
            model_state_dict=unwrapped.state_dict(),
            ema_state_dict=self.ema_model.state_dict(),
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            epoch=epoch,
            global_step=global_step,
            best_val_loss=self.best_val_loss,
        )

    def compute_val_loss(self) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.val_dataloader:
                clean_images = batch
                bs = clean_images.shape[0]
                timesteps = torch.rand(bs, device=self.accelerator.device)
                noisy_images, v_real = self.add_noise(clean_images, timesteps)

                grid = self._grid.expand(bs, -1, -1, -1)
                model_input = torch.cat([noisy_images, grid], dim=1)

                v_pred = self.model(model_input, timesteps * 1000, return_dict=False)[0]
                loss = F.mse_loss(v_pred, v_real)
                total_loss += self.accelerator.gather_for_metrics(loss).mean().item()  # type: ignore[union-attr]

        return total_loss / len(self.val_dataloader)

    def save_samples(self, epoch: int):
        self.model.eval()
        sampler = Sampler(self.accelerator.unwrap_model(self.model))

        sample = sampler.sample_no_condition(
            size=self.config.image_size,
            num_timesteps=self.config.num_sample_timesteps,
            batch_size=1,
            device=self.accelerator.device,
        )

        sample = channel_denormalize(sample, self.config.channel_mean, self.config.channel_std)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(sample[0, 0].cpu().numpy(), cmap='Blues_r')
        axes[0].set_title(f'Concentration — Epoch {epoch}')
        axes[0].axis('off')
        axes[1].imshow(sample[0, 1].cpu().numpy(), cmap='viridis')
        axes[1].set_title(f'Thickness — Epoch {epoch}')
        axes[1].axis('off')
        plt.tight_layout()

        samples_dir = os.path.join(self.output_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)
        save_path = os.path.join(samples_dir, f"epoch_{epoch:04d}.png")
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        logger.info("Sample saved: %s", save_path)

    def train_loop(self):
        global_step = 0

        for epoch in range(self.config.num_epochs):
            self.model.train()
            progress_bar = tqdm(
                total=len(self.train_dataloader),
                disable=not self.accelerator.is_local_main_process,
            )
            progress_bar.set_description(f"Epoch {epoch}")

            for batch in self.train_dataloader:
                clean_images = batch
                bs = clean_images.shape[0]
                timesteps = torch.rand(bs, device=self.accelerator.device)
                noisy_images, v_real = self.add_noise(clean_images, timesteps)

                with self.accelerator.accumulate(self.model):
                    grid = self._grid.expand(bs, -1, -1, -1)
                    model_input = torch.cat([noisy_images, grid], dim=1)

                    v_pred = self.model(model_input, timesteps * 1000, return_dict=False)[0]
                    loss = F.mse_loss(v_pred, v_real)

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    self.ema_model.step(self.unwrapped_model.parameters())

                self.accelerator.log({"train_loss": loss.item()}, step=global_step)
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item())

            progress_bar.close()

            val_loss = self.compute_val_loss()
            self.val_history.append({
                "epoch": epoch,
                "val_loss": float(val_loss),
                "step": global_step,
                "timestamp": datetime.now().isoformat(),
            })
            self.accelerator.log({"val_loss": val_loss}, step=global_step)
            logger.info("Epoch %d — val_loss: %.6f", epoch, val_loss)

            if self.accelerator.is_main_process:
                with open(os.path.join(self.output_dir, "metrics.json"), "w") as f:
                    json.dump(self.val_history, f, indent=4)

                self.save_model_custom("last_model.pth", epoch=epoch, global_step=global_step)

                if epoch % self.config.sample_every_n_epochs == 0:
                    self.save_samples(epoch)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_model_custom("best_model.pth", epoch=epoch, global_step=global_step)
                    logger.info("New best model saved! val_loss: %.6f", val_loss)

                if self.config.push_to_hub:
                    try:
                        upload_folder(
                            repo_id=self.config.hub_model_id,
                            folder_path=self.output_dir,
                            path_in_repo=f"gradient_based/{self.run_name}",
                            commit_message=f"Epoch {epoch} - val_loss {val_loss:.4f}",
                            ignore_patterns=["*.pth", "*.pt", "*.bin"],
                        )
                    except Exception as e:
                        logger.error("Hub upload error: %s", e)

        self.accelerator.end_training()
