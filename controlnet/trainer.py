import platform
import torch
import os
import numpy as np
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
from itertools import cycle

from utils import make_normalized_xy_grid, channel_denormalize, generate_satellite_track_mask
from sampler import ControlNetSampler

torch.set_float32_matmul_precision('high')

logger = get_logger(__name__)


def _default_mixed_precision() -> str:
    return 'bf16' if torch.cuda.is_available() else 'no'


def _default_data_dir() -> str:
    if platform.system() == 'Darwin':
        return '/Users/amir/sciml/sea_ice_data'
    return '/mnt/sciml/a.sadreev/sea_ice_data'


@dataclass
class ControlNetTrainingConfig:
    data_dir_train: str = _default_data_dir() + "/train"
    data_dir_valid: str = _default_data_dir() + "/valid"
    data_dir_satellite_mask: str = _default_data_dir() + "/satellite_samples"

    with open(os.path.join(data_dir_train, "stats.json")) as f:
        stats = json.load(f)
    channel_mean: tuple = tuple(stats["mean"])
    channel_std: tuple = tuple(stats["std"])

    image_size: tuple = (320, 256)
    # UNet input: noisy image (2ch) + XY grid (2ch)
    in_channels: int = 4
    out_channels: int = 2
    # ControlNet conditioning input: mask (1ch) + observed values (2ch)
    controlnet_conditioning_channels: int = 3

    num_workers_train: int = 6
    num_workers_val: int = 4

    train_batch_size: int = 24
    eval_batch_size: int = 1
    num_epochs: int = 20
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    mixed_precision: str = field(default_factory=_default_mixed_precision)
    seed: int = 0

    satellite_n_tracks_range: tuple = (0, 5)

    timestep_sampler: str = 'uniform'
    timestep_beta_params: tuple = (2.0, 1.0)

    masked_loss_weight: float = 0.1

    sample_every_n_epochs: int = 1
    num_sample_timesteps: int = 50

    push_to_hub: bool = True
    hub_model_id: str = 'amirsadreev/sea_ice_diffusion'
    base_output_dir: str = 'checkpoints/controlnet'
    resume_from_checkpoint: str = ""


class ControlNetTrainer:
    def __init__(self, config: ControlNetTrainingConfig, unet, controlnet, optimizer,
                 data_loader_train, data_loader_val, data_loader_satellite_mask,
                 lr_scheduler, add_noise_func):

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
        self.accelerator.init_trackers("controlnet_training", config=loggable_config)

        unet.requires_grad_(False)

        for m in (unet, controlnet):
            if (self.accelerator.device.type == 'cuda'
                    and hasattr(m, "enable_xformers_memory_efficient_attention")):
                m.enable_xformers_memory_efficient_attention()

        (self.model,
         self.controlnet,
         self.optimizer,
         self.train_dataloader,
         self.val_dataloader,
         self.satellite_mask_dataloader,
         self.lr_scheduler) = self.accelerator.prepare(
            unet, controlnet, optimizer,
            data_loader_train, data_loader_val, data_loader_satellite_mask,
            lr_scheduler,
        )

        self.satellite_mask_iter = cycle(self.satellite_mask_dataloader)

        self.ema_controlnet = EMAModel(
            self.accelerator.unwrap_model(self.controlnet).parameters(), decay=0.999
        )
        self.ema_controlnet.to(self.accelerator.device)

        H, W = config.image_size
        self._grid = make_normalized_xy_grid(H, W).to(self.accelerator.device)

        mask_path = os.path.join(os.path.dirname(config.data_dir_train), "mask_padding.npy")
        self._valid_mask = np.load(mask_path).astype(np.float32)

    def _sample_timesteps(self, bs: int) -> torch.Tensor:
        device = self.accelerator.device
        if self.config.timestep_sampler == 'beta':
            alpha, beta = self.config.timestep_beta_params
            return torch.distributions.Beta(alpha, beta).sample((bs,)).to(device)
        return torch.rand(bs, device=device)

    def _make_unet_input(self, noisy_images: torch.Tensor) -> torch.Tensor:
        bs = noisy_images.shape[0]
        grid = self._grid.expand(bs, -1, -1, -1)
        return torch.cat([noisy_images, grid], dim=1)

    def _make_controlnet_cond(
        self, clean_images: torch.Tensor, satellite_mask: torch.Tensor
    ) -> torch.Tensor:
        bs = clean_images.shape[0]
        observed = clean_images * satellite_mask
        mask_exp = satellite_mask.expand(bs, -1, -1, -1)
        return torch.cat([mask_exp, observed], dim=1)

    def _model_forward(
        self,
        unet_input: torch.Tensor,
        controlnet_cond: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        down_samples, mid_sample = self.controlnet(
            sample=unet_input,
            timestep=timesteps * 1000,
            encoder_hidden_states=None,
            controlnet_cond=controlnet_cond,
            return_dict=False,
        )
        v_pred = self.model(
            sample=unet_input,
            timestep=timesteps * 1000,
            encoder_hidden_states=None,
            down_block_additional_residuals=down_samples,
            mid_block_additional_residual=mid_sample,
            return_dict=False,
        )[0]
        return v_pred

    def _masked_mse(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_exp = mask.expand_as(pred)
        n = mask_exp.sum().clamp(min=1.0)
        return (F.mse_loss(pred, target, reduction='none') * mask_exp).sum() / n

    def save_model_custom(self, name: str = "last"):
        os.makedirs(self.output_dir, exist_ok=True)
        torch.save(
            self.accelerator.unwrap_model(self.controlnet).state_dict(),
            os.path.join(self.output_dir, f"controlnet_{name}.pth"),
        )
        torch.save(
            self.ema_controlnet.state_dict(),
            os.path.join(self.output_dir, f"controlnet_ema_{name}.pth"),
        )

    def compute_val_loss(self) -> tuple[float, float, float]:
        self.model.eval()
        self.controlnet.eval()
        total_full = 0.0
        total_masked = 0.0
        n = len(self.val_dataloader)

        with torch.no_grad():
            for batch in self.val_dataloader:
                clean_images = batch
                bs = clean_images.shape[0]
                satellite_mask = next(self.satellite_mask_iter)

                timesteps = self._sample_timesteps(bs)
                noisy_images, v_real = self.add_noise(clean_images, timesteps)

                unet_input = self._make_unet_input(noisy_images)
                controlnet_cond = self._make_controlnet_cond(clean_images, satellite_mask)

                v_pred = self._model_forward(unet_input, controlnet_cond, timesteps)
                loss_full   = F.mse_loss(v_pred, v_real)
                loss_masked = self._masked_mse(v_pred, v_real, satellite_mask)

                total_full   += self.accelerator.gather_for_metrics(loss_full).mean().item()   # type: ignore[union-attr]
                total_masked += self.accelerator.gather_for_metrics(loss_masked).mean().item() # type: ignore[union-attr]

        vf = total_full / n
        vm = total_masked / n
        return vf, vm, vf + self.config.masked_loss_weight * vm

    def save_samples(self, epoch: int):
        self.model.eval()
        self.controlnet.eval()
        device = self.accelerator.device
        sampler = ControlNetSampler(
            unet=self.accelerator.unwrap_model(self.model),
            controlnet=self.accelerator.unwrap_model(self.controlnet),
        )

        clean_images = next(iter(self.val_dataloader))[:1]

        mask = torch.from_numpy(
            generate_satellite_track_mask(
                self.config.image_size, 1, self._valid_mask, self.config.satellite_n_tracks_range
            )
        ).unsqueeze(1).to(device)

        observed = clean_images * mask

        sample = sampler.sample_conditioned(
            mask=mask,
            observed=observed,
            size=self.config.image_size,
            num_timesteps=self.config.num_sample_timesteps,
            device=device,
        )

        truth  = channel_denormalize(clean_images.clone(), self.config.channel_mean, self.config.channel_std)
        sample = channel_denormalize(sample, self.config.channel_mean, self.config.channel_std)
        mask_np = mask[0, 0].cpu().numpy()

        titles = [
            ['Truth — Concentration', 'Track mask', f'Predicted — Concentration (ep {epoch})'],
            ['Truth — Thickness',     '',           f'Predicted — Thickness (ep {epoch})'],
        ]
        cmaps = [['Blues_r', 'gray', 'Blues_r'], ['viridis', 'gray', 'viridis']]
        data  = [
            [truth[0, 0].cpu().numpy(),  mask_np,  sample[0, 0].cpu().numpy()],
            [truth[0, 1].cpu().numpy(),  mask_np,  sample[0, 1].cpu().numpy()],
        ]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        for r in range(2):
            for c in range(3):
                axes[r, c].imshow(data[r][c], cmap=cmaps[r][c])
                axes[r, c].set_title(titles[r][c])
                axes[r, c].axis('off')
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
            self.controlnet.train()
            progress_bar = tqdm(
                total=len(self.train_dataloader),
                disable=not self.accelerator.is_local_main_process,
            )
            progress_bar.set_description(f"Epoch {epoch}")

            for batch in self.train_dataloader:
                clean_images = batch
                satellite_mask = next(self.satellite_mask_iter)
                bs = clean_images.shape[0]
                timesteps = self._sample_timesteps(bs)
                noisy_images, v_real = self.add_noise(clean_images, timesteps)

                with self.accelerator.accumulate(self.controlnet):
                    unet_input = self._make_unet_input(noisy_images)
                    controlnet_cond = self._make_controlnet_cond(clean_images, satellite_mask)

                    v_pred = self._model_forward(unet_input, controlnet_cond, timesteps)
                    loss_full   = F.mse_loss(v_pred, v_real)
                    loss_masked = self._masked_mse(v_pred, v_real, satellite_mask)
                    loss        = loss_full + self.config.masked_loss_weight * loss_masked

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.controlnet.parameters(), 1.0)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    self.ema_controlnet.step(self.accelerator.unwrap_model(self.controlnet).parameters())

                self.accelerator.log({
                    "train_loss":        loss.item(),
                    "train_loss_full":   loss_full.item(),
                    "train_loss_masked": loss_masked.item(),
                }, step=global_step)
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item(), masked=loss_masked.item())

            progress_bar.close()

            val_loss_full, val_loss_masked, val_loss = self.compute_val_loss()
            self.val_history.append({
                "epoch":           epoch,
                "val_loss":        val_loss,
                "val_loss_full":   val_loss_full,
                "val_loss_masked": val_loss_masked,
                "step":            global_step,
                "timestamp":       datetime.now().isoformat(),
            })
            self.accelerator.log({
                "val_loss":        val_loss,
                "val_loss_full":   val_loss_full,
                "val_loss_masked": val_loss_masked,
            }, step=global_step)
            logger.info("Epoch %d — val_loss: %.6f  full: %.6f  masked: %.6f",
                        epoch, val_loss, val_loss_full, val_loss_masked)

            if self.accelerator.is_main_process:
                with open(os.path.join(self.output_dir, "metrics.json"), "w") as f:
                    json.dump(self.val_history, f, indent=4)

                self.save_model_custom("last")

                if epoch % self.config.sample_every_n_epochs == 0:
                    self.save_samples(epoch)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_model_custom("best")
                    logger.info("New best model saved! val_loss: %.6f", val_loss)

                if self.config.push_to_hub:
                    try:
                        upload_folder(
                            repo_id=self.config.hub_model_id,
                            folder_path=self.output_dir,
                            path_in_repo=f"controlnet/{self.run_name}",
                            commit_message=f"Epoch {epoch} - val_loss {val_loss:.4f}",
                            ignore_patterns=["*.pth", "*.pt", "*.bin"],
                        )
                    except Exception as e:
                        logger.error("Hub upload error: %s", e)

        self.accelerator.end_training()
