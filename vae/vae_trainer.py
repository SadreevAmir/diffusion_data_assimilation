import platform
import torch
import os
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict, field
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from huggingface_hub import upload_folder, create_repo
from datetime import datetime

from vae.vae import vae_loss
from utils import channel_denormalize

torch.set_float32_matmul_precision('high')

logger = get_logger(__name__)


def _default_mixed_precision() -> str:
    return 'bf16' if torch.cuda.is_available() else 'no'


def _default_data_dir() -> str:
    if platform.system() == 'Darwin':
        return '/Users/amir/sciml/sea_ice_data'
    return '/mnt/sciml/a.sadreev/sea_ice_data'


@dataclass
class VAETrainingConfig:
    data_dir_train: str = _default_data_dir() + "/train"
    data_dir_valid: str = _default_data_dir() + "/valid"

    with open(os.path.join(data_dir_train, "stats.json")) as f:
        stats = json.load(f)
    channel_mean: tuple = tuple(stats["mean"])
    channel_std: tuple = tuple(stats["std"])

    image_size: tuple = (320, 256)

    num_workers_train: int = 6
    num_workers_val: int = 4

    train_batch_size: int = 16
    eval_batch_size: int = 8
    num_epochs: int = 50
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    mixed_precision: str = field(default_factory=_default_mixed_precision)
    seed: int = 0

    kl_weight: float = 1e-3

    sample_every_n_epochs: int = 5

    push_to_hub: bool = False
    hub_model_id: str = 'amirsadreev/vae_sea_ice'
    base_output_dir: str = 'checkpoints_vae'
    resume_from_checkpoint: str = ""


class VAETrainer:
    def __init__(self, config: VAETrainingConfig, model, optimizer,
                 data_loader_train, data_loader_val, lr_scheduler):

        self.config = config
        self.run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(config.base_output_dir, self.run_name)

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
        self.accelerator.init_trackers("vae_training", config=loggable_config)

        (self.model,
         self.optimizer,
         self.train_dataloader,
         self.val_dataloader,
         self.lr_scheduler) = self.accelerator.prepare(
            model, optimizer, data_loader_train, data_loader_val, lr_scheduler
        )

    def save_model(self, name="last_model.pth"):
        os.makedirs(self.output_dir, exist_ok=True)
        unwrapped = self.accelerator.unwrap_model(self.model)
        torch.save(unwrapped.state_dict(), os.path.join(self.output_dir, name))

    def compute_val_loss(self) -> tuple[float, float, float]:
        self.model.eval()
        total_loss = total_recon = total_kl = 0.0
        n = len(self.val_dataloader)

        with torch.no_grad():
            for batch in self.val_dataloader:
                recon, mu, logvar = self.model(batch)
                loss, recon_loss, kl = vae_loss(recon, batch, mu, logvar, self.config.kl_weight)

                total_loss  += self.accelerator.gather_for_metrics(loss).mean().item()       # type: ignore[union-attr]
                total_recon += self.accelerator.gather_for_metrics(recon_loss).mean().item() # type: ignore[union-attr]
                total_kl    += self.accelerator.gather_for_metrics(kl).mean().item()         # type: ignore[union-attr]

        return total_loss / n, total_recon / n, total_kl / n

    def save_samples(self, epoch: int):
        self.model.eval()
        batch = next(iter(self.val_dataloader))[:4]

        with torch.no_grad():
            recon, _, _ = self.model(batch)

        truth = channel_denormalize(batch.clone(), self.config.channel_mean, self.config.channel_std)
        recon = channel_denormalize(recon, self.config.channel_mean, self.config.channel_std)

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for i in range(4):
            axes[0, i].imshow(truth[i, 0].cpu().numpy(), cmap='Blues_r')
            axes[0, i].set_title(f'Truth {i}')
            axes[0, i].axis('off')
            axes[1, i].imshow(recon[i, 0].cpu().numpy(), cmap='Blues_r')
            axes[1, i].set_title(f'Recon {i}')
            axes[1, i].axis('off')

        plt.suptitle(f'Epoch {epoch}')
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
                with self.accelerator.accumulate(self.model):
                    recon, mu, logvar = self.model(batch)
                    loss, recon_loss, kl = vae_loss(recon, batch, mu, logvar, self.config.kl_weight)

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                self.accelerator.log({
                    "train_loss":       loss.item(),
                    "train_recon_loss": recon_loss.item(),
                    "train_kl":         kl.item(),
                }, step=global_step)
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item(), recon=recon_loss.item(), kl=kl.item())

            progress_bar.close()

            val_loss, val_recon, val_kl = self.compute_val_loss()
            self.val_history.append({
                "epoch":      epoch,
                "val_loss":   val_loss,
                "val_recon":  val_recon,
                "val_kl":     val_kl,
                "step":       global_step,
                "timestamp":  datetime.now().isoformat(),
            })
            self.accelerator.log({
                "val_loss":  val_loss,
                "val_recon": val_recon,
                "val_kl":    val_kl,
            }, step=global_step)
            logger.info("Epoch %d — val_loss: %.6f  recon: %.6f  kl: %.6f",
                        epoch, val_loss, val_recon, val_kl)

            if self.accelerator.is_main_process:
                with open(os.path.join(self.output_dir, "metrics.json"), "w") as f:
                    json.dump(self.val_history, f, indent=4)

                self.save_model("last_model.pth")

                if epoch % self.config.sample_every_n_epochs == 0:
                    self.save_samples(epoch)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_model("best_model.pth")
                    logger.info("New best model saved! val_loss: %.6f", val_loss)

                if self.config.push_to_hub:
                    try:
                        upload_folder(
                            repo_id=self.config.hub_model_id,
                            folder_path=self.output_dir,
                            path_in_repo=self.run_name,
                            commit_message=f"Epoch {epoch} - val_loss {val_loss:.4f}",
                            ignore_patterns=["*.pth", "*.pt", "*.bin"],
                        )
                    except Exception as e:
                        logger.error("Hub upload error: %s", e)

        self.accelerator.end_training()
