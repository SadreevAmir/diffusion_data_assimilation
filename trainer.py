import torch
import os
import torch.nn.functional as F
from dataclasses import dataclass, asdict
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers.training_utils import EMAModel
from huggingface_hub import upload_folder, create_repo
import json
from datetime import datetime

from model import VideoDiffusionModel
from sampler import Sampler

torch.set_float32_matmul_precision('high')


@dataclass
class TrainingConfig:
    train_batch_size: int = 16
    eval_batch_size: int = 24
    num_epochs: int = 15
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    mixed_precision: str = 'bf16'
    seed: int = 0
    push_to_hub: bool = True
    hub_model_id: str = 'amirsadreev/diffusion_data_assimilation'
    save_best_model: bool = True
    base_output_dir: str = 'checkpoints'
    in_channels: int = 2          
    model_dim: int = 64 
    dim_mults: tuple = (1, 2, 4, 8)
    num_frames: int = 1 


def build_model(config: TrainingConfig) -> VideoDiffusionModel:
    model = VideoDiffusionModel(
        in_channels=config.in_channels,
        dim=config.model_dim,
        dim_mults=config.dim_mults,
        temporal_compression=(False, False, False, False),
        self_attns=(False, False, False, True),
        num_frames=config.num_frames,
    )
    return model


class UNetTrainer:
    """
    Тренер — интерфейс полностью совпадает с оригиналом.
    Единственное отличие: add_noise_func теперь работает с 5D тензором (B, C, T, H, W).
    """

    def __init__(self, config, model, optimizer, data_loader_train, data_loader_val,
                 lr_scheduler, add_noise_func):

        self.config = config
        self.run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(config.base_output_dir, self.run_name)

        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = data_loader_train
        self.val_dataloader = data_loader_val
        self.lr_scheduler = lr_scheduler
        self.add_noise = add_noise_func

        self.best_val_loss = float('inf')
        self.val_history = []

    def save_model_custom(self, name="last_model.pth"):
        os.makedirs(self.output_dir, exist_ok=True)

        save_path = os.path.join(self.output_dir, name)
        unwrapped = self.accelerator.unwrap_model(self.model)
        torch.save(unwrapped.state_dict(), save_path)

        ema_path = os.path.join(self.output_dir, f"ema_{name}")
        torch.save(self.ema_model.state_dict(), ema_path)

    def _forward_pass(self, batch: torch.Tensor):
        clean = batch

        if clean.ndim == 4:
            clean = clean.unsqueeze(2)

        B = clean.shape[0]
        timesteps = torch.rand(B, device=self.accelerator.device)   

        noisy, v_real = self.add_noise(clean, timesteps)          
        v_pred = self.model(noisy, timesteps)

        loss = F.mse_loss(v_pred, v_real)
        return loss

    def compute_val_loss(self):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.val_dataloader:
                loss = self._forward_pass(batch)
                avg_loss = self.accelerator.gather_for_metrics(loss).mean()
                total_loss += avg_loss.item()

        return total_loss / len(self.val_dataloader)

    def train_loop(self):
        logging_dir = os.path.join(self.output_dir, "logs")
        accelerator_project_config = ProjectConfiguration(
            project_dir=self.output_dir,
            logging_dir=logging_dir,
        )

        self.accelerator = Accelerator(
            mixed_precision=self.config.mixed_precision,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            log_with="tensorboard",
            project_config=accelerator_project_config,
        )

        if self.accelerator.is_main_process:
            os.makedirs(self.output_dir, exist_ok=True)
            if getattr(self.config, "push_to_hub", False):
                create_repo(repo_id=self.config.hub_model_id, exist_ok=True)

        config_dict = {
            k: str(v) if isinstance(v, (tuple, list)) else v
            for k, v in asdict(self.config).items()
            }
        self.accelerator.init_trackers(
            "diffusion_training",
            config=config_dict,
        )

        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.val_dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model, self.optimizer,
            self.train_dataloader, self.val_dataloader,
            self.lr_scheduler,
        )

        self.unwrapped_model = self.accelerator.unwrap_model(self.model)
        self.ema_model = EMAModel(self.unwrapped_model.parameters(), decay=0.999)
        self.ema_model.to(self.accelerator.device)

        for epoch in range(self.config.num_epochs):
            self.model.train()
            progress_bar = tqdm(
                total=len(self.train_dataloader),
                disable=not self.accelerator.is_local_main_process,
            )
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model):
                    loss = self._forward_pass(batch)

                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    self.ema_model.step(self.unwrapped_model.parameters())

                self.accelerator.log({"train_loss": loss.item()}, step=step)
                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item())

            progress_bar.close()

            val_loss = self.compute_val_loss()

            self.val_history.append({
                "epoch": epoch,
                "val_loss": float(val_loss),
                "step": step,
                "timestamp": datetime.now().isoformat(),
            })

            self.accelerator.log({"val_loss": val_loss}, step=step)

            if self.accelerator.is_main_process:
                with open(os.path.join(self.output_dir, "metrics.json"), "w") as f:
                    json.dump(self.val_history, f, indent=4)

                self.save_model_custom("last_model.pth")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_model_custom("best_model.pth")
                    self.accelerator.print(
                        f"--- New best model saved! Loss: {val_loss:.6f}"
                    )

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
                        print(f"Hub upload error: {e}")

        self.accelerator.end_training()