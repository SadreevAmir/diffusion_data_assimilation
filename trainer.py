import torch
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataclasses import dataclass
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers.training_utils import EMAModel
from huggingface_hub import upload_folder, create_repo
import json
import warnings
from datetime import datetime

from utils import make_normalized_xy_grid
from sampler import Sampler

torch.set_float32_matmul_precision('high')

@dataclass
class TrainingConfig:
    train_batch_size = 2
    eval_batch_size = 24
    num_epochs = 15
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    mixed_precision = 'bf16'
    seed = 0
    push_to_hub = True
    hub_model_id = 'amirsadreev/diffusion_data_assimilation'
    save_best_model = True
    base_output_dir = 'checkpoints' 

class UNetTrainer:
    def __init__(self, config, model, optimizer, data_loader_train, data_loader_val, 
                 lr_scheduler, add_noise_func):
        
        self.config = config
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(config.base_output_dir, run_name)
        
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = data_loader_train
        self.val_dataloader = data_loader_val
        self.lr_scheduler = lr_scheduler
        self.add_noise = add_noise_func
        
        self.best_val_loss = float('inf')
        self.val_history = []

    def save_model_custom(self, name="last_model.pth"):
        save_path = os.path.join(self.output_dir, name)
        torch.save(self.accelerator.unwrap_model(self.model).state_dict(), save_path)
        
        ema_path = os.path.join(self.output_dir, f"ema_{name}")
        torch.save(self.ema_model.state_dict(), ema_path)

    def train_loop(self):
        logging_dir = os.path.join(self.output_dir, "logs")
        accelerator_project_config = ProjectConfiguration(
            project_dir=self.output_dir, 
            logging_dir=logging_dir
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

        self.accelerator.init_trackers("diffusion_training")
        self.model.enable_xformers_memory_efficient_attention()
        
        self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.lr_scheduler
        )
    
        self.unwrapped_model = self.accelerator.unwrap_model(self.model)
        self.ema_model = EMAModel(self.unwrapped_model.parameters(), decay=0.999)
        self.ema_model.to(self.accelerator.device)
    
        global_step = 0
    
        for epoch in range(self.config.num_epochs):
            self.model.train()
            progress_bar = tqdm(total=len(self.train_dataloader), disable=not self.accelerator.is_local_main_process)
            
            for step, batch in enumerate(self.train_dataloader):
                clean_images = batch
                bs = clean_images.shape[0]
                timesteps = torch.rand(bs, device=self.accelerator.device)
                noisy_images, v_real = self.add_noise(clean_images, timesteps)
    
                with self.accelerator.accumulate(self.model):
                    grid_coords = make_normalized_xy_grid().to(self.accelerator.device)
                    grid_coords = grid_coords.expand(bs, -1, -1, -1)
                    model_input = torch.cat([noisy_images, grid_coords], dim=1)
                
                    v_pred = self.model(model_input, timesteps * 1000, return_dict=False)[0]
                    loss = F.mse_loss(v_pred, v_real)
    
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    self.ema_model.step(self.unwrapped_model.parameters())
    
                progress_bar.update(1)
                global_step += 1
            
            progress_bar.close()
        
            val_loss = self.compute_val_loss()
            self.val_history.append({"epoch": epoch, "val_loss": val_loss})
            self.accelerator.log({"val/loss": val_loss}, step=global_step)

            if self.accelerator.is_main_process:
                self.save_model_custom("last_model.pth")
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_model_custom("best_model.pth")
                    self.accelerator.print(f"Новая лучшая модель! Loss: {val_loss:.6f}")
                
                if self.config.push_to_hub:
                    try:
                        upload_folder(
                            repo_id=self.config.hub_model_id,
                            folder_path=self.output_dir,
                            path_in_repo=os.path.basename(self.output_dir),
                            commit_message=f"Epoch {epoch} - run update",
                            ignore_patterns=["*.pth", "*.pt", "*.bin"],
                        )
                    except Exception as e:
                        print(f"Hub error: {e}")

        self.accelerator.end_training()