from __future__ import annotations

import json
import math
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers.training_utils import EMAModel
from tqdm.auto import tqdm

from utils import make_normalized_xy_grid

from .clearml_tracking import ClearMLTracker
from .dashboard import make_multi_case_background_condition_assim_figure
from .sampler import Sampler
from .transforms import channel_denormalize, make_conditioned_model_input


logger = get_logger(__name__)


_DASHBOARD_EVERY_N_EPOCHS = 5
_UNCONDITIONAL_DASHBOARD_EVERY_N_EPOCHS = 1
_DASHBOARD_NUM_CASES = 12
_DASHBOARD_HISTORY_EPOCHS = 1
_DASHBOARD_CASES_PER_PAGE = 12
_DASHBOARD_CHANNELS = (0,)
_DASHBOARD_DPI = 200
_DASHBOARD_PANEL_WIDTH = 3.6
_DASHBOARD_PANEL_HEIGHT = 2.5
_DASHBOARD_LARGE_PANEL_WIDTH = 5.5
_DASHBOARD_LARGE_PANEL_HEIGHT = 4.4
_UNCONDITIONAL_PANEL_WIDTH = 7.0
_UNCONDITIONAL_PANEL_HEIGHT = 5.6


def _debug(message: str) -> None:
    print(f"[assim_lib][trainer] {message}", flush=True)


def _default_mixed_precision() -> str:
    return "bf16" if torch.cuda.is_available() else "no"


@dataclass
class TrainingConfig:
    image_size: tuple[int, int] = (320, 256)
    in_channels: int = 23
    out_channels: int = 4
    train_batch_size: int = 8
    eval_batch_size: int = 4
    num_workers_train: int = 4
    num_workers_val: int = 2
    num_epochs: int = 20
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    mixed_precision: str = field(default_factory=_default_mixed_precision)
    seed: int = 0
    training_objective: str = "diffusion"
    timestep_sampler: str = "uniform"
    timestep_beta_params: tuple[float, float] = (2.0, 1.0)
    obs_loss_weight: float = 0.1
    smoothness_loss_weight: float = 0.0
    background_dropout_probability: float = 0.0
    conditioning_mode_probabilities: dict[str, float] | None = None
    sample_every_n_epochs: int = 1
    num_sample_timesteps: int = 200
    sample_method: str = "euler"
    sample_rtol: float = 1e-3
    sample_atol: float = 1e-4
    sample_use_ema: bool = True
    ema_decay: float = 0.999
    sample_start_mode: str = "background"
    sample_start_noise_level: float = 0.5
    sample_enforce_observations: bool = True
    sample_obs_guidance_scale: float = 0.0
    sample_obs_guidance_eps: float = 1e-8
    base_output_dir: str = "checkpoints/concat_conditioning"
    run_name: str = ""
    tracker: str | None = None
    resume_from_checkpoint: str = ""
    clearml_project_name: str = "concat_conditioning"
    clearml_task_name: str = ""
    clearml_tags: tuple[str, ...] = ()
    clearml_output_uri: str | None = None
    clearml_env_path: str | None = None
    clearml_upload_checkpoints: bool = False
    metric_every_n_epochs: int = 1
    metric_num_cases: int = 24
    metric_num_timesteps: int = 200
    metric_num_ensemble: int = 10
    metric_stride_days: int = 15
    metric_save_ensemble_samples: bool = False
    metric_ensemble_save_dtype: str = "float16"

    block_out_channels: tuple[int, ...] = (64, 128, 256, 512, 512)
    layers_per_block: int = 2
    dropout: float = 0.0
    down_block_types: tuple[str, ...] = (
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
    )
    up_block_types: tuple[str, ...] = (
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    )
    norm_num_groups: int = 32

    @classmethod
    def from_dict(cls, config: dict) -> "TrainingConfig":
        known = {field.name for field in cls.__dataclass_fields__.values()}
        values = {key: value for key, value in config.items() if key in known}
        tuple_keys = {
            "image_size",
            "timestep_beta_params",
            "block_out_channels",
            "down_block_types",
            "up_block_types",
            "clearml_tags",
        }
        for key in tuple_keys:
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


class UNetTrainer:
    def __init__(self, config: TrainingConfig, model, optimizer, data_loader_train,
                 data_loader_val, lr_scheduler, add_noise_func,
                 experiment_config=None, model_config=None, data_config=None,
                 dashboard_dataset=None):
        self.config = config
        self.run_name = config.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(config.base_output_dir, self.run_name)
        self.add_noise = add_noise_func
        self.best_val_loss = float("inf")
        self.val_history = []
        self.dashboard_history = []
        self.clearml = None
        self.data_config = data_config or {}
        self.dashboard_dataset = dashboard_dataset
        self.fields = list(self.data_config.get("fields", [f"ch{i}" for i in range(config.out_channels)]))
        self.channel_means = list(self.data_config.get("means", [0.0] * config.out_channels))
        self.channel_stds = list(self.data_config.get("stds", [1.0] * config.out_channels))
        self._shared_viz_obs_mask: torch.Tensor | None = None
        self._shared_viz_obs_mask_attempted = False

        logging_dir = os.path.join(self.output_dir, "logs")
        _debug(f"output_dir={self.output_dir}")
        _debug(f"creating Accelerator mixed_precision={config.mixed_precision} tracker={config.tracker}")
        accelerator_kwargs = {
            "mixed_precision": config.mixed_precision,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "project_config": ProjectConfiguration(project_dir=self.output_dir, logging_dir=logging_dir),
        }
        if config.tracker:
            accelerator_kwargs["log_with"] = config.tracker
        self.accelerator = Accelerator(**accelerator_kwargs)

        if self.accelerator.is_main_process:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.output_dir, "config.json"), "w") as f:
                json.dump(_jsonable(asdict(config)), f, indent=4)
            run_metadata = {
                "fields": self.fields,
                "normalization_means": self.channel_means,
                "normalization_stds": self.channel_stds,
                "sample_metrics_values_space": "physical",
                "concentration_clipping": "[0, 1]" if "siconc" in self.fields else None,
                "data_config": _jsonable(self.data_config),
                "training_config": _jsonable(asdict(config)),
                "experiment_config": _jsonable(experiment_config) if experiment_config is not None else None,
                "model_config": _jsonable(model_config) if model_config is not None else None,
            }
            with open(os.path.join(self.output_dir, "metadata.json"), "w") as f:
                json.dump(run_metadata, f, indent=4)
            _debug("initializing ClearML on main process")
            self.clearml = ClearMLTracker(
                project_name=config.clearml_project_name,
                task_name=config.clearml_task_name or self.run_name,
                tags=config.clearml_tags,
                output_uri=config.clearml_output_uri,
                env_path=config.clearml_env_path,
            )
            self.clearml.connect("training_config", _jsonable(asdict(config)))
            if experiment_config is not None:
                self.clearml.connect("experiment_config", _jsonable(experiment_config))
            if model_config is not None:
                self.clearml.connect("model_config", _jsonable(model_config))
            if data_config is not None:
                self.clearml.connect("data_config", _jsonable(data_config))
            self.clearml.connect("run_metadata", run_metadata)
        else:
            _debug("non-main process: ClearML init skipped")

        if config.tracker:
            self.accelerator.init_trackers("concat_conditioning_training", config=_jsonable(asdict(config)))

        if (self.accelerator.device.type == "cuda"
                and hasattr(model, "enable_xformers_memory_efficient_attention")):
            model.enable_xformers_memory_efficient_attention()

        (self.model,
         self.optimizer,
         self.train_dataloader,
         self.val_dataloader,
         self.lr_scheduler) = self.accelerator.prepare(
            model, optimizer, data_loader_train, data_loader_val, lr_scheduler
        )
        _debug(f"accelerator ready device={self.accelerator.device}")

        self.unwrapped_model = self.accelerator.unwrap_model(self.model)
        self.ema_model = EMAModel(self.unwrapped_model.parameters(), decay=float(config.ema_decay))
        self.ema_model.to(self.accelerator.device)

        height, width = config.image_size
        self._grid = make_normalized_xy_grid(height, width).to(self.accelerator.device)

    def _sample_timesteps(self, batch_size: int) -> torch.Tensor:
        device = self.accelerator.device
        if self.config.timestep_sampler == "beta":
            alpha, beta = self.config.timestep_beta_params
            return torch.distributions.Beta(alpha, beta).sample((batch_size,)).to(device)
        return torch.rand(batch_size, device=device)

    def _batch_to_device(self, batch: dict) -> dict[str, torch.Tensor]:
        return {
            key: batch[key].to(self.accelerator.device, dtype=torch.float32, non_blocking=True)
            for key in ("truth", "background", "obs_values", "obs_mask", "valid_mask", "water_mask")
        }

    def _conditioned_inputs(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        background = batch["background"]
        background_mask = torch.ones_like(background)
        obs_values = batch["obs_values"]
        obs_mask = batch["obs_mask"]
        mode_probabilities = getattr(self.config, "conditioning_mode_probabilities", None)
        if mode_probabilities is not None:
            names = ("no_background", "no_track", "no_conditioning", "both")
            unexpected = set(mode_probabilities) - set(names)
            if unexpected:
                raise ValueError(f"Unknown conditioning modes: {sorted(unexpected)}")
            probabilities = [float(mode_probabilities.get(name, 0.0)) for name in names]
            if any(probability < 0.0 for probability in probabilities):
                raise ValueError("conditioning_mode_probabilities cannot contain negative values")
            if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError("conditioning_mode_probabilities must sum to 1.0")
            if self.model.training:
                draw = torch.rand((background.shape[0], 1, 1, 1), device=background.device)
                no_background = draw < probabilities[0]
                no_track = (draw >= probabilities[0]) & (draw < probabilities[0] + probabilities[1])
                no_conditioning = (
                    (draw >= probabilities[0] + probabilities[1])
                    & (draw < probabilities[0] + probabilities[1] + probabilities[2])
                )
                keep_background = (~(no_background | no_conditioning)).to(dtype=background.dtype)
                keep_track = (~(no_track | no_conditioning)).to(dtype=obs_values.dtype)
                background = background * keep_background
                background_mask = background_mask * keep_background
                obs_values = obs_values * keep_track
                obs_mask = obs_mask * keep_track.to(dtype=obs_mask.dtype)
            return background, background_mask, obs_values, obs_mask

        dropout_probability = float(getattr(self.config, "background_dropout_probability", 0.0))
        if not 0.0 <= dropout_probability <= 1.0:
            raise ValueError(
                "background_dropout_probability must be within [0, 1], "
                f"got {dropout_probability}"
            )
        if self.model.training and dropout_probability > 0.0:
            keep = (
                torch.rand((background.shape[0], 1, 1, 1), device=background.device) >= dropout_probability
            ).to(dtype=background.dtype)
            background = background * keep
            background_mask = background_mask * keep
        return background, background_mask, obs_values, obs_mask

    def _conditioned_background(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._conditioned_inputs(batch)[0]

    def _make_model_input(
        self,
        noisy_truth: torch.Tensor,
        batch: dict[str, torch.Tensor],
        background: torch.Tensor | None = None,
        background_mask: torch.Tensor | None = None,
        obs_values: torch.Tensor | None = None,
        obs_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = noisy_truth.shape[0]
        grid = self._grid.expand(batch_size, -1, -1, -1)
        if background is None or background_mask is None or obs_values is None or obs_mask is None:
            (
                conditioned_background,
                conditioned_background_mask,
                conditioned_obs_values,
                conditioned_obs_mask,
            ) = self._conditioned_inputs(batch)
            background = conditioned_background if background is None else background
            background_mask = conditioned_background_mask if background_mask is None else background_mask
            obs_values = conditioned_obs_values if obs_values is None else obs_values
            obs_mask = conditioned_obs_mask if obs_mask is None else obs_mask
        return make_conditioned_model_input(
            noisy_truth,
            grid,
            background,
            background_mask,
            obs_values,
            obs_mask,
            batch["water_mask"],
        )

    def _make_training_pair(
        self,
        truth: torch.Tensor,
        batch: dict[str, torch.Tensor],
        timesteps: torch.Tensor,
        residual_background: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.training_objective == "diffusion":
            return self.add_noise(truth, timesteps)
        if self.config.training_objective == "diffusion_residual":
            background = batch["background"] if residual_background is None else residual_background
            return self.add_noise(truth - background, timesteps)
        if self.config.training_objective == "bridge":
            t = timesteps.view(-1, *([1] * (truth.dim() - 1)))
            state = (1.0 - t) * truth + t * batch["background"]
            target_velocity = batch["background"] - truth
            return state, target_velocity
        raise ValueError(
            f"Unknown training_objective={self.config.training_objective!r}; "
            "expected 'diffusion', 'diffusion_residual', or 'bridge'"
        )

    def _sample_target(self) -> str:
        return "residual" if self.config.training_objective == "diffusion_residual" else "state"

    def _sample_solver_kwargs(self) -> dict[str, object]:
        method = str(getattr(self.config, "sample_method", "euler"))
        return {
            "method": method,
            "rtol": float(getattr(self.config, "sample_rtol", 1e-3)),
            "atol": float(getattr(self.config, "sample_atol", 1e-4)),
            "memory_efficient_euler": method == "euler",
        }

    def _physical_metric_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        physical = channel_denormalize(tensor.to(dtype=torch.float32), self.channel_means, self.channel_stds)
        if "siconc" in self.fields:
            concentration_channel = self.fields.index("siconc")
            physical[:, concentration_channel].clamp_(0.0, 1.0)
        return physical

    @contextmanager
    def _sampling_model(self):
        unwrapped = self.accelerator.unwrap_model(self.model)
        if not self.config.sample_use_ema:
            yield unwrapped
            return

        self.ema_model.store(unwrapped.parameters())
        self.ema_model.copy_to(unwrapped.parameters())
        try:
            yield unwrapped
        finally:
            self.ema_model.restore(unwrapped.parameters())

    def _sampling_weight_label(self) -> str:
        if self.config.sample_use_ema:
            return f"EMA weights, decay={float(self.config.ema_decay):.6g}"
        return "raw training weights"

    @staticmethod
    def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.expand_as(pred)
        denom = mask.sum().clamp(min=1.0)
        return (F.mse_loss(pred, target, reduction="none") * mask).sum() / denom

    @staticmethod
    def _masked_smoothness_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.expand_as(pred)
        dy_mask = mask[..., 1:, :] * mask[..., :-1, :]
        dx_mask = mask[..., :, 1:] * mask[..., :, :-1]
        dy = (pred[..., 1:, :] - pred[..., :-1, :]).square() * dy_mask
        dx = (pred[..., :, 1:] - pred[..., :, :-1]).square() * dx_mask
        denom = (dy_mask.sum() + dx_mask.sum()).clamp(min=1.0)
        return (dy.sum() + dx.sum()) / denom

    def _sea_ice_concentration_smoothness_loss(
        self,
        pred: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.smoothness_loss_weight <= 0.0 or "siconc" not in self.fields:
            return pred.new_tensor(0.0)
        channel = self.fields.index("siconc")
        return self._masked_smoothness_loss(
            pred[:, channel:channel + 1],
            mask[:, channel:channel + 1],
        )

    @staticmethod
    def _masked_error_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
        mask = mask.expand_as(pred)
        denom = mask.sum().clamp(min=1.0)
        diff = (pred - target) * mask
        mae = diff.abs().sum() / denom
        rmse = torch.sqrt((diff.square().sum() / denom).clamp(min=0.0))
        return float(mae.item()), float(rmse.item())

    @staticmethod
    def _masked_error_sums(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[float, float, float]:
        mask = mask.expand_as(pred)
        count = mask.sum()
        diff = (pred - target) * mask
        return (
            float(diff.abs().sum().item()),
            float(diff.square().sum().item()),
            float(count.item()),
        )

    def save_model_custom(self, name: str = "last_model.pth"):
        os.makedirs(self.output_dir, exist_ok=True)
        unwrapped = self.accelerator.unwrap_model(self.model)
        torch.save(unwrapped.state_dict(), os.path.join(self.output_dir, name))
        torch.save(self.ema_model.state_dict(), os.path.join(self.output_dir, f"ema_{name}"))

    def _report_train_metrics(self, loss, loss_full, loss_obs, loss_smooth, step: int):
        if self.clearml is None:
            return
        self.clearml.report_scalar("loss_normalized/total", "train", loss.item(), step)
        self.clearml.report_scalar("loss_normalized/full", "train", loss_full.item(), step)
        self.clearml.report_scalar("loss_normalized/observed", "train", loss_obs.item(), step)
        self.clearml.report_scalar("loss_normalized/smoothness", "train", loss_smooth.item(), step)

    def _report_val_metrics(self, loss_full: float, loss_obs: float, loss: float, step: int):
        if self.clearml is None:
            return
        self.clearml.report_scalar("loss_normalized/total", "validation", loss, step)
        self.clearml.report_scalar("loss_normalized/full", "validation", loss_full, step)
        self.clearml.report_scalar("loss_normalized/observed", "validation", loss_obs, step)

    def _report_sample_validation_metrics(self, metrics: dict[str, float], step: int):
        if self.clearml is None or not metrics:
            return

        def _report(title: str, series: str, key: str):
            value = metrics.get(key)
            if value is None:
                return
            value = float(value)
            if math.isfinite(value):
                self.clearml.report_scalar(title, series, value, step)

        for kind in ("mae", "rmse"):
            for field_name in self.fields:
                for region in ("full", "obs"):
                    suffix = f"{field_name}_{region}"
                    for aggregate in ("mean", "min", "max", "of_mean"):
                        _report(
                            f"metric_physical/analysis_{kind}",
                            f"{field_name}/{aggregate}_{region}",
                            f"analysis_{kind}_{aggregate}_{suffix}",
                        )
                    _report(
                        f"metric_physical/analysis_{kind}",
                        f"{field_name}/background_{region}",
                        f"background_{kind}_{suffix}",
                    )
                    _report(
                        f"metric_physical/background_{kind}", f"{field_name}/{region}", f"background_{kind}_{suffix}"
                    )
        for field_name in self.fields:
            for region in ("full", "obs"):
                suffix = f"{field_name}_{region}"
                for aggregate in ("mean", "min", "max", "of_mean"):
                    _report(
                        "metric_physical/analysis_rmse_skill",
                        f"{field_name}/{aggregate}_{region}",
                        f"analysis_rmse_skill_{aggregate}_{suffix}",
                    )
                _report("metric_physical/count", f"{field_name}/{region}", f"metric_count_{suffix}")

    @staticmethod
    def _meta_mean(value):
        if torch.is_tensor(value):
            if value.numel() == 0:
                return None
            return float(value.to(dtype=torch.float32).mean().item())
        if isinstance(value, (list, tuple)):
            numeric = [float(item) for item in value if isinstance(item, (int, float))]
            if not numeric:
                return None
            return float(sum(numeric) / len(numeric))
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _report_condition_diagnostics(self, raw_batch: dict, step: int, series: str):
        if self.clearml is None or "meta" not in raw_batch:
            return
        meta = raw_batch["meta"]
        if not isinstance(meta, dict):
            return
        for key in ("obs_count", "observed_fraction", "sral_files_used", "empty_obs_days"):
            if key not in meta:
                continue
            value = self._meta_mean(meta[key])
            if value is not None:
                self.clearml.report_scalar(f"condition/{key}", series, value, step)

    def _upload_clearml_artifacts(self):
        if self.clearml is None:
            return
        output_dir = os.path.abspath(self.output_dir)
        self.clearml.upload_artifact("training_config", os.path.join(output_dir, "config.json"))
        self.clearml.upload_artifact("run_metadata", os.path.join(output_dir, "metadata.json"))
        self.clearml.upload_artifact("metrics", os.path.join(output_dir, "metrics.json"))
        if self.config.clearml_upload_checkpoints:
            for name in ("last_model.pth", "ema_last_model.pth", "best_model.pth", "ema_best_model.pth"):
                self.clearml.upload_artifact(name, os.path.join(output_dir, name))

    def compute_val_loss(self) -> tuple[float, float, float]:
        self.model.eval()
        total_full = 0.0
        n = len(self.val_dataloader)
        _debug(f"validation start batches={n}")

        with torch.no_grad():
            for raw_batch in self.val_dataloader:
                batch = self._batch_to_device(raw_batch)
                truth = batch["truth"]
                batch_size = truth.shape[0]
                timesteps = self._sample_timesteps(batch_size)
                background, background_mask, obs_values, obs_mask = self._conditioned_inputs(batch)
                model_state, v_real = self._make_training_pair(
                    truth, batch, timesteps, residual_background=background
                )

                model_input = self._make_model_input(
                    model_state,
                    batch,
                    background=background,
                    background_mask=background_mask,
                    obs_values=obs_values,
                    obs_mask=obs_mask,
                )
                v_pred = self.model(model_input, timesteps * 1000, return_dict=False)[0]
                loss_full = F.mse_loss(v_pred, v_real)

                total_full += self.accelerator.gather_for_metrics(loss_full).mean().item()

        loss_full = total_full / max(n, 1)
        loss_obs = 0.0
        _debug(f"validation done loss_full={loss_full:.6f} loss_obs={loss_obs:.6f}")
        return loss_full, loss_obs, loss_full

    @staticmethod
    def _empty_metric_totals() -> dict[str, float]:
        return {
            "analysis_abs": 0.0,
            "analysis_sq": 0.0,
            "background_abs": 0.0,
            "background_sq": 0.0,
            "count": 0.0,
        }

    @staticmethod
    def _finalize_metric_totals(prefix: str, totals: dict[str, float]) -> dict[str, float]:
        count = totals["count"]
        if count <= 0.0:
            return {
                f"analysis_mae_{prefix}": float("nan"),
                f"analysis_rmse_{prefix}": float("nan"),
                f"background_mae_{prefix}": float("nan"),
                f"background_rmse_{prefix}": float("nan"),
                f"analysis_rmse_skill_{prefix}": float("nan"),
                f"metric_count_{prefix}": 0.0,
            }

        analysis_mae = totals["analysis_abs"] / count
        analysis_rmse = math.sqrt(max(totals["analysis_sq"] / count, 0.0))
        background_mae = totals["background_abs"] / count
        background_rmse = math.sqrt(max(totals["background_sq"] / count, 0.0))
        skill = float("nan") if background_rmse <= 0.0 else 1.0 - analysis_rmse / background_rmse
        return {
            f"analysis_mae_{prefix}": analysis_mae,
            f"analysis_rmse_{prefix}": analysis_rmse,
            f"background_mae_{prefix}": background_mae,
            f"background_rmse_{prefix}": background_rmse,
            f"analysis_rmse_skill_{prefix}": skill,
            f"metric_count_{prefix}": count,
        }

    def _validation_cases(self, max_cases: int) -> list[dict[str, torch.Tensor]]:
        cases = []
        if max_cases <= 0:
            return cases

        for raw_batch in self.val_dataloader:
            batch = self._batch_to_device(raw_batch)
            batch_size = batch["truth"].shape[0]
            for batch_idx in range(batch_size):
                cases.append({key: value[batch_idx:batch_idx + 1] for key, value in batch.items()})
                if len(cases) >= max_cases:
                    return cases
        return cases

    def _iter_validation_batches(self, max_cases: int):
        if max_cases <= 0:
            return
        remaining = max_cases
        for raw_batch in self.val_dataloader:
            if remaining <= 0:
                return
            batch = self._batch_to_device(raw_batch)
            batch_size = batch["truth"].shape[0]
            take = min(batch_size, remaining)
            if take < batch_size:
                batch = {key: value[:take] for key, value in batch.items()}
            remaining -= take
            yield batch

    def _metric_case_indices(self, max_cases: int, stride_days: int) -> list[int]:
        val_dataset = getattr(self.val_dataloader, "dataset", None)
        if val_dataset is None or max_cases <= 0:
            return []
        hours_per_day = int(getattr(val_dataset, "hours_per_day", 1))
        if hasattr(val_dataset, "strided_case_indices"):
            indices = val_dataset.strided_case_indices(max_cases, stride_days=stride_days)
            if indices:
                if hours_per_day > 1:
                    metric_hour = min(max(int(getattr(val_dataset, "hour_index", 0)), 0), hours_per_day - 1)
                    return [(idx // hours_per_day) * hours_per_day + metric_hour for idx in indices]
                return indices
        return list(range(min(max_cases, len(val_dataset))))

    def _iter_indexed_batches(self, indices: list[int], batch_size: int):
        val_dataset = getattr(self.val_dataloader, "dataset", None)
        if val_dataset is None or not indices:
            return
        from torch.utils.data import default_collate

        batch_size = max(int(batch_size), 1)
        for start in range(0, len(indices), batch_size):
            chunk = indices[start:start + batch_size]
            samples = [val_dataset[idx] for idx in chunk]
            raw_batch = default_collate(samples)
            yield self._batch_to_device(raw_batch)

    @torch.no_grad()
    def compute_sample_validation_metrics(self, epoch: int | None = None) -> dict[str, float]:
        max_cases = int(self.config.metric_num_cases)
        if max_cases <= 0:
            return {}

        num_ensemble = max(int(getattr(self.config, "metric_num_ensemble", 1)), 1)
        stride_days = max(int(getattr(self.config, "metric_stride_days", 30)), 1)
        metric_timesteps = int(self.config.metric_num_timesteps)
        if metric_timesteps <= 0:
            metric_timesteps = int(self.config.num_sample_timesteps)

        indices = self._metric_case_indices(max_cases, stride_days)
        if not indices:
            return {}

        _debug(
            f"sample validation metrics start cases={len(indices)} "
            f"ensemble={num_ensemble} timesteps={metric_timesteps} stride_days={stride_days}"
        )
        totals = {
            field_name: {
                region: {
                    "background": self._empty_metric_totals(),
                    "members": [self._empty_metric_totals() for _ in range(num_ensemble)],
                    "of_mean": self._empty_metric_totals(),
                }
                for region in ("full", "obs")
            }
            for field_name in self.fields
        }
        total_processed = 0
        save_ensemble = bool(getattr(self.config, "metric_save_ensemble_samples", False)) and epoch is not None
        saved_batches = []
        if save_ensemble:
            save_dtype_name = str(getattr(self.config, "metric_ensemble_save_dtype", "float16")).lower()
            save_dtypes = {
                "float16": torch.float16,
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
            }
            if save_dtype_name not in save_dtypes:
                raise ValueError(
                    "metric_ensemble_save_dtype must be one of "
                    f"{sorted(save_dtypes)}, got {save_dtype_name!r}"
                )
            save_dtype = save_dtypes[save_dtype_name]

        self.model.eval()
        with self._sampling_model() as sample_model:
            sampler = Sampler(sample_model)
            for batch in self._iter_indexed_batches(indices, self.config.eval_batch_size):
                ensemble = []
                for _ in range(num_ensemble):
                    ensemble.append(
                        sampler.sample_conditioned(
                            background=batch["background"],
                            background_mask=torch.ones_like(batch["background"]),
                            obs_values=batch["obs_values"],
                            obs_mask=batch["obs_mask"],
                            water_mask=batch["water_mask"],
                            size=self.config.image_size,
                            num_timesteps=metric_timesteps,
                            device=self.accelerator.device,
                            start_mode=self.config.sample_start_mode,
                            start_noise_level=self.config.sample_start_noise_level,
                            enforce_observations=self.config.sample_enforce_observations,
                            valid_mask=batch["valid_mask"],
                            obs_guidance_scale=self.config.sample_obs_guidance_scale,
                            obs_guidance_eps=self.config.sample_obs_guidance_eps,
                            sample_target=self._sample_target(),
                            **self._sample_solver_kwargs(),
                        )
                    )
                ensemble_tensor = torch.stack(ensemble, dim=1)
                if save_ensemble:
                    saved_batches.append({
                        "samples": ensemble_tensor.detach().to(device="cpu", dtype=save_dtype),
                        "truth": batch["truth"].detach().to(device="cpu", dtype=save_dtype),
                        "background": batch["background"].detach().to(device="cpu", dtype=save_dtype),
                        "obs_values": batch["obs_values"].detach().to(device="cpu", dtype=save_dtype),
                        "obs_mask": batch["obs_mask"].detach().to(device="cpu", dtype=save_dtype),
                        "valid_mask": batch["valid_mask"].detach().to(device="cpu", dtype=save_dtype),
                        "water_mask": batch["water_mask"].detach().to(device="cpu", dtype=save_dtype),
                    })
                n_cases, n_members, n_channels, height, width = ensemble_tensor.shape
                ensemble_physical = self._physical_metric_tensor(
                    ensemble_tensor.reshape(n_cases * n_members, n_channels, height, width)
                ).reshape(n_cases, n_members, n_channels, height, width)
                ensemble_mean = ensemble_physical.mean(dim=1)
                truth_physical = self._physical_metric_tensor(batch["truth"])
                background_physical = self._physical_metric_tensor(batch["background"])
                for channel, field_name in enumerate(self.fields):
                    for region, mask in (("full", batch["valid_mask"]), ("obs", batch["obs_mask"])):
                        channel_mask = mask[:, channel:channel + 1]
                        truth_channel = truth_physical[:, channel:channel + 1]
                        background_abs, background_sq, background_count = self._masked_error_sums(
                            background_physical[:, channel:channel + 1], truth_channel, channel_mask
                        )
                        background_totals = totals[field_name][region]["background"]
                        background_totals["background_abs"] += background_abs
                        background_totals["background_sq"] += background_sq
                        background_totals["count"] += background_count

                        for member_index, analysis in enumerate(ensemble_physical.unbind(dim=1)):
                            analysis_abs, analysis_sq, count = self._masked_error_sums(
                                analysis[:, channel:channel + 1], truth_channel, channel_mask
                            )
                            member_totals = totals[field_name][region]["members"][member_index]
                            member_totals["analysis_abs"] += analysis_abs
                            member_totals["analysis_sq"] += analysis_sq
                            member_totals["count"] += min(count, background_count)

                        analysis_abs, analysis_sq, count = self._masked_error_sums(
                            ensemble_mean[:, channel:channel + 1], truth_channel, channel_mask
                        )
                        mean_totals = totals[field_name][region]["of_mean"]
                        mean_totals["analysis_abs"] += analysis_abs
                        mean_totals["analysis_sq"] += analysis_sq
                        mean_totals["count"] += min(count, background_count)
                total_processed += int(batch["truth"].shape[0])

        if total_processed == 0:
            return {}

        metrics = {
            "metric_num_cases": float(total_processed),
            "metric_num_timesteps": float(metric_timesteps),
            "metric_num_ensemble": float(num_ensemble),
            "metric_stride_days": float(stride_days),
        }
        for field_name in self.fields:
            for region in ("full", "obs"):
                suffix = f"{field_name}_{region}"
                background_totals = totals[field_name][region]["background"]

                def with_background(analysis_totals: dict[str, float]) -> dict[str, float]:
                    return {
                        "analysis_abs": analysis_totals["analysis_abs"],
                        "analysis_sq": analysis_totals["analysis_sq"],
                        "background_abs": background_totals["background_abs"],
                        "background_sq": background_totals["background_sq"],
                        "count": background_totals["count"],
                    }

                member_metrics = [
                    self._finalize_metric_totals(suffix, with_background(member_totals))
                    for member_totals in totals[field_name][region]["members"]
                ]
                mean_field_metrics = self._finalize_metric_totals(
                    suffix, with_background(totals[field_name][region]["of_mean"])
                )
                metrics[f"background_mae_{suffix}"] = mean_field_metrics[f"background_mae_{suffix}"]
                metrics[f"background_rmse_{suffix}"] = mean_field_metrics[f"background_rmse_{suffix}"]
                metrics[f"metric_count_{suffix}"] = mean_field_metrics[f"metric_count_{suffix}"]
                for key in ("analysis_mae", "analysis_rmse"):
                    values = [member[f"{key}_{suffix}"] for member in member_metrics]
                    metrics[f"{key}_mean_{suffix}"] = sum(values) / len(values)
                    metrics[f"{key}_min_{suffix}"] = min(values)
                    metrics[f"{key}_max_{suffix}"] = max(values)
                    metrics[f"{key}_of_mean_{suffix}"] = mean_field_metrics[f"{key}_{suffix}"]
                skill_values = [member[f"analysis_rmse_skill_{suffix}"] for member in member_metrics]
                metrics[f"analysis_rmse_skill_mean_{suffix}"] = sum(skill_values) / len(skill_values)
                metrics[f"analysis_rmse_skill_min_{suffix}"] = min(skill_values)
                metrics[f"analysis_rmse_skill_max_{suffix}"] = max(skill_values)
                metrics[f"analysis_rmse_skill_of_mean_{suffix}"] = mean_field_metrics[
                    f"analysis_rmse_skill_{suffix}"
                ]
        if saved_batches:
            samples_dir = os.path.join(self.output_dir, "samples")
            os.makedirs(samples_dir, exist_ok=True)
            artifact = {
                key: torch.cat([batch[key] for batch in saved_batches], dim=0)
                for key in ("samples", "truth", "background", "obs_values", "obs_mask", "valid_mask", "water_mask")
            }
            artifact.update({
                "case_indices": torch.as_tensor(indices[:total_processed], dtype=torch.long),
                "epoch": int(epoch),
                "num_timesteps": int(metric_timesteps),
                "num_ensemble": int(num_ensemble),
                "stride_days": int(stride_days),
                "save_dtype": save_dtype_name,
                "values_space": "normalized",
                "metrics_values_space": "physical",
                "concentration_clipping": "[0, 1]" if "siconc" in self.fields else None,
            })
            artifact_path = os.path.join(samples_dir, f"epoch_{epoch:04d}_metric_ensemble.pt")
            torch.save(artifact, artifact_path)
            _debug(f"saved sample validation ensemble: {artifact_path}")
        primary_suffix = f"{self.fields[0]}_full"
        analysis_mean = metrics[f"analysis_rmse_mean_{primary_suffix}"]
        analysis_of_mean = metrics[f"analysis_rmse_of_mean_{primary_suffix}"]
        background_rmse = metrics[f"background_rmse_{primary_suffix}"]
        _debug(
            "sample validation metrics done "
            f"analysis_rmse_mean_{primary_suffix}={analysis_mean:.6f} "
            f"analysis_rmse_of_mean_{primary_suffix}={analysis_of_mean:.6f} "
            f"background_rmse_{primary_suffix}={background_rmse:.6f}"
        )
        return metrics

    def save_samples(self, epoch: int):
        self.model.eval()
        raw_batch = next(iter(self.val_dataloader))
        batch = self._batch_to_device(raw_batch)
        one = {key: value[:1] for key, value in batch.items()}
        with self._sampling_model() as sample_model:
            sampler = Sampler(sample_model)
            sample = sampler.sample_conditioned(
                background=one["background"],
                background_mask=torch.ones_like(one["background"]),
                obs_values=one["obs_values"],
                obs_mask=one["obs_mask"],
                water_mask=one["water_mask"],
                size=self.config.image_size,
                num_timesteps=self.config.num_sample_timesteps,
                device=self.accelerator.device,
                start_mode=self.config.sample_start_mode,
                start_noise_level=self.config.sample_start_noise_level,
                enforce_observations=self.config.sample_enforce_observations,
                valid_mask=one["valid_mask"],
                obs_guidance_scale=self.config.sample_obs_guidance_scale,
                obs_guidance_eps=self.config.sample_obs_guidance_eps,
                sample_target=self._sample_target(),
                **self._sample_solver_kwargs(),
            )
        samples_dir = os.path.join(self.output_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)
        torch.save(
            {
                "sample": sample.detach().cpu(),
                "truth": one["truth"].detach().cpu(),
                "background": one["background"].detach().cpu(),
                "obs_values": one["obs_values"].detach().cpu(),
                "obs_mask": one["obs_mask"].detach().cpu(),
                "valid_mask": one["valid_mask"].detach().cpu(),
                "water_mask": one["water_mask"].detach().cpu(),
            },
            os.path.join(samples_dir, f"epoch_{epoch:04d}.pt"),
        )

    def _ensure_shared_viz_obs_mask(self) -> torch.Tensor | None:
        if self._shared_viz_obs_mask is not None or self._shared_viz_obs_mask_attempted:
            return self._shared_viz_obs_mask
        self._shared_viz_obs_mask_attempted = True

        val_dataset = self.dashboard_dataset
        if val_dataset is None:
            val_dataset = getattr(self.val_dataloader, "dataset", None)
        if val_dataset is None:
            return None
        if hasattr(val_dataset, "strided_case_indices"):
            scan_indices = val_dataset.strided_case_indices(30, stride_days=5)
        else:
            scan_indices = list(range(min(30, len(val_dataset))))

        best_mask = None
        best_count = 0.0
        for idx in scan_indices:
            try:
                sample = val_dataset[idx]
            except Exception as exc:
                _debug(f"shared viz mask scan idx={idx} failed: {exc}")
                continue
            meta = sample.get("meta", {})
            if meta.get("mask_kind") != "sral_tracks":
                continue
            if int(meta.get("sral_files_used", 0)) <= 0:
                continue
            mask = sample["obs_mask"]
            if not torch.is_tensor(mask):
                mask = torch.as_tensor(mask)
            count = float(mask.sum().item())
            if count > best_count:
                best_count = count
                best_mask = mask.detach().clone()
                if best_count >= 200.0:
                    break

        if best_mask is None:
            _debug("shared viz obs_mask: no real SRAL case found in scan; using per-case masks")
        else:
            _debug(f"shared viz obs_mask: selected mask with count={best_count:.0f}")
        self._shared_viz_obs_mask = best_mask
        return self._shared_viz_obs_mask

    def _apply_shared_viz_mask(self, cases: list[dict[str, torch.Tensor]]) -> None:
        if not cases:
            return
        shared_mask = self._ensure_shared_viz_obs_mask()
        if shared_mask is None:
            return
        shared_mask = shared_mask.to(device=self.accelerator.device, dtype=torch.float32)
        for case in cases:
            mask = shared_mask.unsqueeze(0).clone()
            case["obs_mask"] = mask
            case["obs_values"] = case["truth"] * mask

    @torch.no_grad()
    def _dashboard_cases(self) -> list[dict[str, torch.Tensor]]:
        max_cases = max(_DASHBOARD_NUM_CASES, 0)
        if max_cases == 0:
            return []

        strided_dataset = self.dashboard_dataset
        if strided_dataset is None:
            strided_dataset = getattr(self.val_dataloader, "dataset", None)
        if strided_dataset is not None and hasattr(strided_dataset, "strided_case_indices"):
            strided_indices = strided_dataset.strided_case_indices(max_cases, stride_days=30)
            if strided_indices:
                from torch.utils.data import default_collate

                samples = [strided_dataset[index] for index in strided_indices]
                labels = [sample.get("meta", {}).get("target_date") for sample in samples]
                cases = self._dashboard_cases_from_raw_batch(default_collate(samples), labels=labels)
                self._apply_shared_viz_mask(cases)
                return cases

        cases = []
        for raw_batch in self.val_dataloader:
            cases.extend(self._dashboard_cases_from_raw_batch(raw_batch))
            if len(cases) >= max_cases:
                cases = cases[:max_cases]
                break
        self._apply_shared_viz_mask(cases)
        return cases

    def _dashboard_cases_from_raw_batch(self, raw_batch: dict, labels=None) -> list[dict[str, torch.Tensor]]:
        batch = self._batch_to_device(raw_batch)
        labels = list(labels or [])
        cases = []
        for batch_idx in range(batch["truth"].shape[0]):
            case = {key: value[batch_idx:batch_idx + 1] for key, value in batch.items()}
            if batch_idx < len(labels) and labels[batch_idx]:
                case["case_label"] = str(labels[batch_idx])
            cases.append(case)
        return cases

    @staticmethod
    def _dashboard_batch(cases: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        keys = ("truth", "background", "obs_values", "obs_mask", "valid_mask", "water_mask")
        return {key: torch.cat([case[key] for case in cases], dim=0) for key in keys}

    def _remember_dashboard_epoch(
        self,
        epoch: int,
        pages: list[dict],
        large_path: str | None = None,
        weights_label: str | None = None,
    ) -> None:
        max_history = max(_DASHBOARD_HISTORY_EPOCHS, 1)
        self.dashboard_history.insert(
            0,
            {
                "epoch": int(epoch),
                "pages": pages,
                "large_path": large_path,
                "weights_label": weights_label or self._sampling_weight_label(),
            },
        )
        del self.dashboard_history[max_history:]

    @staticmethod
    def _dashboard_slot_name(index: int) -> str:
        if index == 0:
            return "latest"
        return f"previous_{index}"

    def _report_dashboard_history(self) -> None:
        if self.clearml is None:
            return
        for slot_idx, entry in enumerate(self.dashboard_history):
            slot = self._dashboard_slot_name(slot_idx)
            epoch = int(entry["epoch"])
            weights_label = str(entry.get("weights_label", self._sampling_weight_label()))
            for page in entry["pages"]:
                page_idx = int(page["page_idx"])
                self.clearml.report_image(
                    title=f"dashboard/{slot}_conditioning_ablation/{weights_label}",
                    series=f"page_{page_idx:02d}",
                    path=page["path"],
                    iteration=epoch,
                )
            if entry.get("large_path"):
                self.clearml.report_image(
                    title=f"dashboard/{slot}_large_conditioned_unconditioned/{weights_label}",
                    series="case_0000",
                    path=entry["large_path"],
                    iteration=epoch,
                )

    @staticmethod
    def _chunks(values: list, size: int):
        size = max(int(size), 1)
        for start in range(0, len(values), size):
            yield start // size, values[start:start + size]

    @torch.no_grad()
    def report_unconditional_dashboard_sample(self, epoch: int):
        if _UNCONDITIONAL_DASHBOARD_EVERY_N_EPOCHS <= 0:
            return
        if epoch % _UNCONDITIONAL_DASHBOARD_EVERY_N_EPOCHS != 0:
            return

        weights_label = self._sampling_weight_label()
        _debug(f"unconditional dashboard sampling start epoch={epoch} weights={weights_label}")
        self.model.eval()
        cases = self._dashboard_cases()
        if not cases:
            _debug(f"unconditional dashboard skipped epoch={epoch}: no validation cases")
            return

        samples_dir = os.path.join(self.output_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)
        dashboard_batch = self._dashboard_batch(cases[:1])
        zero_background = torch.zeros_like(dashboard_batch["background"])
        zero_obs_values = torch.zeros_like(dashboard_batch["obs_values"])
        zero_obs_mask = torch.zeros_like(dashboard_batch["obs_mask"])
        initial_noise = torch.randn_like(dashboard_batch["background"])

        with self._sampling_model() as sample_model:
            sampler = Sampler(sample_model)
            unconditional = sampler.sample_conditioned(
                background=zero_background,
                background_mask=torch.zeros_like(dashboard_batch["background"]),
                obs_values=zero_obs_values,
                obs_mask=zero_obs_mask,
                water_mask=dashboard_batch["water_mask"],
                size=self.config.image_size,
                num_timesteps=self.config.num_sample_timesteps,
                device=self.accelerator.device,
                start_mode=self.config.sample_start_mode,
                start_noise_level=self.config.sample_start_noise_level,
                enforce_observations=self.config.sample_enforce_observations,
                valid_mask=dashboard_batch["valid_mask"],
                obs_guidance_scale=self.config.sample_obs_guidance_scale,
                obs_guidance_eps=self.config.sample_obs_guidance_eps,
                initial_noise=initial_noise,
                sample_target=self._sample_target(),
                **self._sample_solver_kwargs(),
            )

        unconditional_physical = self._physical_metric_tensor(unconditional)[0].detach().cpu()
        water_mask = dashboard_batch["water_mask"][0].detach().cpu()
        valid_mask = dashboard_batch["valid_mask"][0].detach().cpu()
        display_mask = water_mask if water_mask.shape[0] == 1 else valid_mask

        import matplotlib.pyplot as plt
        import numpy as np

        n_channels = int(unconditional_physical.shape[0])
        fig, axes = plt.subplots(
            1,
            n_channels,
            figsize=(max(n_channels, 1) * _UNCONDITIONAL_PANEL_WIDTH, _UNCONDITIONAL_PANEL_HEIGHT),
            squeeze=False,
            constrained_layout=True,
        )
        fig.suptitle(
            f"fully unconditional sample, epoch {epoch}, {weights_label}",
            fontsize=16,
        )

        for channel in range(n_channels):
            field_name = self.fields[channel] if channel < len(self.fields) else f"ch{channel}"
            values = unconditional_physical[channel].numpy()
            if display_mask.ndim == 3:
                if display_mask.shape[0] == 1:
                    channel_mask = display_mask[0].numpy() > 0
                else:
                    channel_mask = display_mask[channel].numpy() > 0
                values = np.where(channel_mask, values, np.nan)

            if channel == 0:
                cmap, vmin, vmax = "Blues_r", 0.0, 1.0
            else:
                cmap = "viridis"
                finite = values[np.isfinite(values)]
                if finite.size:
                    vmin = float(np.nanpercentile(finite, 1.0))
                    vmax = float(np.nanpercentile(finite, 99.0))
                    if vmin == vmax:
                        vmin, vmax = None, None
                else:
                    vmin, vmax = None, None

            ax = axes[0, channel]
            image = ax.imshow(
                np.ma.masked_invalid(values),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            ax.set_title(field_name, fontsize=14)
            ax.axis("off")
            fig.colorbar(image, ax=ax, fraction=0.035, pad=0.015)

        path = os.path.join(samples_dir, f"epoch_{epoch:04d}_dashboard_unconditional_large.png")
        fig.savefig(path, dpi=_DASHBOARD_DPI, bbox_inches="tight")
        plt.close(fig)

        if self.clearml is not None:
            self.clearml.report_image(
                title=f"dashboard/unconditional_large/{weights_label}",
                series="case_0000",
                path=path,
                iteration=epoch,
            )
        _debug(f"unconditional dashboard saved: {path}")

    @torch.no_grad()
    def report_dashboard_samples(self, epoch: int):
        if _DASHBOARD_EVERY_N_EPOCHS <= 0:
            return
        if epoch % _DASHBOARD_EVERY_N_EPOCHS != 0:
            return

        weights_label = self._sampling_weight_label()
        _debug(f"dashboard sampling start epoch={epoch} weights={weights_label}")
        if self.clearml is not None:
            self.clearml.report_single_value("sampling_use_ema", float(bool(self.config.sample_use_ema)))
            self.clearml.report_single_value("ema_decay", float(self.config.ema_decay))
        self.model.eval()
        cases = self._dashboard_cases()
        if not cases:
            _debug(f"dashboard sampling skipped epoch={epoch}: no validation cases")
            return
        samples_dir = os.path.join(self.output_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)
        dashboard_cases = []
        dashboard_batch = self._dashboard_batch(cases)

        with self._sampling_model() as sample_model:
            sampler = Sampler(sample_model)
            initial_noise = torch.randn_like(dashboard_batch["background"])
            zero_background = torch.zeros_like(dashboard_batch["background"])
            zero_obs_values = torch.zeros_like(dashboard_batch["obs_values"])
            zero_obs_mask = torch.zeros_like(dashboard_batch["obs_mask"])
            sample_target = self._sample_target()

            def sample_variant(background, background_mask, obs_values, obs_mask):
                return sampler.sample_conditioned(
                    background=background,
                    background_mask=background_mask,
                    obs_values=obs_values,
                    obs_mask=obs_mask,
                    water_mask=dashboard_batch["water_mask"],
                    size=self.config.image_size,
                    num_timesteps=self.config.num_sample_timesteps,
                    device=self.accelerator.device,
                    start_mode=self.config.sample_start_mode,
                    start_noise_level=self.config.sample_start_noise_level,
                    enforce_observations=self.config.sample_enforce_observations,
                    valid_mask=dashboard_batch["valid_mask"],
                    obs_guidance_scale=self.config.sample_obs_guidance_scale,
                    obs_guidance_eps=self.config.sample_obs_guidance_eps,
                    initial_noise=initial_noise,
                    sample_target=sample_target,
                    **self._sample_solver_kwargs(),
                )

            assim_batch = sample_variant(
                dashboard_batch["background"],
                torch.ones_like(dashboard_batch["background"]),
                dashboard_batch["obs_values"],
                dashboard_batch["obs_mask"],
            )
            assim_background_only_batch = sample_variant(
                dashboard_batch["background"],
                torch.ones_like(dashboard_batch["background"]),
                zero_obs_values,
                zero_obs_mask,
            )
            assim_observation_only_batch = sample_variant(
                zero_background,
                torch.zeros_like(dashboard_batch["background"]),
                dashboard_batch["obs_values"],
                dashboard_batch["obs_mask"],
            )
            assim_neither_batch = sample_variant(
                zero_background,
                torch.zeros_like(dashboard_batch["background"]),
                zero_obs_values,
                zero_obs_mask,
            )

            for case_idx, one in enumerate(cases):
                assim = assim_batch[case_idx]
                assim_physical = self._physical_metric_tensor(assim.unsqueeze(0))[0]
                truth_physical = self._physical_metric_tensor(one["truth"])[0]
                background_physical = self._physical_metric_tensor(one["background"])[0]

                if self.clearml is not None:
                    for channel, field_name in enumerate(self.fields):
                        valid_mask = one["valid_mask"][0, channel:channel + 1]
                        analysis_mae, analysis_rmse = self._masked_error_metrics(
                            assim_physical[channel:channel + 1],
                            truth_physical[channel:channel + 1],
                            valid_mask,
                        )
                        background_mae, background_rmse = self._masked_error_metrics(
                            background_physical[channel:channel + 1],
                            truth_physical[channel:channel + 1],
                            valid_mask,
                        )
                        skill = 0.0 if background_rmse <= 0.0 else 1.0 - analysis_rmse / background_rmse
                        series = f"{field_name}/case_{case_idx:04d}"
                        self.clearml.report_scalar("sample_physical/rmse_analysis", series, analysis_rmse, epoch)
                        self.clearml.report_scalar("sample_physical/rmse_background", series, background_rmse, epoch)
                        self.clearml.report_scalar("sample_physical/mae_analysis", series, analysis_mae, epoch)
                        self.clearml.report_scalar("sample_physical/mae_background", series, background_mae, epoch)
                        self.clearml.report_scalar("sample_physical/rmse_skill", series, skill, epoch)

                dashboard_cases.append({
                    "case_idx": case_idx,
                    "case_label": one.get("case_label"),
                    "background": one["background"][0],
                    "obs_values": one["obs_values"][0],
                    "obs_mask": one["obs_mask"][0],
                    "assim": assim,
                    "assim_background_only": assim_background_only_batch[case_idx],
                    "assim_observation_only": assim_observation_only_batch[case_idx],
                    "assim_neither": assim_neither_batch[case_idx],
                    "truth": one["truth"][0],
                    "valid_mask": one["valid_mask"][0],
                    "water_mask": one["water_mask"][0],
                })

        import matplotlib.pyplot as plt
        current_pages = []
        cases_per_page = max(_DASHBOARD_CASES_PER_PAGE, 1)
        for page_idx, page_cases in self._chunks(dashboard_cases, cases_per_page):
            title = (
                f"conditioning ablation, shared initial noise, epoch {epoch}, "
                f"page {page_idx}, cases {len(page_cases)}, {weights_label}"
            )
            fig = make_multi_case_background_condition_assim_figure(
                cases=page_cases,
                fields=self.fields,
                means=self.channel_means,
                stds=self.channel_stds,
                channels=_DASHBOARD_CHANNELS,
                title=title,
                panel_width=_DASHBOARD_PANEL_WIDTH,
                panel_height=_DASHBOARD_PANEL_HEIGHT,
            )
            figure_path = os.path.join(
                samples_dir,
                f"epoch_{epoch:04d}_dashboard_page_{page_idx:02d}_conditioning_ablation.png",
            )
            fig.savefig(figure_path, dpi=_DASHBOARD_DPI, bbox_inches="tight")
            plt.close(fig)
            current_pages.append({"page_idx": page_idx, "path": figure_path})

        large_case = {
            key: value for key, value in dashboard_cases[0].items()
            if key not in ("assim_background_only", "assim_observation_only")
        }
        large_fig = make_multi_case_background_condition_assim_figure(
            cases=[large_case],
            fields=self.fields,
            means=self.channel_means,
            stds=self.channel_stds,
            channels=range(len(self.fields)),
            title=(
                f"large fully conditioned vs fully unconditioned sample, "
                f"epoch {epoch}, {weights_label}"
            ),
            panel_width=_DASHBOARD_LARGE_PANEL_WIDTH,
            panel_height=_DASHBOARD_LARGE_PANEL_HEIGHT,
        )
        large_path = os.path.join(
            samples_dir, f"epoch_{epoch:04d}_dashboard_large_conditioned_unconditioned.png"
        )
        large_fig.savefig(large_path, dpi=_DASHBOARD_DPI, bbox_inches="tight")
        plt.close(large_fig)

        self._remember_dashboard_epoch(
            epoch,
            current_pages,
            large_path=large_path,
            weights_label=weights_label,
        )
        self._report_dashboard_history()
        _debug(f"dashboard sampling done epoch={epoch}")

    def train_loop(self):
        global_step = 0
        _debug(f"training start epochs={self.config.num_epochs} train_batches={len(self.train_dataloader)}")
        for epoch in range(self.config.num_epochs):
            self.model.train()
            _debug(f"epoch {epoch} start")
            progress_bar = tqdm(total=len(self.train_dataloader), disable=not self.accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            for batch_index, raw_batch in enumerate(self.train_dataloader):
                batch = self._batch_to_device(raw_batch)
                truth = batch["truth"]
                batch_size = truth.shape[0]
                timesteps = self._sample_timesteps(batch_size)
                background, background_mask, obs_values, obs_mask = self._conditioned_inputs(batch)
                model_state, v_real = self._make_training_pair(
                    truth, batch, timesteps, residual_background=background
                )

                with self.accelerator.accumulate(self.model):
                    model_input = self._make_model_input(
                        model_state,
                        batch,
                        background=background,
                        background_mask=background_mask,
                        obs_values=obs_values,
                        obs_mask=obs_mask,
                    )
                    v_pred = self.model(model_input, timesteps * 1000, return_dict=False)[0]
                    loss_full = F.mse_loss(v_pred, v_real)
                    loss_obs = torch.zeros_like(loss_full)
                    loss_smooth = torch.zeros_like(loss_full)
                    loss = loss_full

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    self.ema_model.step(self.unwrapped_model.parameters())

                if batch_index == 0:
                    _debug(
                        f"epoch {epoch} first batch "
                        f"loss={loss.item():.6f} full={loss_full.item():.6f} "
                        f"obs={loss_obs.item():.6f} smooth={loss_smooth.item():.6f}"
                    )
                    self._report_condition_diagnostics(raw_batch, global_step, "train")
                self._report_train_metrics(loss, loss_full, loss_obs, loss_smooth, global_step)
                if self.config.tracker:
                    self.accelerator.log({
                        "train_loss": loss.item(),
                        "train_loss_full": loss_full.item(),
                        "train_loss_obs": loss_obs.item(),
                        "train_loss_smoothness": loss_smooth.item(),
                    }, step=global_step)
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item(), obs=loss_obs.item())

            progress_bar.close()
            val_loss_full, val_loss_obs, val_loss = self.compute_val_loss()
            sample_metrics = {}
            if (self.accelerator.is_main_process
                    and self.config.metric_every_n_epochs > 0
                    and epoch % self.config.metric_every_n_epochs == 0):
                sample_metrics = self.compute_sample_validation_metrics(epoch=epoch)

            history_entry = {
                "epoch": epoch,
                "val_loss": val_loss,
                "val_loss_full": val_loss_full,
                "val_loss_obs": val_loss_obs,
                "step": global_step,
                "timestamp": datetime.now().isoformat(),
            }
            history_entry.update(sample_metrics)
            self.val_history.append(history_entry)
            if self.config.tracker:
                self.accelerator.log({
                    "val_loss": val_loss,
                    "val_loss_full": val_loss_full,
                    "val_loss_obs": val_loss_obs,
                }, step=global_step)
            self._report_val_metrics(val_loss_full, val_loss_obs, val_loss, global_step)
            self._report_sample_validation_metrics(sample_metrics, global_step)
            try:
                self._report_condition_diagnostics(next(iter(self.val_dataloader)), global_step, "validation")
            except StopIteration:
                pass
            _debug(f"epoch {epoch} validation total={val_loss:.6f}")
            logger.info("Epoch %d val_loss %.6f full %.6f obs %.6f", epoch, val_loss, val_loss_full, val_loss_obs)

            if self.accelerator.is_main_process:
                with open(os.path.join(self.output_dir, "metrics.json"), "w") as f:
                    json.dump(self.val_history, f, indent=4)
                _debug("saved metrics.json")
                self.save_model_custom("last_model.pth")
                _debug("saved last checkpoint")
                self.report_unconditional_dashboard_sample(epoch)
                self.report_dashboard_samples(epoch)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_model_custom("best_model.pth")
                    _debug("saved best checkpoint")
                    if self.clearml is not None:
                        self.clearml.report_single_value("best_val_loss", val_loss)

        if self.accelerator.is_main_process:
            _debug("uploading ClearML artifacts")
            self._upload_clearml_artifacts()
            if self.clearml is not None:
                self.clearml.close()
        _debug("training finished")
        self.accelerator.end_training()
        return self.output_dir


def _jsonable(value):
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {key: _jsonable(val) for key, val in value.items()}
    return value
