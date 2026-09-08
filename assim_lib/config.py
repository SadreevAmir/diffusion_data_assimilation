from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def resolve_path(path: str | Path, base_dir: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return Path(base_dir) / candidate


def merge_config_overrides(
    config: Mapping[str, Any],
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Recursively merge experiment overrides without mutating either input."""
    merged = dict(config)
    for key, value in (overrides or {}).items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = merge_config_overrides(current, value)
        else:
            merged[key] = value
    return merged


@dataclass
class TrainingConfig:
    image_size: tuple[int, int] = (320, 256)
    in_channels: int = 13
    out_channels: int = 2
    train_batch_size: int = 8
    eval_batch_size: int = 4
    num_workers_train: int = 4
    num_workers_val: int = 2
    preserve_persistent_worker_rng: bool = False
    num_epochs: int = 20
    gradient_accumulation_steps: int = 1
    activation_checkpointing: bool = False
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    lr_scheduler_total_steps: int = 0
    mixed_precision: str = "no"
    seed: int = 0
    training_objective: str = "flow"
    timestep_sampler: str = "uniform"
    timestep_beta_params: tuple[float, float] = (2.0, 1.0)
    loss_domain: str = "full"
    obs_loss_weight: float = 0.1
    smoothness_loss_weight: float = 0.0
    background_dropout_probability: float = 0.0
    conditioning_mode_probabilities: dict[str, float] | None = None
    sample_every_n_epochs: int = 1
    num_sample_timesteps: int = 200
    sample_method: str = "euler"
    sample_rtol: float = 1e-3
    sample_atol: float = 1e-4
    sample_end_time: float = 0.001
    sample_use_ema: bool = True
    ema_decay: float = 0.999
    ema_use_warmup: bool = False
    ema_update_after_step: int = 0
    ema_min_decay: float = 0.0
    validation_weight_source: str = "raw"
    minimum_optimizer_steps: int = 0
    diagnostic_min_optimizer_steps: int = 0
    sample_start_mode: str = "background"
    sample_start_noise_level: float = 0.5
    sample_enforce_observations: bool = True
    sample_obs_guidance_scale: float = 0.0
    sample_obs_guidance_eps: float = 1e-8
    sample_cfg_mode: str = "none"
    sample_cfg_background_scale: float = 1.0
    sample_cfg_observation_scale: float = 1.0
    base_output_dir: str = "checkpoints/concat_conditioning"
    run_name: str = ""
    tracker: str | None = None
    resume_from_checkpoint: str = ""
    recovery_checkpoint_name: str = "structured_recovery"
    clearml_enabled: bool = True
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
    structured_state_stats_path: str = ""
    structured_state_stats: dict[str, Any] | None = None
    structured_velocity_parameterization: str = "raw"
    validation_seed: int = 2718
    trajectory_horizon_days: int = 0
    trajectory_lead_days: tuple[int, ...] = ()

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> TrainingConfig:
        known = {item.name for item in fields(cls)}
        values = {key: value for key, value in config.items() if key in known}
        objective_aliases = {
            "diffusion": "flow",
            "diffusion_residual": "residual_flow",
        }
        if "training_objective" in values:
            values["training_objective"] = objective_aliases.get(
                values["training_objective"],
                values["training_objective"],
            )
        tuple_keys = {
            "image_size",
            "timestep_beta_params",
            "block_out_channels",
            "down_block_types",
            "up_block_types",
            "clearml_tags",
            "trajectory_lead_days",
        }
        for key in tuple_keys & values.keys():
            values[key] = tuple(values[key])
        parsed = cls(**values)
        if not parsed.trajectory_lead_days:
            parsed.trajectory_lead_days = tuple(range(parsed.trajectory_horizon_days + 1))
        parsed.validate()
        return parsed

    def validate(self) -> None:
        from .flow_parameterization import (
            RAW_VELOCITY,
            validate_velocity_parameterization,
        )

        validate_velocity_parameterization(self.structured_velocity_parameterization)
        if len(self.image_size) != 2 or any(size <= 0 for size in self.image_size):
            raise ValueError(f"image_size must contain two positive values, got {self.image_size}")
        positive_ints = {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "train_batch_size": self.train_batch_size,
            "eval_batch_size": self.eval_batch_size,
            "num_epochs": self.num_epochs,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
        }
        invalid = {name: value for name, value in positive_ints.items() if value <= 0}
        if invalid:
            raise ValueError(f"Training values must be positive: {invalid}")
        if self.num_workers_train < 0 or self.num_workers_val < 0:
            raise ValueError("DataLoader worker counts must be non-negative")
        if self.preserve_persistent_worker_rng and (self.num_workers_train != 0 or self.num_workers_val != 0):
            raise ValueError("preserve_persistent_worker_rng requires zero train and validation workers")
        if self.lr_scheduler_total_steps < 0:
            raise ValueError("lr_scheduler_total_steps must be non-negative")
        if self.training_objective not in {
            "flow",
            "residual_flow",
            "bridge",
            "structured_joint_state_flow",
        }:
            raise ValueError(f"Unknown training_objective={self.training_objective!r}")
        if self.timestep_sampler not in {"uniform", "beta", "stratified_uniform"}:
            raise ValueError(f"Unknown timestep_sampler={self.timestep_sampler!r}")
        if self.loss_domain not in {"full", "valid"}:
            raise ValueError(f"Unknown loss_domain={self.loss_domain!r}; expected 'full' or 'valid'")
        if self.sample_start_mode not in {"noise", "background", "bridge"}:
            raise ValueError(f"Unknown sample_start_mode={self.sample_start_mode!r}")
        if self.training_objective == "residual_flow" and self.sample_start_mode == "bridge":
            raise ValueError("residual_flow is incompatible with sample_start_mode='bridge'")
        if not 0.0 <= self.background_dropout_probability <= 1.0:
            raise ValueError("background_dropout_probability must be within [0, 1]")
        if not 0.0 <= self.sample_end_time < 1.0:
            raise ValueError("sample_end_time must lie in [0, 1)")
        if self.training_objective == "structured_joint_state_flow":
            from .structured_joint_state import LATENT_CHANNELS, validate_structured_state_stats

            if (
                not self.trajectory_lead_days
                or any(value < 0 for value in self.trajectory_lead_days)
                or tuple(sorted(set(self.trajectory_lead_days))) != self.trajectory_lead_days
            ):
                raise ValueError(
                    "trajectory_lead_days must be a non-empty increasing tuple of unique non-negative days"
                )
            if max(self.trajectory_lead_days) != self.trajectory_horizon_days:
                raise ValueError("trajectory_horizon_days must equal max(trajectory_lead_days)")
            expected_latent_channels = LATENT_CHANNELS * len(self.trajectory_lead_days)
            if self.trajectory_horizon_days < 0:
                raise ValueError("trajectory_horizon_days must be non-negative")
            if self.out_channels != expected_latent_channels:
                raise ValueError(
                    "structured_joint_state_flow requires "
                    f"out_channels={expected_latent_channels}, got {self.out_channels}"
                )
            if self.timestep_sampler != "stratified_uniform":
                raise ValueError("structured_joint_state_flow requires timestep_sampler='stratified_uniform'")
            if self.loss_domain != "valid":
                raise ValueError("structured_joint_state_flow requires loss_domain='valid'")
            if self.obs_loss_weight != 0.0 or self.smoothness_loss_weight != 0.0:
                raise ValueError(
                    "structured_joint_state_flow requires zero auxiliary observation and smoothness loss"
                )
            if self.sample_start_mode != "noise":
                raise ValueError("structured_joint_state_flow requires sample_start_mode='noise'")
            if self.sample_end_time != 0.0:
                raise ValueError("structured_joint_state_flow requires sample_end_time=0")
            if self.sample_enforce_observations or self.sample_obs_guidance_scale != 0.0:
                raise ValueError(
                    "structured_joint_state_flow does not use hard observation enforcement or guidance"
                )
            if self.sample_cfg_mode != "none":
                raise ValueError("structured_joint_state_flow requires sample_cfg_mode='none'")
            if self.background_dropout_probability != 0.0:
                raise ValueError("structured_joint_state_flow requires background_dropout_probability=0")
            probabilities = self.conditioning_mode_probabilities
            if probabilities not in (None, {"both": 1.0}):
                raise ValueError(
                    "structured_joint_state_flow requires both-only conditioning "
                    "(conditioning_mode_probabilities null or {'both': 1.0})"
                )
            if self.structured_state_stats is None:
                raise ValueError("structured_joint_state_flow requires structured_state_stats")
            validate_structured_state_stats(self.structured_state_stats)
            if self.validation_weight_source != "ema" or not self.sample_use_ema:
                raise ValueError(
                    "structured_joint_state_flow requires EMA for both validation selection and sampling"
                )
            if self.ema_use_warmup or self.ema_update_after_step != 0 or self.ema_min_decay != 0.0:
                raise ValueError(
                    "structured_joint_state_flow freezes the default diffusers EMA ramp "
                    "(no warmup, update_after_step=0, min_decay=0)"
                )
            if self.minimum_optimizer_steps <= 0:
                raise ValueError(
                    "structured_joint_state_flow requires a positive predeclared minimum_optimizer_steps"
                )
            if not 0 <= self.diagnostic_min_optimizer_steps <= self.minimum_optimizer_steps:
                raise ValueError(
                    "structured_joint_state_flow requires diagnostic_min_optimizer_steps "
                    "within [0, minimum_optimizer_steps]"
                )
            if self.gradient_accumulation_steps != 1:
                raise ValueError(
                    "structured_joint_state_flow currently requires gradient_accumulation_steps=1 "
                    "so each EMA update follows an optimizer update"
                )
            if self.resume_from_checkpoint not in {"", "auto"}:
                raise ValueError("structured_joint_state_flow supports only empty or 'auto' resume policy")
            if not self.run_name:
                raise ValueError(
                    "structured_joint_state_flow requires a stable run_name for fail-safe resume"
                )
            if (
                not self.recovery_checkpoint_name
                or Path(self.recovery_checkpoint_name).name != self.recovery_checkpoint_name
                or self.recovery_checkpoint_name in {".", ".."}
            ):
                raise ValueError("recovery_checkpoint_name must be one safe directory name")
        elif self.structured_velocity_parameterization != RAW_VELOCITY:
            raise ValueError(
                "structured velocity preconditioning is only valid for structured_joint_state_flow"
            )
        elif self.validation_weight_source not in {"raw", "ema"}:
            raise ValueError("validation_weight_source must be 'raw' or 'ema'")


def jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    return value
