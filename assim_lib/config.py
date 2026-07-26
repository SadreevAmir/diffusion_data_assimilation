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
    num_epochs: int = 20
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    mixed_precision: str = "no"
    seed: int = 0
    training_objective: str = "flow"
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
    sample_cfg_mode: str = "none"
    sample_cfg_background_scale: float = 1.0
    sample_cfg_observation_scale: float = 1.0
    base_output_dir: str = "checkpoints/concat_conditioning"
    run_name: str = ""
    tracker: str | None = None
    resume_from_checkpoint: str = ""
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
        }
        for key in tuple_keys & values.keys():
            values[key] = tuple(values[key])
        parsed = cls(**values)
        parsed.validate()
        return parsed

    def validate(self) -> None:
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
        if self.training_objective not in {"flow", "residual_flow", "bridge"}:
            raise ValueError(f"Unknown training_objective={self.training_objective!r}")
        if self.timestep_sampler not in {"uniform", "beta"}:
            raise ValueError(f"Unknown timestep_sampler={self.timestep_sampler!r}")
        if self.sample_start_mode not in {"noise", "background", "bridge"}:
            raise ValueError(f"Unknown sample_start_mode={self.sample_start_mode!r}")
        if self.training_objective == "residual_flow" and self.sample_start_mode == "bridge":
            raise ValueError("residual_flow is incompatible with sample_start_mode='bridge'")
        if not 0.0 <= self.background_dropout_probability <= 1.0:
            raise ValueError("background_dropout_probability must be within [0, 1]")


def jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    return value
