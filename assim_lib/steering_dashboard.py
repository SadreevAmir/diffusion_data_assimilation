from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from utils import get_device

from .dashboard import make_multi_case_background_condition_assim_figure
from .data import build_dataset
from .evaluate import apply_sampler_normalization
from .main import load_json, merge_config_overrides, resolve_path
from .model_io import load_sampler
from .trainer import TrainingConfig


def _sample_target(config: TrainingConfig) -> str:
    return "residual" if config.training_objective == "diffusion_residual" else "state"


def _default_setting() -> dict[str, Any]:
    return {
        "label": "baseline",
        "cfg_mode": "none",
        "cfg_background_scale": 1.0,
        "cfg_observation_scale": 1.0,
    }


def _resolve_setting(setting: dict[str, Any]) -> dict[str, Any]:
    resolved = {**_default_setting(), **dict(setting)}
    if "bg" in resolved:
        resolved["cfg_background_scale"] = resolved["bg"]
    if "obs" in resolved:
        resolved["cfg_observation_scale"] = resolved["obs"]
    return resolved


def make_steering_dashboard(
    config_path: str | Path,
    run_dir: str | Path,
    checkpoint_name: str = "ema_best_model.pth",
    steering_settings: list[dict[str, Any]] | None = None,
    split: str = "valid",
    case_index: int = 0,
    num_timesteps: int | None = None,
    method: str | None = None,
    seed: int = 1234,
    device: str | torch.device | None = None,
    enforce_observations: bool = False,
    obs_guidance_scale: float = 0.0,
    output_path: str | Path | None = None,
    channels=None,
    panel_width: float = 4.4,
    panel_height: float = 3.2,
    dpi: int = 180,
):
    """Generate and render a conditioning-steering dashboard for one validation case."""
    config_path = Path(config_path).resolve()
    experiment = load_json(config_path)
    data_config_path = resolve_path(experiment["data_config"], config_path.parent).resolve()
    model_config_path = resolve_path(experiment["model_config"], config_path.parent).resolve()
    data_config = merge_config_overrides(load_json(data_config_path), experiment.get("data_overrides"))
    raw_model_config = load_json(model_config_path)
    model_config = {**raw_model_config, **experiment.get("training", {})}
    training = TrainingConfig.from_dict(model_config)

    device = torch.device(device or get_device())
    dataset = build_dataset(data_config, split=split)
    sampler = load_sampler(str(run_dir), checkpoint_name, model_config, device=device)
    means, stds = apply_sampler_normalization(dataset, sampler, data_config)

    item = dataset[int(case_index)]
    fields = list(getattr(dataset, "config", data_config).get("fields", data_config.get("fields", [])))
    case_id = item.get("meta", {}).get("case_id", f"index_{int(case_index):06d}")
    batch = {
        key: item[key].unsqueeze(0).to(device, dtype=torch.float32)
        for key in ("truth", "background", "obs_values", "obs_mask", "valid_mask", "water_mask")
    }

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    shared_noise = torch.randn(batch["background"].shape, generator=generator, device=device)

    resolved_settings = [_resolve_setting(setting) for setting in (steering_settings or [_default_setting()])]
    num_timesteps = int(num_timesteps or training.num_sample_timesteps)
    method = str(method or training.sample_method)
    dashboard_cases = []

    with torch.no_grad():
        for setting in resolved_settings:
            sample = sampler.sample_conditioned(
                background=batch["background"],
                background_mask=torch.ones_like(batch["background"]),
                obs_values=batch["obs_values"],
                obs_mask=batch["obs_mask"],
                water_mask=batch["water_mask"],
                valid_mask=batch["valid_mask"],
                size=tuple(training.image_size),
                num_timesteps=num_timesteps,
                device=device,
                method=method,
                rtol=float(training.sample_rtol),
                atol=float(training.sample_atol),
                start_mode=training.sample_start_mode,
                start_noise_level=float(training.sample_start_noise_level),
                enforce_observations=bool(enforce_observations),
                obs_guidance_scale=float(obs_guidance_scale),
                obs_guidance_eps=float(training.sample_obs_guidance_eps),
                initial_noise=shared_noise,
                sample_target=_sample_target(training),
                memory_efficient_euler=method == "euler",
                cfg_mode=str(setting["cfg_mode"]),
                cfg_background_scale=float(setting["cfg_background_scale"]),
                cfg_observation_scale=float(setting["cfg_observation_scale"]),
            )
            dashboard_cases.append({
                "case_idx": int(case_index),
                "case_label": str(setting["label"]),
                "background": batch["background"][0].detach().cpu(),
                "obs_values": batch["obs_values"][0].detach().cpu(),
                "obs_mask": batch["obs_mask"][0].detach().cpu(),
                "assim": sample[0].detach().cpu(),
                "truth": batch["truth"][0].detach().cpu(),
                "valid_mask": batch["valid_mask"][0].detach().cpu(),
                "water_mask": batch["water_mask"][0].detach().cpu(),
            })

    title = (
        f"conditioning steering, split={split}, case={case_id}, steps={num_timesteps}, "
        f"method={method}, seed={seed}"
    )
    if channels is None:
        channels = range(len(fields))
    fig = make_multi_case_background_condition_assim_figure(
        cases=dashboard_cases,
        fields=fields,
        means=means,
        stds=stds,
        channels=channels,
        title=title,
        panel_width=panel_width,
        panel_height=panel_height,
    )

    saved_path = None
    if output_path:
        saved_path = Path(output_path)
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(saved_path, dpi=int(dpi), bbox_inches="tight")

    metadata = {
        "config_path": str(config_path),
        "data_config_path": str(data_config_path),
        "model_config_path": str(model_config_path),
        "run_dir": str(Path(run_dir).resolve()),
        "checkpoint_name": checkpoint_name,
        "split": split,
        "case_index": int(case_index),
        "case_id": case_id,
        "fields": fields,
        "num_timesteps": num_timesteps,
        "method": method,
        "seed": int(seed),
        "device": str(device),
        "enforce_observations": bool(enforce_observations),
        "obs_guidance_scale": float(obs_guidance_scale),
        "steering_settings": resolved_settings,
        "output_path": str(saved_path) if saved_path is not None else None,
    }
    if saved_path is not None:
        saved_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return fig, metadata
