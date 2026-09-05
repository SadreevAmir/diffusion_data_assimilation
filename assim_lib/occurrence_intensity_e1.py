"""Fail-closed executable handoff for the E1 engineering sentinel."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from .structured_sic import (
    decode_cfm_targets,
    exact_one_policy,
    make_cfm_targets,
    make_lagged_observation_channels,
)

REVIEWED_MODE = "occurrence_intensity_e1_engineering_sentinel"
CONFIG_KEYS = {
    "mode", "project_name", "task_name", "clearml", "protocol", "cases",
    "selection_role", "rank_gate", "dataset", "target",
}
DATASET_KEYS = {"lags_days", "channel_layout_per_lag", "observed_finite_only"}
CHANNEL_LAYOUT_PER_LAG = [
    "innovation", "value", "mask", "age", "geometry_real",
    "geometry_synthetic", "value_real", "value_synthetic",
]


def validate_config(config: dict[str, Any]) -> None:
    if set(config) != CONFIG_KEYS:
        raise ValueError("E1 config keys must be exact")
    if config.get("mode") != REVIEWED_MODE:
        raise ValueError("unreviewed E1 mode")
    if config.get("project_name") != "generative-sea-ice-da":
        raise ValueError("unexpected E1 project_name")
    if config.get("task_name") != "occurrence-intensity-e1-engineering-sentinel":
        raise ValueError("unexpected E1 task_name")
    if config.get("clearml") != {"enabled": True}:
        raise ValueError("E1 requires literal clearml.enabled=true")
    if config.get("protocol") != "primary_real":
        raise ValueError("E1 primary sentinel permits real observations only")
    if config.get("cases") != 8 or config.get("selection_role") != "engineering_only":
        raise ValueError("E1 must be the fixed eight-case engineering sentinel")
    if config.get("rank_gate") is not False:
        raise ValueError("E1 engineering sentinel cannot contain a rank gate")
    dataset = config.get("dataset", {})
    if not isinstance(dataset, dict) or set(dataset) != DATASET_KEYS:
        raise ValueError("E1 dataset keys must be exact")
    if dataset.get("lags_days") != [0, 1, 2] or dataset.get("observed_finite_only") is not True:
        raise ValueError("invalid lag or finite-mask contract")
    if dataset.get("channel_layout_per_lag") != CHANNEL_LAYOUT_PER_LAG:
        raise ValueError("invalid channel_layout_per_lag contract")
    target = config.get("target", {})
    expected = {
        "occurrence": "continuous_dequantized_exact_zero",
        "zero_intensity_auxiliary": "uniform_0_1",
        "exact_one_policy_source": "train_inventory",
    }
    if target != expected:
        raise ValueError("invalid occurrence/intensity target law")


def controller_request(config_path: str | Path) -> dict[str, Any]:
    """Validate the literal local handoff; this function never launches work."""
    path = Path(config_path)
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    return {
        "reviewed_mode": REVIEWED_MODE,
        "resource_kind": "server_gpu",
        "admission_required": True,
        "admission_status": "PENDING_INDEPENDENT_REVIEW",
        "launch_authorized": False,
        "clearml_enabled": True,
        "cases": 8,
        "scientific_gate": False,
    }


def integrate_dataset_record(record: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | str]:
    """Convert one dataset record into the frozen E1 training representation."""
    required = {
        "truth", "background_trajectory", "obs_values", "obs_masks", "ages_days",
        "geometry_provenance", "value_provenance", "occurrence_uniform",
        "zero_intensity_uniform", "training_concentration",
    }
    if set(record) != required:
        raise ValueError(f"E1 dataset record keys differ: {sorted(set(record) ^ required)}")
    truth = record["truth"]
    if truth.ndim != 4 or truth.shape[1] != 1:
        raise ValueError("truth must have shape [B,1,H,W]")
    conditioning = make_lagged_observation_channels(
        record["background_trajectory"], record["obs_values"], record["obs_masks"],
        record["ages_days"], record["geometry_provenance"], record["value_provenance"],
    )
    occurrence, intensity = make_cfm_targets(
        truth, record["occurrence_uniform"], record["zero_intensity_uniform"]
    )
    return {
        "conditioning": conditioning,
        "target": torch.cat((occurrence, intensity), dim=1),
        "exact_one_policy": exact_one_policy(record["training_concentration"]),
    }


class E1SentinelVelocity(torch.nn.Module):
    """Minimal trainable velocity used only to exercise the E1 contract."""

    def __init__(self, conditioning_channels: int):
        super().__init__()
        self.net = torch.nn.Conv2d(conditioning_channels + 2, 2, kernel_size=1)

    def forward(self, state: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((state, conditioning), dim=1))


def train_sentinel_step(
    model: E1SentinelVelocity, batch: dict[str, torch.Tensor | str]
) -> float:
    """Execute one finite CFM optimization step; this is not scientific training."""
    target = batch["target"]
    conditioning = batch["conditioning"]
    assert isinstance(target, torch.Tensor) and isinstance(conditioning, torch.Tensor)
    generator = torch.Generator(device=target.device).manual_seed(1103)
    source = torch.rand(target.shape, generator=generator, device=target.device, dtype=target.dtype)
    time = torch.full((target.shape[0], 1, 1, 1), 0.5, device=target.device)
    state = (1 - time) * source + time * target
    desired_velocity = target - source
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss = torch.mean((model(state, conditioning) - desired_velocity) ** 2)
    if not torch.isfinite(loss):
        raise ValueError("E1 sentinel training loss is not finite")
    loss.backward()
    optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def sample_sentinel(
    model: E1SentinelVelocity, conditioning: torch.Tensor, *, members: int = 10, steps: int = 4
) -> torch.Tensor:
    """Exercise continuous generation and atom-aware decoding without clipping."""
    if steps != 4:
        raise ValueError("E1 engineering sampler uses exactly four Euler steps")
    if members != 10:
        raise ValueError("E1 engineering sampler uses exactly ten members")
    generator = torch.Generator(device=conditioning.device).manual_seed(2207)
    repeated_conditioning = conditioning.repeat_interleave(members, dim=0)
    state = torch.rand(
        (repeated_conditioning.shape[0], 2, *conditioning.shape[-2:]), generator=generator,
        device=conditioning.device, dtype=conditioning.dtype,
    )
    for _ in range(steps):
        state = state + model(state, repeated_conditioning) / steps
    # A smooth bounded parameterization is applied once; no physical concentration is clipped.
    coordinates = torch.sigmoid(state)
    decoded = decode_cfm_targets(coordinates[:, :1], coordinates[:, 1:])
    if not torch.all(torch.isfinite(decoded)) or not torch.all((decoded >= 0) & (decoded <= 1)):
        raise ValueError("E1 sentinel sampler emitted an invalid concentration")
    return decoded.reshape(conditioning.shape[0], members, 1, *conditioning.shape[-2:])


def run_engineering_sentinel(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Run the local eight-case executable contract and emit compact artifacts."""
    config_bytes = Path(config_path).read_bytes()
    config = json.loads(config_bytes)
    validate_config(config)
    cases, height, width = config["cases"], 5, 4
    generator = torch.Generator().manual_seed(901)
    truth = torch.rand((cases, 1, height, width), generator=generator)
    truth[:, :, 0, 0] = 0.0
    truth[0, :, 0, 1] = 1.0
    backgrounds = torch.rand((cases, 3, height, width), generator=generator)
    values = torch.rand((cases, 3, height, width), generator=generator)
    masks = (torch.rand((cases, 3, height, width), generator=generator) > 0.65).float()
    values[masks == 0] = math.nan
    backgrounds[masks == 0] = math.nan
    value_provenance = torch.zeros_like(values)
    value_provenance[masks == 0] = math.nan
    geometry = torch.zeros((cases, 3))
    geometry[:, 1] = 1
    record = {
        "truth": truth,
        "background_trajectory": backgrounds,
        "obs_values": values,
        "obs_masks": masks,
        "ages_days": torch.arange(3).expand(cases, 3),
        "geometry_provenance": geometry,
        "value_provenance": value_provenance,
        "occurrence_uniform": torch.rand(truth.shape, generator=generator),
        "zero_intensity_uniform": torch.rand(truth.shape, generator=generator),
        "training_concentration": truth.clone(),
    }
    batch = integrate_dataset_record(record)
    conditioning = batch["conditioning"]
    assert isinstance(conditioning, torch.Tensor)
    torch.manual_seed(3301)
    model = E1SentinelVelocity(conditioning.shape[1])
    loss = train_sentinel_step(model, batch)
    samples = sample_sentinel(model, conditioning)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    status = {
        "status": "completed", "selection_role": "engineering_only", "cases": cases,
        "clearml_enabled": True, "scientific_gate": False,
    }
    manifest = {
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "dataset_records": cases, "conditioning_shape": list(conditioning.shape),
        "target_shape": list(batch["target"].shape),
        "exact_one_policy": batch["exact_one_policy"], "training_loss": loss,
        "sample_shape": list(samples.shape), "sample_finite": True,
        "geometry_value_provenance_separate": True, "lag_specific_backgrounds": True,
        "raw_arrays": "not_persisted",
    }
    (output / "run_status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    (output / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"run_status": status, "artifact_manifest": manifest}
