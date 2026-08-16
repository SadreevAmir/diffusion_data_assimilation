"""Frozen locked-MC-dropout sampling for the final-EMA learned ensemble.

The public entry point is an admission-controlled server orchestrator.  The
private ``--worker`` entry point is invoked in a child process and replaces
only ``compare_3dvar.generate_ensemble``.  No shared sampler implementation is
modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from contextlib import nullcontext
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from . import compare_3dvar
from .deep_ensemble import (
    COMPACT_OUTPUTS,
    _atomic_csv,
    _atomic_json,
    _canonical_hash,
    _file_hash,
    _run_checked,
    _sample_manifest,
)
from .evaluate import _seed_member
from .latent_temperature_ensemble import _validate_source

MODE = "validation_locked_mc_dropout_sampling"
CANDIDATE_METHOD = "locked_mc_dropout_p010_final_ema_ensemble"
SOURCE_EXPERIMENT = "joint_full_condition_validation_2022"
DEPENDENT_GATE_MODE = "validation_locked_mc_dropout_gate"
CASE_COUNT = 40
MEMBER_COUNT = 10
BASE_NOISE_SEED = 1234
DROPOUT_PROBABILITY = 0.1
MASK_SEED_BASE = 271_828_000
EXPECTED_DATES = tuple(date(2022, 1, 1) + timedelta(days=5 * i) for i in range(CASE_COUNT))
SOURCE_CONFIG_SHA256 = "8623a048735b0f739807c90dd038cf52e9e834eae29fd32ff3b73ad4e8505634"
SOURCE_METADATA_SHA256 = "5fcb930c084d7bb7c6367a81bd5272e9bdee2614d949780d6d18c9d82553c95f"
SOURCE_METRICS_SHA256 = "c95dad7a279ec947b936a28b98b922b09222b438be97f8dc0839c793ff5b9ae3"
CHECKPOINT_NAME = "ema_last_model.pth"
CHECKPOINT_SHA256 = "cd73cedc97a9f19d15c70ba31d28d3248a77dc6ef568edf8793326763810fdfe"
CONTRACT_CONFIG = "config/experiments/locked_mc_dropout_p010_final_ema_ensemble.json"
WORKER_MANIFEST_ENV = "LOCKED_MC_DROPOUT_MANIFEST"

ACTIVE_DROPOUT_LAYERS = (
    "down_blocks.0.resnets.0.dropout",
    "down_blocks.1.resnets.0.dropout",
    "down_blocks.2.resnets.0.dropout",
    "down_blocks.3.resnets.0.dropout",
    "mid_block.resnets.0.dropout",
    "mid_block.resnets.1.dropout",
    "up_blocks.0.resnets.0.dropout",
    "up_blocks.0.resnets.1.dropout",
    "up_blocks.1.resnets.0.dropout",
    "up_blocks.1.resnets.1.dropout",
    "up_blocks.2.resnets.0.dropout",
    "up_blocks.2.resnets.1.dropout",
    "up_blocks.3.resnets.0.dropout",
    "up_blocks.3.resnets.1.dropout",
)
ZERO_DROPOUT_LAYERS = (
    "down_blocks.2.attentions.0.to_out.1",
    "up_blocks.1.attentions.0.to_out.1",
    "up_blocks.1.attentions.1.to_out.1",
    "mid_block.attentions.0.to_out.1",
)
MODEL_IDENTITY = {
    "class": "UNet2DModel",
    "sample_size": [320, 256],
    "in_channels": 13,
    "out_channels": 2,
    "layers_per_block": 1,
    "block_out_channels": [96, 192, 384, 384],
    "down_block_types": [
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
    ],
    "up_block_types": ["UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"],
    "norm_num_groups": 32,
}


def noise_seed(case_index: int, member_index: int) -> int:
    if not 0 <= case_index < CASE_COUNT or not 0 <= member_index < MEMBER_COUNT:
        raise ValueError("case or member index outside frozen schedule")
    return BASE_NOISE_SEED + case_index * MEMBER_COUNT + member_index


def mask_seed(case_index: int, member_index: int, layer_index: int) -> int:
    if not 0 <= case_index < CASE_COUNT or not 0 <= member_index < MEMBER_COUNT:
        raise ValueError("case or member index outside frozen schedule")
    if not 0 <= layer_index < len(ACTIVE_DROPOUT_LAYERS):
        raise ValueError("layer index outside frozen inventory")
    return MASK_SEED_BASE + 140 * case_index + 14 * member_index + layer_index


def frozen_contract() -> dict[str, Any]:
    """Return the JSON-serializable, predeclared scientific contract."""
    return {
        "schema_version": 1,
        "mode": MODE,
        "candidate_method": CANDIDATE_METHOD,
        "source_experiment": SOURCE_EXPERIMENT,
        "checkpoint": {"name": CHECKPOINT_NAME, "sha256": CHECKPOINT_SHA256},
        "source_anchors": {
            "config_sha256": SOURCE_CONFIG_SHA256,
            "metadata_sha256": SOURCE_METADATA_SHA256,
            "metrics_sha256": SOURCE_METRICS_SHA256,
        },
        "model": MODEL_IDENTITY,
        "schedule": {
            "dataset_split": "valid",
            "start_date": "2022-01-01",
            "end_date": "2022-07-15",
            "case_stride_days": 5,
            "case_count": CASE_COUNT,
            "member_count": MEMBER_COUNT,
            "base_noise_seed": BASE_NOISE_SEED,
            "noise_seed_formula": "1234 + 10*case_index + member_index",
        },
        "sampler": {
            "method": "dopri5",
            "num_timesteps": 25,
            "rtol": 1e-5,
            "atol": 1e-6,
            "precision": "float32",
            "conditioning_mode": "full",
            "cfg_mode": "none",
            "sample_batch_size": MEMBER_COUNT,
        },
        "locked_dropout": {
            "probability": DROPOUT_PROBABILITY,
            "active_layers": list(ACTIVE_DROPOUT_LAYERS),
            "excluded_zero_probability_layers": list(ZERO_DROPOUT_LAYERS),
            "mask_seed_formula": "271828000 + 140*case_index + 14*member_index + layer_index",
            "generator_device": "cpu",
            "generator_dtype": "float32",
            "keep_rule": "torch.rand(shape, float32, cpu_generator) >= 0.1",
            "inverted_dropout_scale": 1.0 / (1.0 - DROPOUT_PROBABILITY),
            "mask_lifetime": "one case/member/layer for all DOPRI RHS evaluations",
        },
        "dependent_gate": {
            "mode": DEPENDENT_GATE_MODE,
            "implementation": "existing deep_ensemble_gate full gate",
            "families": ["proper", "reliability", "boundary", "spatial", "operational"],
            "paired_bootstrap_draws": 20_000,
            "block_sensitivity_is_diagnostic_only": True,
            "no_compensation": True,
        },
        "artifact_policy": "four compact top-level files; arrays remain server-side",
    }


def _validate_contract_file(repo: Path) -> str:
    path = repo / CONTRACT_CONFIG
    if not path.is_file() or path.is_symlink():
        raise ValueError("frozen locked-dropout contract is missing or is a symlink")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload != frozen_contract():
        raise ValueError("locked-dropout contract differs from the compiled contract")
    return _file_hash(path)


def _resolve_parent(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    parts = name.split(".")
    parent: nn.Module = model
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        if not isinstance(parent, nn.Module):
            raise TypeError(f"{name} does not resolve through nn.Module objects")
    return parent, parts[-1]


def _packed_mask_hash(mask: torch.Tensor) -> str:
    array = mask.detach().cpu().numpy().astype(np.uint8, copy=False).reshape(-1)
    packed = np.packbits(array, bitorder="little")
    return hashlib.sha256(packed.tobytes()).hexdigest()


class LockedDropout(nn.Module):
    """Inference-only dropout whose masks are fixed across ODE evaluations."""

    def __init__(self, controller: LockedDropoutController, layer_name: str, layer_index: int):
        super().__init__()
        self.controller = controller
        self.layer_name = layer_name
        self.layer_index = layer_index
        self._mask: torch.Tensor | None = None
        self._shape: tuple[int, ...] | None = None

    def reset(self) -> None:
        self._mask = None
        self._shape = None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        members = self.controller.member_indices
        if members is None or self.controller.case_index is None:
            raise RuntimeError("locked dropout used outside an active case/member batch")
        if value.ndim < 2 or value.shape[0] != len(members):
            raise ValueError("locked dropout received an unexpected batch shape")
        shape = tuple(int(item) for item in value.shape)
        if self._mask is None:
            generator = torch.Generator(device="cpu")
            masks: list[torch.Tensor] = []
            for batch_index, member_index in enumerate(members):
                seed = mask_seed(self.controller.case_index, member_index, self.layer_index)
                generator.manual_seed(seed)
                member_mask = (
                    torch.rand(shape[1:], dtype=torch.float32, device="cpu", generator=generator)
                    >= DROPOUT_PROBABILITY
                )
                masks.append(member_mask)
                self.controller.record_mask(
                    member_index=member_index,
                    layer_index=self.layer_index,
                    layer_name=self.layer_name,
                    seed=seed,
                    shape=shape[1:],
                    mask_sha256=_packed_mask_hash(member_mask),
                    keep_fraction=float(member_mask.float().mean().item()),
                )
            self._mask = torch.stack(masks).to(device=value.device)
            self._shape = shape
        elif shape != self._shape:
            raise ValueError("locked dropout activation shape changed within an ODE trajectory")
        if self._mask.device != value.device:
            raise ValueError("locked dropout activation device changed within an ODE trajectory")
        return value * self._mask.to(dtype=value.dtype) / (1.0 - DROPOUT_PROBABILITY)


class LockedDropoutController:
    """Install, activate, account for, and restore the exact layer inventory."""

    def __init__(self) -> None:
        self.case_index: int | None = None
        self.member_indices: tuple[int, ...] | None = None
        self.wrappers: list[LockedDropout] = []
        self._originals: list[tuple[nn.Module, str, nn.Module]] = []
        self._records: dict[int, dict[int, dict[str, Any]]] = {}
        self.model_class: str | None = None
        self._model: nn.Module | None = None

    def install(self, model: nn.Module) -> None:
        if self.wrappers:
            if self._model is not model:
                raise ValueError("sampler model changed after locked-dropout installation")
            return
        if model.training:
            raise ValueError("locked MC dropout requires the source model to remain in eval mode")
        if type(model).__name__ != MODEL_IDENTITY["class"]:
            raise ValueError("sampler model class differs from the frozen identity")
        config = getattr(model, "config", None)
        if config is None:
            raise ValueError("sampler model does not expose a frozen configuration")
        expected_config = {
            "sample_size": tuple(MODEL_IDENTITY["sample_size"]),
            "in_channels": MODEL_IDENTITY["in_channels"],
            "out_channels": MODEL_IDENTITY["out_channels"],
            "layers_per_block": MODEL_IDENTITY["layers_per_block"],
            "block_out_channels": tuple(MODEL_IDENTITY["block_out_channels"]),
            "down_block_types": tuple(MODEL_IDENTITY["down_block_types"]),
            "up_block_types": tuple(MODEL_IDENTITY["up_block_types"]),
            "norm_num_groups": MODEL_IDENTITY["norm_num_groups"],
            "dropout": DROPOUT_PROBABILITY,
        }
        for key, expected in expected_config.items():
            if getattr(config, key, None) != expected:
                raise ValueError(f"sampler model configuration differs for {key}")
        batch_norm = [
            name
            for name, module in model.named_modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
        ]
        if batch_norm:
            raise ValueError(f"BatchNorm is forbidden in locked-dropout inference: {batch_norm}")
        dropout = {name: module for name, module in model.named_modules() if type(module) is nn.Dropout}
        expected_names = set(ACTIVE_DROPOUT_LAYERS) | set(ZERO_DROPOUT_LAYERS)
        if set(dropout) != expected_names:
            missing = sorted(expected_names - set(dropout))
            extra = sorted(set(dropout) - expected_names)
            raise ValueError(f"dropout inventory differs; missing={missing}, extra={extra}")
        for name in ACTIVE_DROPOUT_LAYERS:
            if not math.isclose(float(dropout[name].p), DROPOUT_PROBABILITY, abs_tol=1e-15):
                raise ValueError(f"active dropout probability differs at {name}")
        for name in ZERO_DROPOUT_LAYERS:
            if float(dropout[name].p) != 0.0:
                raise ValueError(f"excluded attention dropout is not zero at {name}")
        for index, name in enumerate(ACTIVE_DROPOUT_LAYERS):
            parent, attribute = _resolve_parent(model, name)
            original = getattr(parent, attribute)
            wrapper = LockedDropout(self, name, index)
            setattr(parent, attribute, wrapper)
            self._originals.append((parent, attribute, original))
            self.wrappers.append(wrapper)
        self.model_class = type(model).__name__
        self._model = model

    def activate(self, case_index: int, member_indices: tuple[int, ...]) -> None:
        if not self.wrappers:
            raise RuntimeError("locked-dropout controller is not installed")
        if not 0 <= case_index < CASE_COUNT:
            raise ValueError("case index outside frozen schedule")
        if not member_indices or len(member_indices) != len(set(member_indices)):
            raise ValueError("member batch is empty or contains duplicates")
        if any(not 0 <= member < MEMBER_COUNT for member in member_indices):
            raise ValueError("member index outside frozen schedule")
        self.case_index = case_index
        self.member_indices = member_indices
        self._records.setdefault(case_index, {})
        for member_index in member_indices:
            if member_index in self._records[case_index]:
                raise ValueError("case/member mask schedule was activated more than once")
            self._records[case_index][member_index] = {"layers": []}
        for wrapper in self.wrappers:
            wrapper.reset()

    def record_mask(
        self,
        *,
        member_index: int,
        layer_index: int,
        layer_name: str,
        seed: int,
        shape: tuple[int, ...],
        mask_sha256: str,
        keep_fraction: float,
    ) -> None:
        if self.case_index is None:
            raise RuntimeError("cannot record a mask outside an active case")
        layers = self._records[self.case_index][member_index]["layers"]
        if len(layers) != layer_index:
            raise ValueError("dropout layers were evaluated outside the frozen order")
        layers.append(
            {
                "layer_index": layer_index,
                "layer_name": layer_name,
                "mask_seed": seed,
                "mask_shape": list(shape),
                "mask_sha256": mask_sha256,
                "keep_fraction": keep_fraction,
            }
        )

    def finalize_batch(self) -> None:
        if self.case_index is None or self.member_indices is None:
            raise RuntimeError("no active locked-dropout batch")
        for member_index in self.member_indices:
            layers = self._records[self.case_index][member_index]["layers"]
            if len(layers) != len(ACTIVE_DROPOUT_LAYERS):
                raise ValueError("not every active dropout layer produced a locked mask")
        self.case_index = None
        self.member_indices = None

    def restore(self) -> None:
        for parent, attribute, original in reversed(self._originals):
            setattr(parent, attribute, original)
        self._originals.clear()
        self.wrappers.clear()
        self._model = None

    def manifest(self) -> dict[str, Any]:
        cases: list[dict[str, Any]] = []
        for case_index in range(CASE_COUNT):
            members = self._records.get(case_index)
            if members is None or set(members) != set(range(MEMBER_COUNT)):
                raise ValueError("locked-mask accounting is incomplete")
            cases.append(
                {
                    "case_index": case_index,
                    "members": [
                        {"member_index": member_index, **members[member_index]}
                        for member_index in range(MEMBER_COUNT)
                    ],
                }
            )
        return {
            "schema_version": 1,
            "candidate_method": CANDIDATE_METHOD,
            "model_class": self.model_class,
            "torch_version": torch.__version__,
            "active_layers": list(ACTIVE_DROPOUT_LAYERS),
            "excluded_zero_probability_layers": list(ZERO_DROPOUT_LAYERS),
            "cases": cases,
        }


_WORKER_CONTROLLER = LockedDropoutController()


def generate_locked_ensemble(
    sampler,
    background: torch.Tensor,
    obs_values: torch.Tensor,
    obs_mask: torch.Tensor,
    water_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    config,
    ensemble_size: int,
    sample_batch_size: int,
    num_timesteps: int,
    method: str,
    device: torch.device,
    seed: int,
    case_order: int,
    sample_target: str,
    rtol: float = 1e-3,
    atol: float = 1e-4,
    autocast_dtype: torch.dtype | None = None,
    progress=None,
    background_mask: torch.Tensor | None = None,
    initial_noise_hashes: list[str] | None = None,
    initial_noise_scale: float = 1.0,
    scaled_initial_noise_hashes: list[str] | None = None,
) -> torch.Tensor:
    """Generate the exact frozen ensemble with one locked mask per trajectory."""
    expected = (
        ensemble_size == MEMBER_COUNT
        and sample_batch_size == MEMBER_COUNT
        and num_timesteps == 25
        and method == "dopri5"
        and seed == BASE_NOISE_SEED
        and math.isclose(float(rtol), 1e-5, abs_tol=0.0)
        and math.isclose(float(atol), 1e-6, abs_tol=0.0)
        and float(initial_noise_scale) == 1.0
        and autocast_dtype is None
    )
    if not expected:
        raise ValueError("sampling arguments differ from the locked-MC-dropout contract")
    if config.sample_cfg_mode != "none" or config.sample_start_mode != "noise":
        raise ValueError("locked dropout requires full non-CFG sampling from unit latent noise")
    if tuple(config.image_size) != tuple(MODEL_IDENTITY["sample_size"]):
        raise ValueError("sample size differs from the frozen model identity")
    _WORKER_CONTROLLER.install(sampler.model)

    samples: list[torch.Tensor] = []
    for start in range(0, ensemble_size, sample_batch_size):
        batch_size = min(sample_batch_size, ensemble_size - start)
        noise: list[torch.Tensor] = []
        for member in range(start, start + batch_size):
            _seed_member(noise_seed(case_order, member))
            member_noise = torch.randn_like(background)
            canonical = member_noise.detach().cpu().to(dtype=torch.float32).contiguous().numpy()
            digest = hashlib.sha256(canonical.tobytes()).hexdigest()
            if initial_noise_hashes is not None:
                initial_noise_hashes.append(digest)
            if scaled_initial_noise_hashes is not None:
                scaled_initial_noise_hashes.append(digest)
            noise.append(member_noise)
        initial_noise = torch.cat(noise, dim=0)
        members = tuple(range(start, start + batch_size))
        _WORKER_CONTROLLER.activate(case_order, members)
        autocast_context = nullcontext()
        with autocast_context:
            sample = sampler.sample_conditioned(
                background=background.expand(batch_size, -1, -1, -1),
                background_mask=(
                    background_mask.expand(batch_size, -1, -1, -1) if background_mask is not None else None
                ),
                obs_values=obs_values.expand(batch_size, -1, -1, -1),
                obs_mask=obs_mask.expand(batch_size, -1, -1, -1),
                water_mask=water_mask.expand(batch_size, -1, -1, -1),
                size=tuple(config.image_size),
                num_timesteps=num_timesteps,
                device=device,
                method=method,
                rtol=rtol,
                atol=atol,
                start_mode=config.sample_start_mode,
                start_noise_level=config.sample_start_noise_level,
                enforce_observations=config.sample_enforce_observations,
                valid_mask=valid_mask.expand(batch_size, -1, -1, -1),
                obs_guidance_scale=config.sample_obs_guidance_scale,
                obs_guidance_eps=config.sample_obs_guidance_eps,
                initial_noise=initial_noise,
                sample_target=sample_target,
                memory_efficient_euler=False,
                cfg_mode=config.sample_cfg_mode,
                cfg_background_scale=config.sample_cfg_background_scale,
                cfg_observation_scale=config.sample_cfg_observation_scale,
            )
        if not bool(torch.isfinite(sample).all()):
            raise ValueError("locked-MC-dropout sampler produced non-finite values")
        _WORKER_CONTROLLER.finalize_batch()
        samples.append(sample.cpu())
        if progress is not None:
            progress.update(batch_size)
    return torch.cat(samples, dim=0)


def _validate_sampling_metadata(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    expected = {
        "dataset_split": "valid",
        "start_date": "2022-01-01",
        "end_date": "2022-07-15",
        "num_cases": CASE_COUNT,
        "case_stride_days": 5,
        "ensemble_size": MEMBER_COUNT,
        "sample_batch_size": MEMBER_COUNT,
        "num_timesteps": 25,
        "method": "dopri5",
        "rtol": 1e-5,
        "atol": 1e-6,
        "inference_precision": "float32",
        "seed": BASE_NOISE_SEED,
        "initial_noise_scale": 1.0,
        "conditioning_mode": "full",
        "cfg_mode": "none",
        "save_ensembles": True,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"sampling metadata differs for {key}")
    if Path(str(metadata.get("checkpoint", ""))).name != CHECKPOINT_NAME:
        raise ValueError("sampling used an unexpected checkpoint")
    if metadata.get("checkpoint_selection") != (
        "fixed final epoch; legacy 2023 validation was not used for checkpoint selection"
    ):
        raise ValueError("checkpoint selection differs from the frozen contract")
    cases = metadata.get("cases")
    if not isinstance(cases, list) or len(cases) != CASE_COUNT:
        raise ValueError("sampling metadata does not contain exactly forty cases")
    return cases


def _validate_mask_manifest(payload: dict[str, Any]) -> list[dict[str, Any]]:
    expected_root_keys = {
        "schema_version",
        "candidate_method",
        "model_class",
        "torch_version",
        "active_layers",
        "excluded_zero_probability_layers",
        "cases",
    }
    if set(payload) != expected_root_keys:
        raise ValueError("locked-mask manifest schema differs")
    if payload.get("schema_version") != 1 or payload.get("candidate_method") != CANDIDATE_METHOD:
        raise ValueError("locked-mask manifest identity differs")
    if payload.get("model_class") != MODEL_IDENTITY["class"]:
        raise ValueError("locked-mask manifest model class differs")
    if payload.get("active_layers") != list(ACTIVE_DROPOUT_LAYERS):
        raise ValueError("active locked-dropout inventory differs")
    if payload.get("excluded_zero_probability_layers") != list(ZERO_DROPOUT_LAYERS):
        raise ValueError("zero-probability dropout inventory differs")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != CASE_COUNT:
        raise ValueError("locked-mask manifest does not contain forty cases")
    seen_seeds: set[int] = set()
    for case_index, case in enumerate(cases):
        if not isinstance(case, dict) or set(case) != {"case_index", "members"}:
            raise ValueError("locked-mask case schema differs")
        if case["case_index"] != case_index:
            raise ValueError("locked-mask cases are out of order")
        members = case["members"]
        if not isinstance(members, list) or len(members) != MEMBER_COUNT:
            raise ValueError("locked-mask case does not contain ten members")
        for member_index, member in enumerate(members):
            if (
                not isinstance(member, dict)
                or set(member) != {"member_index", "layers"}
                or member.get("member_index") != member_index
            ):
                raise ValueError("locked-mask member order differs")
            layers = member.get("layers")
            if not isinstance(layers, list) or len(layers) != len(ACTIVE_DROPOUT_LAYERS):
                raise ValueError("locked-mask member inventory is incomplete")
            for layer_index, layer in enumerate(layers):
                expected_seed = mask_seed(case_index, member_index, layer_index)
                mask_hash = str(layer.get("mask_sha256", ""))
                mask_shape = layer.get("mask_shape")
                if (
                    set(layer)
                    != {
                        "layer_index",
                        "layer_name",
                        "mask_seed",
                        "mask_shape",
                        "mask_sha256",
                        "keep_fraction",
                    }
                    or layer.get("layer_index") != layer_index
                    or layer.get("layer_name") != ACTIVE_DROPOUT_LAYERS[layer_index]
                    or layer.get("mask_seed") != expected_seed
                    or len(mask_hash) != 64
                    or not isinstance(mask_shape, list)
                    or not mask_shape
                    or any(type(size) is not int or size <= 0 for size in mask_shape)
                    or not 0.85 <= float(layer.get("keep_fraction", -1.0)) <= 0.95
                ):
                    raise ValueError("locked-mask layer provenance differs")
                try:
                    int(mask_hash, 16)
                except ValueError as error:
                    raise ValueError("locked-mask hash is not hexadecimal") from error
                if expected_seed in seen_seeds:
                    raise ValueError("locked-mask seed schedule is not unique")
                seen_seeds.add(expected_seed)
    if len(seen_seeds) != CASE_COUNT * MEMBER_COUNT * len(ACTIVE_DROPOUT_LAYERS):
        raise ValueError("locked-mask seed accounting is incomplete")
    return cases


def run(output_dir: Path, source_experiment: str, run_dir: Path) -> None:
    if source_experiment != SOURCE_EXPERIMENT:
        raise ValueError("source_experiment differs from the frozen contract")
    if output_dir.is_symlink() or run_dir.is_symlink():
        raise ValueError("input and output directories cannot be symlinks")
    output_dir = output_dir.resolve()
    run_dir = run_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    unexpected = (
        {path.name for path in output_dir.iterdir()}
        - COMPACT_OUTPUTS
        - {
            "internal",
            "samples",
        }
    )
    if unexpected:
        raise ValueError(f"output_dir contains unexpected entries: {sorted(unexpected)}")
    if (output_dir / "samples").exists():
        raise ValueError("candidate samples already exist")
    repo = Path(__file__).resolve().parents[1]
    internal = output_dir / "internal"
    sampling_root = internal / "sampling"
    mask_manifest_path = sampling_root / "locked_dropout_masks.json"
    if internal.is_symlink() or sampling_root.exists():
        raise ValueError("internal sampling path must be a new ordinary directory")
    internal.mkdir(exist_ok=True)
    try:
        checkpoint_record = _validate_source(run_dir)
        contract_sha256 = _validate_contract_file(repo)
        old_manifest = os.environ.get(WORKER_MANIFEST_ENV)
        os.environ[WORKER_MANIFEST_ENV] = str(mask_manifest_path)
        try:
            _run_checked(
                [
                    sys.executable,
                    "-m",
                    "assim_lib.locked_mc_dropout_ensemble",
                    "--worker",
                    "--config",
                    str(repo / "config/experiments/experiment_m2m_flow_modes.json"),
                    "--mode",
                    "validation_clean_checkpoint_member_sampling",
                    "--run-dir",
                    str(run_dir),
                    "--checkpoint-name",
                    CHECKPOINT_NAME,
                    "--output-dir",
                    str(sampling_root),
                    "--ensemble-size",
                    str(MEMBER_COUNT),
                    "--sample-batch-size",
                    str(MEMBER_COUNT),
                    "--seed",
                    str(BASE_NOISE_SEED),
                    "--initial-noise-scale",
                    "1.0",
                    "--save-ensembles",
                ],
                repo,
                output_dir,
                "sampling_locked_mc_dropout_p010",
                0,
                total_units=CASE_COUNT,
            )
        finally:
            if old_manifest is None:
                os.environ.pop(WORKER_MANIFEST_ENV, None)
            else:
                os.environ[WORKER_MANIFEST_ENV] = old_manifest

        sampling_metadata = json.loads((sampling_root / "metadata.json").read_text(encoding="utf-8"))
        sampling_cases = _validate_sampling_metadata(sampling_metadata)
        mask_payload = json.loads(mask_manifest_path.read_text(encoding="utf-8"))
        mask_cases = _validate_mask_manifest(mask_payload)
        sample_source = sampling_root / "samples"
        samples_dir = output_dir / "samples"
        sample_source.replace(samples_dir)

        output_paths: list[Path] = []
        manifest_cases: list[dict[str, Any]] = []
        per_case: list[dict[str, object]] = []
        for case_index, expected_date in enumerate(EXPECTED_DATES):
            filename = f"{case_index:03d}_{expected_date.isoformat()}_h23.npz"
            path = samples_dir / filename
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"missing frozen sample {filename}")
            with np.load(path, allow_pickle=False) as payload:
                ensemble = np.asarray(payload["analysis_ensemble"])
                if ensemble.ndim != 3 or ensemble.shape[0] != MEMBER_COUNT:
                    raise ValueError("candidate sample does not contain exactly ten 2D members")
                if not np.all(np.isfinite(ensemble)):
                    raise ValueError("candidate sample contains non-finite values")
            case = sampling_cases[case_index]
            if case.get("case_order") != case_index or case.get("target_date") != expected_date.isoformat():
                raise ValueError("case identity differs from the frozen schedule")
            base_hashes = case.get("initial_noise_sha256")
            scaled_hashes = case.get("scaled_initial_noise_sha256")
            if not isinstance(base_hashes, list) or not isinstance(scaled_hashes, list):
                raise ValueError("initial-noise hashes are missing")
            if len(base_hashes) != MEMBER_COUNT or base_hashes != scaled_hashes:
                raise ValueError("unit latent noise does not match the raw frozen schedule")
            members: list[dict[str, Any]] = []
            for member_index, initial_hash in enumerate(base_hashes):
                if not isinstance(initial_hash, str) or len(initial_hash) != 64:
                    raise ValueError("initial-noise hash is invalid")
                layers = mask_cases[case_index]["members"][member_index]["layers"]
                actual_seed = noise_seed(case_index, member_index)
                members.append(
                    {
                        "member_id": f"case{case_index:02d}-ema_last-z{actual_seed}-locked-p010",
                        "checkpoint_hash": CHECKPOINT_SHA256,
                        "noise_seed": actual_seed,
                        "initial_noise_hash": initial_hash,
                        "locked_dropout_layers": layers,
                        "finite": True,
                    }
                )
            output_paths.append(path)
            manifest_cases.append({"case_index": case_index, "members": members})
            per_case.append(
                {
                    "case_index": case_index,
                    "target_date": expected_date.isoformat(),
                    "ensemble_size": MEMBER_COUNT,
                    "dropout_probability": DROPOUT_PROBABILITY,
                    "member_manifest_sha256": _canonical_hash(members),
                    "sample_sha256": _file_hash(path),
                }
            )
        if len(list(samples_dir.glob("*.npz"))) != CASE_COUNT:
            raise ValueError("sample directory contains files outside the frozen schedule")

        admission_manifest = {
            "schema_version": 1,
            "candidate_method": CANDIDATE_METHOD,
            "source_experiment": SOURCE_EXPERIMENT,
            "source_config_sha256": SOURCE_CONFIG_SHA256,
            "source_metadata_sha256": SOURCE_METADATA_SHA256,
            "source_metrics_sha256": SOURCE_METRICS_SHA256,
            "checkpoint_record": checkpoint_record,
            "contract_sha256": contract_sha256,
            "active_dropout_layers": list(ACTIVE_DROPOUT_LAYERS),
            "excluded_zero_probability_layers": list(ZERO_DROPOUT_LAYERS),
            "dropout_probability": DROPOUT_PROBABILITY,
            "base_noise_seed": BASE_NOISE_SEED,
            "mask_seed_base": MASK_SEED_BASE,
            "library_identity": {
                "python": sys.version.split()[0],
                "torch": mask_payload["torch_version"],
                "model_class": mask_payload["model_class"],
            },
            "cases": manifest_cases,
        }
        _atomic_csv(output_dir / "per_case_metrics.csv", per_case)
        metadata = {
            "status": "completed",
            "mode": MODE,
            "candidate_method": CANDIDATE_METHOD,
            "source_experiment": SOURCE_EXPERIMENT,
            "dataset_split": "valid",
            "date_range": ["2022-01-01", "2022-07-15"],
            "case_stride_days": 5,
            "num_cases": CASE_COUNT,
            "ensemble_size": MEMBER_COUNT,
            "checkpoint_record": checkpoint_record,
            "dropout_probability": DROPOUT_PROBABILITY,
            "base_noise_seed": BASE_NOISE_SEED,
            "mask_seed_base": MASK_SEED_BASE,
            "conditioning_mode": "full",
            "sampler": {"method": "dopri5", "num_timesteps": 25, "rtol": 1e-5, "atol": 1e-6},
            "admission_manifest": admission_manifest,
            "admission_manifest_sha256": _canonical_hash(admission_manifest),
            "sample_manifest_sha256": _sample_manifest(output_paths),
            "dependent_gate": frozen_contract()["dependent_gate"],
            "fallback_member_count": 0,
            "raw_arrays_retrieved": False,
            "test_data_used_for_this_selection": False,
            "gate_pending": True,
            "scientific_role": "predeclared model-space uncertainty diagnostic",
            "scientific_limitations": [
                "Locked activation masks are an approximate variational diagnostic, "
                "not an exact Bayesian posterior.",
                "The dropout probability was inherited from training and is not tuned on these cases.",
                "The reused 2022 development period is not independent test evidence.",
            ],
        }
        _atomic_json(output_dir / "metadata.json", metadata)
        _atomic_json(
            output_dir / "aggregate_case_mean_metrics.json",
            {
                "schema_version": 1,
                "status": "sampling_completed_gate_pending",
                "candidate_method": CANDIDATE_METHOD,
                "num_cases": CASE_COUNT,
                "ensemble_size": MEMBER_COUNT,
                "dropout_probability": DROPOUT_PROBABILITY,
                "admission_manifest_sha256": metadata["admission_manifest_sha256"],
            },
        )
        _atomic_json(
            output_dir / "run_status.json",
            {
                "status": "completed",
                "experiment_id": output_dir.parent.name,
                "completed_cases": CASE_COUNT,
                "total_cases": CASE_COUNT,
                "progress_percent": 100.0,
                "gate_pending": True,
                "dependent_gate_mode": DEPENDENT_GATE_MODE,
            },
        )
    except Exception as error:
        for name in COMPACT_OUTPUTS - {"run_status.json"}:
            (output_dir / name).unlink(missing_ok=True)
        _atomic_json(
            output_dir / "run_status.json",
            {
                "status": "failed",
                "experiment_id": output_dir.parent.name,
                "error_type": type(error).__name__,
            },
        )
        raise


def _worker_main() -> int:
    manifest_value = os.environ.get(WORKER_MANIFEST_ENV)
    if not manifest_value:
        raise ValueError(f"{WORKER_MANIFEST_ENV} is required for the trusted worker")
    manifest_path = Path(manifest_value)
    if manifest_path.is_symlink():
        raise ValueError("worker manifest path cannot be a symlink")
    manifest_path = manifest_path.resolve()
    original = compare_3dvar.generate_ensemble
    compare_3dvar.generate_ensemble = generate_locked_ensemble
    try:
        compare_3dvar.run(compare_3dvar.parse_args())
        _atomic_json(manifest_path, _WORKER_CONTROLLER.manifest())
    finally:
        compare_3dvar.generate_ensemble = original
        _WORKER_CONTROLLER.restore()
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        del sys.argv[1]
        return _worker_main()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-experiment", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.output_dir, args.source_experiment, args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
