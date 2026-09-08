"""Audit two-stage data and benchmark bounded native-resolution candidates."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
from diffusers.training_utils import EMAModel

from .clearml_tracking import ClearMLTracker
from .config import TrainingConfig, load_json
from .data import build_dataset
from .flow_parameterization import reconstruct_velocity, velocity_model_state
from .model_io import build_unet
from .runtime import build_dataloader, make_normalized_xy_grid, seed_everything
from .structured_archive_audit import build_archive_semantics_audit
from .structured_joint_state import (
    canonical_mapping_sha256,
    encode_structured_joint_trajectory,
)
from .structured_joint_stats import build_structured_state_stats
from .trainer import configure_activation_checkpointing

SCHEMA_VERSION = "two_stage_native_speed_admission_v1"
WARMUP_STEPS = 10
MEASURED_STEPS = 30


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bind_data_contract(protocol_path: Path, output_dir: Path) -> tuple[dict, dict]:
    config = load_json(protocol_path)
    role = "assimilation" if "assimilation" in protocol_path.stem else "dynamics"
    audit_path = output_dir / "contracts" / f"{role}_archive_audit.json"
    _atomic_json(audit_path, build_archive_semantics_audit(config))
    config["archive_semantics_audit_path"] = str(audit_path.resolve())
    config["archive_semantics_audit_sha256"] = _sha256(audit_path)
    stats = build_structured_state_stats(config, protocol_path)
    update = stats["conditioning_normalization"]["data_config_update_required"]
    config["means"] = update["means"]
    config["stds"] = update["stds"]
    if update.get("dynamic_forcing_stats") is not None:
        config["dynamic_forcing_stats"] = update["dynamic_forcing_stats"]
    if canonical_mapping_sha256(config) != stats["data_config_sha256"]:
        raise ValueError("runtime data contract differs from its train-only statistics")
    _atomic_json(output_dir / "contracts" / f"{role}_data.json", config)
    _atomic_json(output_dir / "contracts" / f"{role}_stats.json", stats)
    return config, stats


def _masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    expanded = mask.expand_as(prediction)
    return ((prediction - target).square() * expanded).sum() / expanded.sum().clamp(min=1)


def _one_step(
    model,
    optimizer,
    ema_model,
    batch: dict,
    config: TrainingConfig,
    grid: torch.Tensor,
    generator: torch.Generator,
) -> float:
    physical = batch["structured_physical_truth"].cuda(non_blocking=True)
    valid_mask = batch["valid_mask"][:, :1].cuda(non_blocking=True)
    flow_mask = batch["structured_flow_mask"].cuda(non_blocking=True)
    condition = batch["structured_conditioning"].cuda(non_blocking=True)
    clean = encode_structured_joint_trajectory(
        physical,
        valid_mask,
        config.structured_state_stats,
        generator=generator,
    )
    active = flow_mask.expand_as(clean) > 0
    clean = torch.where(active, clean, torch.zeros_like(clean))
    noise = torch.randn(clean.shape, device="cuda", dtype=clean.dtype, generator=generator)
    noise = torch.where(active, noise, torch.zeros_like(noise))
    jitter = torch.rand(clean.shape[0], device="cuda", generator=generator)
    timestep = (torch.arange(clean.shape[0], device="cuda") + jitter) / clean.shape[0]
    time_view = timestep.view(-1, 1, 1, 1)
    state = (1.0 - time_view) * clean + time_view * noise
    target = noise - clean
    model_input = torch.cat(
        (velocity_model_state(state, timestep, config.structured_velocity_parameterization), grid, condition),
        dim=1,
    )
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        residual = model(model_input, timestep * 1000, return_dict=False)[0]
        prediction = reconstruct_velocity(
            residual.float(), state, timestep, config.structured_velocity_parameterization
        )
        loss = _masked_mse(prediction, target, flow_mask)
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite speed-admission loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    ema_model.step(model.parameters())
    return float(loss.detach().item())


def _benchmark_variant(
    *,
    name: str,
    data_config: dict,
    stats: dict,
    method_path: Path,
    widths: tuple[int, ...],
    batch_size: int,
) -> dict:
    method = load_json(method_path)
    method.update(
        {
            "block_out_channels": list(widths),
            "train_batch_size": batch_size,
            "eval_batch_size": min(batch_size, 4),
            "num_workers_train": 2,
            "num_workers_val": 1,
            "activation_checkpointing": False,
            "structured_state_stats": stats,
            "minimum_optimizer_steps": 1,
            "diagnostic_min_optimizer_steps": 0,
            "num_epochs": 1,
            "run_name": f"speed_{name}",
            "clearml_enabled": False,
        }
    )
    config = TrainingConfig.from_dict(method)
    dataset = build_dataset(data_config, split="train")
    if dataset.conditioned_input_channels != config.in_channels:
        raise ValueError(f"{name}: model/data input channel mismatch")
    loader = build_dataloader(dataset, config.train_batch_size, config.num_workers_train, shuffle=True)
    seed_everything(config.seed)
    model = build_unet(config).cuda().train()
    checkpointed = configure_activation_checkpointing(model, config.activation_checkpointing)
    if checkpointed:
        raise RuntimeError("activation checkpointing must be empty")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    ema_model = EMAModel(model.parameters(), decay=0.999)
    ema_model.to("cuda")
    grid = make_normalized_xy_grid(*config.image_size, batch_size=batch_size, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(config.seed + 91)
    iterator = iter(loader)
    fetch_seconds = []
    step_seconds = []
    losses = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(WARMUP_STEPS + MEASURED_STEPS):
        fetch_start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        fetch_elapsed = time.perf_counter() - fetch_start
        torch.cuda.synchronize()
        step_start = time.perf_counter()
        loss = _one_step(model, optimizer, ema_model, batch, config, grid, generator)
        torch.cuda.synchronize()
        step_elapsed = time.perf_counter() - step_start
        if step >= WARMUP_STEPS:
            fetch_seconds.append(fetch_elapsed)
            step_seconds.append(step_elapsed)
            losses.append(loss)
    result = {
        "name": name,
        "status": "passed",
        "input_channels": config.in_channels,
        "output_channels": config.out_channels,
        "batch_size": batch_size,
        "block_out_channels": list(widths),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "activation_checkpointing": False,
        "checkpointed_modules": [],
        "ema_in_benchmark": True,
        "workers_train": config.num_workers_train,
        "median_fetch_seconds": float(torch.tensor(fetch_seconds).median().item()),
        "median_step_seconds": float(torch.tensor(step_seconds).median().item()),
        "p95_step_seconds": float(torch.tensor(step_seconds).quantile(0.95).item()),
        "examples_per_second": batch_size / (sum(step_seconds) / len(step_seconds)),
        "first_measured_loss": losses[0],
        "last_measured_loss": losses[-1],
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "device_total_bytes": int(torch.cuda.get_device_properties(0).total_memory),
    }
    if not all(
        math.isfinite(result[key])
        for key in (
            "median_fetch_seconds",
            "median_step_seconds",
            "p95_step_seconds",
            "examples_per_second",
            "first_measured_loss",
            "last_measured_loss",
        )
    ):
        raise FloatingPointError(f"{name}: non-finite benchmark result")
    del iterator, loader, dataset, optimizer, ema_model, model, grid
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run(output_dir: Path) -> dict:
    if torch.cuda.device_count() != 1:
        raise RuntimeError("speed admission requires exactly one visible GPU")
    tracker = ClearMLTracker(
        project_name="sea_ice_two_stage",
        task_name="two_stage_native_speed_admission_v1",
        tags=("speed-admission", "one-gpu", "no-checkpointing", "valid-only"),
    )
    root = Path(__file__).resolve().parents[1]
    variants = []
    try:
        assimilation_data, assimilation_stats = _bind_data_contract(
            root / "config/data/m2m_2f_two_stage_assimilation_protocol.json",
            output_dir,
        )
        dynamics_data, dynamics_stats = _bind_data_contract(
            root / "config/data/m2m_2f_two_stage_dynamics_protocol.json",
            output_dir,
        )
        declarations = (
            (
                "assimilation_s_native",
                assimilation_data,
                assimilation_stats,
                root / "config/methods/two_stage_assimilation_s_native.json",
                (32, 64, 128, 128),
                8,
            ),
            (
                "dynamics_s_native",
                dynamics_data,
                dynamics_stats,
                root / "config/methods/two_stage_dynamics_s_native.json",
                (32, 64, 128, 128),
                8,
            ),
            (
                "dynamics_m_native",
                dynamics_data,
                dynamics_stats,
                root / "config/methods/two_stage_dynamics_s_native.json",
                (64, 128, 256, 256),
                4,
            ),
        )
        for variant_index, (name, data, stats, method, widths, batch_size) in enumerate(declarations):
            try:
                result = _benchmark_variant(
                    name=name,
                    data_config=data,
                    stats=stats,
                    method_path=method,
                    widths=widths,
                    batch_size=batch_size,
                )
                variants.append(result)
                for metric in (
                    "median_fetch_seconds",
                    "median_step_seconds",
                    "examples_per_second",
                    "peak_allocated_bytes",
                ):
                    tracker.report_scalar(
                        "speed_admission", f"{name}/{metric}", result[metric], variant_index
                    )
            except torch.cuda.OutOfMemoryError as error:
                variants.append(
                    {
                        "name": name,
                        "status": "cuda_out_of_memory",
                        "error": str(error)[:1000],
                    }
                )
                gc.collect()
                torch.cuda.empty_cache()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "warmup_steps": WARMUP_STEPS,
            "measured_steps": MEASURED_STEPS,
            "variants": variants,
        }
        result_path = output_dir / "speed_admission.json"
        _atomic_json(result_path, payload)
        tracker.upload_artifact("speed_admission", result_path)
        return payload
    finally:
        tracker.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        output_dir / "run_status.json",
        {"schema_version": SCHEMA_VERSION, "status": "running"},
    )
    try:
        result = run(output_dir)
    except Exception as error:
        _atomic_json(
            output_dir / "run_status.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error)[:2000],
            },
        )
        raise
    _atomic_json(
        output_dir / "run_status.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "variant_count": len(result["variants"]),
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
