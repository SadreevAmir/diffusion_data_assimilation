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
from torch.utils.data import default_collate

from .clearml_tracking import ClearMLTracker
from .config import TrainingConfig, load_json
from .data import build_dataset
from .flow_parameterization import reconstruct_velocity, velocity_model_state
from .model_io import build_unet
from .runtime import build_dataloader, make_normalized_xy_grid, seed_everything
from .forecast import forecast_records
from .structured_archive_audit import (
    _manifest_sha256,
    _selected_records_and_ranges,
    _sral_provenance,
    _static_mask_provenance,
    build_archive_semantics_audit,
    validate_trajectory_time_contract,
)
from .structured_joint_state import (
    canonical_mapping_sha256,
    encode_structured_joint_trajectory,
    validate_conditioning_normalization,
    validate_structured_state_stats,
)
from .structured_joint_stats import build_structured_state_stats
from .trainer import configure_activation_checkpointing

SCHEMA_VERSION = "two_stage_native_speed_admission_v1"
WARMUP_STEPS = 10
MEASURED_STEPS = 30
SHM_SAFETY_FRACTION = 0.70
TRAINING_BATCH_KEYS = (
    "structured_physical_truth",
    "valid_mask",
    "structured_flow_mask",
    "structured_conditioning",
)
REUSED_CONTRACT_SHA256S = {
    "assimilation": {
        "archive_audit": "b65c122488f3a9b27f8a4eaa9d88f2239dbd074233c672d9aa0341209a647959",
        "data": "7b6d05fcd690c9670d9dd5c0821225bb2e09ca75ff7ab76471507529c483f9d8",
        "stats": "5b87df05d93930b05a0f52625bd9293b553dfb187c8396e32a53c3c1bd3ead14",
    },
    "dynamics": {
        "archive_audit": "00155552aac6a9e7439e5e7d8143eb710182e8af8562d88ae91dee311077e526",
        "data": "cb1737273a80a2dc58363dc1216556bdc0b6f6d556ae8b0b119a267e67d9f668",
        "stats": "d1603a841b8126aa5f8d6499076d2a6f3197a998ab114dbc1e4caeda7f7ac7bc",
    },
}


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


def _compact_training_collate(samples: list[dict]) -> dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("speed-admission collate requires at least one sample")
    missing = [key for key in TRAINING_BATCH_KEYS if any(key not in sample for sample in samples)]
    if missing:
        raise KeyError(f"speed-admission samples miss required keys: {missing}")
    return {
        key: default_collate([sample[key] for sample in samples])
        for key in TRAINING_BATCH_KEYS
    }


def _tensor_bytes(batch: dict[str, torch.Tensor]) -> int:
    if set(batch) != set(TRAINING_BATCH_KEYS):
        raise ValueError("speed-admission batch contains non-training payload")
    return sum(value.numel() * value.element_size() for value in batch.values())


def _estimated_peak_ipc_bytes(
    single_sample_bytes: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
) -> int:
    if min(single_sample_bytes, batch_size) <= 0 or num_workers < 0 or prefetch_factor < 1:
        raise ValueError("invalid DataLoader IPC estimate inputs")
    retained_batches = 1 + num_workers * prefetch_factor
    return single_sample_bytes * batch_size * retained_batches


def _available_shared_memory_bytes() -> int:
    stats = os.statvfs("/dev/shm")
    return int(stats.f_bavail * stats.f_frsize)


def _ipc_preflight(dataset, batch_size: int, num_workers: int, prefetch_factor: int) -> dict:
    probe_batch = _compact_training_collate([dataset[0]])
    single_sample_bytes = _tensor_bytes(probe_batch)
    estimated_peak = _estimated_peak_ipc_bytes(
        single_sample_bytes,
        batch_size,
        num_workers,
        prefetch_factor,
    )
    available = _available_shared_memory_bytes()
    if estimated_peak > SHM_SAFETY_FRACTION * available:
        raise RuntimeError(
            "compact DataLoader IPC estimate exceeds shared-memory safety budget: "
            f"estimated={estimated_peak}, available={available}"
        )
    return {
        "single_sample_bytes": single_sample_bytes,
        "estimated_peak_ipc_bytes": estimated_peak,
        "shared_memory_available_bytes": available,
    }


def _load_reused_contract(
    protocol_path: Path,
    output_dir: Path,
    reuse_contract_dir: Path,
) -> tuple[dict, dict]:
    role = "assimilation" if "assimilation" in protocol_path.stem else "dynamics"
    expected = REUSED_CONTRACT_SHA256S[role]
    paths = {
        "archive_audit": reuse_contract_dir / f"{role}_archive_audit.json",
        "data": reuse_contract_dir / f"{role}_data.json",
        "stats": reuse_contract_dir / f"{role}_stats.json",
    }
    pinned_bytes: dict[str, bytes] = {}
    pinned_json: dict[str, dict] = {}
    for kind, path in paths.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"trusted reused {role} {kind} is missing")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected[kind]:
            raise ValueError(f"trusted reused {role} {kind} hash differs")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"trusted reused {role} {kind} must be an object")
        pinned_bytes[kind] = raw
        pinned_json[kind] = value

    protocol = load_json(protocol_path)
    config = pinned_json["data"]
    stats = pinned_json["stats"]
    audit = pinned_json["archive_audit"]
    derived = {
        "archive_semantics_audit_path",
        "archive_semantics_audit_sha256",
        "means",
        "stds",
        "dynamic_forcing_stats",
    }
    if {key: value for key, value in config.items() if key not in derived} != {
        key: value for key, value in protocol.items() if key not in derived
    }:
        raise ValueError(f"trusted reused {role} protocol differs from current source")
    if audit.get("status") != "data_semantics_verified":
        raise ValueError(f"trusted reused {role} archive audit did not pass")
    if config.get("archive_semantics_audit_sha256") != expected["archive_audit"]:
        raise ValueError(f"trusted reused {role} audit binding differs")
    validate_structured_state_stats(stats)
    validate_conditioning_normalization(config, stats)
    if canonical_mapping_sha256(config) != stats["data_config_sha256"]:
        raise ValueError(f"trusted reused {role} data/stats binding differs")

    records = forecast_records(
        Path(config["dataset_dir"]) / "preds",
        int(config["lead_time_hours"]),
    )
    selected, _ = _selected_records_and_ranges(config, records)
    if (
        len(selected) != int(audit.get("record_count", -1))
        or not selected
        or selected[0].date.isoformat() != audit.get("first_record_date")
        or selected[-1].date.isoformat() != audit.get("last_record_date")
        or _manifest_sha256(selected) != audit.get("forecast_metadata_manifest_sha256")
    ):
        raise ValueError(f"trusted reused {role} forecast inventory drifted")
    raw_land, static_provenance = _static_mask_provenance(config)
    if static_provenance != audit.get("static_mask_provenance"):
        raise ValueError(f"trusted reused {role} static mask drifted")
    ocean = raw_land <= 0 if bool(config.get("mask_true_is_invalid", True)) else raw_land > 0
    if _sral_provenance(config, records, ocean) != audit.get("sral_provenance"):
        raise ValueError(f"trusted reused {role} SRAL provenance drifted")
    validate_trajectory_time_contract(config, audit)
    dataset = build_dataset(config, split="train")
    dataset.validate_structured_sral_audit_contract(audit)
    if dataset.provenance()["pair_manifest_sha256"] != stats["pair_manifest_sha256"]:
        raise ValueError(f"trusted reused {role} train pair manifest drifted")

    for kind in paths:
        target_name = f"{role}_{kind}.json" if kind != "archive_audit" else f"{role}_archive_audit.json"
        target = output_dir / "contracts" / target_name
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        temporary.write_bytes(pinned_bytes[kind])
        os.replace(temporary, target)
    return config, stats


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
    seed_everything(config.seed)
    dataset = build_dataset(data_config, split="train")
    if dataset.conditioned_input_channels != config.in_channels:
        raise ValueError(f"{name}: model/data input channel mismatch")
    ipc = _ipc_preflight(
        dataset,
        config.train_batch_size,
        config.num_workers_train,
        1,
    )
    loader_generator = torch.Generator().manual_seed(config.seed)
    loader = build_dataloader(
        dataset,
        config.train_batch_size,
        config.num_workers_train,
        shuffle=True,
        collate_fn=_compact_training_collate,
        prefetch_factor=1,
        generator=loader_generator,
    )
    iterator = iter(loader)
    first_batch = next(iterator)
    batch_bytes = _tensor_bytes(first_batch)
    if batch_bytes > ipc["single_sample_bytes"] * config.train_batch_size:
        raise RuntimeError("observed compact batch exceeds pre-worker IPC estimate")
    model = build_unet(config).cuda().train()
    checkpointed = configure_activation_checkpointing(model, config.activation_checkpointing)
    if checkpointed:
        raise RuntimeError("activation checkpointing must be empty")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    ema_model = EMAModel(model.parameters(), decay=0.999)
    ema_model.to("cuda")
    grid = make_normalized_xy_grid(*config.image_size, batch_size=batch_size, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(config.seed + 91)
    fetch_seconds = []
    step_seconds = []
    losses = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(WARMUP_STEPS + MEASURED_STEPS):
        fetch_start = time.perf_counter()
        if step == 0:
            batch = first_batch
        else:
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
        "prefetch_factor": 1,
        "compact_batch_keys": list(TRAINING_BATCH_KEYS),
        "compact_batch_bytes": batch_bytes,
        "estimated_peak_ipc_bytes": ipc["estimated_peak_ipc_bytes"],
        "shared_memory_available_bytes": ipc["shared_memory_available_bytes"],
        "shared_memory_safety_fraction": SHM_SAFETY_FRACTION,
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


def run(output_dir: Path, reuse_contract_dir: Path | None = None) -> dict:
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
        bind = (
            (lambda path: _load_reused_contract(path, output_dir, reuse_contract_dir))
            if reuse_contract_dir is not None
            else (lambda path: _bind_data_contract(path, output_dir))
        )
        assimilation_data, assimilation_stats = bind(
            root / "config/data/m2m_2f_two_stage_assimilation_protocol.json"
        )
        dynamics_data, dynamics_stats = bind(
            root / "config/data/m2m_2f_two_stage_dynamics_protocol.json"
        )
        progress_path = output_dir / "speed_admission_progress.json"
        _atomic_json(
            progress_path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "contracts_ready",
                "reused_contracts": reuse_contract_dir is not None,
                "variants": [],
            },
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
                _atomic_json(
                    progress_path,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "status": "benchmarking",
                        "reused_contracts": reuse_contract_dir is not None,
                        "variants": variants,
                    },
                )
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
                _atomic_json(
                    progress_path,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "status": "benchmarking",
                        "reused_contracts": reuse_contract_dir is not None,
                        "variants": variants,
                    },
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
    parser.add_argument("--reuse-contract-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        output_dir / "run_status.json",
        {"schema_version": SCHEMA_VERSION, "status": "running"},
    )
    try:
        reuse_contract_dir = args.reuse_contract_dir.resolve() if args.reuse_contract_dir else None
        result = run(output_dir, reuse_contract_dir)
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
