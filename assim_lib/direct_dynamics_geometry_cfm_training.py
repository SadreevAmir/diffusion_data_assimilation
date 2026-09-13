"""Matched full-CFM continuation from EMA6 with a fixed geometry metric.

Both arms see the same all-hour examples, interpolation times, Gaussian noise,
dropout seeds, optimizer and compute.  The sole experimental difference is
``geometry_weight``: zero for control and the frozen value for treatment.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import signal
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader, Subset

from .clearml_tracking import ClearMLTracker
from .config import TrainingConfig, load_json
from .data import build_dataset
from .direct_dynamics_geometry_cfm import GeometryCFMSpec, geometry_weighted_cfm_loss
from .direct_dynamics_geometry_preflight import _sha256, _tensor_sha256
from .direct_dynamics_training import (
    DIRECT_INPUT_CHANNELS,
    DIRECT_OUTPUT_CHANNELS,
    validate_direct_dataset,
)
from .model_io import load_sampler
from .runtime import make_normalized_xy_grid, seed_everything
from .trainer import UNetTrainer, _atomic_json


SCHEMA_VERSION = "direct_dynamics_geometry_full_cfm_ab_v1"
ARMS = ("control", "treatment")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_torch_save(payload: Any, path: Path) -> str:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(path)


def _merge_status(path: Path, **updates: Any) -> dict[str, Any]:
    current: dict[str, Any] = {}
    if path.exists():
        loaded = load_json(path)
        if isinstance(loaded, dict):
            current = loaded
    current.update(updates)
    current["updated_at"] = _utc_now()
    _atomic_json(path, current)
    return current


def _record_terminal_failure(
    status_path: Path,
    tracker: ClearMLTracker | None,
    error: BaseException,
) -> dict[str, Any]:
    cleanup_errors: list[str] = []
    failure = {
        "status": "failed_terminal_non_resumable",
        "error_type": type(error).__name__,
        "error": str(error),
        "updated_at": _utc_now(),
    }
    try:
        failure = _merge_status(status_path, **failure)
    except Exception as cleanup_error:
        cleanup_errors.append(f"failure_status_write: {cleanup_error}")
    if tracker is not None:
        try:
            tracker.task.mark_failed(
                status_reason=type(error).__name__, status_message=str(error)[:1000]
            )
        except Exception as cleanup_error:
            cleanup_errors.append(f"clearml_mark_failed: {cleanup_error}")
        try:
            tracker.close()
        except Exception as cleanup_error:
            cleanup_errors.append(f"clearml_close: {cleanup_error}")
    if cleanup_errors:
        failure["cleanup_errors"] = cleanup_errors
        try:
            failure = _merge_status(status_path, cleanup_errors=cleanup_errors)
        except Exception:
            pass
    return failure


def _repository_identity(repository_root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise RuntimeError("preflight requires an exact clean git worktree")
    return {"git_commit": commit, "git_clean": True}


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_frozen_metric(path: Path, expected_sha256: str) -> tuple[GeometryCFMSpec, float, dict]:
    if _sha256(path) != expected_sha256:
        raise ValueError("frozen geometry metric config SHA mismatch")
    admission = load_json(path)
    metric = admission.get("metric")
    if not isinstance(metric, dict):
        raise ValueError("frozen geometry metric is missing")
    expected_fixed = {
        "multiscale_factors": [2, 4],
        "multiscale_grid_shapes": [[160, 128], [80, 64]],
        "pooling": "equal_valid_ocean_cell_block_mean_not_area_weighted",
        "field_stds": [0.36890927421384667, 0.4359687842884708],
        "product_scale": 0.3980696029516097,
        "edge_source": "valid_valid_d0_sic_differences_only_with_masked_5x5_smoothing",
        "regions_enabled": False,
        "group_weights": {
            "multiscale_sum_of_two_levels": 0.125,
            "lead_difference": 0.25,
            "d0_linearized_product": 0.25,
            "d0_ice_edge_tube": 0.25,
            "regions": 0.0,
        },
        "geometry_weight_lambda": 0.1,
        "selection_policy": "single_train_only_fixed_value_no_validation_sweep",
        "future_truth_weights": False,
    }
    mismatches = {
        key: (metric.get(key), value)
        for key, value in expected_fixed.items()
        if metric.get(key) != value
    }
    if mismatches:
        raise ValueError(f"unreviewed frozen metric values: {mismatches}")
    group = metric["group_weights"]
    spec = GeometryCFMSpec(
        multiscale_factors=tuple(metric["multiscale_factors"]),
        field_stds=tuple(metric["field_stds"]),
        product_scale=float(metric["product_scale"]),
        multiscale_group_weight=float(group["multiscale_sum_of_two_levels"]),
        lead_group_weight=float(group["lead_difference"]),
        product_group_weight=float(group["d0_linearized_product"]),
        edge_group_weight=float(group["d0_ice_edge_tube"]),
        region_group_weight=float(group["regions"]),
    )
    spec.validate()
    return spec, float(metric["geometry_weight_lambda"]), admission


def _check_protocol(protocol: dict[str, Any]) -> None:
    expected = {
        "arms": ["control", "treatment"],
        "updates_per_arm": 512,
        "batch_size": 8,
        "num_workers": 4,
        "prefetch_factor": 2,
        "learning_rate": 1e-5,
        "weight_decay": 0.0,
        "optimizer": "AdamW",
        "lr_schedule": "constant",
        "gradient_clip_norm": 1.0,
        "ema_decay": 0.999,
        "checkpoint_updates": [128, 256, 512],
        "network_precision": "bf16",
        "loss_precision": "fp32",
        "timestep_sampler": "source_beta_1.0_1.5",
        "activation_checkpointing": False,
        "torch_compile": False,
        "seed": 47117,
        "test_2023_accessed": False,
    }
    if protocol != expected:
        mismatches = {
            key: (protocol.get(key), value)
            for key, value in expected.items()
            if protocol.get(key) != value
        }
        extras = sorted(set(protocol) - set(expected))
        raise ValueError(f"unreviewed full-CFM A/B protocol: mismatches={mismatches}, extras={extras}")


def make_matched_schedule(dataset_size: int, protocol: dict[str, Any]) -> dict[str, Any]:
    updates = int(protocol["updates_per_arm"])
    batch_size = int(protocol["batch_size"])
    count = updates * batch_size
    if count > dataset_size:
        raise ValueError("matched schedule must not repeat examples")
    generator = np.random.default_rng(int(protocol["seed"]))
    indices = generator.permutation(dataset_size)[:count].astype(np.int64).tolist()
    state = torch.random.get_rng_state()
    try:
        torch.manual_seed(int(protocol["seed"]) + 1)
        beta = torch.distributions.Beta(1.0, 1.5)
        timesteps = beta.sample((updates, batch_size)).to(torch.float32).tolist()
    finally:
        torch.random.set_rng_state(state)
    noise_seeds = generator.integers(1, 2**31 - 1, size=updates, dtype=np.int64).tolist()
    dropout_seeds = generator.integers(1, 2**31 - 1, size=updates, dtype=np.int64).tolist()
    schedule = {
        "dataset_size": dataset_size,
        "updates": updates,
        "batch_size": batch_size,
        "indices": indices,
        "timesteps": timesteps,
        "noise_seeds": noise_seeds,
        "dropout_seeds": dropout_seeds,
    }
    schedule["sha256"] = _json_sha256(schedule)
    return schedule


def training_contract_sha256(experiment: dict[str, Any]) -> str:
    """Hash the scientific law while excluding the later preflight file binding."""
    contract = copy.deepcopy(experiment)
    contract.pop("required_preflight", None)
    return _json_sha256(contract)


def _make_loader(dataset, schedule: dict[str, Any], protocol: dict[str, Any]) -> DataLoader:
    return DataLoader(
        Subset(dataset, schedule["indices"]),
        batch_size=int(protocol["batch_size"]),
        shuffle=False,
        num_workers=int(protocol["num_workers"]),
        pin_memory=True,
        persistent_workers=int(protocol["num_workers"]) > 0,
        prefetch_factor=int(protocol["prefetch_factor"]),
        drop_last=True,
    )


def _assert_train_batch(raw_batch: dict[str, Any]) -> None:
    meta = raw_batch.get("meta")
    if not isinstance(meta, dict) or meta.get("split") != ["train"] * len(meta.get("case_id", [])):
        raise RuntimeError("matched batch escaped train split")

    def strings(value: Any):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from strings(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from strings(child)

    if any("2023" in Path(value).name for value in strings(meta)):
        raise RuntimeError("test-2023 path entered matched training")


def _batch_to_device(raw_batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    _assert_train_batch(raw_batch)
    truth = raw_batch["truth"].to(device, torch.float32, non_blocking=True)
    valid = raw_batch["valid_mask"][:, :1].to(device, torch.float32, non_blocking=True)
    condition = raw_batch["structured_conditioning"].to(device, torch.float32, non_blocking=True)
    physical_d0 = raw_batch["structured_physical_background"][:, :2].to(
        device, torch.float32, non_blocking=True
    )
    if truth.shape[1] != DIRECT_OUTPUT_CHANNELS or condition.shape[1] != 15:
        raise ValueError("matched batch violates direct-dynamics channel contract")
    return {
        "truth": truth,
        "valid": valid,
        "condition": condition,
        "initial_sic": physical_d0[:, 0:1],
        "initial_sit": physical_d0[:, 1:2],
    }


def make_cfm_pair(
    batch: dict[str, torch.Tensor],
    timesteps: torch.Tensor,
    noise_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    truth = batch["truth"]
    valid = batch["valid"].expand_as(truth) > 0
    clean = torch.where(valid, truth, torch.zeros_like(truth))
    generator = torch.Generator(device="cpu").manual_seed(int(noise_seed))
    noise_cpu = torch.randn(clean.shape, generator=generator, dtype=torch.float32, device="cpu")
    noise = noise_cpu.to(clean.device, non_blocking=True)
    noise = torch.where(valid, noise, torch.zeros_like(noise))
    time = timesteps.to(clean.device, torch.float32).view(-1, 1, 1, 1)
    state = (1.0 - time) * clean + time * noise
    return state, noise - clean, noise


def compute_matched_loss(
    prediction: torch.Tensor,
    target_velocity: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    geometry_weight: float,
    spec: GeometryCFMSpec,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prediction_fp32 = prediction.to(torch.float32)
    target_fp32 = target_velocity.to(torch.float32)
    if prediction_fp32.dtype != torch.float32 or target_fp32.dtype != torch.float32:
        raise RuntimeError("matched loss did not enter FP32")
    return geometry_weighted_cfm_loss(
        prediction_fp32,
        target_fp32,
        batch["valid"],
        batch["initial_sic"],
        batch["initial_sit"],
        geometry_weight=float(geometry_weight),
        region_masks=None,
        spec=spec,
    )


def _gradient_norm(model: torch.nn.Module) -> float:
    values = [parameter.grad.detach().float() for parameter in model.parameters() if parameter.grad is not None]
    if not values or not all(torch.isfinite(value).all() for value in values):
        raise FloatingPointError("missing or non-finite matched training gradient")
    norm = float(torch.sqrt(sum(value.square().sum() for value in values)).cpu())
    if not math.isfinite(norm) or norm <= 0.0:
        raise FloatingPointError("matched training gradient norm must be finite and strictly positive")
    return norm


def _load_source(experiment: dict, repository_root: Path, device: torch.device):
    source = experiment["source"]
    run_dir = Path(source["run_dir"])
    expected_sha = source["sha256"]
    if set(expected_sha) != {
        "metadata.json",
        "config.json",
        "epoch_snapshots/epoch_0006/ema_last_model.pth",
    }:
        raise ValueError("source SHA map is incomplete")
    for relative, digest in expected_sha.items():
        if _sha256(run_dir / relative) != digest:
            raise ValueError(f"source SHA mismatch: {relative}")
    metadata = load_json(run_dir / "metadata.json")
    config = TrainingConfig.from_dict(metadata["training_config"])
    if (
        tuple(config.image_size) != (320, 256)
        or config.in_channels != DIRECT_INPUT_CHANNELS
        or config.out_channels != DIRECT_OUTPUT_CHANNELS
        or config.training_objective != "flow"
        or config.timestep_sampler != "beta"
        or tuple(config.timestep_beta_params) != (1.0, 1.5)
        or config.activation_checkpointing
    ):
        raise ValueError("EMA6 source training contract mismatch")
    dataset = build_dataset(metadata["data_config"], split="train")
    sentinel = validate_direct_dataset(dataset)
    sampler = load_sampler(str(run_dir), source["checkpoint"], metadata["training_config"], device=device)
    return sampler.model, config, dataset, sentinel, metadata


def _prepare_static_contract(config_path: Path):
    experiment = load_json(config_path)
    if experiment.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected matched full-CFM schema")
    protocol = experiment["protocol"]
    _check_protocol(protocol)
    repository_root = config_path.parents[2]
    metric_binding = experiment["metric_admission"]
    metric_path = repository_root / metric_binding["path"]
    spec, treatment_weight, admission = _load_frozen_metric(metric_path, metric_binding["sha256"])
    if treatment_weight != 0.1:
        raise ValueError("unreviewed treatment geometry weight")
    return experiment, protocol, spec, treatment_weight, admission, repository_root, metric_path


def _prepare_contract(config_path: Path, device: torch.device):
    (
        experiment,
        protocol,
        spec,
        treatment_weight,
        admission,
        repository_root,
        _metric_path,
    ) = _prepare_static_contract(config_path)
    model, model_config, dataset, sentinel, metadata = _load_source(experiment, repository_root, device)
    schedule = make_matched_schedule(len(dataset), protocol)
    return experiment, protocol, spec, treatment_weight, admission, model, model_config, dataset, sentinel, metadata, schedule


def _implementation_manifest(
    config_path: Path,
    experiment: dict[str, Any],
    repository_root: Path,
    metric_path: Path,
) -> dict[str, Any]:
    checkpoint = str(experiment["source"]["checkpoint"])
    sha_map = experiment["source"]["sha256"]
    if checkpoint not in sha_map:
        raise ValueError("selected source checkpoint is absent from the verified SHA map")
    checkpoint_path = Path(experiment["source"]["run_dir"]) / checkpoint
    checkpoint_sha = _sha256(checkpoint_path)
    if checkpoint_sha != sha_map[checkpoint]:
        raise ValueError("selected source checkpoint does not match its exact SHA-map entry")
    identity = _repository_identity(repository_root)
    identity.update(
        {
            "runner_path": str(Path(__file__).resolve().relative_to(repository_root)),
            "runner_sha256": _sha256(Path(__file__)),
            "objective_path": str(
                Path(__file__).with_name("direct_dynamics_geometry_cfm.py").resolve().relative_to(repository_root)
            ),
            "objective_sha256": _sha256(Path(__file__).with_name("direct_dynamics_geometry_cfm.py")),
            "metric_config_path": str(metric_path.resolve().relative_to(repository_root)),
            "metric_config_sha256": _sha256(metric_path),
            "preflight_config_sha256": _sha256(config_path),
            "scientific_contract_sha256": training_contract_sha256(experiment),
            "source_checkpoint_relative_path": checkpoint,
            "source_checkpoint_sha256": checkpoint_sha,
            "source_checkpoint_sha_map_entry": sha_map[checkpoint],
        }
    )
    return identity


def _make_forward_evidence(
    batch: dict[str, torch.Tensor],
    timesteps: torch.Tensor,
    noise_seed: int,
    grid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    state, target, noise = make_cfm_pair(batch, timesteps, noise_seed)
    model_input = torch.cat((state, grid.expand(state.shape[0], -1, -1, -1), batch["condition"]), dim=1)
    if model_input.shape[1] != DIRECT_INPUT_CHANNELS:
        raise ValueError("matched model input has wrong channel count")
    return {
        "truth": batch["truth"],
        "condition": batch["condition"],
        "valid": batch["valid"],
        "initial_sic": batch["initial_sic"],
        "initial_sit": batch["initial_sit"],
        "noise": noise,
        "timesteps": timesteps,
        "model_input": model_input,
        "target_velocity": target,
    }


def _one_forward(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    timesteps: torch.Tensor,
    noise_seed: int,
    dropout_seed: int,
    grid: torch.Tensor,
    geometry_weight: float,
    spec: GeometryCFMSpec,
    prediction_callback: Callable[[torch.Tensor], None] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    evidence = _make_forward_evidence(batch, timesteps, noise_seed, grid)
    torch.manual_seed(int(dropout_seed))
    torch.cuda.manual_seed_all(int(dropout_seed))
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        prediction = model(
            evidence["model_input"],
            timesteps.to(evidence["model_input"].device) * 1000,
            return_dict=False,
        )[0]
    prediction_fp32 = prediction.float()
    if prediction_callback is not None:
        prediction_callback(prediction_fp32.detach())
    loss, diagnostics = compute_matched_loss(
        prediction_fp32,
        evidence["target_velocity"],
        batch,
        geometry_weight=geometry_weight,
        spec=spec,
    )
    evidence["prediction_fp32"] = prediction_fp32
    return loss, diagnostics, evidence


def _persist_preflight_evidence(
    path: Path,
    common: dict[str, torch.Tensor],
    predictions: dict[str, torch.Tensor],
) -> str:
    return _atomic_torch_save(
        {
            "common": {key: value.detach().cpu() for key, value in common.items()},
            "predictions": {
                key: value.detach().cpu() for key, value in predictions.items()
            },
            "prediction_sha256": {
                key: _tensor_sha256(value) for key, value in predictions.items()
            },
            "schedule_update": 0,
            "optimizer_steps": 0,
        },
        path,
    )


def _assert_exact_matched_prediction(
    control: torch.Tensor,
    treatment: torch.Tensor,
) -> None:
    if not torch.equal(control, treatment):
        raise RuntimeError("matched control/treatment predictions differ")


def run_preflight(config_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    status_path = output_dir / "status.json"
    _merge_status(status_path, status="initializing", optimizer_steps=0, test_2023_accessed=False)
    tracker: ClearMLTracker | None = None
    previous = {
        signal_number: signal.getsignal(signal_number)
        for signal_number in (signal.SIGINT, signal.SIGTERM)
    }

    def terminate(signum, _frame):
        raise InterruptedError(f"zero-update preflight received signal {signum}")

    for signal_number in previous:
        signal.signal(signal_number, terminate)
    try:
        (
            experiment,
            protocol,
            spec,
            treatment_weight,
            admission,
            repository_root,
            metric_path,
        ) = _prepare_static_contract(config_path)
        manifest = _implementation_manifest(
            config_path, experiment, repository_root, metric_path
        )
        tracker = ClearMLTracker(
            experiment["project_name"],
            f"{experiment['task_name']}-preflight-{output_dir.name}",
            tags=[*experiment["clearml"]["tags"], "zero-update-preflight"],
            env_path=experiment["clearml"]["env_path"],
        )
        tracker.connect(
            "zero_update_preflight_contract",
            {"experiment": experiment, "implementation_manifest": manifest},
        )
        _merge_status(
            status_path,
            status="reserved_before_gpu",
            clearml_task_id=str(getattr(tracker.task, "id", "")),
            implementation_manifest=manifest,
        )
        if torch.cuda.device_count() != 1:
            raise RuntimeError("zero-update preflight requires exactly one visible GPU")
        device = torch.device("cuda:0")
        model, model_config, dataset, sentinel, _metadata = _load_source(
            experiment, repository_root, device
        )
        schedule = make_matched_schedule(len(dataset), protocol)
        _atomic_json(output_dir / "schedule.json", schedule)
        loader = _make_loader(dataset, schedule, protocol)
        raw = next(iter(loader))
        batch = _batch_to_device(raw, device)
        timesteps = torch.tensor(schedule["timesteps"][0], dtype=torch.float32, device=device)
        grid = make_normalized_xy_grid(*model_config.image_size, device=device, dtype=torch.float32)
        source_parameter_sha = _json_sha256(
            {name: _tensor_sha256(value) for name, value in model.state_dict().items()}
        )
        torch.cuda.reset_peak_memory_stats(device)

        # Save the exact fixed inputs before the first model forward. Then save
        # every obtained prediction before checking it, so failed admission is
        # itself inspectable.
        fixed_inputs_path = output_dir / "fixed_inputs_and_predictions.pt"
        common_evidence = _make_forward_evidence(
            batch, timesteps, schedule["noise_seeds"][0], grid
        )
        saved_predictions: dict[str, torch.Tensor] = {}
        fixed_inputs_sha = _persist_preflight_evidence(
            fixed_inputs_path, common_evidence, saved_predictions
        )
        _merge_status(
            status_path,
            status="fixed_inputs_saved_before_forward",
            fixed_inputs_and_predictions_sha256=fixed_inputs_sha,
        )
        prediction_evidence: dict[str, dict[str, torch.Tensor]] = {}

        def prediction_saver(key: str, phase: str):
            def save(prediction: torch.Tensor) -> None:
                nonlocal fixed_inputs_sha
                saved_predictions[key] = prediction
                fixed_inputs_sha = _persist_preflight_evidence(
                    fixed_inputs_path, common_evidence, saved_predictions
                )
                _merge_status(
                    status_path,
                    status=phase,
                    fixed_inputs_and_predictions_sha256=fixed_inputs_sha,
                )

            return save

        with torch.no_grad():
            for arm, weight in (("control", 0.0), ("treatment", treatment_weight)):
                model.train()
                loss, diagnostics, evidence = _one_forward(
                    model,
                    batch,
                    timesteps,
                    schedule["noise_seeds"][0],
                    schedule["dropout_seeds"][0],
                    grid,
                    weight,
                    spec,
                    prediction_callback=prediction_saver(
                        f"{arm}_evidence_forward",
                        f"{arm}_prediction_saved_before_scoring",
                    ),
                )
                prediction_evidence[arm] = {
                    "prediction": evidence["prediction_fp32"].detach().cpu(),
                    "native": diagnostics["native"].detach().cpu(),
                    "loss": loss.detach().cpu(),
                }
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite {arm} preflight loss")
                if arm == "control":
                    native = UNetTrainer._masked_mse(
                        evidence["prediction_fp32"], evidence["target_velocity"], batch["valid"]
                    )
                    if not torch.equal(loss.detach(), native.detach()):
                        raise RuntimeError("real control path differs from native masked MSE")
        _assert_exact_matched_prediction(
            prediction_evidence["control"]["prediction"],
            prediction_evidence["treatment"]["prediction"],
        )
        if not torch.equal(
            prediction_evidence["control"]["native"],
            prediction_evidence["treatment"]["native"],
        ):
            raise RuntimeError("matched control/treatment native losses differ")
        _merge_status(
            status_path,
            status="fixed_inputs_and_predictions_saved_before_backward",
            fixed_inputs_and_predictions_sha256=fixed_inputs_sha,
        )
        if os.environ.get("DIRECT_DYNAMICS_CFM_PREFLIGHT_FAIL_AFTER_EVIDENCE") == "1":
            raise RuntimeError("injected failure after durable preflight evidence")

        branches: dict[str, dict[str, Any]] = {}
        expected_prediction_sha = _tensor_sha256(
            prediction_evidence["control"]["prediction"]
        )
        for arm, weight in (("control", 0.0), ("treatment", treatment_weight)):
            model.zero_grad(set_to_none=True)
            model.train()
            loss, diagnostics, evidence = _one_forward(
                model,
                batch,
                timesteps,
                schedule["noise_seeds"][0],
                schedule["dropout_seeds"][0],
                grid,
                weight,
                spec,
                prediction_callback=prediction_saver(
                    f"{arm}_backward_forward",
                    f"{arm}_backward_prediction_saved_before_scoring",
                ),
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite {arm} backward loss")
            if _tensor_sha256(evidence["prediction_fp32"]) != expected_prediction_sha:
                raise RuntimeError(f"{arm} backward forward differs from saved prediction")
            loss.backward()
            branches[arm] = {
                "loss": float(loss.detach().cpu()),
                "native": float(diagnostics["native"].detach().cpu()),
                "geometry": float(diagnostics["geometry"].detach().cpu()),
                "gradient_norm": _gradient_norm(model),
                "prediction_sha256": _tensor_sha256(evidence["prediction_fp32"]),
                "prediction_dtype_before_loss": str(evidence["prediction_fp32"].dtype),
                "target_dtype_at_loss": str(evidence["target_velocity"].dtype),
                "input_fingerprints": {
                    key: _tensor_sha256(value)
                    for key, value in evidence.items()
                    if key != "prediction_fp32"
                },
            }
            _merge_status(status_path, status=f"backward_{arm}_passed")
        if branches["control"]["input_fingerprints"] != branches["treatment"]["input_fingerprints"]:
            raise RuntimeError("preflight arms did not receive exact common inputs")
        if branches["control"]["prediction_sha256"] != branches["treatment"]["prediction_sha256"]:
            raise RuntimeError("preflight branch predictions are not exact matches")
        if branches["control"]["native"] != branches["treatment"]["native"]:
            raise RuntimeError("preflight branch native losses are not exact matches")
        final_parameter_sha = _json_sha256(
            {name: _tensor_sha256(value) for name, value in model.state_dict().items()}
        )
        if source_parameter_sha != final_parameter_sha:
            raise RuntimeError("zero-update preflight changed model parameters")
        result = {
            "schema_version": f"{SCHEMA_VERSION}_zero_update_preflight",
            "status": "passed_pending_clearml_close",
            "optimizer_steps": 0,
            "test_2023_accessed": False,
            "source_parameter_sha256": source_parameter_sha,
            "parameters_unchanged": True,
            "fixed_inputs_and_predictions_sha256": fixed_inputs_sha,
            "schedule_sha256": schedule["sha256"],
            "dataset_sentinel": sentinel,
            "metric_admission": admission,
            "implementation_manifest": manifest,
            "branches": branches,
            "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(device) / 2**20),
            "config_sha256": _sha256(config_path),
            "training_contract_sha256": training_contract_sha256(experiment),
            "code_sha256": _sha256(Path(__file__)),
            "completed_at": _utc_now(),
        }
        result_path = output_dir / "preflight.json"
        _atomic_json(result_path, result)
        _merge_status(status_path, status="closing_clearml", preflight_sha256=_sha256(result_path))
        tracker.close()
        tracker = None
        result["status"] = "passed"
        result["clearml_closed_before_terminal_success"] = True
        _atomic_json(result_path, result)
        _merge_status(
            status_path,
            status="passed",
            preflight_sha256=_sha256(result_path),
            clearml_closed_before_terminal_success=True,
        )
        return result
    except BaseException as error:
        _record_terminal_failure(status_path, tracker, error)
        raise
    finally:
        for signal_number, handler in previous.items():
            signal.signal(signal_number, handler)


def _save_checkpoint(
    output_dir: Path,
    arm: str,
    update: int,
    model: torch.nn.Module,
    ema: EMAModel,
    optimizer: torch.optim.Optimizer,
) -> dict[str, str]:
    root = output_dir / arm / f"update_{update:04d}"
    root.mkdir(parents=True, exist_ok=False)
    raw_sha = _atomic_torch_save(model.state_dict(), root / "model.pth")
    ema_sha = _atomic_torch_save(ema.state_dict(), root / "ema_model.pth")
    recovery_sha = _atomic_torch_save(
        {"arm": arm, "completed_update": update, "optimizer": optimizer.state_dict()},
        root / "optimizer_recovery.pth",
    )
    return {"model": raw_sha, "ema_model": ema_sha, "optimizer_recovery": recovery_sha}


def run_training(config_path: Path, output_dir: Path) -> dict[str, Any]:
    if torch.cuda.device_count() != 1:
        raise RuntimeError("matched training requires exactly one visible GPU")
    device = torch.device("cuda:0")
    (
        experiment,
        protocol,
        spec,
        treatment_weight,
        admission,
        source_model,
        model_config,
        dataset,
        sentinel,
        metadata,
        schedule,
    ) = _prepare_contract(config_path, device)
    preflight = experiment["required_preflight"]
    preflight_path = Path(preflight["path"])
    if _sha256(preflight_path) != preflight["sha256"]:
        raise ValueError("required zero-update preflight SHA mismatch")
    preflight_payload = load_json(preflight_path)
    preflight_manifest = preflight_payload.get("implementation_manifest", {})
    objective_path = Path(__file__).with_name("direct_dynamics_geometry_cfm.py")
    selected_checkpoint = experiment["source"]["checkpoint"]
    selected_checkpoint_sha = experiment["source"]["sha256"].get(selected_checkpoint)
    if (
        preflight_payload.get("status") != "passed"
        or preflight_payload.get("optimizer_steps") != 0
        or preflight_payload.get("test_2023_accessed") is not False
        or preflight_payload.get("training_contract_sha256") != training_contract_sha256(experiment)
        or preflight_payload.get("code_sha256") != _sha256(Path(__file__))
        or preflight_payload.get("schedule_sha256") != schedule["sha256"]
        or preflight_manifest.get("runner_sha256") != _sha256(Path(__file__))
        or preflight_manifest.get("objective_sha256") != _sha256(objective_path)
        or preflight_manifest.get("metric_config_sha256") != experiment["metric_admission"]["sha256"]
        or preflight_manifest.get("source_checkpoint_relative_path") != selected_checkpoint
        or preflight_manifest.get("source_checkpoint_sha256") != selected_checkpoint_sha
        or preflight_manifest.get("source_checkpoint_sha_map_entry") != selected_checkpoint_sha
        or preflight_manifest.get("scientific_contract_sha256")
        != training_contract_sha256(experiment)
    ):
        raise ValueError("required preflight is not bound to this exact training law")
    del source_model
    torch.cuda.empty_cache()
    output_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(output_dir / "schedule.json", schedule)
    status_path = output_dir / "status.json"
    _atomic_json(status_path, {"status": "initializing", "completed_updates": 0})
    tracker = ClearMLTracker(
        experiment["project_name"],
        f"{experiment['task_name']}-{output_dir.name}",
        tags=experiment["clearml"]["tags"],
        env_path=experiment["clearml"]["env_path"],
    )
    tracker.connect(
        "matched_full_cfm_contract",
        {"source": experiment["source"], "protocol": protocol, "metric": admission, "schedule": schedule},
    )
    grid = make_normalized_xy_grid(*model_config.image_size, device=device, dtype=torch.float32)
    histories: dict[str, list[dict[str, Any]]] = {}
    checkpoints: dict[str, dict[str, dict[str, str]]] = {}
    interrupted = False

    def terminate(_signum, _frame):
        nonlocal interrupted
        interrupted = True

    previous = {signal_number: signal.getsignal(signal_number) for signal_number in (signal.SIGINT, signal.SIGTERM)}
    for signal_number in previous:
        signal.signal(signal_number, terminate)
    try:
        for arm in ARMS:
            weight = 0.0 if arm == "control" else treatment_weight
            sampler = load_sampler(
                experiment["source"]["run_dir"],
                experiment["source"]["checkpoint"],
                metadata["training_config"],
                device=device,
            )
            model = sampler.model.train()
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(protocol["learning_rate"]),
                weight_decay=float(protocol["weight_decay"]),
            )
            ema = EMAModel(model.parameters(), decay=float(protocol["ema_decay"]))
            ema.to(device)
            loader = _make_loader(dataset, schedule, protocol)
            history = []
            checkpoints[arm] = {}
            for update_index, raw_batch in enumerate(loader, start=1):
                if interrupted:
                    raise InterruptedError("matched training received termination signal")
                batch = _batch_to_device(raw_batch, device)
                timesteps = torch.tensor(
                    schedule["timesteps"][update_index - 1], dtype=torch.float32, device=device
                )
                optimizer.zero_grad(set_to_none=True)
                loss, diagnostics, evidence = _one_forward(
                    model,
                    batch,
                    timesteps,
                    schedule["noise_seeds"][update_index - 1],
                    schedule["dropout_seeds"][update_index - 1],
                    grid,
                    weight,
                    spec,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite matched full-CFM training loss")
                loss.backward()
                unclipped_norm = _gradient_norm(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(protocol["gradient_clip_norm"]))
                optimizer.step()
                ema.step(model.parameters())
                record = {
                    "update": update_index,
                    "loss": float(loss.detach().cpu()),
                    "native": float(diagnostics["native"].detach().cpu()),
                    "geometry": float(diagnostics["geometry"].detach().cpu()),
                    "gradient_norm_before_clip": unclipped_norm,
                    "case_ids": list(raw_batch["meta"]["case_id"]),
                    "input_sha256": {
                        "truth": _tensor_sha256(evidence["truth"]),
                        "noise": _tensor_sha256(evidence["noise"]),
                        "timesteps": _tensor_sha256(evidence["timesteps"]),
                    },
                }
                history.append(record)
                tracker.report_scalar("matched_full_cfm/loss", arm, record["loss"], update_index)
                tracker.report_scalar("matched_full_cfm/native", arm, record["native"], update_index)
                tracker.report_scalar("matched_full_cfm/geometry", arm, record["geometry"], update_index)
                if update_index in protocol["checkpoint_updates"]:
                    checkpoints[arm][str(update_index)] = _save_checkpoint(
                        output_dir, arm, update_index, model, ema, optimizer
                    )
                _atomic_json(
                    status_path,
                    {
                        "status": "training",
                        "arm": arm,
                        "completed_updates": update_index,
                        "schedule_sha256": schedule["sha256"],
                        "latest_record": record,
                        "updated_at": _utc_now(),
                    },
                )
            histories[arm] = history
            _atomic_json(output_dir / arm / "history.json", history)
            del ema, optimizer, model, sampler, loader
            torch.cuda.empty_cache()
        if len(histories["control"]) != len(histories["treatment"]):
            raise RuntimeError("matched arms completed different update counts")
        for control, treatment in zip(histories["control"], histories["treatment"], strict=True):
            if (
                control["case_ids"] != treatment["case_ids"]
                or control["input_sha256"] != treatment["input_sha256"]
            ):
                raise RuntimeError("matched arm evidence diverged")
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed_pending_paired_validation",
            "completed_updates_per_arm": int(protocol["updates_per_arm"]),
            "schedule_sha256": schedule["sha256"],
            "checkpoints": checkpoints,
            "test_2023_accessed": False,
            "training_contract_sha256": training_contract_sha256(experiment),
            "config_sha256": _sha256(config_path),
            "code_sha256": _sha256(Path(__file__)),
            "completed_at": _utc_now(),
        }
        _atomic_json(output_dir / "training_result.json", result)
        _atomic_json(status_path, result)
        tracker.close()
        return result
    except BaseException as error:
        _atomic_json(
            status_path,
            {"status": "failed", "error_type": type(error).__name__, "error": str(error), "updated_at": _utc_now()},
        )
        try:
            tracker.close()
        finally:
            raise
    finally:
        for signal_number, handler in previous.items():
            signal.signal(signal_number, handler)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", choices=("preflight", "train"), required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.mode == "preflight":
        result = run_preflight(args.config.resolve(), args.output_dir.resolve())
    else:
        result = run_training(args.config.resolve(), args.output_dir.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
