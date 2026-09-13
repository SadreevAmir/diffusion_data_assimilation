"""Zero-update actual-EMA6 admission for the native geometry-score A/B."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import signal
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .clearml_tracking import ClearMLTracker
from .config import TrainingConfig, load_json
from .data import build_dataset
from .direct_dynamics_cascade_coarse_proper_refinement import (
    proper_objective,
)
from .direct_dynamics_geometry_score import geometry_energy_score
from .direct_dynamics_suffix import (
    FROZEN_INTERVALS,
    TOTAL_INTERVALS,
    exact_two_pass_score_vjp,
    frozen_prefix,
    integrate_intervals,
)
from .direct_dynamics_training import (
    DIRECT_OUTPUT_CHANNELS,
    _repeat_field_stats,
    validate_direct_dataset,
)
from .model_io import load_sampler
from .runtime import make_normalized_xy_grid, seed_everything
from .trainer import _atomic_json
from .transforms import channel_denormalize


SCHEMA_VERSION = "direct_dynamics_geometry_objective_actual_preflight_v1"
SOURCE_SHA_KEYS = {
    "metadata.json",
    "config.json",
    "epoch_snapshots/epoch_0006/ema_last_model.pth",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    """Stable evidence hash over a contiguous CPU tensor and its contract."""

    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _atomic_torch_save(payload: Any, path: Path) -> str:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(path)


def _finite_scalars(value: Any, path: str = "result") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _finite_scalars(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_scalars(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError(f"non-finite scalar at {path}")


def _repeat_members(value: torch.Tensor, members: int) -> torch.Tensor:
    return value[:, None].expand(-1, members, *value.shape[1:]).flatten(0, 1)


def _schedule_indices(length: int, updates: int, slices_per_day: int = 24) -> list[int]:
    if length % slices_per_day:
        raise ValueError("all-hour train length is not divisible by 24")
    days = length // slices_per_day
    if updates < 2 or updates > days:
        raise ValueError("update count must select distinct train anchor dates")
    day_indices = [round(index * (days - 1) / (updates - 1)) for index in range(updates)]
    if len(set(day_indices)) != updates:
        raise RuntimeError("seasonal schedule contains duplicate dates")
    return [day * slices_per_day + 23 for day in day_indices]


def _cell_area_proxy(
    latitude_path: Path,
    longitude_path: Path,
    mask: torch.Tensor,
    image_size: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    latitude = np.deg2rad(np.load(latitude_path).astype(np.float64))
    longitude = np.deg2rad(np.load(longitude_path).astype(np.float64))
    mask_2d = mask.squeeze()
    if mask_2d.ndim != 2:
        raise ValueError("unpacked archive mask must be exactly two-dimensional")
    if latitude.shape != longitude.shape or latitude.shape != tuple(mask_2d.shape):
        raise ValueError("lat/lon centers differ from unpadded archive mask")
    lat_y, lat_x = np.gradient(latitude)
    lon_y, lon_x = np.gradient(longitude)
    area = 6371.0**2 * np.cos(latitude) * np.abs(lat_y * lon_x - lat_x * lon_y)
    valid = mask_2d.cpu().numpy().astype(bool)
    if not np.all(np.isfinite(area[valid])) or np.any(area[valid] <= 0):
        raise ValueError("center-Jacobian cell-area proxy is invalid on water")
    padded = np.zeros(image_size, dtype=np.float32)
    padded[: area.shape[0], : area.shape[1]] = area.astype(np.float32)
    tensor = torch.from_numpy(padded)[None, None].to(device)
    return tensor, {
        "kind": "spherical_center_coordinate_jacobian_proxy_not_verified_corners",
        "latitude_path": str(latitude_path),
        "latitude_sha256": _sha256(latitude_path),
        "longitude_path": str(longitude_path),
        "longitude_sha256": _sha256(longitude_path),
        "valid_q05_q50_q95_km2": np.quantile(area[valid], (0.05, 0.5, 0.95)).tolist(),
    }


def _production_sample(
    sampler,
    item: dict[str, Any],
    condition: torch.Tensor,
    valid: torch.Tensor,
    noise: torch.Tensor,
    image_size: tuple[int, int],
    members: int,
    device: torch.device,
) -> torch.Tensor:
    def repeated(key: str) -> torch.Tensor:
        return item[key].unsqueeze(0).repeat(members, *([1] * item[key].ndim)).to(
            device=device, dtype=torch.float32
        )

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return sampler.sample_conditioned(
            background=repeated("background"),
            background_mask=torch.ones_like(repeated("background")),
            obs_values=repeated("obs_values"),
            obs_mask=repeated("obs_mask"),
            water_mask=repeated("water_mask"),
            size=image_size,
            num_timesteps=17,
            device=device,
            method="rk4",
            rtol=1e-5,
            atol=1e-6,
            start_mode="noise",
            initial_noise=noise,
            sample_target="state",
            model_conditioning=_repeat_members(condition, members),
            state_channels=DIRECT_OUTPUT_CHANNELS,
            end_time=0.0,
            state_mask=_repeat_members(valid, members).expand(-1, DIRECT_OUTPUT_CHANNELS, -1, -1),
        ).float()


def _gradient_summary(model: torch.nn.Module) -> dict[str, float]:
    gradients = [
        parameter.grad.detach().float()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradients or not all(torch.all(torch.isfinite(value)) for value in gradients):
        raise FloatingPointError("model gradients are absent or non-finite")
    norm = torch.sqrt(sum(value.square().sum() for value in gradients))
    if not torch.isfinite(norm) or norm <= 0:
        raise FloatingPointError("model gradient norm is dead or non-finite")
    return {
        "parameter_gradient_norm": float(norm.cpu()),
        "parameters_with_gradient": float(len(gradients)),
    }


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    experiment = load_json(config_path)
    if experiment.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("geometry preflight schema differs from reviewed v1")
    protocol = experiment["protocol"]
    expected_protocol = {
        "members": 4,
        "updates_per_arm": 256,
        "batch_size": 1,
        "total_rk4_intervals": TOTAL_INTERVALS,
        "frozen_prefix_intervals": FROZEN_INTERVALS,
        "trainable_suffix_intervals": TOTAL_INTERVALS - FROZEN_INTERVALS,
        "network_precision": "bf16",
        "state_precision": "fp32",
        "score_precision": "fp32",
        "control_objective": "0.75_fair_native_crps_plus_0.25_unbiased_native_joint_energy",
        "treatment_addition": "0.25_unbiased_geometry_observable_energy",
        "geometry_weight": 0.25,
        "product_scale": 0.3980696029516097,
        "product_scale_source": "142_train_anchors_x_d0_d3_d6_d9_slice23_population_std",
        "product_semantics": "mean_sic_times_mean_sit_proxy_not_verified_hourly_volume",
        "seed": 81373,
        "production_replay_atol": 0.00002,
        "step_zero_atol": 0.000001,
        "suffix_replay_atol": 0.0,
        "rank_or_hard_event_training_loss": False,
        "new_support_decoder": False,
        "optimizer_steps_in_preflight": 0,
        "test_2023_accessed": False,
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(f"reviewed protocol mismatch for {key}")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("actual preflight requires exactly one visible GPU")
    if os.environ.get("CLEARML_REQUIRE_ONLINE") != "1":
        raise RuntimeError("actual preflight requires online ClearML logging")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to reuse output {output_dir}")
    output_dir.mkdir(parents=True)
    status_path = output_dir / "status.json"
    _atomic_json(status_path, {"status": "initializing", "optimizer_steps": 0})
    tracker = None
    previous_signal_handlers: dict[int, Any] = {}

    def terminate(signum, _frame):
        raise InterruptedError(f"received termination signal {signum}")

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous_signal_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, terminate)
    try:
        source = experiment["source"]
        if source.get("checkpoint") != "epoch_snapshots/epoch_0006/ema_last_model.pth":
            raise ValueError("source checkpoint is not the reviewed EMA6 binding")
        if set(source.get("sha256", {})) != SOURCE_SHA_KEYS:
            raise ValueError("source SHA map must bind metadata, config, and EMA6 checkpoint exactly")
        run_dir = Path(source["run_dir"])
        verified = {}
        for relative, expected in source["sha256"].items():
            actual = _sha256(run_dir / relative)
            if actual != expected:
                raise ValueError(f"source SHA mismatch for {relative}: {actual}")
            verified[relative] = actual
        repository_root = config_path.parents[2]
        audit_binding = experiment["product_scale_audit"]
        audit_path = repository_root / audit_binding["path"]
        audit_script_path = repository_root / audit_binding["script_path"]
        if _sha256(audit_path) != audit_binding["sha256"]:
            raise ValueError("train-only product-scale audit SHA mismatch")
        if _sha256(audit_script_path) != audit_binding["script_sha256"]:
            raise ValueError("train-only product-scale script SHA mismatch")
        audit_payload = json.loads(audit_path.read_text())
        if (
            audit_payload.get("schema_version") != "train_geometry_product_scale_v1"
            or audit_payload.get("split") != "train-2016-2021"
            or audit_payload.get("test_2023_read") is not False
            or audit_payload.get("area_times_sit_population_std") != protocol["product_scale"]
        ):
            raise ValueError("product scale is not bound to the reviewed train-only audit")
        code_identity = {
            "config_sha256": _sha256(config_path),
            "preflight_sha256": _sha256(Path(__file__)),
            "geometry_score_sha256": _sha256(
                repository_root / "assim_lib/direct_dynamics_geometry_score.py"
            ),
            "suffix_sha256": _sha256(
                repository_root / "assim_lib/direct_dynamics_suffix.py"
            ),
            "product_scale_audit_sha256": audit_binding["sha256"],
            "product_scale_script_sha256": audit_binding["script_sha256"],
        }
        metadata = json.loads((run_dir / "metadata.json").read_text())
        training_config = metadata["training_config"]
        model_config = TrainingConfig.from_dict(training_config)
        if (
            tuple(model_config.image_size) != (320, 256)
            or model_config.in_channels != 23
            or model_config.out_channels != 6
            or model_config.training_objective != "flow"
            or model_config.dropout != 0.1
            or model_config.structured_velocity_parameterization != "raw"
            or model_config.num_sample_timesteps != 17
            or model_config.sample_method != "rk4"
            or model_config.sample_end_time != 0.0
        ):
            raise ValueError("EMA6 architecture/training contract differs from expected direct flow")
        seed_everything(int(protocol["seed"]))
        dataset = build_dataset(metadata["data_config"], split="train")
        dataset_sentinel = validate_direct_dataset(dataset)
        schedule = _schedule_indices(len(dataset), int(protocol["updates_per_arm"]))
        item = dataset[schedule[0]]
        if dataset.split != "train":
            raise RuntimeError("actual preflight escaped the train split")
        target_paths = tuple(item["meta"]["target_trajectory_paths"])
        if not target_paths or any("2023" in Path(value).name for value in target_paths):
            raise RuntimeError("test-year path entered preflight or target provenance is absent")

        device = torch.device("cuda:0")
        tracker = ClearMLTracker(
            experiment["project_name"],
            f"{experiment['task_name']}-{output_dir.name}",
            tags=experiment["clearml"]["tags"],
            env_path=experiment["clearml"]["env_path"],
        )
        _atomic_json(
            status_path,
            {"status": "gpu_preflight", "optimizer_steps": 0, "clearml_task_id": str(tracker.task.id)},
        )
        sampler = load_sampler(
            str(run_dir), source["checkpoint"], training_config, device=device
        )
        frozen_model = sampler.model.eval()
        for parameter in frozen_model.parameters():
            parameter.requires_grad_(False)
        condition = item["structured_conditioning"].unsqueeze(0).to(device, torch.float32)
        valid = item["valid_mask"].unsqueeze(0)[:, :1].to(device, torch.float32)
        truth = item["truth"].unsqueeze(0).to(device, torch.float32)
        physical_truth = item["structured_physical_truth"].unsqueeze(0).to(device, torch.float32)
        initial_sic = item["structured_physical_background"][0:1].unsqueeze(0).to(
            device, torch.float32
        )
        members = int(protocol["members"])
        generator = torch.Generator(device=device).manual_seed(int(protocol["seed"]) + 1)
        noise = torch.randn(
            (members, DIRECT_OUTPUT_CHANNELS, *model_config.image_size),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        member_valid = _repeat_members(valid, members)
        masked_noise = torch.where(
            member_valid.expand_as(noise) > 0,
            noise,
            torch.zeros_like(noise),
        )
        grid = make_normalized_xy_grid(*model_config.image_size, device=device, dtype=torch.float32)
        cell_area, cell_area_evidence = _cell_area_proxy(
            Path(experiment["cell_geometry"]["latitude_path"]),
            Path(experiment["cell_geometry"]["longitude_path"]),
            torch.from_numpy(~np.load(metadata["data_config"]["mask_path"])).unsqueeze(0),
            tuple(model_config.image_size),
            device,
        )
        means = _repeat_field_stats(dataset.means)
        stds = _repeat_field_stats(dataset.stds)
        input_evidence_sha256 = _atomic_torch_save(
            {
                "condition": condition.detach().cpu(),
                "valid": valid.detach().cpu(),
                "truth": truth.detach().cpu(),
                "physical_truth": physical_truth.detach().cpu(),
                "initial_sic": initial_sic.detach().cpu(),
                "noise": noise.detach().cpu(),
                "masked_noise": masked_noise.detach().cpu(),
                "source": source,
                "code_identity": code_identity,
                "train_index": int(schedule[0]),
            },
            output_dir / "step0_inputs.pth",
        )
        prefix = torch.stack(
            [
                frozen_prefix(
                    frozen_model,
                    noise[member : member + 1],
                    condition,
                    valid,
                    grid,
                )[0]
                for member in range(members)
            ],
            dim=0,
        ).unsqueeze(0)
        prefix_evidence_sha256 = _atomic_torch_save(
            prefix.detach().cpu(), output_dir / "step0_prefix.pth"
        )
        with torch.no_grad():
            custom_full = torch.cat(
                [
                    integrate_intervals(
                        frozen_model,
                        masked_noise[member : member + 1],
                        condition,
                        valid,
                        grid,
                        first_interval=0,
                        final_interval=TOTAL_INTERVALS,
                    )
                    for member in range(members)
                ],
                dim=0,
            )
        custom_evidence_sha256 = _atomic_torch_save(
            custom_full.detach().cpu(), output_dir / "step0_custom_full.pth"
        )
        with torch.no_grad():
            production = torch.cat(
                [
                    _production_sample(
                        sampler,
                        item,
                        condition,
                        valid,
                        noise[member : member + 1],
                        tuple(model_config.image_size),
                        1,
                        device,
                    )
                    for member in range(members)
                ],
                dim=0,
            )
        production_evidence_sha256 = _atomic_torch_save(
            production.detach().cpu(), output_dir / "step0_production.pth"
        )
        production_replay_max_abs = float((custom_full - production).abs().max().cpu())
        if production_replay_max_abs > float(protocol["production_replay_atol"]):
            raise RuntimeError("custom direct RK4 does not replay production EMA6")

        def physical(outputs: torch.Tensor) -> torch.Tensor:
            return channel_denormalize(outputs.float(), means, stds)

        arms = {}
        output_hashes = set()
        for arm in ("control", "treatment"):
            candidate = copy.deepcopy(frozen_model).eval()
            for parameter in candidate.parameters():
                parameter.requires_grad_(True)
            candidate.zero_grad(set_to_none=True)

            def score(outputs: torch.Tensor) -> torch.Tensor:
                native, _, _ = proper_objective(outputs, truth, valid)
                if arm == "control":
                    return native
                geometry, _ = geometry_energy_score(
                    physical(outputs),
                    physical_truth,
                    valid,
                    initial_sic,
                    cell_area,
                    sic_scale=float(dataset.stds[0]),
                    sit_scale=float(dataset.stds[1]),
                    product_scale=float(protocol["product_scale"]),
                )
                return native + float(protocol["geometry_weight"]) * geometry

            def persist_score_evidence(
                outputs: torch.Tensor,
                objective: torch.Tensor,
                output_gradient: torch.Tensor,
            ) -> None:
                _atomic_torch_save(
                    {
                        "arm": arm,
                        "outputs": outputs.cpu(),
                        "objective": objective.cpu(),
                        "output_gradient": output_gradient.cpu(),
                        "source": source,
                        "code_identity": code_identity,
                    },
                    output_dir / f"step0_{arm}_score_vjp.pth",
                )

            torch.cuda.reset_peak_memory_stats(device)
            outputs, objective, vjp = exact_two_pass_score_vjp(
                candidate,
                prefix,
                condition,
                valid,
                grid,
                score,
                replay_atol=float(protocol["suffix_replay_atol"]),
                evidence_callback=persist_score_evidence,
            )
            step_zero_max_abs = float(
                (outputs.flatten(0, 1) - custom_full).abs().max().cpu()
            )
            if step_zero_max_abs > float(protocol["step_zero_atol"]):
                raise RuntimeError(f"{arm} step-zero law differs from EMA6")
            arms[arm] = {
                "objective": float(objective.cpu()),
                "step_zero_vs_ema6_max_abs": step_zero_max_abs,
                "step_zero_output_sha256": _tensor_sha256(outputs),
                "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(device) / 2**20),
                "peak_gpu_reserved_mib": float(torch.cuda.max_memory_reserved(device) / 2**20),
                **vjp,
                **_gradient_summary(candidate),
            }
            output_hashes.add(arms[arm]["step_zero_output_sha256"])
            _atomic_json(
                output_dir / "step0_arm_progress.json",
                {"completed_arms": arms, "optimizer_steps": 0},
            )
            del candidate, outputs
            torch.cuda.empty_cache()

        if len(output_hashes) != 1:
            raise RuntimeError("control/treatment step-zero laws are not identical")

        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "preflight_passed",
            "training_performed": False,
            "optimizer_steps": 0,
            "source": source,
            "code_identity": code_identity,
            "verified_sha256": verified,
            "protocol": protocol,
            "dataset_sentinel": dataset_sentinel,
            "schedule": {
                "count": len(schedule),
                "first_index": schedule[0],
                "last_index": schedule[-1],
                "unique_indices": len(set(schedule)),
                "fixed_archive_slice": 23,
            },
            "cell_area": cell_area_evidence,
            "production_replay_max_abs": production_replay_max_abs,
            "evidence_sha256": {
                "step0_inputs_file": input_evidence_sha256,
                "step0_prefix_file": prefix_evidence_sha256,
                "step0_custom_full_file": custom_evidence_sha256,
                "step0_production_file": production_evidence_sha256,
                "noise": _tensor_sha256(noise),
                "masked_noise": _tensor_sha256(masked_noise),
                "frozen_prefix": _tensor_sha256(prefix),
                "custom_full": _tensor_sha256(custom_full),
                "production": _tensor_sha256(production),
            },
            "arms": arms,
            "clearml_task_id": str(tracker.task.id),
            "test_2023_accessed": False,
        }
        _finite_scalars(result)
        _atomic_json(output_dir / "preflight.json", result)
        tracker.connect("geometry_objective_preflight", result)
        tracker.upload_artifact("geometry_objective_preflight", output_dir / "preflight.json")
        tracker.close()
        tracker = None
        _atomic_json(status_path, {"status": "preflight_passed", "optimizer_steps": 0})
        return result
    except BaseException as error:
        durable_artifacts = {}
        try:
            durable_artifacts = {
                path.name: _sha256(path)
                for path in sorted(output_dir.iterdir())
                if path.is_file() and path != status_path
            }
        except Exception:
            pass
        try:
            _atomic_json(
                status_path,
                {
                    "status": "failed",
                    "optimizer_steps": 0,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "durable_artifacts": durable_artifacts,
                },
            )
        except Exception:
            pass
        if tracker is not None:
            try:
                tracker.task.mark_failed(
                    status_reason=f"{type(error).__name__}: {error}"
                )
            except Exception:
                pass
        raise
    finally:
        for signum, previous in previous_signal_handlers.items():
            signal.signal(signum, previous)
        if tracker is not None:
            try:
                tracker.close()
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.config.resolve(), arguments.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
