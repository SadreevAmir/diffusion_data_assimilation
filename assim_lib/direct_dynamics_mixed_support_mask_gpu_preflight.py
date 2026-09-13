"""Zero-update single-GPU preflight for the IDEA-F1 mask-only A/B runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
from pathlib import Path
from typing import Any

import torch

from .clearml_tracking import ClearMLTracker
from .config import load_json
from .data import build_dataset
from .direct_dynamics_mixed_support_admission import (
    MODE as ADMISSION_MODE,
    binary_dequantized_logit,
)
from .direct_dynamics_mixed_support_mask_runner import (
    build_matched_models,
    coarse_mask_batch_from_item,
    sample_masks,
)


MODE = "direct_dynamics_mixed_support_mask_gpu_preflight_v1"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(payload: Any, path: Path) -> str:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _sha256(path)


def _terminate(signum, _frame) -> None:
    raise RuntimeError(f"termination signal {signum}")


def _injected_failure(stage: str) -> None:
    if os.environ.get("IDEA_F1_PREFLIGHT_FAIL_STAGE") == stage:
        raise RuntimeError(f"injected preflight failure after {stage}")


def _finalize_failure(
    status_path: Path,
    reservation: dict[str, Any],
    error: BaseException,
    tracker: ClearMLTracker | None,
    evidence_sha256: dict[str, str],
) -> None:
    """Preserve the primary error while independently attempting cleanup."""
    failure = {
        **reservation,
        "status": "failed",
        "error_type": type(error).__name__,
        "error": str(error)[:2000],
        "evidence_sha256": dict(evidence_sha256),
        "cleanup_errors": [],
    }
    try:
        _atomic_json(status_path, failure)
    except Exception as cleanup_error:
        failure["cleanup_errors"].append(f"status_write: {cleanup_error}")
    if tracker is not None:
        try:
            tracker.task.mark_failed(status_reason=str(error)[:1000])
        except Exception as cleanup_error:
            failure["cleanup_errors"].append(f"clearml_mark_failed: {cleanup_error}")
        try:
            tracker.close()
        except Exception as cleanup_error:
            failure["cleanup_errors"].append(f"clearml_close: {cleanup_error}")
    try:
        _atomic_json(status_path, failure)
    except Exception:
        pass


def _finite_positive_gradients(model: torch.nn.Module) -> dict[str, Any]:
    norms = {
        name: float(parameter.grad.float().norm().item()) if parameter.grad is not None else 0.0
        for name, parameter in model.named_parameters()
    }
    return {
        "norms": norms,
        "all_finite_positive": all(value > 0 and torch.isfinite(torch.tensor(value)) for value in norms.values()),
    }


def run(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    config_path, output_dir = Path(config_path), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse preflight output: {output_dir}")
    output_dir.mkdir(parents=True, mode=0o700)
    status_path = output_dir / "status.json"
    reservation = {"status": "initializing", "optimizer_steps": 0}
    _atomic_json(status_path, reservation)
    tracker: ClearMLTracker | None = None
    evidence_sha256: dict[str, str] = {}
    try:
        commit = os.environ.get("IDEA_F1_CODE_COMMIT", "")
        if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise ValueError("IDEA_F1_CODE_COMMIT must be an exact commit")
        config = load_json(config_path)
        if config.get("mode") != ADMISSION_MODE:
            raise ValueError("preflight requires the admitted IDEA-F1 config")
        if config.get("dataset_split") != "train" or bool(config.get("test_2023_access_allowed", True)):
            raise ValueError("preflight is train-only with test-2023 sealed")
        if os.environ.get("CLEARML_REQUIRE_ONLINE") != "1":
            raise RuntimeError("IDEA-F1 GPU preflight requires online ClearML")
        tracker = ClearMLTracker(
            "sea_ice_two_stage",
            f"direct_dynamics_mixed_support_mask_gpu_preflight_v1-{output_dir.name}",
            tags=["idea-f1", "gpu-preflight", "zero-update", "mask-only"],
            env_path="/home/.env",
        )
        tracker.connect("preflight_contract", config)
        reservation.update({
            "code_commit": commit,
            "clearml_task_id": str(tracker.task.id),
        })
        _atomic_json(status_path, {**reservation, "status": "tracker_online"})
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("preflight requires exactly one visible CUDA device")
        device = torch.device("cuda:0")
        torch.cuda.reset_peak_memory_stats(device)
        dataset_config = dict(load_json(config["dataset_config"]))
        dataset_config["dynamic_forcing_stats"] = config["dynamic_forcing_stats"]
        dataset = build_dataset(dataset_config, "train")
        indices = dataset.strided_case_indices(max_cases=1, stride_days=int(config["stride_days"]))
        if len(indices) != 1:
            raise RuntimeError("no train preflight case")
        batch_cpu = coarse_mask_batch_from_item(
            dataset[indices[0]], dataset_config["means"], dataset_config["stds"]
        )
        batch = type(batch_cpu)(*(value.to(device) for value in (
            batch_cpu.target, batch_cpu.condition, batch_cpu.d0_occurrence, batch_cpu.valid
        )))
        seed = int(config["seed"])
        candidate, control = build_matched_models(seed=seed, hidden_channels=8)
        parameter_count = sum(parameter.numel() for parameter in candidate.parameters())
        initial_parameters = {
            label: {name: value.detach().clone() for name, value in model.state_dict().items()}
            for label, model in (("candidate", candidate), ("control", control))
        }
        generator = torch.Generator(device=device).manual_seed(seed + 1)
        uniform = torch.rand(batch.target.shape, generator=generator, device=device).clamp(1e-5, 1 - 1e-5)
        noise = torch.randn(batch.target.shape, generator=generator, device=device)
        time = torch.tensor([0.43], device=device)
        clean = binary_dequantized_logit(batch.target, uniform)
        clean = torch.where(batch.valid > 0, clean, torch.zeros_like(clean))
        masked_noise = torch.where(batch.valid > 0, noise, torch.zeros_like(noise))
        state = (1 - time[:, None, None, None]) * clean + time[:, None, None, None] * masked_noise
        target_velocity = masked_noise - clean
        fixed_path = output_dir / "fixed_inputs.pt"
        evidence_sha256[fixed_path.name] = _atomic_torch_save({
            "target": batch.target.cpu(),
            "condition": batch.condition.cpu(),
            "d0_occurrence": batch.d0_occurrence.cpu(),
            "valid": batch.valid.cpu(),
            "uniform": uniform.cpu(),
            "noise": noise.cpu(),
            "time": time.cpu(),
            "state": state.cpu(),
            "target_velocity": target_velocity.cpu(),
            "initial_parameters": initial_parameters,
            "case_id": dataset[indices[0]]["meta"]["case_id"],
            "code_commit": commit,
        }, fixed_path)
        _atomic_json(status_path, {**reservation, "status": "fixed_evidence", "evidence_sha256": evidence_sha256})
        _injected_failure("fixed_evidence")
        arms = {}
        for label, model in (("candidate", candidate), ("control", control)):
            model.to(device).train()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = model(
                    state, time, batch.condition, batch.d0_occurrence, batch.valid
                )
            prediction_path = output_dir / f"{label}_prediction.pt"
            evidence_sha256[prediction_path.name] = _atomic_torch_save(
                {"prediction": prediction.detach().cpu(), "code_commit": commit}, prediction_path
            )
            _atomic_json(status_path, {
                **reservation, "status": f"{label}_prediction_evidence",
                "evidence_sha256": evidence_sha256,
            })
            _injected_failure(f"{label}_prediction_evidence")
            numerator = (
                (prediction - target_velocity).square() * batch.valid
            ).reshape(state.shape[0], -1).sum(dim=1)
            denominator = batch.valid.reshape(state.shape[0], -1).sum(dim=1) * state.shape[1]
            if torch.any(denominator <= 0):
                raise ValueError("each preflight case requires positive ocean support")
            per_case = numerator / denominator
            loss = per_case.mean()
            if not torch.isfinite(loss) or not torch.isfinite(per_case).all():
                raise FloatingPointError(f"{label} preflight loss is not finite")
            loss.backward()
            gradients = _finite_positive_gradients(model)
            if not gradients["all_finite_positive"]:
                raise FloatingPointError(f"{label} has missing or invalid gradients")
            terminal_path = output_dir / f"{label}_terminal_latent.pt"

            def save_terminal_latent(latent: torch.Tensor, *, path=terminal_path, arm=label) -> None:
                evidence_sha256[path.name] = _atomic_torch_save(
                    {"terminal_latent": latent.cpu(), "code_commit": commit}, path
                )
                _atomic_json(status_path, {
                    **reservation,
                    "status": f"{arm}_terminal_latent_evidence",
                    "evidence_sha256": evidence_sha256,
                })

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                decoded = sample_masks(
                    model,
                    condition=batch.condition,
                    d0_occurrence=batch.d0_occurrence,
                    valid=batch.valid,
                    noise=noise,
                    steps=1,
                    before_decode=save_terminal_latent,
                )
            sample_path = output_dir / f"{label}_sample.pt"
            evidence_sha256[sample_path.name] = _atomic_torch_save(
                {"decoded": decoded.detach().cpu(), "code_commit": commit}, sample_path
            )
            _atomic_json(status_path, {
                **reservation, "status": f"{label}_sample_evidence",
                "evidence_sha256": evidence_sha256,
            })
            _injected_failure(f"{label}_sample_evidence")
            if not torch.isfinite(decoded).all() or torch.any(decoded[batch.valid.expand_as(decoded) == 0] != 0):
                raise FloatingPointError(f"{label} sampling path violates finite/land contract")
            parameters_unchanged = all(
                torch.equal(value.detach().cpu(), initial_parameters[label][name])
                for name, value in model.state_dict().items()
            )
            if not parameters_unchanged:
                raise RuntimeError(f"{label} parameters changed during zero-update preflight")
            arms[label] = {
                "loss": float(loss.detach().item()),
                "per_case_loss": [float(value) for value in per_case.detach()],
                "gradients": gradients,
                "decoded_ice_cells": int(decoded.sum().item()),
                "parameters_unchanged": True,
                "prediction_evidence_sha256": evidence_sha256[prediction_path.name],
                "terminal_latent_evidence_sha256": evidence_sha256[terminal_path.name],
                "sample_evidence_sha256": evidence_sha256[sample_path.name],
            }
            model.zero_grad(set_to_none=True)
            model.to("cpu")
            torch.cuda.empty_cache()
        result = {
            "schema_version": MODE,
            "status": "preflight_passed",
            "code_commit": commit,
            "resource_kind": "single_gpu",
            "device_name": torch.cuda.get_device_name(device),
            "dataset_split": "train",
            "test_2023_accessed": False,
            "case_id": dataset[indices[0]]["meta"]["case_id"],
            "common_random_numbers": True,
            "parameter_count_each_arm": parameter_count,
            "precision": "bf16_autocast_fp32_parameters",
            "optimizer_created": False,
            "optimizer_steps": 0,
            "sampling_steps_per_arm": 1,
            "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(device) / 2**20),
            "arms": arms,
            "evidence_sha256": evidence_sha256,
            "clearml_task_id": str(tracker.task.id),
            "gpu_training_authorized": False,
            "claim_boundary": "zero-update execution and memory only; no skill or calibration claim",
        }
        metrics_path = output_dir / "preflight.json"
        _atomic_json(metrics_path, result)
        tracker.connect("preflight_result", result)
        for name in sorted(evidence_sha256):
            tracker.upload_artifact(f"idea_f1_{name}", output_dir / name)
        tracker.upload_artifact("idea_f1_preflight", metrics_path)
        tracker.close()
        tracker = None
        _atomic_json(status_path, {
            **reservation,
            "status": "preflight_passed",
            "optimizer_steps": 0,
            "evidence_sha256": evidence_sha256,
        })
        return result
    except BaseException as error:
        _finalize_failure(status_path, reservation, error, tracker, evidence_sha256)
        tracker = None
        raise
    finally:
        if tracker is not None:
            try:
                tracker.close()
            except Exception:
                pass


def main() -> None:
    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("output_dir")
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.config, arguments.output_dir), indent=2))


if __name__ == "__main__":
    main()
