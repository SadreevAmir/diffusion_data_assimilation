"""Zero-update single-GPU preflight for the IDEA-F1 mask-only A/B runner."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import torch

from .config import load_json
from .data import build_dataset
from .direct_dynamics_mixed_support_admission import MODE as ADMISSION_MODE
from .direct_dynamics_mixed_support_mask_runner import (
    build_matched_models,
    coarse_mask_batch_from_item,
    masked_cfm_loss,
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
    _atomic_json(status_path, {"status": "initializing", "optimizer_steps": 0})
    try:
        commit = os.environ.get("IDEA_F1_CODE_COMMIT", "")
        if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise ValueError("IDEA_F1_CODE_COMMIT must be an exact commit")
        config = load_json(config_path)
        if config.get("mode") != ADMISSION_MODE:
            raise ValueError("preflight requires the admitted IDEA-F1 config")
        if config.get("dataset_split") != "train" or bool(config.get("test_2023_access_allowed", True)):
            raise ValueError("preflight is train-only with test-2023 sealed")
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
        generator = torch.Generator(device=device).manual_seed(seed + 1)
        uniform = torch.rand(batch.target.shape, generator=generator, device=device).clamp(1e-5, 1 - 1e-5)
        noise = torch.randn(batch.target.shape, generator=generator, device=device)
        time = torch.tensor([0.43], device=device)
        arms = {}
        for label, model in (("candidate", candidate), ("control", control)):
            model.to(device).train()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, per_case = masked_cfm_loss(
                    model, batch, time=time, noise=noise, uniform=uniform
                )
            if not torch.isfinite(loss) or not torch.isfinite(per_case).all():
                raise FloatingPointError(f"{label} preflight loss is not finite")
            loss.backward()
            gradients = _finite_positive_gradients(model)
            if not gradients["all_finite_positive"]:
                raise FloatingPointError(f"{label} has missing or invalid gradients")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                decoded = sample_masks(
                    model,
                    condition=batch.condition,
                    d0_occurrence=batch.d0_occurrence,
                    valid=batch.valid,
                    noise=noise,
                    steps=1,
                )
            if not torch.isfinite(decoded).all() or torch.any(decoded[batch.valid.expand_as(decoded) == 0] != 0):
                raise FloatingPointError(f"{label} sampling path violates finite/land contract")
            arms[label] = {
                "loss": float(loss.detach().item()),
                "per_case_loss": [float(value) for value in per_case.detach()],
                "gradients": gradients,
                "decoded_ice_cells": int(decoded.sum().item()),
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
            "gpu_training_authorized": False,
            "claim_boundary": "zero-update execution and memory only; no skill or calibration claim",
        }
        _atomic_json(output_dir / "preflight.json", result)
        _atomic_json(status_path, {"status": "preflight_passed", "optimizer_steps": 0})
        return result
    except BaseException as error:
        _atomic_json(status_path, {
            "status": "failed", "optimizer_steps": 0,
            "error_type": type(error).__name__, "error": str(error),
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("output_dir")
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.config, arguments.output_dir), indent=2))


if __name__ == "__main__":
    main()
