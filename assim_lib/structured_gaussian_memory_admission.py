"""Final bounded memory admission for the checkpointed Gaussian-path pilot.

This is execution evidence only.  It runs a CPU parity check, compares CUDA
activation peaks on one production-resolution member, and exercises exactly two
training batches plus one validation batch through the real Trainer/Accelerate/
ClearML path.  It never samples a forecast or opens the test split.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any

import torch
from diffusers.training_utils import EMAModel
from torch.utils.data import Dataset

from .config import TrainingConfig, load_json
from .main import main as train_main
from .model_io import build_unet
from .runtime import build_dataloader
from .trainer import UNetTrainer, configure_activation_checkpointing

SCHEMA_VERSION = "structured_gaussian_memory_admission_v1"
EXPECTED_EXPERIMENT = (
    "config/experiments/"
    "admit_structured_joint_gaussian_preconditioned_checkpointed.json"
)
EXPECTED_METHOD = (
    "config/methods/"
    "structured_joint_gaussian_preconditioned_checkpointed_pilot_2f.json"
)
EXPECTED_PROTOCOL = "paper/STRUCTURED_GAUSSIAN_MEMORY_ADMISSION_PROTOCOL.json"
EXPECTED_TRAIN_BATCHES = 2
EXPECTED_VALIDATION_BATCHES = 1
PEAK_RESERVED_FRACTION_LIMIT = 0.90
GRADIENT_ATOL = 1e-6
GRADIENT_RTOL = 1e-5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _tiny_config() -> TrainingConfig:
    return TrainingConfig(
        image_size=(8, 8),
        in_channels=4,
        out_channels=2,
        block_out_channels=(32, 32),
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
        layers_per_block=1,
        norm_num_groups=8,
        dropout=0.0,
    )


def _run_cpu_path(initial: dict[str, torch.Tensor], checkpointing: bool) -> dict[str, Any]:
    model = build_unet(_tiny_config())
    model.load_state_dict(initial)
    modules = configure_activation_checkpointing(model, checkpointing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: 1.0 / (step + 1)
    )
    ema = EMAModel(model.parameters(), decay=0.999)
    torch.manual_seed(9127)
    value = torch.randn(2, 4, 8, 8)
    timestep = torch.tensor([125.0, 875.0])
    before = torch.random.get_rng_state().clone()
    optimizer.zero_grad(set_to_none=True)
    output = model(value, timestep, return_dict=False)[0]
    output.square().mean().backward()
    after = torch.random.get_rng_state().clone()
    gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    optimizer.step()
    scheduler.step()
    ema.step(model.parameters())
    return {
        "modules": modules,
        "output": output.detach(),
        "gradients": gradients,
        "rng_before": before,
        "rng_after": after,
        "parameters": {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        },
        "ema": [parameter.detach().clone() for parameter in ema.shadow_params],
        "optimizer_steps": 1,
        "scheduler_steps": 1,
        "ema_updates": 1,
    }


def cpu_parity_record() -> dict[str, Any]:
    """Prove the audited recomputation path preserves one complete update."""
    torch.manual_seed(5151)
    initial = copy.deepcopy(build_unet(_tiny_config()).state_dict())
    reference = _run_cpu_path(initial, False)
    candidate = _run_cpu_path(initial, True)
    if not torch.equal(reference["output"], candidate["output"]):
        raise RuntimeError("checkpointed CPU forward is not bitwise equal")
    if not torch.equal(reference["rng_before"], candidate["rng_before"]) or not torch.equal(
        reference["rng_after"], candidate["rng_after"]
    ):
        raise RuntimeError("checkpointed CPU path changed the RNG state")
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for name, reference_gradient in reference["gradients"].items():
        candidate_gradient = candidate["gradients"][name]
        if (reference_gradient is None) != (candidate_gradient is None):
            raise RuntimeError(f"gradient presence differs for {name}")
        if reference_gradient is None:
            continue
        if not bool(torch.isfinite(reference_gradient).all()) or not bool(
            torch.isfinite(candidate_gradient).all()
        ):
            raise FloatingPointError(f"non-finite CPU gradient for {name}")
        difference = (candidate_gradient - reference_gradient).abs()
        maximum_absolute = max(maximum_absolute, float(difference.max().item()))
        relative = difference / reference_gradient.abs().clamp_min(1e-12)
        maximum_relative = max(maximum_relative, float(relative.max().item()))
        if not torch.allclose(
            candidate_gradient,
            reference_gradient,
            atol=GRADIENT_ATOL,
            rtol=GRADIENT_RTOL,
        ):
            raise RuntimeError(f"checkpointed CPU gradient differs for {name}")
    for name, reference_parameter in reference["parameters"].items():
        if not torch.equal(reference_parameter, candidate["parameters"][name]):
            raise RuntimeError(f"post-Adam parameter differs for {name}")
    if len(reference["ema"]) != len(candidate["ema"]):
        raise RuntimeError("EMA parameter counts differ")
    if not all(
        torch.equal(reference_shadow, candidate_shadow)
        for reference_shadow, candidate_shadow in zip(
            reference["ema"], candidate["ema"], strict=True
        )
    ):
        raise RuntimeError("post-update EMA state differs")
    return {
        "status": "passed",
        "forward_bitwise_equal": True,
        "rng_state_equal": True,
        "gradient_none_pattern_equal": True,
        "gradients_finite": True,
        "gradient_atol": GRADIENT_ATOL,
        "gradient_rtol": GRADIENT_RTOL,
        "gradient_max_abs_difference": maximum_absolute,
        "gradient_max_relative_difference": maximum_relative,
        "post_adam_parameters_bitwise_equal": True,
        "post_ema_parameters_bitwise_equal": True,
        "checkpointed_modules": list(candidate["modules"]),
        "optimizer_steps": 1,
        "scheduler_steps": 1,
        "ema_updates": 1,
    }


def _memory_record(device: torch.device) -> dict[str, int]:
    free, total = torch.cuda.mem_get_info(device)
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "free_bytes": int(free),
        "total_bytes": int(total),
    }


def gpu_activation_probe(config: TrainingConfig, checkpointing: bool) -> dict[str, Any]:
    """Measure activation memory without allocating optimizer state."""
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    model = build_unet(config)
    modules = configure_activation_checkpointing(model, checkpointing)
    model.enable_xformers_memory_efficient_attention()
    model.to(device).train()
    value = torch.randn(
        (1, config.in_channels, *config.image_size),
        device=device,
        dtype=torch.float32,
    )
    timestep = torch.tensor([500.0], device=device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(value, timestep, return_dict=False)[0]
        loss = output.float().square().mean()
    loss.backward()
    torch.cuda.synchronize(device)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("activation memory probe produced a non-finite loss")
    record = {
        "checkpointing": bool(checkpointing),
        "batch_shape": [1, config.in_channels, *config.image_size],
        "checkpointed_modules": list(modules),
        "loss_finite": True,
        "memory": _memory_record(device),
    }
    del loss, output, value, timestep, model
    torch.cuda.empty_cache()
    return record


class _PrefixDataset(Dataset):
    def __init__(self, dataset, length: int):
        if length <= 0 or len(dataset) < length:
            raise ValueError("admission dataset prefix is unavailable")
        self.dataset = dataset
        self.length = int(length)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        if not 0 <= index < self.length:
            raise IndexError(index)
        return self.dataset[index]

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)


class MemoryAdmissionTrainer(UNetTrainer):
    """Bounded adapter that keeps the production Trainer code path intact."""

    last_record: dict[str, Any] | None = None

    def __init__(self, config, model, optimizer, data_loader_train, data_loader_val, *args, **kwargs):
        if not config.activation_checkpointing:
            raise ValueError("memory admission requires activation_checkpointing=true")
        if config.train_batch_size != 16 or config.eval_batch_size != 8:
            raise ValueError("memory admission requires production batch sizes 16/8")
        if config.gradient_accumulation_steps != 1 or config.mixed_precision != "bf16":
            raise ValueError("memory admission requires accumulation=1 and bf16")
        train_dataset = _PrefixDataset(data_loader_train.dataset, 32)
        validation_dataset = _PrefixDataset(data_loader_val.dataset, 8)
        bounded_train = build_dataloader(
            train_dataset,
            config.train_batch_size,
            0,
            shuffle=True,
        )
        bounded_validation = build_dataloader(
            validation_dataset,
            config.eval_batch_size,
            0,
            shuffle=False,
        )
        bounded_config = replace(
            config,
            num_epochs=1,
            minimum_optimizer_steps=2,
            diagnostic_min_optimizer_steps=2,
            sample_every_n_epochs=0,
            metric_every_n_epochs=0,
        )
        super().__init__(
            bounded_config,
            model,
            optimizer,
            bounded_train,
            bounded_validation,
            *args,
            **kwargs,
        )
        if self.clearml is None or os.environ.get("CLEARML_OFFLINE_MODE"):
            raise RuntimeError("memory admission requires online ClearML")
        self._admission_trace: list[str] = []
        self._validation_calls = 0
        self._wrap_call(self.accelerator, "backward", "backward")
        self._wrap_call(self.optimizer, "step", "optimizer")
        self._wrap_call(self.lr_scheduler, "step", "scheduler")
        self._wrap_call(self.ema_model, "step", "ema")
        self._clearml_task_id = str(getattr(self.clearml.task, "id", ""))
        if not self._clearml_task_id:
            raise RuntimeError("ClearML did not return a task id")

    def _wrap_call(self, owner, name: str, label: str) -> None:
        original = getattr(owner, name)

        def traced(*args, **kwargs):
            self._admission_trace.append(label)
            return original(*args, **kwargs)

        setattr(owner, name, traced)

    def compute_val_loss(self) -> tuple[float, float, float]:
        self._validation_calls += 1
        if self.config.validation_weight_source != "ema" or not self.config.sample_use_ema:
            raise RuntimeError("memory admission validation must use EMA")
        return super().compute_val_loss()

    def train_loop(self):
        device = self.accelerator.device
        if device.type != "cuda":
            raise RuntimeError("memory admission requires CUDA")
        torch.cuda.reset_peak_memory_stats(device)
        output = super().train_loop()
        expected_trace = [
            "backward",
            "optimizer",
            "scheduler",
            "ema",
        ] * EXPECTED_TRAIN_BATCHES
        if self._admission_trace != expected_trace:
            raise RuntimeError(
                f"training call order differs: {self._admission_trace} != {expected_trace}"
            )
        if self._validation_calls != EXPECTED_VALIDATION_BATCHES:
            raise RuntimeError("memory admission did not execute exactly one validation pass")
        completion = load_json(Path(output) / "structured_training_completion.json")
        if completion.get("completed_optimizer_steps") != EXPECTED_TRAIN_BATCHES:
            raise RuntimeError("memory admission optimizer-step count differs")
        memory = _memory_record(device)
        if memory["peak_reserved_bytes"] > int(
            PEAK_RESERVED_FRACTION_LIMIT * memory["total_bytes"]
        ):
            raise RuntimeError("checkpointed production path exceeds the 90% memory limit")
        type(self).last_record = {
            "status": "passed",
            "train_batches": EXPECTED_TRAIN_BATCHES,
            "validation_batches": EXPECTED_VALIDATION_BATCHES,
            "batch_size": self.config.train_batch_size,
            "validation_batch_size": self.config.eval_batch_size,
            "image_size": list(self.config.image_size),
            "mixed_precision": self.config.mixed_precision,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "validation_weight_source": self.config.validation_weight_source,
            "call_trace": list(self._admission_trace),
            "checkpointed_modules": list(self.activation_checkpointed_modules),
            "clearml_online": True,
            "clearml_task_id": self._clearml_task_id,
            "memory": memory,
        }
        return output


def _validate_pilot_config(config: TrainingConfig) -> None:
    expected = {
        "image_size": (320, 256),
        "in_channels": 50,
        "out_channels": 16,
        "train_batch_size": 16,
        "eval_batch_size": 8,
        "num_epochs": 16,
        "gradient_accumulation_steps": 1,
        "activation_checkpointing": True,
        "mixed_precision": "bf16",
        "training_objective": "structured_joint_state_flow",
        "structured_velocity_parameterization": "gaussian_path_preconditioned",
        "timestep_sampler": "stratified_uniform",
        "minimum_optimizer_steps": 2128,
        "validation_weight_source": "ema",
    }
    actual = {name: getattr(config, name) for name in expected}
    if actual != expected:
        raise ValueError(f"checkpointed pilot contract differs: {actual} != {expected}")


def run_admission(experiment_path: Path, output_dir: Path) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[1]
    if experiment_path.resolve() != (repo / EXPECTED_EXPERIMENT).resolve():
        raise ValueError("memory admission requires its exact frozen experiment config")
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("memory admission output already exists")
    output_dir.mkdir(parents=True)
    protocol_path = repo / EXPECTED_PROTOCOL
    protocol = load_json(protocol_path)
    if (
        protocol.get("schema_version") != SCHEMA_VERSION
        or protocol.get("single_attempt") is not True
        or protocol.get("purpose") != "execution_memory_only_not_model_quality"
        or protocol.get("terminal_rule")
        != "failure_closes_checkpointing_memory_recovery_without_another_workaround"
    ):
        raise ValueError("memory admission protocol differs from the frozen contract")
    identity = protocol.get("required_identity_sha256")
    if not isinstance(identity, dict) or not identity:
        raise ValueError("memory admission protocol has no identity manifest")
    for relative, expected_sha256 in identity.items():
        relative_path = PurePosixPath(relative) if isinstance(relative, str) else None
        path = repo / relative if isinstance(relative, str) else repo
        if (
            relative_path is None
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or path.is_symlink()
            or not path.is_file()
            or _sha256(path) != expected_sha256
        ):
            raise ValueError(f"memory admission identity differs: {relative}")
    experiment = load_json(experiment_path)
    method_path = (experiment_path.parent / experiment["model_config"]).resolve()
    if method_path != (repo / EXPECTED_METHOD).resolve():
        raise ValueError("memory admission method path differs")
    method = load_json(method_path)
    stats_path = (method_path.parent / method["structured_state_stats_path"]).resolve()
    method["structured_state_stats"] = load_json(stats_path)
    config = TrainingConfig.from_dict(method)
    _validate_pilot_config(config)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("memory admission requires exactly one visible CUDA device")

    parity = cpu_parity_record()
    uncheckpointed = gpu_activation_probe(config, False)
    checkpointed = gpu_activation_probe(config, True)
    if checkpointed["memory"]["peak_reserved_bytes"] >= uncheckpointed["memory"][
        "peak_reserved_bytes"
    ]:
        raise RuntimeError("activation checkpointing did not reduce peak reserved CUDA memory")

    runtime = copy.deepcopy(experiment)
    runtime["training"] = {
        **runtime.get("training", {}),
        "base_output_dir": str((output_dir / "training").resolve()),
        "run_name": "resource_admission",
        "resume_from_checkpoint": "",
        "num_workers_train": 0,
        "num_workers_val": 0,
        "preserve_persistent_worker_rng": True,
        "sample_every_n_epochs": 0,
        "metric_every_n_epochs": 0,
        "metric_save_ensemble_samples": False,
    }
    MemoryAdmissionTrainer.last_record = None
    train_main(
        runtime,
        experiment_path.resolve().parent,
        trainer_class=MemoryAdmissionTrainer,
    )
    production = MemoryAdmissionTrainer.last_record
    if not isinstance(production, dict) or production.get("status") != "passed":
        raise RuntimeError("bounded production Trainer admission did not complete")
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "purpose": "execution_memory_only_not_model_quality",
        "test_2023_used": False,
        "full_training_permitted": False,
        "experiment_config": EXPECTED_EXPERIMENT,
        "experiment_config_sha256": _sha256(experiment_path),
        "method_config": EXPECTED_METHOD,
        "method_config_sha256": _sha256(method_path),
        "protocol": EXPECTED_PROTOCOL,
        "protocol_sha256": _sha256(protocol_path),
        "cpu_parity": parity,
        "activation_memory_comparison": {
            "uncheckpointed": uncheckpointed,
            "checkpointed": checkpointed,
            "checkpointed_peak_strictly_lower": True,
        },
        "production_path": production,
        "evaluation_identity": {
            path: _sha256(repo / path)
            for path in (
                "assim_lib/structured_preconditioned_pilot.py",
                "paper/STRUCTURED_PAIRED_PILOT_PANEL.json",
                "paper/STRUCTURED_GAUSSIAN_PRECONDITIONED_PILOT_PROTOCOL.json",
            )
        },
        "consumed_oom_attempts": [
            "structured_joint_gaussian_preconditioned_pilot_v1",
            "structured_joint_gaussian_preconditioned_pilot_retry1",
        ],
    }
    _atomic_json(output_dir / "memory_admission.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(
        json.dumps(
            run_admission(arguments.experiment, arguments.output_dir),
            indent=2,
            sort_keys=True,
        )
    )
