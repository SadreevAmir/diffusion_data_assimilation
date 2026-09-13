"""Matched 512-update IDEA-F1 mask-only A/B training screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .clearml_tracking import ClearMLTracker
from .config import load_json
from .data import build_dataset
from .direct_dynamics_mixed_support_mask_gpu_preflight import (
    _atomic_json,
    _atomic_torch_save,
    _finalize_failure,
    _sha256,
    _terminate,
)
from .direct_dynamics_mixed_support_mask_runner import (
    CfmBatch,
    build_matched_models,
    coarse_mask_batch_from_item,
    masked_cfm_loss,
)
from .direct_dynamics_mixed_support_train_stats import apply_verified_conditioning_stats


MODE = "direct_dynamics_mixed_support_mask_ab_training_v1"


def _dates_sha(dates: list[str]) -> str:
    return hashlib.sha256("\n".join(dates).encode()).hexdigest()


class _CoarseMaskDataset(Dataset):
    def __init__(self, dataset, indices: list[int], means: list[float], stds: list[float]):
        self.dataset, self.indices, self.means, self.stds = dataset, indices, means, stds

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict[str, torch.Tensor]:
        batch = coarse_mask_batch_from_item(
            self.dataset[self.indices[position]], self.means, self.stds
        )
        return {
            "target": batch.target[0], "condition": batch.condition[0],
            "d0_occurrence": batch.d0_occurrence[0], "valid": batch.valid[0],
        }


def _select_year_indices(dataset, years: set[int]) -> tuple[list[int], list[str]]:
    indices, dates = [], []
    lead = max(dataset.trajectory_lead_days)
    for index, (_, anchor) in enumerate(dataset.calendar_pairs):
        if anchor.date.year in years and (anchor.date + timedelta(days=lead)).year in years:
            indices.append(index)
            dates.append(anchor.date.isoformat())
    return indices, dates


def _validate_contract(config: dict[str, Any], repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if config.get("schema_version") != MODE:
        raise ValueError("unreviewed IDEA-F1 training schema")
    expected = {
        "train_years": [2016, 2017, 2018, 2019, 2020], "heldout_year": 2021,
        "target_slice_index": 23, "trajectory_lead_days": [3, 6, 9],
        "coarse_shape": [80, 64], "updates_per_arm": 512, "batch_size": 8,
        "hidden_channels": 8, "learning_rate": 0.001, "weight_decay": 0.0,
        "adam_betas": [0.9, 0.999], "ema_decay": 0.99,
        "checkpoint_updates": [128, 256, 384, 512], "precision": "bf16",
        "workers": 4, "prefetch_factor": 2, "seed": 91731, "test_2023": "closed",
    }
    if config.get("protocol") != expected:
        raise ValueError("IDEA-F1 training protocol differs from reviewed contract")
    admission_spec = config["admission_config"]
    admission_path = repo / admission_spec["path"]
    if _sha256(admission_path) != admission_spec["sha256"]:
        raise ValueError("admission config SHA mismatch")
    admission = load_json(admission_path)
    preflight_spec = config["preflight"]
    preflight_path = Path(preflight_spec["path"])
    if _sha256(preflight_path) != preflight_spec["sha256"]:
        raise ValueError("preflight SHA mismatch")
    preflight = load_json(preflight_path)
    if (
        preflight.get("status") != "preflight_passed"
        or preflight.get("optimizer_steps") != 0
        or preflight.get("test_2023_accessed") is not False
        or preflight.get("code_commit") != preflight_spec["code_commit"]
        or preflight.get("clearml_task_id") != preflight_spec["clearml_task_id"]
    ):
        raise ValueError("GPU preflight binding is not accepted")
    return admission, preflight


def _ema_update(ema: dict[str, torch.Tensor], model: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for name, value in model.state_dict().items():
            ema[name].mul_(decay).add_(value.detach(), alpha=1 - decay)


def _finite_model(model: torch.nn.Module, label: str) -> None:
    for name, value in model.state_dict().items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"{label} parameter {name} is not finite")


def run(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    config_path, output_dir = Path(config_path), Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("refusing to reuse IDEA-F1 training output")
    output_dir.mkdir(parents=True, mode=0o700)
    status_path = output_dir / "status.json"
    reservation = {"status": "initializing", "optimizer_steps_per_arm": 0}
    _atomic_json(status_path, reservation)
    tracker: ClearMLTracker | None = None
    evidence: dict[str, str] = {}
    try:
        commit = os.environ.get("IDEA_F1_TRAIN_CODE_COMMIT", "")
        if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise ValueError("exact training commit is required")
        repo = Path(__file__).resolve().parents[1]
        config = load_json(config_path)
        admission, preflight = _validate_contract(config, repo)
        if os.environ.get("CLEARML_REQUIRE_ONLINE") != "1":
            raise RuntimeError("online ClearML is required")
        tracker = ClearMLTracker(
            config["project_name"], f"{config['task_name']}-{output_dir.name}",
            tags=config["clearml"]["tags"], env_path=config["clearml"]["env_path"],
        )
        tracker.connect("training_contract", config)
        reservation.update({"code_commit": commit, "clearml_task_id": str(tracker.task.id)})
        _atomic_json(status_path, {**reservation, "status": "tracker_online"})
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("training requires exactly one visible CUDA GPU")
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        data_config = dict(load_json(admission["dataset_config"]))
        data_config, stats_binding = apply_verified_conditioning_stats(
            data_config, admission["conditioning_stats_artifact"]
        )
        dataset = build_dataset(data_config, "train")
        train_indices, train_dates = _select_year_indices(dataset, set(config["protocol"]["train_years"]))
        heldout_indices, heldout_dates = _select_year_indices(dataset, {config["protocol"]["heldout_year"]})
        if len(train_indices) != 1803 or not heldout_indices:
            raise ValueError("train/heldout inventory differs from frozen calendar")
        train_data = _CoarseMaskDataset(dataset, train_indices, data_config["means"], data_config["stds"])
        loader_generator = torch.Generator().manual_seed(config["protocol"]["seed"])
        loader = DataLoader(
            train_data, batch_size=8, shuffle=True, drop_last=True, num_workers=4,
            prefetch_factor=2, persistent_workers=True, pin_memory=True,
            generator=loader_generator,
        )
        candidate, control = build_matched_models(seed=config["protocol"]["seed"], hidden_channels=8)
        models = {"candidate": candidate.to(device), "control": control.to(device)}
        optimizers = {
            label: torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.0)
            for label, model in models.items()
        }
        ema = {label: {name: value.detach().clone() for name, value in model.state_dict().items()}
               for label, model in models.items()}
        cuda_generator = torch.Generator(device=device).manual_seed(config["protocol"]["seed"] + 1)
        history = {"candidate": [], "control": []}
        iterator = iter(loader)
        for update in range(1, 513):
            try:
                raw = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                raw = next(iterator)
            batch = CfmBatch(*(raw[name].to(device, non_blocking=True) for name in (
                "target", "condition", "d0_occurrence", "valid"
            )))
            uniform = torch.rand(batch.target.shape, device=device, generator=cuda_generator).clamp(1e-5, 1 - 1e-5)
            noise = torch.randn(batch.target.shape, device=device, generator=cuda_generator)
            times = torch.rand((batch.target.shape[0],), device=device, generator=cuda_generator).clamp(1e-4, 1 - 1e-4)
            for label in ("candidate", "control"):
                optimizer, model = optimizers[label], models[label]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss, _ = masked_cfm_loss(
                        model, batch, time=times, noise=noise, uniform=uniform
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"{label} nonfinite loss at update {update}")
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError(f"{label} nonfinite gradient at update {update}")
                optimizer.step()
                _finite_model(model, label)
                _ema_update(ema[label], model, 0.99)
                value = float(loss.detach().item())
                history[label].append(value)
                if update == 1 or update % 8 == 0:
                    tracker.report_scalar("train_cfm_loss", label, value, update)
            if update in {128, 256, 384, 512}:
                checkpoint_path = output_dir / f"checkpoint_{update:04d}.pt"
                evidence[checkpoint_path.name] = _atomic_torch_save({
                    "schema_version": MODE, "code_commit": commit, "update": update,
                    "optimizer_steps_per_arm": update, "models": {k: v.state_dict() for k, v in models.items()},
                    "ema": ema, "optimizers": {k: v.state_dict() for k, v in optimizers.items()},
                    "stats_binding": stats_binding, "preflight": config["preflight"],
                    "train_dates_sha256": _dates_sha(train_dates),
                    "heldout_dates_sha256": _dates_sha(heldout_dates),
                }, checkpoint_path)
                _atomic_json(status_path, {
                    **reservation, "status": "running", "optimizer_steps_per_arm": update,
                    "evidence_sha256": evidence,
                })
        result = {
            "schema_version": MODE, "status": "training_complete", "code_commit": commit,
            "clearml_task_id": str(tracker.task.id), "optimizer_steps_per_arm": 512,
            "batch_size": 8, "presentations_per_arm": 4096, "unique_train_anchors": len(train_indices),
            "train_years": config["protocol"]["train_years"], "heldout_year": 2021,
            "heldout_values_accessed": False, "test_2023_accessed": False,
            "train_dates_sha256": _dates_sha(train_dates), "heldout_dates_sha256": _dates_sha(heldout_dates),
            "conditioning_stats_binding": stats_binding, "preflight_binding": config["preflight"],
            "loss": {label: {"first32_mean": sum(values[:32]) / 32,
                              "last32_mean": sum(values[-32:]) / 32,
                              "final": values[-1]} for label, values in history.items()},
            "checkpoint_sha256": evidence, "evaluation_authorized": False,
            "claim_boundary": "bounded 80x64 occurrence A/B training; no validation, skill or calibration claim",
        }
        metrics_path = output_dir / "training.json"
        _atomic_json(metrics_path, result)
        tracker.connect("training_result", result)
        tracker.upload_artifact("idea_f1_training", metrics_path)
        for name in sorted(evidence):
            tracker.upload_artifact(f"idea_f1_{name}", output_dir / name)
        tracker.close()
        tracker = None
        _atomic_json(status_path, {**reservation, "status": "training_complete",
                                  "optimizer_steps_per_arm": 512, "evidence_sha256": evidence})
        return result
    except BaseException as error:
        _finalize_failure(status_path, reservation, error, tracker, evidence)
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
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
