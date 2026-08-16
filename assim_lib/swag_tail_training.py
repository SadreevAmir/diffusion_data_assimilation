"""Frozen single-run training extension for the SWAG-tail fallback."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import torch

from . import main as training_entrypoint
from .deep_ensemble import _atomic_json, _canonical_hash, _file_hash
from .trainer import UNetTrainer

TRAINING_SEED = 1701
EPOCHS = 40
TAIL_START_EPOCH = 30
SNAPSHOT_EPOCHS = tuple(range(TAIL_START_EPOCH, EPOCHS))
WARMUP_STEPS = 500
BASE_LEARNING_RATE = 1e-4
RUN_NAME = "swag_tail"
RESUME_SCHEMA_VERSION = 1

_SCHEDULER_RECORD: dict[str, Any] = {}
_CAPTURE_RECORDS: list[dict[str, Any]] = []


def _resume_contract_hash() -> str:
    return _canonical_hash(
        {
            "training_seed": TRAINING_SEED,
            "epochs": EPOCHS,
            "tail_start_epoch_zero_based": TAIL_START_EPOCH,
            "snapshot_epochs_zero_based": list(SNAPSHOT_EPOCHS),
            "warmup_steps": WARMUP_STEPS,
            "base_learning_rate": BASE_LEARNING_RATE,
            "scheduler": "cosine_frozen_at_epoch_30_boundary",
        }
    )


def frozen_tail_factor(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int = WARMUP_STEPS,
    tail_start_step: int,
) -> float:
    """Cosine warmup/decay frozen exactly at the 75% epoch boundary."""
    if total_steps <= warmup_steps or not warmup_steps < tail_start_step < total_steps:
        raise ValueError("training steps do not admit the frozen SWAG tail")
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    effective_step = min(int(step), int(tail_start_step))
    progress = float(effective_step - warmup_steps) / float(total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_frozen_tail_scheduler(*, optimizer, num_warmup_steps: int, num_training_steps: int):
    if int(num_warmup_steps) != WARMUP_STEPS:
        raise ValueError("warmup differs from the frozen SWAG contract")
    if int(num_training_steps) % EPOCHS:
        raise ValueError("total training steps are not divisible by forty epochs")
    steps_per_epoch = int(num_training_steps) // EPOCHS
    if len(optimizer.param_groups) != 1:
        raise ValueError("trusted SWAG optimizer must have exactly one parameter group")
    group = optimizer.param_groups[0]
    optimizer_contract = {
        "class": type(optimizer).__name__,
        "learning_rate": float(group["lr"]),
        "betas": [float(value) for value in group["betas"]],
        "epsilon": float(group["eps"]),
        "weight_decay": float(group["weight_decay"]),
        "amsgrad": bool(group["amsgrad"]),
    }
    if optimizer_contract != {
        "class": "AdamW",
        "learning_rate": BASE_LEARNING_RATE,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "weight_decay": 0.01,
        "amsgrad": False,
    }:
        raise ValueError("optimizer differs from the frozen SWAG contract")
    tail_start_step = TAIL_START_EPOCH * steps_per_epoch
    tail_factor = frozen_tail_factor(
        tail_start_step,
        total_steps=int(num_training_steps),
        warmup_steps=WARMUP_STEPS,
        tail_start_step=tail_start_step,
    )
    _SCHEDULER_RECORD.clear()
    _SCHEDULER_RECORD.update(
        {
            "total_steps": int(num_training_steps),
            "steps_per_epoch": steps_per_epoch,
            "warmup_steps": WARMUP_STEPS,
            "tail_start_step": tail_start_step,
            "tail_factor": tail_factor,
            "tail_learning_rate": BASE_LEARNING_RATE * tail_factor,
            "optimizer": optimizer_contract,
        }
    )
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: frozen_tail_factor(
            step,
            total_steps=int(num_training_steps),
            warmup_steps=WARMUP_STEPS,
            tail_start_step=tail_start_step,
        ),
    )


class SwagTailTrainer(UNetTrainer):
    """Capture the ten predeclared raw SGD states without altering the loop."""

    def save_model_custom(self, name: str = "last_model.pth"):
        super().save_model_custom(name)
        if name != "last_model.pth":
            return
        epoch = len(self.val_history) - 1
        if epoch not in SNAPSHOT_EPOCHS:
            return
        snapshot_dir = Path(self.output_dir) / "swag_snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        target = snapshot_dir / f"raw_epoch_{epoch:02d}.pth"
        if target.exists() or target.is_symlink():
            raise ValueError("SWAG snapshot target already exists")
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        state = {
            key: value.detach().cpu().clone()
            for key, value in self.accelerator.unwrap_model(self.model).state_dict().items()
        }
        if not state or not all(value.is_floating_point() for value in state.values()):
            raise ValueError("SWAG snapshot state inventory differs")
        if not all(bool(torch.isfinite(value).all()) for value in state.values()):
            raise ValueError("SWAG snapshot contains non-finite weights")
        torch.save(state, temporary)
        temporary.replace(target)
        learning_rates = [float(group["lr"]) for group in self.optimizer.param_groups]
        if len(learning_rates) != 1 or not math.isfinite(learning_rates[0]):
            raise ValueError("SWAG capture requires one finite optimizer learning rate")
        _CAPTURE_RECORDS.append(
            {
                "epoch_zero_based": epoch,
                "path": str(target.resolve()),
                "sha256": _file_hash(target),
                "learning_rate": learning_rates[0],
            }
        )

    @property
    def _resume_root(self) -> Path:
        return Path(self.output_dir) / "swag_resume"

    def _validate_capture_records(self, next_epoch: int) -> None:
        expected_epochs = [epoch for epoch in SNAPSHOT_EPOCHS if epoch < next_epoch]
        if [row.get("epoch_zero_based") for row in _CAPTURE_RECORDS] != expected_epochs:
            raise ValueError("resumed SWAG snapshot accounting differs")
        snapshot_dir = (Path(self.output_dir) / "swag_snapshots").resolve()
        for row in _CAPTURE_RECORDS:
            path = Path(str(row.get("path", "")))
            if (
                path.parent.resolve() != snapshot_dir
                or path.name != f"raw_epoch_{row['epoch_zero_based']:02d}.pth"
                or not path.is_file()
                or path.is_symlink()
                or _file_hash(path) != row.get("sha256")
            ):
                raise ValueError("resumed SWAG snapshot provenance differs")

    def _training_loop_start(self) -> tuple[int, int]:
        if self.accelerator.num_processes != 1:
            raise ValueError("trusted SWAG training requires exactly one process/GPU")
        latest_path = self._resume_root / "latest.json"
        if not latest_path.exists():
            if latest_path.is_symlink():
                raise ValueError("SWAG resume pointer cannot be a symlink")
            if self._resume_root.exists():
                if self._resume_root.is_symlink() or not self._resume_root.is_dir():
                    raise ValueError("SWAG resume root has unsafe type")
                for child in self._resume_root.iterdir():
                    if child.is_symlink() or not child.is_dir():
                        raise ValueError("unexpected uncommitted SWAG resume entry")
                    if child.name.startswith(".epoch_") or (
                        child.name.startswith("epoch_") and len(child.name) == 8 and child.name[6:].isdigit()
                    ):
                        shutil.rmtree(child)
                    else:
                        raise ValueError("unexpected uncommitted SWAG resume directory")
            return 0, 0
        if latest_path.is_symlink() or not latest_path.is_file():
            raise ValueError("SWAG resume pointer must be an ordinary file")
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        if not isinstance(latest, dict) or latest.get("schema_version") != RESUME_SCHEMA_VERSION:
            raise ValueError("SWAG resume pointer schema differs")
        state_name = latest.get("state_dir")
        if (
            not isinstance(state_name, str)
            or len(state_name) != 8
            or not state_name.startswith("epoch_")
            or not state_name[6:].isdigit()
        ):
            raise ValueError("SWAG resume state identity differs")
        state_dir = self._resume_root / state_name
        state_path = state_dir / "resume.json"
        ema_path = state_dir / "ema_state.pth"
        if (
            state_dir.is_symlink()
            or not state_dir.is_dir()
            or state_path.is_symlink()
            or not state_path.is_file()
            or ema_path.is_symlink()
            or not ema_path.is_file()
            or _file_hash(state_path) != latest.get("resume_sha256")
        ):
            raise ValueError("SWAG resume state is incomplete")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        next_epoch = state.get("next_epoch")
        global_step = state.get("global_step")
        expected_step = int(next_epoch) * len(self.train_dataloader) if type(next_epoch) is int else -1
        if (
            state.get("schema_version") != RESUME_SCHEMA_VERSION
            or state.get("contract_sha256") != _resume_contract_hash()
            or type(next_epoch) is not int
            or not 1 <= next_epoch <= EPOCHS
            or type(global_step) is not int
            or global_step != expected_step
            or state.get("steps_per_epoch") != len(self.train_dataloader)
        ):
            raise ValueError("SWAG resume state contract differs")
        history = state.get("validation_history")
        records = state.get("snapshot_records")
        if (
            not isinstance(history, list)
            or len(history) != next_epoch
            or [row.get("epoch") for row in history] != list(range(next_epoch))
            or not isinstance(records, list)
        ):
            raise ValueError("SWAG resume history differs")
        self.accelerator.load_state(str(state_dir / "accelerator"))
        try:
            ema_state = torch.load(ema_path, map_location="cpu", weights_only=True)
        except TypeError:
            ema_state = torch.load(ema_path, map_location="cpu")
        self.ema_model.load_state_dict(ema_state)
        self.val_history = history
        self.best_val_loss = float(state.get("best_val_loss"))
        if not math.isfinite(self.best_val_loss):
            raise ValueError("SWAG resumed best validation loss is non-finite")
        _CAPTURE_RECORDS[:] = records
        self._validate_capture_records(next_epoch)

        snapshot_dir = Path(self.output_dir) / "swag_snapshots"
        for epoch in SNAPSHOT_EPOCHS:
            orphan = snapshot_dir / f"raw_epoch_{epoch:02d}.pth"
            if epoch >= next_epoch and (orphan.exists() or orphan.is_symlink()):
                if orphan.is_symlink() or not orphan.is_file():
                    raise ValueError("uncommitted SWAG snapshot has unsafe type")
                orphan.unlink()
        for child in self._resume_root.iterdir():
            if child.name in {"latest.json", state_name}:
                continue
            if child.is_symlink() or not child.is_dir():
                raise ValueError("unexpected SWAG resume entry")
            if child.name.startswith(".epoch_") or child.name.startswith("epoch_"):
                shutil.rmtree(child)
            else:
                raise ValueError("unexpected SWAG resume directory")
        return next_epoch, global_step

    def _after_training_epoch(self, epoch: int, global_step: int) -> None:
        if not self.accelerator.is_main_process:
            raise ValueError("trusted SWAG resume is single-process only")
        next_epoch = epoch + 1
        if global_step != next_epoch * len(self.train_dataloader):
            raise ValueError("SWAG global-step accounting differs")
        finite_fields = ("val_loss", "val_loss_full", "val_loss_obs")
        if len(self.val_history) != next_epoch or any(
            not math.isfinite(float(self.val_history[-1].get(name, float("nan")))) for name in finite_fields
        ):
            raise ValueError("SWAG validation history contains non-finite values")
        if not math.isfinite(float(self.best_val_loss)):
            raise ValueError("SWAG best validation loss is non-finite")
        self._validate_capture_records(next_epoch)
        if self._resume_root.is_symlink() or (self._resume_root.exists() and not self._resume_root.is_dir()):
            raise ValueError("SWAG resume root has unsafe type")
        self._resume_root.mkdir(parents=True, exist_ok=True)
        final_dir = self._resume_root / f"epoch_{next_epoch:02d}"
        staging = self._resume_root / f".epoch_{next_epoch:02d}.{os.getpid()}.incomplete"
        if final_dir.exists() or final_dir.is_symlink() or staging.exists() or staging.is_symlink():
            raise ValueError("SWAG resume target already exists")
        staging.mkdir()
        try:
            accelerator_dir = staging / "accelerator"
            self.accelerator.save_state(str(accelerator_dir))
            torch.save(self.ema_model.state_dict(), staging / "ema_state.pth")
            state = {
                "schema_version": RESUME_SCHEMA_VERSION,
                "contract_sha256": _resume_contract_hash(),
                "next_epoch": next_epoch,
                "global_step": global_step,
                "steps_per_epoch": len(self.train_dataloader),
                "best_val_loss": self.best_val_loss,
                "validation_history": self.val_history,
                "snapshot_records": list(_CAPTURE_RECORDS),
            }
            _atomic_json(staging / "resume.json", state)
            staging.replace(final_dir)
            latest_path = self._resume_root / "latest.json"
            previous_name = None
            if latest_path.is_file() and not latest_path.is_symlink():
                previous_name = json.loads(latest_path.read_text(encoding="utf-8")).get("state_dir")
            _atomic_json(
                latest_path,
                {
                    "schema_version": RESUME_SCHEMA_VERSION,
                    "state_dir": final_dir.name,
                    "resume_sha256": _file_hash(final_dir / "resume.json"),
                },
            )
            if isinstance(previous_name, str) and previous_name != final_dir.name:
                if (
                    len(previous_name) != 8
                    or not previous_name.startswith("epoch_")
                    or not previous_name[6:].isdigit()
                ):
                    raise ValueError("previous SWAG resume state identity differs")
                previous = self._resume_root / previous_name
                if previous.is_symlink() or not previous.is_dir():
                    raise ValueError("previous SWAG resume state has unsafe type")
                shutil.rmtree(previous)
        except Exception:
            if staging.is_dir() and not staging.is_symlink():
                shutil.rmtree(staging)
            raise


def _validate_config(config: dict[str, Any]) -> None:
    expected_swag = {
        "schema_version": 1,
        "training_seed": TRAINING_SEED,
        "epochs": EPOCHS,
        "initialization": "from_scratch_with_current_UNet_and_training_split",
        "optimizer": {
            "class": "AdamW",
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "weight_decay": 0.01,
            "amsgrad": False,
            "state_policy": "retained continuously for all 40 epochs",
        },
        "learning_rate": BASE_LEARNING_RATE,
        "warmup_steps": WARMUP_STEPS,
        "pre_tail_schedule": "current cosine schedule through the epoch-30 boundary",
        "tail_schedule": "learning rate frozen at the epoch-30 boundary for epochs 30..39",
        "tail_start_epoch_zero_based": TAIL_START_EPOCH,
        "snapshot_epochs_zero_based": list(SNAPSHOT_EPOCHS),
        "snapshot_state": "raw non-EMA end-of-epoch weights",
        "weight_sample_seeds": [170_100 + index for index in range(10)],
        "covariance": "standard SWAG diagonal plus rank-9; canonical one-half split",
        "deviation_definition": "each snapshot minus the final ten-snapshot mean",
        "matrix_chunk_parameters": 262_144,
        "resume": "epoch boundary only; exact model/optimizer/scheduler/RNG/EMA restore or fail",
        "validation_selection": False,
        "member_count": 10,
        "latent_seed_formula": "1234 + 10*case_index + member_index",
        "sampling": {
            "dataset_split": "valid",
            "start_date": "2022-01-01",
            "end_date": "2022-07-15",
            "case_stride_days": 5,
            "case_count": 40,
            "method": "dopri5",
            "num_timesteps": 25,
            "rtol": 1e-5,
            "atol": 1e-6,
            "precision": "float32",
            "conditioning_mode": "full",
            "cfg_mode": "none",
        },
        "dependent_gate": {
            "mode": "validation_swag_tail_weight_posterior_gate",
            "implementation": "existing deep_ensemble_gate full no-compensation gate",
            "families": ["proper", "reliability", "boundary", "spatial", "operational"],
            "paired_bootstrap_draws": 20_000,
            "block_sensitivity_is_diagnostic_only": True,
            "no_compensation": True,
        },
    }
    if config.get("swag") != expected_swag:
        raise ValueError("SWAG training contract differs")
    expected_training = {
        "base_output_dir": "training",
        "run_name": RUN_NAME,
        "seed": TRAINING_SEED,
        "num_workers_train": 0,
        "num_workers_val": 0,
        "sample_every_n_epochs": 0,
        "metric_every_n_epochs": 0,
        "metric_num_cases": 0,
    }
    if config.get("training") != expected_training:
        raise ValueError("effective SWAG training overrides differ")
    if config.get("clearml") != {"enabled": False, "upload_checkpoints": False}:
        raise ValueError("SWAG training must not use a network tracker")


def _load_completed_training(output_dir: Path) -> bool:
    manifest_path = output_dir / "swag_training_manifest.json"
    if not manifest_path.exists():
        if manifest_path.is_symlink():
            raise ValueError("SWAG training manifest cannot be a symlink")
        return False
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("SWAG training manifest must be an ordinary file")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics_path = output_dir / "metrics.json"
    if metrics_path.is_symlink() or not metrics_path.is_file():
        raise ValueError("completed SWAG metrics are missing")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    records = manifest.get("snapshots")
    scheduler = manifest.get("scheduler")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("training_seed") != TRAINING_SEED
        or manifest.get("training_completed_normally") is not True
        or manifest.get("epochs_completed") != EPOCHS
        or manifest.get("resume_contract_sha256") != _resume_contract_hash()
        or not isinstance(metrics, list)
        or [row.get("epoch") for row in metrics] != list(range(EPOCHS))
        or any(
            not math.isfinite(float(row.get(name, float("nan"))))
            for row in metrics
            for name in ("val_loss", "val_loss_full", "val_loss_obs")
        )
        or not isinstance(records, list)
        or [row.get("epoch_zero_based") for row in records] != list(SNAPSHOT_EPOCHS)
        or not isinstance(scheduler, dict)
    ):
        raise ValueError("completed SWAG training contract differs")
    snapshot_dir = (output_dir / "swag_snapshots").resolve()
    for epoch, row in zip(SNAPSHOT_EPOCHS, records, strict=True):
        path = Path(str(row.get("path", "")))
        if (
            path.parent.resolve() != snapshot_dir
            or path.name != f"raw_epoch_{epoch:02d}.pth"
            or path.is_symlink()
            or not path.is_file()
            or _file_hash(path) != row.get("sha256")
        ):
            raise ValueError("completed SWAG snapshot provenance differs")
    _CAPTURE_RECORDS[:] = records
    _SCHEDULER_RECORD.clear()
    _SCHEDULER_RECORD.update(scheduler)
    return True


def _source_anchors(config_path: Path, config: dict[str, Any]) -> dict[str, str]:
    module_root = Path(__file__).resolve().parent
    paths = {
        "contract_config": config_path.resolve(),
        "data_config": (config_path.resolve().parent / str(config["data_config"])).resolve(),
        "model_config": (config_path.resolve().parent / str(config["model_config"])).resolve(),
        "training_module": Path(__file__).resolve(),
        "training_entrypoint": module_root / "main.py",
        "base_trainer": module_root / "trainer.py",
    }
    anchors: dict[str, str] = {}
    for name, path in paths.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"SWAG source anchor {name} is unavailable")
        anchors[f"{name}_sha256"] = _file_hash(path)
    return anchors


def run(config_path: Path) -> Path:
    if config_path.is_symlink() or not config_path.is_file():
        raise ValueError("config must be an existing ordinary file")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config must contain one JSON object")
    _validate_config(config)
    expected_output_dir = (Path.cwd() / "training" / RUN_NAME).resolve()
    _CAPTURE_RECORDS.clear()
    _SCHEDULER_RECORD.clear()
    if _load_completed_training(expected_output_dir):
        return expected_output_dir
    result = training_entrypoint.main(
        config,
        config_path.resolve().parent,
        trainer_class=SwagTailTrainer,
        scheduler_factory=build_frozen_tail_scheduler,
    )
    output_dir = Path(result["output_dir"]).resolve()
    if output_dir != expected_output_dir:
        raise ValueError("SWAG training output path differs from the frozen contract")
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    if not isinstance(metrics, list) or len(metrics) != EPOCHS:
        raise ValueError("SWAG training did not complete exactly forty epochs")
    if [row["epoch"] for row in metrics] != list(range(EPOCHS)):
        raise ValueError("SWAG training epoch accounting differs")
    if [row["epoch_zero_based"] for row in _CAPTURE_RECORDS] != list(SNAPSHOT_EPOCHS):
        raise ValueError("SWAG snapshot accounting differs")
    tail_lr = float(_SCHEDULER_RECORD["tail_learning_rate"])
    if any(
        not math.isclose(float(row["learning_rate"]), tail_lr, rel_tol=1e-6, abs_tol=1e-12)
        for row in _CAPTURE_RECORDS
    ):
        raise ValueError("SWAG snapshot learning rate differs from the frozen tail")
    manifest = {
        "schema_version": 1,
        "training_seed": TRAINING_SEED,
        "training_completed_normally": True,
        "epochs_completed": EPOCHS,
        "validation_selected": False,
        "test_data_used": False,
        "optimizer": dict(_SCHEDULER_RECORD["optimizer"]),
        "optimizer_state_retained_through_tail": True,
        "weight_state": "raw non-EMA end-of-epoch model",
        "resume_contract_sha256": _resume_contract_hash(),
        "epoch_boundary_resume": "Accelerate model/optimizer/scheduler/RNG plus EMA; fail-closed",
        "source_anchors": _source_anchors(config_path, config),
        "library_identity": {
            "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "torch": torch.__version__,
        },
        "scheduler": dict(_SCHEDULER_RECORD),
        "snapshots": list(_CAPTURE_RECORDS),
    }
    _atomic_json(output_dir / "swag_training_manifest.json", manifest)
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    output_dir = run(args.config)
    print(json.dumps({"output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
