from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


CHECKPOINT_FORMAT_VERSION = 1


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def module_name(obj: Any) -> str:
    cls = obj.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def model_config(model: Any) -> Any:
    config = getattr(model, "config", None)
    if config is None:
        return None
    if hasattr(config, "to_dict"):
        return to_jsonable(config.to_dict())
    if isinstance(config, dict):
        return to_jsonable(config)
    return to_jsonable(dict(config)) if hasattr(config, "items") else to_jsonable(config)


def build_training_checkpoint(
    *,
    model: torch.nn.Module,
    config: Any,
    run_name: str,
    checkpoint_name: str,
    model_state_dict: dict[str, Any] | None = None,
    ema_state_dict: dict[str, Any] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    lr_scheduler: Any | None = None,
    epoch: int | None = None,
    global_step: int | None = None,
    best_val_loss: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "saved_at": datetime.now().isoformat(),
        "run_name": run_name,
        "checkpoint_name": checkpoint_name,
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "config": to_jsonable(config),
        "model_class": module_name(model),
        "model_config": model_config(model),
        "model_state_dict": model_state_dict if model_state_dict is not None else model.state_dict(),
    }
    if ema_state_dict is not None:
        checkpoint["ema_state_dict"] = ema_state_dict
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if lr_scheduler is not None and hasattr(lr_scheduler, "state_dict"):
        checkpoint["lr_scheduler_state_dict"] = lr_scheduler.state_dict()
    if extra:
        checkpoint["extra"] = to_jsonable(extra)
    return checkpoint


def save_training_checkpoint(path: str, **kwargs: Any) -> None:
    torch.save(build_training_checkpoint(checkpoint_name=Path(path).name, **kwargs), path)


def safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(checkpoint: Any, keys: Iterable[str] = ("model_state_dict",)) -> Any:
    if isinstance(checkpoint, dict):
        for key in keys:
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def checkpoint_metadata(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, dict) or "format_version" not in checkpoint:
        return {"format_version": 0}
    skip = {"model_state_dict", "ema_state_dict", "optimizer_state_dict", "lr_scheduler_state_dict"}
    return to_jsonable({k: v for k, v in checkpoint.items() if k not in skip})
