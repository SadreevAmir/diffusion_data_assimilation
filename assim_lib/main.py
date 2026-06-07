from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: str | Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def resolve_path(path: str | Path, base_dir: Path) -> Path:
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    return base_dir / path


def merge_config_overrides(config: dict, overrides: dict | None) -> dict:
    merged = dict(config)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def _debug(message: str) -> None:
    print(f"[assim_lib] {message}", flush=True)


def _loader(dataset, batch_size: int, num_workers: int, shuffle: bool):
    import torch
    from torch.utils.data import DataLoader

    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=shuffle,
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **kwargs)


def main(config: dict, config_dir: Path):
    _debug(f"run config project={config.get('project_name')} task={config.get('task_name')}")
    data_config_path = resolve_path(config["data_config"], config_dir)
    model_config_path = resolve_path(config["model_config"], config_dir)
    _debug(f"loading data config: {data_config_path}")
    data_config = merge_config_overrides(load_json(data_config_path), config.get("data_overrides"))
    _debug(f"loading model config: {model_config_path}")
    model_config_raw = load_json(model_config_path)

    _debug("importing torch/diffusers and assim_lib modules")
    import numpy as np
    import torch
    from diffusers.optimization import get_cosine_schedule_with_warmup

    from utils import add_noise

    from .data import build_dataset
    from .model_io import build_unet
    from .trainer import TrainingConfig, UNetTrainer

    _debug("imports complete")
    model_config = {**model_config_raw, **config.get("training", {})}
    clearml_config = config.get("clearml", {})
    model_config["clearml_project_name"] = config["project_name"]
    model_config["clearml_task_name"] = config["task_name"]
    model_config.setdefault("clearml_tags", clearml_config.get("tags", []))
    model_config.setdefault("clearml_output_uri", clearml_config.get("output_uri"))
    model_config.setdefault("clearml_env_path", clearml_config.get("env_path"))
    model_config.setdefault("clearml_upload_checkpoints", bool(clearml_config.get("upload_checkpoints", False)))
    train_config = TrainingConfig.from_dict(model_config)
    _debug(
        "training config "
        f"image_size={train_config.image_size} in_channels={train_config.in_channels} "
        f"out_channels={train_config.out_channels} epochs={train_config.num_epochs} "
        f"objective={train_config.training_objective}"
    )

    _debug(f"setting random seed: {train_config.seed}")
    torch.manual_seed(train_config.seed)
    np.random.seed(train_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_config.seed)

    _debug(f"building train dataset: {data_config.get('dataset_name')}")
    train_dataset = build_dataset(data_config, split="train")
    _debug(f"building valid dataset: {data_config.get('dataset_name')}")
    valid_dataset = build_dataset(data_config, split="valid")
    _debug(f"dataset sizes train={len(train_dataset)} valid={len(valid_dataset)}")
    train_loader = _loader(train_dataset, train_config.train_batch_size, train_config.num_workers_train, shuffle=True)
    valid_loader = _loader(valid_dataset, train_config.eval_batch_size, train_config.num_workers_val, shuffle=False)
    _debug(f"dataloader batches train={len(train_loader)} valid={len(valid_loader)}")

    _debug("building UNet")
    model = build_unet(train_config)
    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    _debug(f"model parameters={num_parameters:,} ({num_parameters / 1_000_000:.2f}M)")
    _debug("building optimizer and scheduler")
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=train_config.lr_warmup_steps,
        num_training_steps=max(1, len(train_loader) * train_config.num_epochs),
    )
    _debug("initializing trainer and ClearML task")
    trainer = UNetTrainer(
        config=train_config,
        model=model,
        optimizer=optimizer,
        data_loader_train=train_loader,
        data_loader_val=valid_loader,
        lr_scheduler=lr_scheduler,
        add_noise_func=add_noise,
        experiment_config=config,
        model_config=model_config_raw,
        data_config=data_config,
        dashboard_dataset=valid_dataset,
    )
    _debug("starting training loop")
    output_dir = trainer.train_loop()
    result = {"output_dir": str(Path(output_dir).resolve()), "metrics_path": str((Path(output_dir) / "metrics.json").resolve())}
    print(json.dumps(result, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Train concat-conditioned background+observation diffusion model.")
    parser.add_argument("--config", required=True, help="Path to experiment JSON config")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_path = Path(args.config)
    _debug(f"entrypoint config={config_path}")
    main(load_json(config_path), config_path.resolve().parent)
