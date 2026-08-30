from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import TrainingConfig, load_json, merge_config_overrides, resolve_path
from .runtime import add_noise, build_dataloader, seed_everything


def _debug(message: str) -> None:
    print(f"[assim_lib] {message}", flush=True)


def main(config: dict, config_dir: Path):
    _debug(f"run config project={config.get('project_name')} task={config.get('task_name')}")
    data_config_path = resolve_path(config["data_config"], config_dir)
    model_config_path = resolve_path(config["model_config"], config_dir)
    _debug(f"loading data config: {data_config_path}")
    data_config = merge_config_overrides(load_json(data_config_path), config.get("data_overrides"))
    _debug(f"loading model config: {model_config_path}")
    model_config_raw = load_json(model_config_path)

    _debug("importing torch/diffusers and assim_lib modules")
    import torch
    from diffusers.optimization import get_cosine_schedule_with_warmup

    from .data import build_dataset
    from .model_io import build_unet
    from .trainer import UNetTrainer

    _debug("imports complete")
    model_config = {**model_config_raw, **config.get("training", {})}
    clearml_config = config.get("clearml", {})
    model_config["clearml_project_name"] = config["project_name"]
    model_config["clearml_task_name"] = config["task_name"]
    model_config.setdefault("clearml_enabled", bool(clearml_config.get("enabled", True)))
    model_config.setdefault("clearml_tags", clearml_config.get("tags", []))
    model_config.setdefault("clearml_output_uri", clearml_config.get("output_uri"))
    model_config.setdefault("clearml_env_path", clearml_config.get("env_path"))
    model_config.setdefault(
        "clearml_upload_checkpoints", bool(clearml_config.get("upload_checkpoints", False))
    )
    train_config = TrainingConfig.from_dict(model_config)
    _debug(
        "training config "
        f"image_size={train_config.image_size} in_channels={train_config.in_channels} "
        f"out_channels={train_config.out_channels} epochs={train_config.num_epochs} "
        f"objective={train_config.training_objective}"
    )

    _debug(f"setting random seed: {train_config.seed}")
    seed_everything(train_config.seed)

    _debug(f"building train dataset: {data_config.get('dataset_name')}")
    train_dataset = build_dataset(data_config, split="train")
    _debug(f"building valid dataset: {data_config.get('dataset_name')}")
    valid_dataset = build_dataset(data_config, split="valid")
    expected_in_channels = getattr(train_dataset, "conditioned_input_channels", None)
    if expected_in_channels is not None and train_config.in_channels != expected_in_channels:
        raise ValueError(
            f"Model in_channels={train_config.in_channels} does not match dataset-conditioned "
            f"input channels={expected_in_channels}"
        )
    expected_out_channels = len(getattr(train_dataset, "indices", ()))
    if expected_out_channels and train_config.out_channels != expected_out_channels:
        raise ValueError(
            f"Model out_channels={train_config.out_channels} does not match dataset fields="
            f"{expected_out_channels}"
        )
    _debug(f"dataset sizes train={len(train_dataset)} valid={len(valid_dataset)}")
    train_loader = build_dataloader(
        train_dataset,
        train_config.train_batch_size,
        train_config.num_workers_train,
        shuffle=True,
    )
    valid_loader = build_dataloader(
        valid_dataset,
        train_config.eval_batch_size,
        train_config.num_workers_val,
        shuffle=False,
    )
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
        dataset_provenance={
            "train": train_dataset.provenance(),
            "valid": valid_dataset.provenance(),
        },
        dashboard_dataset=valid_dataset,
    )
    _debug("starting training loop")
    output_dir = trainer.train_loop()
    result = {
        "output_dir": str(Path(output_dir).resolve()),
        "metrics_path": str((Path(output_dir) / "metrics.json").resolve()),
    }
    print(json.dumps(result, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a flow-matching model conditioned on background and observations."
    )
    parser.add_argument("--config", required=True, help="Path to experiment JSON config")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_path = Path(args.config)
    _debug(f"entrypoint config={config_path}")
    main(load_json(config_path), config_path.resolve().parent)
