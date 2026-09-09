"""Bounded 512-update overfit gate for direct SIC/SIT dynamics.

This is a mechanics/representation test, not a validation experiment.  It uses
four immutable training-day anchors, all 24 archive slices for each day, and
the unchanged full-resolution direct-dynamics backbone.  No result from this
module may be reported as generalization evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Subset, default_collate

from .bounded_clean_state_flow import (
    BoundedCleanStateSampler,
    bounded_clean_state_loss,
)
from .clearml_tracking import ClearMLTracker
from .config import TrainingConfig, load_json, merge_config_overrides, resolve_path
from .data import build_dataset
from .direct_dynamics_training import (
    DIRECT_CONDITION_CHANNELS,
    DIRECT_INPUT_CHANNELS,
    DIRECT_LEADS,
    DIRECT_OUTPUT_CHANNELS,
    _repeat_field_stats,
    validate_direct_dataset,
)
from .model_io import build_unet
from .runtime import build_dataloader, make_normalized_xy_grid, seed_everything
from .structured_trajectory_evaluation import make_structured_trajectory_figure
from .trainer import _atomic_json
from .transforms import channel_denormalize


ANCHOR_DAY_OFFSETS = (14, 105, 196, 287)
DIAGNOSTIC_UPDATES = (64, 256, 512)
MAX_UPDATES = DIAGNOSTIC_UPDATES[-1]
TRAIN_BATCH_SIZE = 16
TRAIN_WORKERS = 4
ENSEMBLE_MEMBERS = 2
SAMPLER_STEPS = 17
ENDPOINT_EPSILON = 0.01
SIT_SCALE_METERS = 1.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(payload: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"refusing stale checkpoint temporary file: {temporary}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def anchor_indices(dataset_length: int) -> tuple[int, ...]:
    indices = tuple(
        day * 24 + hour for day in ANCHOR_DAY_OFFSETS for hour in range(24)
    )
    if len(indices) != 96 or len(set(indices)) != 96:
        raise RuntimeError("bounded overfit anchors are not exactly four days x 24 slices")
    if min(indices) < 0 or max(indices) >= int(dataset_length):
        raise ValueError("bounded overfit anchor schedule lies outside the training dataset")
    return indices


def diagnostic_indices(dataset_length: int) -> tuple[int, ...]:
    indices = tuple(day * 24 + 23 for day in ANCHOR_DAY_OFFSETS)
    if max(indices) >= int(dataset_length):
        raise ValueError("bounded diagnostic schedule lies outside the training dataset")
    return indices


def _device_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _masked_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask = valid.expand_as(value)
    return (value * mask).sum() / mask.sum().clamp(min=1)


def _masked_rmse(
    prediction: torch.Tensor, truth: torch.Tensor, valid: torch.Tensor
) -> float:
    return float(torch.sqrt(_masked_mean((prediction - truth).square(), valid)).item())


def _physical_spread(
    ensemble: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    if ensemble.shape[1] < 2:
        raise ValueError("spread requires at least two ensemble members")
    return ensemble.std(dim=1, unbiased=True) * valid.expand_as(ensemble[:, 0])


@torch.no_grad()
def _diagnostic(
    *,
    model,
    batch: dict[str, Any],
    update: int,
    output_dir: Path,
    tracker: ClearMLTracker,
    means: tuple[float, ...],
    stds: tuple[float, ...],
    checkpoint: dict[str, Any],
    sampler_steps: int = SAMPLER_STEPS,
) -> dict[str, Any]:
    valid = batch["valid_mask"][:, :1]
    case_count = batch["truth"].shape[0]
    condition = batch["structured_conditioning"]
    if condition.shape[1] != DIRECT_CONDITION_CHANNELS:
        raise ValueError("bounded overfit diagnostic received the wrong conditioning layout")
    repeated_condition = condition.repeat_interleave(ENSEMBLE_MEMBERS, dim=0)
    repeated_valid = valid.repeat_interleave(ENSEMBLE_MEMBERS, dim=0)
    generator = torch.Generator(device=batch["truth"].device)
    generator.manual_seed(89173)
    initial_noise = torch.randn(
        (
            case_count * ENSEMBLE_MEMBERS,
            DIRECT_OUTPUT_CHANNELS,
            *batch["truth"].shape[-2:],
        ),
        generator=generator,
        dtype=torch.float32,
        device=batch["truth"].device,
    )
    sampler = BoundedCleanStateSampler(
        model,
        means=means,
        stds=stds,
        sit_scale=SIT_SCALE_METERS,
        epsilon=ENDPOINT_EPSILON,
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        sampled = sampler.sample(
            initial_noise,
            repeated_condition,
            repeated_valid,
            num_steps=int(sampler_steps),
        )
    ensemble = sampled.physical.unflatten(
        0, (case_count, ENSEMBLE_MEMBERS)
    ).float()
    terminal_physical = channel_denormalize(
        sampled.terminal_ode_state.float(), means, stds
    ).unflatten(0, (case_count, ENSEMBLE_MEMBERS))
    truth = batch["structured_physical_truth"].float()
    persistence = batch["structured_physical_background"].float()
    endpoint_spread = _physical_spread(ensemble, valid)
    terminal_spread = _physical_spread(terminal_physical, valid)
    ensemble_mean = ensemble.mean(dim=1)
    metrics: dict[str, Any] = {
        "update": int(update),
        "case_count": int(case_count),
        "ensemble_members": ENSEMBLE_MEMBERS,
        "sampler_steps": int(sampler_steps),
        "epsilon": ENDPOINT_EPSILON,
        "checkpoint": checkpoint,
        "support": {
            "sic_below_zero": int(((ensemble[:, :, 0::2] < 0) & (valid[:, None] > 0)).sum()),
            "sic_above_one": int(((ensemble[:, :, 0::2] > 1) & (valid[:, None] > 0)).sum()),
            "sit_below_zero": int(((ensemble[:, :, 1::2] < 0) & (valid[:, None] > 0)).sum()),
        },
        "leads": {},
    }
    for lead_index, lead_day in enumerate(DIRECT_LEADS):
        lead_metrics = {}
        for offset, field_name in enumerate(("sic", "sit")):
            channel = 2 * lead_index + offset
            end_spread = _masked_mean(
                endpoint_spread[:, channel : channel + 1], valid
            )
            ode_spread = _masked_mean(
                terminal_spread[:, channel : channel + 1], valid
            )
            lead_metrics[field_name] = {
                "ensemble_mean_rmse": _masked_rmse(
                    ensemble_mean[:, channel : channel + 1],
                    truth[:, channel : channel + 1],
                    valid,
                ),
                "member0_rmse": _masked_rmse(
                    ensemble[:, 0, channel : channel + 1],
                    truth[:, channel : channel + 1],
                    valid,
                ),
                "endpoint_spread": float(end_spread.item()),
                "terminal_ode_spread": float(ode_spread.item()),
                "endpoint_to_terminal_spread_ratio": float(
                    (end_spread / ode_spread.clamp(min=1e-8)).item()
                ),
            }
        metrics["leads"][f"d{lead_day}"] = lead_metrics

    stage_dir = output_dir / "diagnostics" / f"update_{update:04d}"
    stage_dir.mkdir(parents=True, exist_ok=False)
    torch.save(
        {
            "ensemble": ensemble.cpu(),
            "terminal_ode_physical": terminal_physical.cpu(),
            "truth": truth.cpu(),
            "persistence": persistence.cpu(),
            "valid_mask": valid.cpu(),
            "initial_noise": initial_noise.cpu(),
        },
        stage_dir / "samples.pt",
    )
    _atomic_json(stage_dir / "metrics.json", metrics)
    for case_index in range(case_count):
        figure = make_structured_trajectory_figure(
            truth[case_index],
            persistence[case_index],
            ensemble[case_index, 0],
            valid[case_index],
            title=(
                f"bounded clean-state overfit update {update}; "
                f"anchor {case_index}, member 0"
            ),
            origin="upper",
            lead_days=DIRECT_LEADS,
        )
        image_path = stage_dir / f"anchor_{case_index:02d}_member0.png"
        figure.savefig(image_path, dpi=180, bbox_inches="tight")
        import matplotlib.pyplot as plt

        plt.close(figure)
        tracker.report_image(
            "bounded_overfit/samples",
            f"anchor_{case_index:02d}_member0",
            image_path,
            update,
        )
    for lead_name, lead in metrics["leads"].items():
        for field_name, field in lead.items():
            for metric_name, value in field.items():
                tracker.report_scalar(
                    f"bounded_overfit/{metric_name}",
                    f"{lead_name}_{field_name}",
                    value,
                    update,
                )
    return metrics


def _final_gate(losses: list[float], diagnostics: dict[int, dict[str, Any]]) -> dict[str, Any]:
    if len(losses) != MAX_UPDATES or set(diagnostics) != set(DIAGNOSTIC_UPDATES):
        raise ValueError("bounded overfit result is incomplete")
    if not all(math.isfinite(value) for value in losses):
        raise FloatingPointError("bounded overfit loss history contains NaN/Inf")
    for update, diagnostic in diagnostics.items():
        for lead_name in ("d3", "d6", "d9"):
            for field_name in ("sic", "sit"):
                field = diagnostic["leads"][lead_name][field_name]
                for metric_name in (
                    "ensemble_mean_rmse",
                    "member0_rmse",
                    "endpoint_spread",
                    "terminal_ode_spread",
                    "endpoint_to_terminal_spread_ratio",
                ):
                    if not math.isfinite(float(field[metric_name])):
                        raise FloatingPointError(
                            "bounded overfit diagnostic contains NaN/Inf: "
                            f"update={update} {lead_name}/{field_name}/{metric_name}"
                        )
    first = sum(losses[:32]) / 32
    last = sum(losses[-32:]) / 32
    initial = diagnostics[DIAGNOSTIC_UPDATES[0]]
    final = diagnostics[DIAGNOSTIC_UPDATES[-1]]
    rmse_ratios = []
    spread_ratios = []
    endpoint_spreads = []
    for lead_name in ("d3", "d6", "d9"):
        for field_name in ("sic", "sit"):
            initial_field = initial["leads"][lead_name][field_name]
            field = final["leads"][lead_name][field_name]
            values = (
                initial_field["ensemble_mean_rmse"],
                field["ensemble_mean_rmse"],
                field["endpoint_to_terminal_spread_ratio"],
                field["endpoint_spread"],
                field["terminal_ode_spread"],
            )
            if not all(math.isfinite(float(value)) for value in values):
                raise FloatingPointError(
                    f"bounded overfit diagnostic contains NaN/Inf for {lead_name}/{field_name}"
                )
            if float(initial_field["ensemble_mean_rmse"]) <= 0.0:
                raise ValueError("initial diagnostic RMSE must be positive")
            rmse_ratios.append(
                float(field["ensemble_mean_rmse"])
                / float(initial_field["ensemble_mean_rmse"])
            )
            spread_ratios.append(float(field["endpoint_to_terminal_spread_ratio"]))
            endpoint_spreads.append(float(field["endpoint_spread"]))
    reasons = []
    if first <= 0.0 or last >= 0.65 * first:
        reasons.append("last-32 clean-state loss did not fall by at least 35%")
    if sum(rmse_ratios) / len(rmse_ratios) >= 0.85 or max(rmse_ratios) > 1.05:
        reasons.append(
            "within-field RMSE ratios did not improve by 15% on average without regression"
        )
    if any(value != 0 for value in final["support"].values()):
        reasons.append("bounded physical endpoint violated SIC/SIT support")
    return {
        "status": (
            "passed_numeric_pending_solver_and_visual_review"
            if not reasons
            else "failed"
        ),
        "reasons": reasons,
        "first_32_loss_mean": first,
        "last_32_loss_mean": last,
        "loss_ratio": last / first,
        "mean_within_field_rmse_ratio_update512_to64": sum(rmse_ratios) / len(rmse_ratios),
        "maximum_within_field_rmse_ratio_update512_to64": max(rmse_ratios),
        "minimum_endpoint_to_terminal_spread_ratio": min(spread_ratios),
        "minimum_endpoint_spread": min(endpoint_spreads),
        "spread_interpretation": (
            "diagnostic_only: one empirical target per condition has a delta-law overfit optimum"
        ),
        "visual_review_required": True,
        "paired_fp32_solver_review_required": True,
        "permits_long_training": False,
    }


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    if torch.cuda.device_count() != 1:
        raise RuntimeError("bounded overfit gate requires exactly one visible GPU")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to reuse bounded overfit output: {output_dir}")
    experiment = load_json(config_path)
    config_dir = config_path.resolve().parent
    data_path = resolve_path(experiment["data_config"], config_dir)
    model_path = resolve_path(experiment["model_config"], config_dir)
    data_config = merge_config_overrides(
        load_json(data_path), experiment.get("data_overrides")
    )
    model_raw = load_json(model_path)
    model_config = TrainingConfig.from_dict(model_raw)
    if (
        model_config.image_size != (320, 256)
        or model_config.in_channels != DIRECT_INPUT_CHANNELS
        or model_config.out_channels != DIRECT_OUTPUT_CHANNELS
        or model_config.activation_checkpointing
    ):
        raise ValueError("bounded overfit must retain the direct full-resolution backbone")
    if TRAIN_BATCH_SIZE != model_config.train_batch_size or TRAIN_WORKERS != model_config.num_workers_train:
        raise ValueError("bounded overfit batch/worker contract drifted from direct dynamics")

    seed_everything(model_config.seed)
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    dataset = build_dataset(data_config, split="train")
    dataset_sentinel = validate_direct_dataset(dataset)
    train_indices = anchor_indices(len(dataset))
    sample_indices = diagnostic_indices(len(dataset))
    train_subset = Subset(dataset, train_indices)
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(model_config.seed + 41)
    loader = build_dataloader(
        train_subset,
        TRAIN_BATCH_SIZE,
        TRAIN_WORKERS,
        shuffle=True,
        generator=loader_generator,
    )
    if len(loader) != 6:
        raise ValueError("bounded overfit must have exactly six batches per anchor epoch")
    diagnostic_batch = default_collate([dataset[index] for index in sample_indices])
    anchor_evidence = [
        {
            "dataset_index": int(index),
            "case_id": dataset[index]["meta"]["case_id"],
            "archive_slice_index": int(dataset[index]["meta"]["archive_slice_index"]),
        }
        for index in sample_indices
    ]

    output_dir.mkdir(parents=True, exist_ok=False)
    tracker = ClearMLTracker(
        "sea_ice_two_stage",
        f"bounded_clean_state_overfit_{output_dir.name}",
        tags=[
            "bounded-clean-state",
            "overfit-mechanics-only",
            "train-anchors-only",
            "full-resolution",
            "one-gpu",
            "no-activation-checkpointing",
            "not-generalization-evidence",
        ],
        env_path=experiment.get("clearml", {}).get("env_path"),
    )
    contract = {
        "purpose": "mechanics_and_representation_gate_only",
        "generalization_claim_permitted": False,
        "dataset_split": "train",
        "anchor_day_offsets": ANCHOR_DAY_OFFSETS,
        "anchor_indices": train_indices,
        "diagnostic_indices": sample_indices,
        "anchor_evidence": anchor_evidence,
        "updates": MAX_UPDATES,
        "diagnostic_updates": DIAGNOSTIC_UPDATES,
        "batch_size": TRAIN_BATCH_SIZE,
        "workers": TRAIN_WORKERS,
        "ensemble_members": ENSEMBLE_MEMBERS,
        "sampler_steps": SAMPLER_STEPS,
        "epsilon": ENDPOINT_EPSILON,
        "sit_scale_meters": SIT_SCALE_METERS,
        "dataset_sentinel": dataset_sentinel,
        "effective_data_config": data_config,
        "source_configs": {
            "experiment": {
                "path": str(config_path),
                "sha256": _sha256_file(config_path),
            },
            "data": {"path": str(data_path), "sha256": _sha256_file(data_path)},
            "model": {"path": str(model_path), "sha256": _sha256_file(model_path)},
        },
        "source_model_config": model_raw,
        "effective_method": {
            "objective": "bounded_clean_state_normalized_mse",
            "path": "z_t=(1-t)*normalized_clean+t*standard_normal",
            "time_sampler": "batch_stratified_uniform",
            "time_range": [ENDPOINT_EPSILON, 1.0],
            "sic_decoder": "sigmoid",
            "sit_decoder": "softplus_times_1_meter",
            "optimizer": "AdamW",
            "learning_rate": model_config.learning_rate,
            "lr_schedule": "32_update_linear_warmup_then_cosine_to_zero",
            "training_weight_source": "raw",
            "ema": False,
            "training_precision": "bf16_network_fp32_decoded_loss",
            "sampling_precision": "bf16_network_fp32_rk4_state",
            "sampler": f"RK4-{SAMPLER_STEPS}_epsilon-{ENDPOINT_EPSILON}",
            "activation_checkpointing": False,
        },
    }
    tracker.connect("bounded_overfit_contract", contract)
    _atomic_json(output_dir / "contract.json", contract)

    device = torch.device("cuda:0")
    model = build_unet(model_config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=model_config.learning_rate)

    def lr_factor(step: int) -> float:
        warmup = 32
        if step < warmup:
            return float(step + 1) / warmup
        progress = (step - warmup) / max(MAX_UPDATES - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    means = _repeat_field_stats(dataset.means)
    stds = _repeat_field_stats(dataset.stds)
    grid = make_normalized_xy_grid(
        *model_config.image_size, device=device, dtype=torch.float32
    ).expand(TRAIN_BATCH_SIZE, -1, -1, -1)
    losses: list[float] = []
    diagnostics: dict[int, dict[str, Any]] = {}
    checkpoints: dict[int, dict[str, Any]] = {}
    iterator = iter(loader)
    try:
        for update in range(1, MAX_UPDATES + 1):
            try:
                raw_batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                raw_batch = next(iterator)
            batch = _device_batch(raw_batch, device)
            truth = batch["truth"].float()
            valid = batch["valid_mask"][:, :1]
            active = valid.expand_as(truth) > 0
            clean = torch.where(active, truth, torch.zeros_like(truth))
            permutation = torch.randperm(TRAIN_BATCH_SIZE, device=device)
            jitter = torch.rand(TRAIN_BATCH_SIZE, device=device)
            time = ENDPOINT_EPSILON + (1.0 - ENDPOINT_EPSILON) * (
                (permutation.float() + jitter) / TRAIN_BATCH_SIZE
            )
            noise = torch.where(active, torch.randn_like(clean), torch.zeros_like(clean))
            state = (1.0 - time[:, None, None, None]) * clean + time[:, None, None, None] * noise
            model_input = torch.cat(
                (state, grid, batch["structured_conditioning"].float()), dim=1
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(model_input, time * 1000.0, return_dict=False)[0]
                loss = bounded_clean_state_loss(
                    logits.float(),
                    clean,
                    valid,
                    means=means,
                    stds=stds,
                    sit_scale=SIT_SCALE_METERS,
                )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("bounded overfit gradient norm is NaN/Inf")
            optimizer.step()
            scheduler.step()
            loss_value = float(loss.detach().item())
            losses.append(loss_value)
            if update == 1 or update % 8 == 0:
                tracker.report_scalar("bounded_overfit/train", "clean_state_loss", loss_value, update)
                tracker.report_scalar(
                    "bounded_overfit/train",
                    "learning_rate",
                    optimizer.param_groups[0]["lr"],
                    update,
                )
            if update in DIAGNOSTIC_UPDATES:
                # Preserve the exact learned state before plotting/sampling so a
                # diagnostic failure cannot erase the completed optimization.
                checkpoint_path = output_dir / f"model_step_{update:04d}.pth"
                _atomic_torch_save(model.state_dict(), checkpoint_path)
                checkpoint = {
                    "path": str(checkpoint_path),
                    "sha256": _sha256_file(checkpoint_path),
                    "update": update,
                    "weight_source": "raw",
                }
                checkpoints[update] = checkpoint
                checkpoint_manifest = output_dir / f"model_step_{update:04d}.json"
                _atomic_json(checkpoint_manifest, checkpoint)
                tracker.connect(f"bounded_overfit_snapshot_{update:04d}", checkpoint)
                tracker.upload_artifact(
                    f"bounded_overfit_snapshot_{update:04d}_manifest",
                    checkpoint_manifest,
                )
                diagnostics[update] = _diagnostic(
                    model=model,
                    batch=_device_batch(diagnostic_batch, device),
                    update=update,
                    output_dir=output_dir,
                    tracker=tracker,
                    means=means,
                    stds=stds,
                    checkpoint=checkpoint,
                )
                model.train()
                _atomic_json(output_dir / "losses.json", losses)
        gate = _final_gate(losses, diagnostics)
        result = {
            "status": "complete",
            "clearml_task_id": str(tracker.task.id),
            "parameter_count": parameter_count,
            "contract": contract,
            "diagnostics": diagnostics,
            "checkpoints": checkpoints,
            "numeric_gate": gate,
        }
        _atomic_json(output_dir / "result.json", result)
        tracker.connect("bounded_overfit_result", result)
        tracker.upload_artifact("bounded_overfit_result", output_dir / "result.json")
        tracker.upload_artifact("bounded_overfit_contract", output_dir / "contract.json")
        return result
    finally:
        tracker.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiments/train_direct_dynamics_all_hours_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(arguments.config.resolve(), arguments.output.resolve()),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
