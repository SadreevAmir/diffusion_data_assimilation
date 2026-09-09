"""512-update real-data mechanics gate for a censored joint generator.

This is deliberately a train-anchor memorization experiment.  It cannot
support a generalization or calibration claim and never evaluates val/test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Subset, default_collate

from .bounded_clean_state_overfit import anchor_indices, diagnostic_indices
from .clearml_tracking import ClearMLTracker
from .config import TrainingConfig, load_json, merge_config_overrides, resolve_path
from .data import build_dataset
from .direct_dynamics_training import (
    DIRECT_CONDITION_CHANNELS,
    DIRECT_INPUT_CHANNELS,
    DIRECT_LEADS,
    DIRECT_OUTPUT_CHANNELS,
    DirectDynamicsTrainer,
    _repeat_field_stats,
    validate_direct_dataset,
)
from .model_io import build_unet
from .runtime import build_dataloader, make_normalized_xy_grid, seed_everything
from .structured_trajectory_evaluation import make_structured_trajectory_figure
from .trainer import _atomic_json
from .transforms import channel_denormalize


DIAGNOSTIC_UPDATES = (0, 64, 256, 512)
MAX_UPDATES = 512
CONDITION_BATCH_SIZE = 8
ENSEMBLE_MEMBERS = 2
TRAIN_WORKERS = 4
PREFETCH_FACTOR = 4
DIAGNOSTIC_MEMBERS = 8
DIAGNOSTIC_CHUNK = 4
FIXED_TIMESTEP = 500.0
BASE_LEARNING_RATE = 1e-4
WARMUP_UPDATES = 32
SOURCE_CONFIG_SHA256 = {
    "experiment": "06c335a798d5c446c601616a73754d2d158f16c513d18d9ae426e9e27a9cd635",
    "data": "6fb71cc079f17fc0888dd4b4c0d7f06dd57c217b0729403ee3916ac3f82903de",
    "model": "bbac44789470eb88e9e47b8b93a1b63e3da99e3ff4de154a6a4f3497caca1029",
}
ZERO_GATE = {
    "sic_mean_max": 0.001,
    "sic_p95_max": 0.005,
    "sit_mean_max_m": 0.001,
    "sit_p95_max_m": 0.005,
    "sit_gt_0p01_fraction_max": 0.01,
}
ZERO_DENOMINATOR_RMSE_MAX = {"sic": 0.001, "sit": 0.005}
RMSE_DENOMINATOR_EPSILON = 1e-8


def _sha256(path: Path) -> str:
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


def _finite_scalars(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _finite_scalars(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_scalars(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError(f"{path} is NaN/Inf")


def _tensor_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(child) for child in value)
    return 0


def _ipc_preflight(dataset, indices: tuple[int, ...]) -> dict[str, int]:
    """Materialize one full collate and reserve worst-case worker prefetch IPC."""

    if len(indices) < CONDITION_BATCH_SIZE:
        raise ValueError("IPC preflight lacks one complete condition batch")
    batch = default_collate(
        [dataset[index] for index in indices[:CONDITION_BATCH_SIZE]]
    )
    batch_bytes = _tensor_bytes(batch)
    if batch_bytes <= 0:
        raise ValueError("IPC preflight collate contains no tensors")
    shm = Path("/dev/shm")
    if not shm.is_dir():
        raise FileNotFoundError("workers4 require /dev/shm")
    available = shutil.disk_usage(shm).free
    resident_batches = TRAIN_WORKERS * PREFETCH_FACTOR + 2
    required = batch_bytes * resident_batches
    if available < required:
        raise RuntimeError(
            f"insufficient /dev/shm: available={available}, required={required}"
        )
    return {
        "full_collate_tensor_bytes": batch_bytes,
        "resident_batch_budget": resident_batches,
        "required_bytes": required,
        "available_bytes": available,
    }


def censor_sic_sit(
    latent_normalized: torch.Tensor,
    valid: torch.Tensor,
    *,
    means: tuple[float, ...],
    stds: tuple[float, ...],
) -> torch.Tensor:
    """Decode and censor the joint physical state without target-dependent masks."""

    if latent_normalized.ndim != 4 or latent_normalized.shape[1] != DIRECT_OUTPUT_CHANNELS:
        raise ValueError("censored joint latent must have shape [B,6,H,W]")
    if valid.shape != latent_normalized.shape[:1] + (1,) + latent_normalized.shape[-2:]:
        raise ValueError("valid mask shape differs from censored joint latent")
    active = valid.expand_as(latent_normalized) > 0
    safe = torch.where(active, latent_normalized, torch.zeros_like(latent_normalized))
    if not torch.all(torch.isfinite(safe[active])):
        raise FloatingPointError("active censored joint latent contains NaN/Inf")
    physical = channel_denormalize(safe.float(), means, stds)
    sic = physical[:, 0::2].clamp(0.0, 1.0)
    sit = physical[:, 1::2].clamp_min(0.0)
    censored = torch.stack(
        (sic[:, 0], sit[:, 0], sic[:, 1], sit[:, 1], sic[:, 2], sit[:, 2]),
        dim=1,
    )
    censored = torch.where(active, censored, torch.zeros_like(censored))
    if not torch.all(torch.isfinite(censored[active])):
        raise FloatingPointError("active censored joint sample contains NaN/Inf")
    return censored


def joint_field_energy_score(
    members: torch.Tensor,
    truth: torch.Tensor,
    valid: torch.Tensor,
    *,
    stds: tuple[float, ...],
) -> torch.Tensor:
    """Unbiased energy score over the complete six-field valid-ocean vector."""

    if members.ndim != 5 or truth.shape != members.shape[:1] + members.shape[2:]:
        raise ValueError("members/truth must have shapes [B,M,6,H,W] and [B,6,H,W]")
    if members.shape[1] < 2 or valid.shape != truth.shape[:1] + (1,) + truth.shape[-2:]:
        raise ValueError("joint energy score requires M>=2 and a common valid mask")
    scales = members.new_tensor(stds).view(1, 1, -1, 1, 1)
    mask = valid[:, None].expand_as(members) > 0
    if not torch.any(mask) or not torch.all(torch.isfinite(members[mask])):
        raise FloatingPointError("joint energy members are empty or non-finite")
    truth_mask = valid.expand_as(truth) > 0
    if not torch.all(torch.isfinite(truth[truth_mask])):
        raise FloatingPointError("joint energy truth is non-finite")
    normalization = torch.sqrt(
        valid.flatten(1).sum(dim=1).clamp(min=1) * members.shape[2]
    )
    difference = torch.where(
        mask,
        (members - truth[:, None]) / scales,
        torch.zeros_like(members),
    )
    observation_distance = torch.linalg.vector_norm(
        difference.flatten(2), dim=2
    ) / normalization[:, None]
    pair_difference = torch.where(
        valid[:, None, None].expand(
            -1, members.shape[1], members.shape[1], members.shape[2], -1, -1
        )
        > 0,
        (members[:, :, None] - members[:, None, :]) / scales[:, None],
        torch.zeros_like(members[:, :, None] - members[:, None, :]),
    )
    pair_distance = torch.linalg.vector_norm(
        pair_difference.flatten(3), dim=3
    ) / normalization[:, None, None]
    count = members.shape[1]
    score = observation_distance.mean(dim=1) - pair_distance.sum(dim=(1, 2)) / (
        2.0 * count * (count - 1)
    )
    result = score.mean()
    if not torch.isfinite(result):
        raise FloatingPointError("joint field energy score is NaN/Inf")
    return result


def _initialize_small_nonzero_head(model: torch.nn.Module, *, seed: int) -> dict[str, float]:
    head = getattr(model, "conv_out", None)
    if head is None or not hasattr(head, "weight"):
        raise ValueError("UNet lacks the required output convolution")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    with torch.no_grad():
        initialized = 1e-3 * torch.randn(
            head.weight.shape, generator=generator, dtype=head.weight.dtype
        )
        head.weight.copy_(initialized.to(head.weight.device))
        if head.bias is not None:
            head.bias.zero_()
    norm = float(torch.linalg.vector_norm(head.weight.detach().float()).cpu())
    if not math.isfinite(norm) or norm <= 0:
        raise RuntimeError("output head initialization is not finite and nonzero")
    return {"weight_std": 1e-3, "weight_norm": norm, "bias": 0.0}


def _generate_members(
    model,
    *,
    condition: torch.Tensor,
    background: torch.Tensor,
    valid: torch.Tensor,
    noise: torch.Tensor,
    grid: torch.Tensor,
    means: tuple[float, ...],
    stds: tuple[float, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, member_count = noise.shape[:2]
    flat_noise = noise.flatten(0, 1)
    flat_valid = valid.repeat_interleave(member_count, dim=0)
    active = flat_valid.expand_as(flat_noise) > 0
    flat_noise = torch.where(active, flat_noise, torch.zeros_like(flat_noise))
    flat_condition = condition.repeat_interleave(member_count, dim=0)
    flat_background = background.repeat_interleave(member_count, dim=0)
    model_input = torch.cat(
        (flat_noise, grid.expand(flat_noise.shape[0], -1, -1, -1), flat_condition),
        dim=1,
    )
    if model_input.shape[1] != DIRECT_INPUT_CHANNELS:
        raise ValueError("censored joint model input has the wrong channel count")
    time = torch.full(
        (flat_noise.shape[0],), FIXED_TIMESTEP, device=flat_noise.device
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        residual = model(model_input, time, return_dict=False)[0]
    latent = flat_background.float() + residual.float()
    physical = censor_sic_sit(latent, flat_valid, means=means, stds=stds)
    return (
        physical.unflatten(0, (batch_size, member_count)),
        latent.unflatten(0, (batch_size, member_count)),
    )


def _masked_rmse(prediction: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor) -> float:
    active = mask.expand_as(prediction) > 0
    if not torch.any(active):
        raise ValueError("RMSE mask is empty")
    return float(torch.sqrt((prediction[active] - truth[active]).square().mean()))


def _diagnostic_metrics(
    ensemble: torch.Tensor,
    latent: torch.Tensor,
    truth: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, Any]:
    active = valid[:, None].expand_as(ensemble) > 0
    if not torch.all(torch.isfinite(ensemble[active])) or not torch.all(
        torch.isfinite(latent[active])
    ):
        raise FloatingPointError("diagnostic ensemble/latent contains NaN/Inf")
    ensemble_mean = ensemble.mean(dim=1)
    outputs = {}
    for lead_index, lead_day in enumerate(DIRECT_LEADS):
        for offset, field in enumerate(("sic", "sit")):
            channel = 2 * lead_index + offset
            prediction = ensemble_mean[:, channel : channel + 1]
            target = truth[:, channel : channel + 1]
            positive = valid * (target > 0).to(valid.dtype)
            outputs[f"d{lead_day}_{field}"] = {
                "rmse": _masked_rmse(prediction, target, valid),
                "positive_region_rmse": _masked_rmse(prediction, target, positive),
                "positive_prediction_mean": float(prediction[positive > 0].mean()),
                "positive_truth_mean": float(target[positive > 0].mean()),
            }
    zero_case_leads = []
    ocean = valid[:, 0] > 0
    for case in range(truth.shape[0]):
        if torch.all(truth[case][:, ocean[case]] == 0):
            for lead_index, lead_day in enumerate(DIRECT_LEADS):
                sic = ensemble[case, :, 2 * lead_index][:, ocean[case]]
                sit = ensemble[case, :, 2 * lead_index + 1][:, ocean[case]]
                zero_case_leads.append(
                    {
                        "case": case,
                        "lead": f"d{lead_day}",
                        "sic_mean": float(sic.mean()),
                        "sic_p95": float(torch.quantile(sic, 0.95)),
                        "sit_mean_m": float(sit.mean()),
                        "sit_p95_m": float(torch.quantile(sit, 0.95)),
                        "sit_gt_0p01_fraction": float((sit > 0.01).float().mean()),
                    }
                )
    if not zero_case_leads:
        raise ValueError("diagnostic anchors contain no fully zero case")
    metrics = {
        "outputs": outputs,
        "zero_case_leads": zero_case_leads,
        "support": {
            "sic_below_zero": int(((ensemble[:, :, 0::2] < 0) & (valid[:, None] > 0)).sum()),
            "sic_above_one": int(((ensemble[:, :, 0::2] > 1) & (valid[:, None] > 0)).sum()),
            "sit_below_zero": int(((ensemble[:, :, 1::2] < 0) & (valid[:, None] > 0)).sum()),
        },
    }
    _finite_scalars(metrics)
    return metrics


@torch.no_grad()
def _diagnostic(
    *,
    model,
    batch: dict[str, Any],
    noises: torch.Tensor,
    update: int,
    output_dir: Path,
    tracker: ClearMLTracker,
    means: tuple[float, ...],
    stds: tuple[float, ...],
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    try:
        condition = batch["structured_conditioning"].float()
        background = batch["background"].float()
        valid = batch["valid_mask"][:, :1].float()
        grid = make_normalized_xy_grid(
            *batch["truth"].shape[-2:], device=batch["truth"].device
        )
        physical_chunks = []
        latent_chunks = []
        for start in range(0, DIAGNOSTIC_MEMBERS, DIAGNOSTIC_CHUNK):
            physical, latent = _generate_members(
                model,
                condition=condition,
                background=background,
                valid=valid,
                noise=noises[:, start : start + DIAGNOSTIC_CHUNK],
                grid=grid,
                means=means,
                stds=stds,
            )
            physical_chunks.append(physical)
            latent_chunks.append(latent)
        ensemble = torch.cat(physical_chunks, dim=1)
        latent = torch.cat(latent_chunks, dim=1)
        truth = batch["structured_physical_truth"].float()
        persistence = batch["structured_physical_background"].float()
        metrics = _diagnostic_metrics(ensemble, latent, truth, valid)
        metrics.update({"update": update, "checkpoint": checkpoint})
        stage = output_dir / "diagnostics" / f"update_{update:04d}"
        stage.mkdir(parents=True, exist_ok=False)
        _atomic_torch_save(
            {
                "physical_ensemble": ensemble.cpu(),
                "uncensored_latent_normalized": latent.cpu(),
                "truth": truth.cpu(),
                "persistence": persistence.cpu(),
                "valid_mask": valid.cpu(),
                "initial_noise": noises.cpu(),
            },
            stage / "samples.pt",
        )
        _atomic_json(stage / "metrics.json", metrics)
        for case in range(truth.shape[0]):
            for member in (0, 1):
                figure = make_structured_trajectory_figure(
                    truth[case],
                    persistence[case],
                    ensemble[case, member],
                    valid[case],
                    title=f"censored joint energy update {update}; anchor {case}, member {member}",
                    origin="upper",
                    lead_days=DIRECT_LEADS,
                )
                image_path = stage / f"anchor_{case:02d}_member{member}.png"
                figure.savefig(image_path, dpi=180, bbox_inches="tight")
                import matplotlib.pyplot as plt

                plt.close(figure)
                tracker.report_image(
                    "censored_joint_energy_overfit/samples",
                    f"anchor_{case:02d}_member{member}",
                    image_path,
                    update,
                )
        for name, values in metrics["outputs"].items():
            for metric, value in values.items():
                tracker.report_scalar(
                    f"censored_joint_energy_overfit/{metric}", name, value, update
                )
        return metrics
    finally:
        model.train(was_training)


def _final_gate(
    losses: list[float], diagnostics: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    if len(losses) != MAX_UPDATES or set(diagnostics) != set(DIAGNOSTIC_UPDATES):
        raise ValueError("censored joint mechanics result is incomplete")
    _finite_scalars(losses, "losses")
    initial = diagnostics[0]
    final = diagnostics[512]
    ratios = []
    positive_ratios = []
    rmse_comparisons = []
    positive_rmse_comparisons = []
    positive_mean_ratios = []
    for name, initial_values in initial["outputs"].items():
        final_values = final["outputs"][name]
        field = name.rsplit("_", 1)[-1]
        for key, destination, comparisons, ratio_limit in (
            ("rmse", ratios, rmse_comparisons, 1.05),
            ("positive_region_rmse", positive_ratios, positive_rmse_comparisons, 1.10),
        ):
            denominator = initial_values[key]
            if denominator > RMSE_DENOMINATOR_EPSILON:
                value = final_values[key] / denominator
                destination.append(value)
                comparisons.append(
                    {
                        "output": name,
                        "mode": "relative_ratio",
                        "value": value,
                        "threshold": ratio_limit,
                        "passed": value <= ratio_limit,
                    }
                )
            else:
                value = final_values[key]
                threshold = ZERO_DENOMINATOR_RMSE_MAX[field]
                comparisons.append(
                    {
                        "output": name,
                        "mode": "absolute_final_rmse",
                        "value": value,
                        "threshold": threshold,
                        "passed": value <= threshold,
                    }
                )
        positive_mean_ratios.append(
            final_values["positive_prediction_mean"]
            / max(final_values["positive_truth_mean"], 1e-8)
        )
    zero_maxima = {
        "sic_mean": max(record["sic_mean"] for record in final["zero_case_leads"]),
        "sic_p95": max(record["sic_p95"] for record in final["zero_case_leads"]),
        "sit_mean_m": max(record["sit_mean_m"] for record in final["zero_case_leads"]),
        "sit_p95_m": max(record["sit_p95_m"] for record in final["zero_case_leads"]),
        "sit_gt_0p01_fraction": max(
            record["sit_gt_0p01_fraction"] for record in final["zero_case_leads"]
        ),
    }
    criteria = {
        "zero_sic_mean": zero_maxima["sic_mean"] <= ZERO_GATE["sic_mean_max"],
        "zero_sic_p95": zero_maxima["sic_p95"] <= ZERO_GATE["sic_p95_max"],
        "zero_sit_mean": zero_maxima["sit_mean_m"] <= ZERO_GATE["sit_mean_max_m"],
        "zero_sit_p95": zero_maxima["sit_p95_m"] <= ZERO_GATE["sit_p95_max_m"],
        "zero_sit_tail": zero_maxima["sit_gt_0p01_fraction"]
        <= ZERO_GATE["sit_gt_0p01_fraction_max"],
        "rmse_mean_ratio": not ratios or sum(ratios) / len(ratios) <= 0.85,
        "rmse_each_output": all(item["passed"] for item in rmse_comparisons),
        "positive_rmse_each_output": all(
            item["passed"] for item in positive_rmse_comparisons
        ),
        "positive_mass_not_collapsed": min(positive_mean_ratios) >= 0.50,
        "physical_support": all(value == 0 for value in final["support"].values()),
    }
    result = {
        "status": "passed_numeric_pending_visual_review" if all(criteria.values()) else "failed",
        "criteria": criteria,
        "zero_anchor_maxima": zero_maxima,
        "rmse_comparisons": rmse_comparisons,
        "positive_region_rmse_comparisons": positive_rmse_comparisons,
        "mean_rmse_ratio_final_to_initial": sum(ratios) / len(ratios) if ratios else 0.0,
        "maximum_rmse_ratio_final_to_initial": max(ratios) if ratios else 0.0,
        "maximum_positive_region_rmse_ratio": max(positive_ratios) if positive_ratios else 0.0,
        "minimum_positive_prediction_to_truth_mean_ratio": min(positive_mean_ratios),
        "visual_review_required": True,
        "rank_or_diversity_gate_applicable": False,
        "permits_heldout_pilot_review": all(criteria.values()),
        "permits_substantive_training": False,
    }
    _finite_scalars(result, "gate")
    return result


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    if torch.cuda.device_count() != 1:
        raise RuntimeError("censored joint mechanics gate requires exactly one visible GPU")
    if os.environ.get("CLEARML_REQUIRE_ONLINE") != "1":
        raise RuntimeError("censored joint mechanics gate requires online ClearML")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to reuse censored joint output: {output_dir}")
    experiment = load_json(config_path)
    config_dir = config_path.resolve().parent
    data_path = resolve_path(experiment["data_config"], config_dir)
    model_path = resolve_path(experiment["model_config"], config_dir)
    for name, path in (("experiment", config_path), ("data", data_path), ("model", model_path)):
        if _sha256(path) != SOURCE_CONFIG_SHA256[name]:
            raise ValueError(f"{name} config differs from audited source")
    data_config = merge_config_overrides(load_json(data_path), experiment.get("data_overrides"))
    raw_model = load_json(model_path)
    effective_model = dict(raw_model)
    effective_model.update(
        {
            "dropout": 0.0,
            "train_batch_size": CONDITION_BATCH_SIZE,
            "activation_checkpointing": False,
        }
    )
    model_config = TrainingConfig.from_dict(effective_model)
    if (
        model_config.image_size != (320, 256)
        or model_config.in_channels != DIRECT_INPUT_CHANNELS
        or model_config.out_channels != DIRECT_OUTPUT_CHANNELS
        or model_config.dropout != 0.0
        or model_config.activation_checkpointing
    ):
        raise ValueError("censored joint mechanics model contract differs")
    seed_everything(model_config.seed)
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    dataset = build_dataset(data_config, split="train")
    sentinel = validate_direct_dataset(dataset)
    train_indices = anchor_indices(len(dataset))
    sample_indices = diagnostic_indices(len(dataset))
    ipc_preflight = _ipc_preflight(dataset, train_indices)
    loader_generator = torch.Generator(device="cpu").manual_seed(model_config.seed + 41)
    loader = build_dataloader(
        Subset(dataset, train_indices),
        CONDITION_BATCH_SIZE,
        TRAIN_WORKERS,
        shuffle=True,
        prefetch_factor=PREFETCH_FACTOR,
        generator=loader_generator,
    )
    if len(loader) != 12:
        raise ValueError("censored joint loader must have twelve batches per anchor epoch")
    diagnostic_batch = default_collate([dataset[index] for index in sample_indices])
    output_dir.mkdir(parents=True, exist_ok=False)
    tracker = ClearMLTracker(
        "sea_ice_two_stage",
        f"censored_joint_energy_overfit_{output_dir.name}",
        tags=[
            "censored-joint-law",
            "energy-score",
            "real-data-mechanics",
            "train-anchors-only",
            "one-gpu",
            "no-ode",
            "no-member-mse",
            "dropout-zero",
        ],
        env_path=experiment.get("clearml", {}).get("env_path"),
    )
    contract = {
        "purpose": "real_data_mechanics_only",
        "generalization_claim_permitted": False,
        "dataset_split": "train",
        "train_indices": train_indices,
        "diagnostic_indices": sample_indices,
        "updates": MAX_UPDATES,
        "diagnostic_updates": DIAGNOSTIC_UPDATES,
        "condition_batch_size": CONDITION_BATCH_SIZE,
        "network_input_batch_size": CONDITION_BATCH_SIZE * ENSEMBLE_MEMBERS,
        "training_members": ENSEMBLE_MEMBERS,
        "diagnostic_members": DIAGNOSTIC_MEMBERS,
        "workers": TRAIN_WORKERS,
        "prefetch_factor": PREFETCH_FACTOR,
        "ipc_preflight": ipc_preflight,
        "fixed_timestep": FIXED_TIMESTEP,
        "objective": "unbiased_joint_energy_score",
        "distance": "train_std_normalized_l2_all_six_fields_valid_ocean",
        "generator": "persistence_centered_censored_joint_residual",
        "sic_censor": "clamp_0_1_inside_train_and_sample_law",
        "sit_censor": "relu_inside_train_and_sample_law",
        "per_member_mse": False,
        "antithetic_noise": False,
        "straight_through_gradient": False,
        "future_support_mask": False,
        "dropout": 0.0,
        "ema": False,
        "activation_checkpointing": False,
        "optimizer": "AdamW",
        "learning_rate": BASE_LEARNING_RATE,
        "warmup_updates": WARMUP_UPDATES,
        "post_warmup_schedule": "constant",
        "precision": "bf16_network_fp32_objective",
        "zero_anchor_gate": ZERO_GATE,
        "zero_denominator_rmse_max": ZERO_DENOMINATOR_RMSE_MAX,
        "rmse_denominator_epsilon": RMSE_DENOMINATOR_EPSILON,
        "dataset_sentinel": sentinel,
        "effective_data_config": data_config,
        "source_config_sha256": SOURCE_CONFIG_SHA256,
        "source_model_config": raw_model,
        "effective_model_config": effective_model,
    }
    tracker.connect("censored_joint_energy_overfit_contract", contract)
    _atomic_json(output_dir / "contract.json", contract)
    device = torch.device("cuda:0")
    model = build_unet(model_config).to(device)
    head_init = _initialize_small_nonzero_head(model, seed=model_config.seed + 313)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LEARNING_RATE)
    means = _repeat_field_stats(dataset.means)
    stds = _repeat_field_stats(dataset.stds)
    grid = make_normalized_xy_grid(*model_config.image_size, device=device)
    noise_generator = torch.Generator(device=device).manual_seed(model_config.seed + 101)
    diagnostic_generator = torch.Generator(device="cpu").manual_seed(model_config.seed + 202)
    diagnostic_noise = torch.randn(
        (
            len(sample_indices),
            DIAGNOSTIC_MEMBERS,
            DIRECT_OUTPUT_CHANNELS,
            *model_config.image_size,
        ),
        generator=diagnostic_generator,
        dtype=torch.float32,
    )
    diagnostic_valid = diagnostic_batch["valid_mask"][:, :1]
    diagnostic_noise = torch.where(
        diagnostic_valid[:, None].expand_as(diagnostic_noise) > 0,
        diagnostic_noise,
        torch.zeros_like(diagnostic_noise),
    ).to(device)
    diagnostic_batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in diagnostic_batch.items()
    }
    losses = []
    diagnostics: dict[int, dict[str, Any]] = {}
    checkpoints: dict[int, dict[str, Any]] = {}

    def snapshot(update: int) -> None:
        checkpoint_path = output_dir / f"model_step_{update:04d}.pth"
        _atomic_torch_save(model.state_dict(), checkpoint_path)
        checkpoint = {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            "update": update,
            "weight_source": "raw",
        }
        checkpoints[update] = checkpoint
        _atomic_json(output_dir / f"model_step_{update:04d}.json", checkpoint)
        tracker.connect(f"censored_joint_snapshot_{update:04d}", checkpoint)
        diagnostics[update] = _diagnostic(
            model=model,
            batch=diagnostic_batch,
            noises=diagnostic_noise,
            update=update,
            output_dir=output_dir,
            tracker=tracker,
            means=means,
            stds=stds,
            checkpoint=checkpoint,
        )

    iterator = iter(loader)
    try:
        snapshot(0)
        for update in range(1, MAX_UPDATES + 1):
            try:
                raw_batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                raw_batch = next(iterator)
            batch = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in raw_batch.items()
            }
            condition = batch["structured_conditioning"].float()
            background = batch["background"].float()
            valid = batch["valid_mask"][:, :1].float()
            noise = torch.randn(
                (
                    CONDITION_BATCH_SIZE,
                    ENSEMBLE_MEMBERS,
                    DIRECT_OUTPUT_CHANNELS,
                    *model_config.image_size,
                ),
                generator=noise_generator,
                device=device,
            )
            members, _ = _generate_members(
                model,
                condition=condition,
                background=background,
                valid=valid,
                noise=noise,
                grid=grid,
                means=means,
                stds=stds,
            )
            loss = joint_field_energy_score(
                members,
                batch["structured_physical_truth"].float(),
                valid,
                stds=stds,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            if not torch.isfinite(gradient_norm) or (update == 1 and gradient_norm <= 0):
                raise FloatingPointError("censored joint gradient is non-finite or dead")
            learning_rate = BASE_LEARNING_RATE * min(update / WARMUP_UPDATES, 1.0)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.step()
            value = float(loss.detach())
            losses.append(value)
            if update == 1 or update % 8 == 0:
                tracker.report_scalar(
                    "censored_joint_energy_overfit/train", "energy_score", value, update
                )
                tracker.report_scalar(
                    "censored_joint_energy_overfit/train", "learning_rate", learning_rate, update
                )
            if update in DIAGNOSTIC_UPDATES:
                snapshot(update)
                _atomic_json(output_dir / "losses.json", losses)
        gate = _final_gate(losses, diagnostics)
        result = {
            "status": "complete",
            "clearml_task_id": str(tracker.task.id),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "head_initialization": head_init,
            "contract": contract,
            "loss_first_32_mean": sum(losses[:32]) / 32,
            "loss_last_32_mean": sum(losses[-32:]) / 32,
            "diagnostics": diagnostics,
            "checkpoints": checkpoints,
            "numeric_gate": gate,
        }
        _finite_scalars(result)
        _atomic_json(output_dir / "result.json", result)
        tracker.connect("censored_joint_energy_overfit_result", result)
        tracker.upload_artifact("censored_joint_energy_overfit_result", output_dir / "result.json")
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
    print(json.dumps(run(arguments.config.resolve(), arguments.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
