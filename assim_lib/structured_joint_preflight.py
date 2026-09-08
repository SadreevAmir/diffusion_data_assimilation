"""Fail-closed CPU preflight for the structured joint SIC/SIT training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import TrainingConfig, load_json, resolve_path
from .data import (
    STRUCTURED_ASSIMILATION_CONDITIONING_LAYOUT,
    STRUCTURED_DYNAMICS_CONDITIONING_LAYOUT,
    STRUCTURED_TRAJECTORY_CONDITIONING_LAYOUT,
    M2MForecastDataset,
    build_dataset,
)
from .runtime import (
    dataloader_batch_count,
    validate_optimizer_step_budget,
)
from .structured_archive_audit import validate_bound_archive_audit
from .structured_joint_state import (
    LATENT_CHANNELS,
    canonical_mapping_sha256,
    decode_structured_joint_trajectory,
    encode_structured_joint_trajectory,
    validate_conditioning_normalization,
)


def _require_binary(value: torch.Tensor, name: str) -> None:
    if not torch.all((value == 0) | (value == 1)):
        raise ValueError(f"{name} must be binary")


def run_preflight(experiment_path: str | Path, *, build_model: bool = True) -> dict:
    experiment_path = Path(experiment_path).resolve()
    experiment = load_json(experiment_path)
    data_path = resolve_path(experiment["data_config"], experiment_path.parent).resolve()
    method_path = resolve_path(experiment["model_config"], experiment_path.parent).resolve()
    data_config = load_json(data_path)
    archive_audit = validate_bound_archive_audit(data_config, data_path)
    blocking_reasons = []
    method_config = {**load_json(method_path), **experiment.get("training", {})}
    stats_path = resolve_path(method_config["structured_state_stats_path"], method_path.parent).resolve()
    stats = load_json(stats_path)
    method_config["structured_state_stats"] = stats
    config = TrainingConfig.from_dict(method_config)
    validate_conditioning_normalization(data_config, stats)

    if canonical_mapping_sha256(data_config) != stats["data_config_sha256"]:
        raise ValueError("train-only stats do not match the exact data configuration")
    train_dataset = build_dataset(data_config, split="train")
    valid_dataset = build_dataset(data_config, split="valid")
    train_dataset.validate_structured_sral_audit_contract(archive_audit)
    valid_dataset.validate_structured_sral_audit_contract(archive_audit)
    train_provenance = train_dataset.provenance()
    if train_provenance["pair_manifest_sha256"] != stats["pair_manifest_sha256"]:
        raise ValueError("train-only stats do not match the exact train pair manifest")
    if config.in_channels != train_dataset.conditioned_input_channels:
        raise ValueError("configured input channels do not match the dataset contract")
    expected_outputs = LATENT_CHANNELS * len(config.trajectory_lead_days)
    if config.out_channels != expected_outputs:
        raise ValueError("configured output channels do not match the structured codec")
    batches_per_epoch = dataloader_batch_count(len(train_dataset), config.train_batch_size, drop_last=True)
    steps_per_epoch, planned_optimizer_steps = validate_optimizer_step_budget(config, batches_per_epoch)
    ema_step = max(0, planned_optimizer_steps - config.ema_update_after_step - 1)
    terminal_ema_decay = max(
        config.ema_min_decay,
        min(config.ema_decay, (1.0 + ema_step) / (10.0 + ema_step)),
    )
    audit = stats.get("audit", {})
    expected_samples = len(train_dataset) * len(config.trajectory_lead_days)
    if int(audit.get("sample_count", -1)) != expected_samples:
        raise ValueError(
            f"stats sample_count={audit.get('sample_count')} does not match train size={expected_samples}"
        )

    checked_cases = []
    layout = data_config.get("conditioning_layout")
    for split_name, dataset in (("train", train_dataset), ("valid", valid_dataset)):
        candidates = sorted({0, len(dataset) // 2, len(dataset) - 1})
        for index in candidates:
            item = dataset[index]
            condition = item["structured_conditioning"]
            expected_condition_channels = config.in_channels - config.out_channels - 2
            if tuple(condition.shape) != (
                expected_condition_channels,
                *config.image_size,
            ):
                raise ValueError(f"unexpected condition shape for {split_name}[{index}]")
            for name in (
                "truth",
                "background",
                "valid_mask",
                "structured_conditioning",
                "structured_physical_truth",
                "structured_physical_background",
                "structured_lag0_physical_values",
            ):
                if not torch.all(torch.isfinite(item[name])):
                    raise ValueError(f"non-finite {name} in {split_name}[{index}]")
            _require_binary(item["valid_mask"], "valid_mask")
            if layout == STRUCTURED_DYNAMICS_CONDITIONING_LAYOUT:
                initial = condition[:2]
                initial_mask = condition[2:3]
                _require_binary(initial_mask, "initial_state_mask")
                if torch.any(initial[initial_mask.expand_as(initial) == 0] != 0):
                    raise ValueError("initial state leaks outside its mask")
                forcing_count = len(data_config.get("dynamic_forcing_indices", ()))
                forcing_values = condition[3 : 3 + forcing_count]
                forcing_masks = condition[3 + forcing_count : 3 + 2 * forcing_count]
                _require_binary(forcing_masks, "dynamic_forcing_masks")
                if torch.any(forcing_values[forcing_masks.expand_as(forcing_values) == 0] != 0):
                    raise ValueError("dynamic forcing leaks outside its masks")
                if torch.any(item["obs_mask"] != 0):
                    raise ValueError("state-only dynamics exposed an observation mask")
            else:
                background_steps = (
                    len(config.trajectory_lead_days)
                    if layout == STRUCTURED_TRAJECTORY_CONDITIONING_LAYOUT
                    else 1
                )
                background_channels = 3 * background_steps
                for lead in range(background_steps):
                    value = condition[3 * lead : 3 * lead + 2]
                    mask = condition[3 * lead + 2 : 3 * lead + 3]
                    _require_binary(mask, f"background_lead{lead}_mask")
                    if torch.any(value[mask.expand_as(value) == 0] != 0):
                        raise ValueError(f"background lead {lead} leaks outside its mask")
                lag_width = 3 if layout == STRUCTURED_ASSIMILATION_CONDITIONING_LAYOUT else 5
                for lag in range(3):
                    start = background_channels + lag * lag_width
                    value = condition[start : start + 2]
                    mask = condition[start + lag_width - 1 : start + lag_width]
                    _require_binary(mask, f"lag{lag}_mask")
                    expanded_mask = mask.expand_as(value)
                    if torch.any(value[expanded_mask == 0] != 0):
                        raise ValueError(f"lag{lag} value leaks outside its mask")
                    if lag_width == 5:
                        innovation = condition[start + 2 : start + 4]
                        if torch.any(innovation[expanded_mask == 0] != 0):
                            raise ValueError(f"lag{lag} innovation leaks outside its mask")
            valid = item["valid_mask"][:1]
            physical_truth = item["structured_physical_truth"].unsqueeze(0)
            latent = encode_structured_joint_trajectory(
                physical_truth,
                valid.unsqueeze(0),
                stats,
                generator=torch.Generator().manual_seed(index + (0 if split_name == "train" else 1)),
            )
            decoded = decode_structured_joint_trajectory(latent, stats)
            expanded_valid = valid.unsqueeze(0).bool().expand_as(decoded)
            if not torch.allclose(
                decoded[expanded_valid],
                physical_truth[expanded_valid],
                atol=2e-6,
                rtol=2e-6,
            ):
                raise ValueError(f"codec round-trip failed for {split_name}[{index}]")
            flow_mask = item["structured_flow_mask"]
            lag0_mask = item["structured_lag0_mask"]
            if layout == STRUCTURED_DYNAMICS_CONDITIONING_LAYOUT:
                if torch.any(flow_mask != valid.expand_as(flow_mask)):
                    raise ValueError("all dynamics targets must remain active on water")
            else:
                if torch.any(flow_mask[:4][lag0_mask.expand(4, -1, -1) > 0] != 0):
                    raise ValueError("lag0 exact observations were not removed from the flow subspace")
                if torch.any(flow_mask[4:] != valid.expand_as(flow_mask[4:])):
                    raise ValueError("future trajectory channels must remain fully active on water")
            checked_cases.append(f"{split_name}:{item['meta']['case_id']}")

    forecast_anchor = None
    if layout == STRUCTURED_TRAJECTORY_CONDITIONING_LAYOUT:
        forecast_anchor = archive_audit["trajectory_date_audit"]["splits"]["valid"]["first_eligible_anchor"]
        forecast_item = M2MForecastDataset.build_structured_forecast_item(data_config, forecast_anchor)
        if (
            "truth" in forecast_item
            or "structured_physical_truth" in forecast_item
            or forecast_item["meta"].get("target_trajectory_paths") != []
        ):
            raise ValueError("truth-free forecast builder exposed future targets")
        if tuple(forecast_item["structured_conditioning"].shape) != (
            config.in_channels - config.out_channels - 2,
            *config.image_size,
        ):
            raise ValueError("truth-free forecast conditioning shape differs")

    parameter_count = None
    smoke_output_shape = None
    if build_model:
        from .model_io import build_unet

        model = build_unet(config).cpu().eval()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        smoke_size = (32, 32)
        smoke_input = torch.zeros((1, config.in_channels, *smoke_size), dtype=torch.float32)
        with torch.no_grad():
            smoke_output = model(smoke_input, torch.zeros(1), return_dict=False)[0]
        smoke_output_shape = list(smoke_output.shape)
        if smoke_output_shape != [1, expected_outputs, *smoke_size]:
            raise ValueError(f"unexpected UNet smoke output shape: {smoke_output_shape}")

    return {
        "status": "ready" if not blocking_reasons else "blocked",
        "blocking_reasons": blocking_reasons,
        "experiment": str(experiment_path),
        "data_config_sha256": stats["data_config_sha256"],
        "train_pair_manifest_sha256": stats["pair_manifest_sha256"],
        "archive_semantics_audit_sha256": data_config["archive_semantics_audit_sha256"],
        "forecast_metadata_manifest_sha256": archive_audit["forecast_metadata_manifest_sha256"],
        "forecast_audited_content_sha256": archive_audit["forecast_audited_content_sha256"],
        "static_mask_content_sha256": archive_audit["static_mask_provenance"]["content_sha256"],
        "sral_content_manifest_sha256": archive_audit["sral_provenance"].get("content_manifest_sha256"),
        "trajectory_semantics": archive_audit["trajectory_semantics"],
        "target_slice_index": archive_audit["target_slice_index"],
        "utc_time_coordinate_verified": archive_audit["utc_time_coordinate_verified"],
        "time_claim_policy": archive_audit["time_claim_policy"],
        "train_cases": len(train_dataset),
        "valid_cases": len(valid_dataset),
        "train_drop_last": True,
        "train_batches_per_epoch": batches_per_epoch,
        "steps_per_epoch": steps_per_epoch,
        "planned_optimizer_steps": planned_optimizer_steps,
        "terminal_instantaneous_ema_decay": terminal_ema_decay,
        "checked_cases": checked_cases,
        "truth_free_forecast_anchor": forecast_anchor,
        "in_channels": config.in_channels,
        "out_channels": config.out_channels,
        "parameter_count": parameter_count,
        "smoke_output_shape": smoke_output_shape,
        "cuda_initialized": torch.cuda.is_initialized(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--skip-model-build", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_preflight(args.experiment, build_model=not args.skip_model_build)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
