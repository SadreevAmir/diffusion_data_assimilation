"""Frozen paired evaluation of coarse SIC/SIT dynamics checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from .clearml_tracking import ClearMLTracker
from .config import TrainingConfig, load_json
from .data import build_dataset
from .direct_dynamics_cascade_coarse import CoarseCascadeSampler, coarse_target
from .direct_dynamics_cascade_coarse_residual import (
    CoarsePersistenceResidualSampler,
    CoarseStandardizedPersistenceResidualSampler,
    coarse_persistence_from_condition,
)
from .direct_dynamics_cascade_fine_training import (
    _calendar_inventory,
    _canonical_sha256,
    _clean_code_identity,
    _fine_collate,
)
from .direct_dynamics_training import DIRECT_LEADS, _repeat_field_stats, validate_direct_dataset
from .model_io import build_unet
from .trainer import _atomic_json
from .transforms import channel_denormalize


ENSEMBLE_SIZE = 8
SAMPLE_STEPS = 33
SOURCE_LABELS = (
    "standardized_1024",
    "standardized_2048",
    "unwhitened_2048",
    "absolute_2048",
)
REQUIRED_SOURCE_ARTIFACTS = (
    "config.json",
    "metadata.json",
    "coarse_cascade_manifest.json",
    "coarse_cascade_dataset_sentinel.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _member_seed(case_id: str, member: int) -> int:
    payload = f"cascade-v1|coarse|{case_id}|member={member}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _verify_source_files(source: dict[str, Any]) -> None:
    root = Path(source["run_dir"])
    required = {
        *REQUIRED_SOURCE_ARTIFACTS,
        source["checkpoint"],
        source["samples_artifact"],
    }
    missing = required.difference(source["sha256"])
    if missing:
        raise ValueError(
            f"source inventory is incomplete for {source['label']}: {sorted(missing)}"
        )
    for relative, expected in source["sha256"].items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe source artifact path: {source['label']}:{relative}")
        path = root / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"frozen source mismatch: {source['label']}:{relative}")


def _verify_common_source_contract(sources: list[dict[str, Any]]) -> dict[str, Any]:
    """Read only hash-verified metadata and prove a common causal data law."""
    metadata = [load_json(Path(source["run_dir"]) / "metadata.json") for source in sources]
    manifests = [
        load_json(Path(source["run_dir"]) / "coarse_cascade_manifest.json")
        for source in sources
    ]
    reference = metadata[0]
    data_config = reference["data_config"]
    data_hash = _canonical_sha256(data_config)
    normalization = {
        "means": reference["normalization_means"],
        "stds": reference["normalization_stds"],
    }
    for source, candidate, manifest in zip(sources, metadata, manifests, strict=True):
        if _canonical_sha256(candidate["data_config"]) != data_hash:
            raise ValueError(f"source data law differs for {source['label']}")
        candidate_normalization = {
            "means": candidate["normalization_means"],
            "stds": candidate["normalization_stds"],
        }
        if candidate_normalization != normalization:
            raise ValueError(f"source normalization differs for {source['label']}")
        if manifest.get("code_commit") != source["code_commit"]:
            raise ValueError(f"source commit mismatch for {source['label']}")
        expected_sampler = {
            "standardized": "CoarseStandardizedPersistenceResidualSampler",
            "unwhitened": "CoarsePersistenceResidualSampler",
            "absolute": "CoarseCascadeSampler",
        }[source["kind"]]
        if manifest.get("sampler") != expected_sampler:
            raise ValueError(f"source sampler mismatch for {source['label']}")
        training_config = TrainingConfig.from_dict(
            load_json(Path(source["run_dir"]) / "config.json")
        )
        if tuple(training_config.image_size) != (160, 128):
            raise ValueError(f"source image size differs for {source['label']}")
        if source["kind"] == "standardized":
            statistics_path = Path(manifest["residual_statistics_path"])
            expected = manifest["residual_statistics_sha256"]
            if not statistics_path.is_file() or _sha256(statistics_path) != expected:
                raise ValueError(
                    f"standardized source statistics differ for {source['label']}"
                )
    if data_config.get("means") != normalization["means"] or data_config.get(
        "stds"
    ) != normalization["stds"]:
        raise ValueError("metadata normalization differs from the common data law")
    expected_causal_contract = {
        "background_strategy": "none",
        "conditioning_layout": "structured_sic_sit_dynamics_v1",
        "trajectory_semantics": "state_only_forecast_snapshots_d_plus_3_6_9",
        "trajectory_lead_days": list(DIRECT_LEADS),
        "future_horizon_days": max(DIRECT_LEADS),
        "resample_observation_masks_each_epoch": False,
        "observation_mask": {"kind": "none"},
        "dynamic_forcing_indices": [6, 7, 13, 14],
    }
    actual = {key: data_config.get(key) for key in expected_causal_contract}
    if actual != expected_causal_contract:
        raise ValueError(f"source causal data contract differs: {actual}")
    return {
        "metadata": reference,
        "data_config_sha256": data_hash,
        "normalization_sha256": _canonical_sha256(normalization),
        "causal_contract": expected_causal_contract,
    }


def _atomic_torch_save(payload: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_source_model(source: dict[str, Any], device: torch.device):
    _verify_source_files(source)
    root = Path(source["run_dir"])
    manifest = load_json(root / "coarse_cascade_manifest.json")
    if manifest.get("code_commit") != source["code_commit"]:
        raise ValueError(f"source commit mismatch for {source['label']}")
    expected_sampler = {
        "standardized": "CoarseStandardizedPersistenceResidualSampler",
        "unwhitened": "CoarsePersistenceResidualSampler",
        "absolute": "CoarseCascadeSampler",
    }[source["kind"]]
    if manifest.get("sampler") != expected_sampler:
        raise ValueError(f"source sampler mismatch for {source['label']}")
    config = TrainingConfig.from_dict(load_json(root / "config.json"))
    model = build_unet(config)
    state = torch.load(root / source["checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device=device, dtype=torch.float32).eval()
    if source["kind"] == "standardized":
        statistics_path = Path(manifest["residual_statistics_path"])
        expected = manifest["residual_statistics_sha256"]
        if not statistics_path.is_file() or _sha256(statistics_path) != expected:
            raise ValueError("standardized source statistics artifact differs")
        statistics = load_json(statistics_path)
        return CoarseStandardizedPersistenceResidualSampler(model, statistics), config
    if source["kind"] == "unwhitened":
        return CoarsePersistenceResidualSampler(model), config
    return CoarseCascadeSampler(model), config


def _weighted_case_rmse(
    estimate: torch.Tensor, truth: torch.Tensor, fraction: torch.Tensor
) -> tuple[float, list[float]]:
    if estimate.shape != truth.shape or estimate.ndim != 4 or estimate.shape[1] != 1:
        raise ValueError("case-equal RMSE requires matching [B,1,H,W] fields")
    weight = fraction.to(dtype=torch.float64)
    error = (estimate - truth).to(dtype=torch.float64)
    denominator = weight.sum(dim=(1, 2, 3))
    case_mse = (error.square() * weight).sum(dim=(1, 2, 3)) / denominator
    values = torch.sqrt(case_mse)
    return float(torch.sqrt(case_mse.mean()).item()), values.tolist()


def _weighted_case_fair_crps(
    members: torch.Tensor, truth: torch.Tensor, fraction: torch.Tensor
) -> tuple[float, list[float]]:
    if members.ndim != 5 or members.shape[2] != 1 or truth.shape != members[:, 0].shape:
        raise ValueError("fair CRPS requires [B,M,1,H,W] and [B,1,H,W]")
    count = members.shape[1]
    if count < 2:
        raise ValueError("fair CRPS requires at least two members")
    members = members.to(dtype=torch.float64)
    truth = truth.to(dtype=torch.float64)
    weight = fraction[:, None].to(dtype=torch.float64)
    accuracy = (members - truth[:, None]).abs().mean(dim=1)
    pair_sum = torch.zeros_like(accuracy)
    for first in range(count):
        for second in range(first + 1, count):
            pair_sum += (members[:, first] - members[:, second]).abs()
    pointwise = accuracy - pair_sum / (count * (count - 1))
    denominator = fraction.to(dtype=torch.float64).sum(dim=(1, 2, 3))
    values = (pointwise * weight[:, 0]).sum(dim=(1, 2, 3)) / denominator
    return float(values.mean().item()), values.tolist()


def _weighted_fractional_rank(
    members: torch.Tensor, truth: torch.Tensor, fraction: torch.Tensor
) -> dict[str, Any]:
    count = members.shape[1]
    less = (members < truth[:, None]).sum(dim=1)
    equal = (members == truth[:, None]).sum(dim=1)
    case_probabilities = []
    for case in range(members.shape[0]):
        mass = torch.zeros(count + 1, dtype=torch.float64)
        weight = fraction[case].to(dtype=torch.float64)
        for rank in range(count + 1):
            selected = (rank >= less[case]) & (rank <= less[case] + equal[case])
            contribution = torch.where(
                selected,
                weight / (equal[case].to(dtype=torch.float64) + 1.0),
                torch.zeros_like(weight),
            )
            mass[rank] = contribution.sum()
        case_probabilities.append(mass / mass.sum())
    probabilities = torch.stack(case_probabilities).mean(dim=0)
    uniform = torch.full_like(probabilities, 1.0 / (count + 1))
    return {
        "case_equal_fractional_rank_mass": (probabilities * members.shape[0]).tolist(),
        "case_equal_fractional_rank_frequencies": probabilities.tolist(),
        "rank_tv_to_uniform": float(0.5 * (probabilities - uniform).abs().sum().item()),
        "normalized_mean_rank": float(
            (probabilities * torch.arange(count + 1, dtype=torch.float64)).sum().item()
            / count
        ),
    }


def _weighted_case_rms(value: torch.Tensor, fraction: torch.Tensor) -> tuple[float, list[float]]:
    if value.ndim != 4 or value.shape[1] != 1:
        raise ValueError("weighted RMS requires [B,1,H,W]")
    value = value.to(dtype=torch.float64)
    weight = fraction.to(dtype=torch.float64)
    denominator = weight.sum(dim=(1, 2, 3))
    case_variance = (value.square() * weight).sum(dim=(1, 2, 3)) / denominator
    result = torch.sqrt(case_variance)
    return float(torch.sqrt(case_variance.mean()).item()), result.tolist()


def _roughness(value: torch.Tensor, active: torch.Tensor) -> float:
    if value.ndim != 5:
        raise ValueError("roughness requires [B,M,1,H,W]")
    results = []
    for case in range(value.shape[0]):
        mask = active[case, 0] > 0
        mask_y = mask[1:] & mask[:-1]
        mask_x = mask[:, 1:] & mask[:, :-1]
        for member in range(value.shape[1]):
            field = value[case, member, 0].to(dtype=torch.float64)
            differences = torch.cat(
                ((field[1:] - field[:-1]).abs()[mask_y], (field[:, 1:] - field[:, :-1]).abs()[mask_x])
            )
            results.append(differences.mean())
    return float(torch.stack(results).mean().item())


def _support_metrics(values: torch.Tensor, active: torch.Tensor, field: str) -> dict[str, Any]:
    per_case: dict[str, list[float]] = {
        "frequency": [],
        "mean_all": [],
        "p95_all": [],
        "max": [],
    }
    for case in range(values.shape[0]):
        selected = values[case][:, :, active[case, 0] > 0].to(dtype=torch.float64)
        if field == "sic":
            excess = torch.maximum((-selected).clamp_min(0), (selected - 1).clamp_min(0))
        elif field == "sit":
            excess = (-selected).clamp_min(0)
        else:
            raise ValueError(field)
        per_case["frequency"].append(float((excess > 0).double().mean().item()))
        per_case["mean_all"].append(float(excess.mean().item()))
        per_case["p95_all"].append(float(torch.quantile(excess.flatten(), 0.95).item()))
        per_case["max"].append(float(excess.max().item()))
    return {
        "case_equal_mean_frequency": float(
            sum(per_case["frequency"]) / len(per_case["frequency"])
        ),
        "case_equal_mean_excess": float(
            sum(per_case["mean_all"]) / len(per_case["mean_all"])
        ),
        "case_equal_mean_p95_excess": float(
            sum(per_case["p95_all"]) / len(per_case["p95_all"])
        ),
        "global_max_excess": max(per_case["max"]),
        "case_values": per_case,
    }


def _low_ice_sit(
    sit_members: torch.Tensor,
    truth_sic: torch.Tensor,
    active: torch.Tensor,
) -> dict[str, Any]:
    rms, positive_p95, negative_p95 = [], [], []
    for case in range(sit_members.shape[0]):
        selector = (active[case, 0] > 0) & (truth_sic[case, 0] < 0.01)
        if not torch.any(selector):
            continue
        selected = sit_members[case, :, 0][:, selector].to(dtype=torch.float64)
        rms.append(float(torch.sqrt(selected.square().mean()).item()))
        positive_p95.append(float(torch.quantile(selected.clamp_min(0).flatten(), 0.95).item()))
        negative_p95.append(float(torch.quantile((-selected).clamp_min(0).flatten(), 0.95).item()))
    if not rms:
        raise ValueError("paired evaluation has no low-ice cells")
    return {
        "case_equal_rms": math.sqrt(sum(value * value for value in rms) / len(rms)),
        "case_equal_mean_positive_p95": sum(positive_p95) / len(positive_p95),
        "case_equal_mean_negative_p95": sum(negative_p95) / len(negative_p95),
        "case_values": {
            "rms": rms,
            "positive_p95": positive_p95,
            "negative_p95": negative_p95,
        },
    }


def _joint_energy_score(
    ensemble: torch.Tensor,
    truth: torch.Tensor,
    fraction: torch.Tensor,
) -> tuple[float, list[float]]:
    """Fair joint energy score in one common six-channel normalized coordinate."""
    count = ensemble.shape[1]
    weight = fraction[:, None].to(dtype=torch.float64)
    denominator = fraction.sum(dim=(1, 2, 3)).to(dtype=torch.float64) * truth.shape[1]

    def norm(value: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((value.to(dtype=torch.float64).square() * weight).sum(dim=(2, 3, 4)) / denominator[:, None])

    accuracy = norm(ensemble - truth[:, None]).mean(dim=1)
    pair_sum = torch.zeros_like(accuracy)
    for first in range(count):
        for second in range(first + 1, count):
            pair_sum += norm(ensemble[:, first : first + 1] - ensemble[:, second : second + 1])[:, 0]
    score = accuracy - pair_sum / (count * (count - 1))
    return float(score.mean().item()), score.tolist()


def _joint_point_mass_energy(
    value: torch.Tensor, truth: torch.Tensor, fraction: torch.Tensor
) -> tuple[float, list[float]]:
    weight = fraction.to(dtype=torch.float64)
    error = (value - truth).to(dtype=torch.float64)
    denominator = weight.sum(dim=(1, 2, 3)) * truth.shape[1]
    result = torch.sqrt(
        (error.square() * weight).sum(dim=(1, 2, 3)) / denominator
    )
    return float(result.mean().item()), result.tolist()


def _require_finite_scalars(value: Any, path: str = "result") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _require_finite_scalars(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _require_finite_scalars(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError(f"non-finite scalar at {path}")


def score_ensemble(
    normalized: torch.Tensor,
    physical: torch.Tensor,
    truth_normalized: torch.Tensor,
    truth_physical: torch.Tensor,
    persistence_physical: torch.Tensor,
    active: torch.Tensor,
    fraction: torch.Tensor,
) -> dict[str, Any]:
    result: dict[str, Any] = {"outputs": {}, "support": {}}
    for lead_index, lead in enumerate(DIRECT_LEADS):
        for offset, field in enumerate(("sic", "sit")):
            channel = 2 * lead_index + offset
            key = f"d{lead}_{field}"
            members = physical[:, :, channel : channel + 1]
            truth = truth_physical[:, channel : channel + 1]
            persistence = persistence_physical[:, channel : channel + 1]
            mean_rmse, case_rmse = _weighted_case_rmse(members.mean(dim=1), truth, fraction)
            persistence_rmse, persistence_case_rmse = _weighted_case_rmse(
                persistence, truth, fraction
            )
            fair_crps, case_crps = _weighted_case_fair_crps(members, truth, fraction)
            persistence_mae = []
            for case in range(truth.shape[0]):
                weight = fraction[case].to(dtype=torch.float64)
                error = (persistence[case] - truth[case]).abs().to(dtype=torch.float64)
                persistence_mae.append(float((error * weight).sum().item() / weight.sum().item()))
            spread, case_spread = _weighted_case_rms(
                members.std(dim=1, unbiased=True), fraction
            )
            mean_field = members.mean(dim=1, keepdim=True)
            result["outputs"][key] = {
                "case_equal_ensemble_mean_rmse": mean_rmse,
                "case_rmse": case_rmse,
                "case_equal_persistence_rmse": persistence_rmse,
                "persistence_case_rmse": persistence_case_rmse,
                "skill_over_persistence": 1.0 - mean_rmse / persistence_rmse,
                "case_equal_fair_crps": fair_crps,
                "case_fair_crps": case_crps,
                "case_equal_persistence_point_mass_crps": sum(persistence_mae)
                / len(persistence_mae),
                "case_equal_spread": spread,
                "case_spread": case_spread,
                "spread_skill_ratio": spread / mean_rmse,
                **_weighted_fractional_rank(members, truth, fraction),
                "member_roughness": _roughness(members, active),
                "ensemble_mean_roughness": _roughness(mean_field, active),
                "anomaly_roughness": _roughness(members - mean_field, active),
                "truth_roughness": _roughness(truth[:, None], active),
                "raw_support": _support_metrics(members, active, field),
            }
            if field == "sit":
                result["outputs"][key]["low_ice_sit"] = _low_ice_sit(
                    members,
                    truth_physical[:, 2 * lead_index : 2 * lead_index + 1],
                    active,
                )
    for offset, field in enumerate(("sic", "sit")):
        result["support"][field] = _support_metrics(
            physical[:, :, offset::2],
            active,
            field,
        )
    energy, case_energy = _joint_energy_score(
        normalized, truth_normalized, fraction
    )
    result["joint_energy_score_common_normalized_coordinate"] = energy
    result["case_joint_energy_score"] = case_energy
    return result


def _save_contact_sheet(
    path: Path,
    label: str,
    case_id: str,
    ensemble: torch.Tensor,
    truth: torch.Tensor,
    persistence: torch.Tensor,
    active: torch.Tensor,
    limits: dict[str, tuple[float, float]],
) -> None:
    rows = [("truth", truth), ("persistence", persistence)] + [
        (f"member {member}", ensemble[member]) for member in range(ensemble.shape[0])
    ]
    figure, axes = plt.subplots(len(rows), 6, figsize=(18, 2.45 * len(rows)))
    ocean = active[0].numpy() > 0
    for row, (row_label, fields) in enumerate(rows):
        for channel in range(6):
            field = "sic" if channel % 2 == 0 else "sit"
            image = np.where(ocean, fields[channel].numpy(), np.nan)
            axes[row, channel].imshow(
                image,
                origin="upper",
                cmap="Blues" if field == "sic" else "viridis",
                vmin=limits[field][0],
                vmax=limits[field][1],
            )
            axes[row, channel].set_xticks([])
            axes[row, channel].set_yticks([])
            if row == 0:
                axes[row, channel].set_title(f"d{DIRECT_LEADS[channel // 2]} {field.upper()}")
            if channel == 0:
                axes[row, channel].set_ylabel(row_label)
    figure.suptitle(f"{label}: all raw members; {case_id}")
    figure.tight_layout(rect=(0, 0, 1, 0.99))
    figure.savefig(path, dpi=130)
    plt.close(figure)


def _launch_status(status: str, **details: Any) -> None:
    raw = os.environ.get("COARSE_PAIRED_EVAL_STATUS_PATH", "").strip()
    if not raw:
        return
    path = Path(raw)
    if not path.is_absolute() or path.name != "status.json":
        raise ValueError("COARSE_PAIRED_EVAL_STATUS_PATH must be an absolute status.json")
    _atomic_json(path, {"status": status, **details})


@torch.no_grad()
def _run_impl(
    config_path: Path, output: Path, lifecycle: dict[str, Any]
) -> dict[str, Any]:
    if torch.cuda.device_count() != 1:
        raise RuntimeError("paired coarse evaluation requires exactly one visible GPU")
    if os.environ.get("CLEARML_REQUIRE_ONLINE") != "1":
        raise RuntimeError("paired coarse evaluation requires online ClearML")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to reuse paired evaluation output: {output}")
    experiment = load_json(config_path)
    sources = experiment["sources"]
    if [source["label"] for source in sources] != list(SOURCE_LABELS):
        raise ValueError("paired evaluation requires the four preregistered sources")
    if int(experiment["members"]) != ENSEMBLE_SIZE or int(experiment["rk4_steps"]) != SAMPLE_STEPS:
        raise ValueError("paired evaluation requires eight members and RK4-33")
    for source in sources:
        _verify_source_files(source)
    source_contract = _verify_common_source_contract(sources)
    code_identity = _clean_code_identity(Path(__file__).resolve().parents[1])
    output.mkdir(parents=True, exist_ok=False)
    lifecycle["output"] = output
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")

    candidate_root = Path(sources[0]["run_dir"])
    candidate_metadata = source_contract["metadata"]
    dataset = build_dataset(candidate_metadata["data_config"], split="valid")
    dataset_validation = validate_direct_dataset(dataset)
    calendar_inventory = _calendar_inventory(dataset, split="valid")
    sentinel = load_json(candidate_root / "coarse_cascade_dataset_sentinel.json")
    subset = sentinel["pilot_subset"]
    indices = [
        subset["validation_indices"][position]
        for position in subset["diagnostic_subset_positions"]
    ]
    batch = _fine_collate([dataset[index] for index in indices])
    case_ids = list(batch["meta"]["case_id"])
    if case_ids != subset["diagnostic_case_ids"] or len(case_ids) != 4:
        raise ValueError("paired evaluation cases differ from frozen diagnostics")
    valid = batch["valid_mask"][:, :1].float()
    truth_normalized, active, fraction = coarse_target(batch["truth"], valid)
    persistence_normalized, persistence_active, persistence_fraction = (
        coarse_persistence_from_condition(batch["structured_conditioning"], valid)
    )
    if not torch.equal(active, persistence_active) or not torch.equal(
        fraction, persistence_fraction
    ):
        raise ValueError("paired truth and persistence supports differ")
    means = _repeat_field_stats(dataset.means)
    stds = _repeat_field_stats(dataset.stds)
    truth_physical = channel_denormalize(truth_normalized, means, stds)
    persistence_physical = channel_denormalize(persistence_normalized, means, stds)

    noise_members = []
    seeds = []
    for member in range(ENSEMBLE_SIZE):
        case_noise, case_seeds = [], []
        for case_id in case_ids:
            seed = _member_seed(case_id, member)
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            case_noise.append(
                torch.randn((1, 6, 160, 128), generator=generator, device=device)
            )
            case_seeds.append(seed)
        noise_members.append(torch.cat(case_noise, dim=0).cpu())
        seeds.append(case_seeds)
    noise = torch.stack(noise_members, dim=1)
    for source in sources:
        payload = torch.load(
            Path(source["run_dir"]) / source["samples_artifact"],
            map_location="cpu",
            weights_only=True,
        )
        if payload["validation_case_ids"] != case_ids:
            raise ValueError(f"saved case identities differ for {source['label']}")
        torch.testing.assert_close(
            payload["raw_coarse_noise_normalized"], noise[:, :2], rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            payload["coarse_truth_physical"], truth_physical, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            payload["coarse_persistence_physical"],
            persistence_physical,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            payload["coarse_active_mask"], active, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            payload["coarse_ocean_fraction"], fraction, rtol=0.0, atol=0.0
        )

    tracker = ClearMLTracker(
        experiment["project_name"],
        experiment["task_name"],
        tags=experiment["clearml"]["tags"],
        env_path=experiment["clearml"]["env_path"],
    )
    lifecycle["tracker"] = tracker
    contract = {
        "schema_version": "coarse_paired_evaluation_v1",
        "code_identity": code_identity,
        "split": "valid",
        "case_ids": case_ids,
        "dataset_indices": indices,
        "members": ENSEMBLE_SIZE,
        "rk4_steps": SAMPLE_STEPS,
        "network_and_ode_precision": "float32",
        "tf32": False,
        "optimizer_steps": 0,
        "raw_unclipped_scoring": True,
        "common_joint_energy_coordinate": "six normalized SIC/SIT channels",
        "member_noise_seeds": seeds,
        "sources": sources,
        "source_contract": {
            key: value for key, value in source_contract.items() if key != "metadata"
        },
        "dataset_validation": dataset_validation,
        "calendar_inventory": calendar_inventory,
    }
    tracker.connect("paired_evaluation_contract", contract)
    _launch_status(
        "sampling",
        code_commit=code_identity["git_commit"],
        output_dir=str(output),
        clearml_task_id=str(tracker.task.id),
    )
    _atomic_json(output / "contract.json", contract)
    _atomic_torch_save(
        {
            "noise": noise,
            "structured_conditioning": batch["structured_conditioning"].float(),
            "fine_valid_mask": batch["valid_mask"].float(),
            "sampler_valid_mask": valid,
            "truth_normalized": truth_normalized,
            "truth_physical": truth_physical,
            "persistence_normalized": persistence_normalized,
            "persistence_physical": persistence_physical,
            "active": active,
            "ocean_fraction": fraction,
            "case_ids": case_ids,
        },
        output / "fixed_inputs.pt",
    )

    ensembles: dict[str, dict[str, torch.Tensor]] = {}
    metrics: dict[str, Any] = {}
    condition = batch["structured_conditioning"].float().to(device)
    valid_device = valid.to(device)
    for source in sources:
        label = source["label"]
        sampler, training_config = _load_source_model(source, device)
        if tuple(training_config.image_size) != (160, 128):
            raise ValueError("paired source uses the wrong coarse image size")
        members = []
        for member in range(ENSEMBLE_SIZE):
            members.append(
                sampler.sample_conditioned(
                    structured_conditioning=condition,
                    valid_mask=valid_device,
                    initial_noise=noise[:, member].to(device),
                    num_timesteps=SAMPLE_STEPS,
                    device=device,
                    method="rk4",
                    end_time=0.0,
                ).cpu()
            )
        normalized = torch.stack(members, dim=1)
        physical = channel_denormalize(
            normalized.flatten(0, 1), means, stds
        ).unflatten(0, (len(case_ids), ENSEMBLE_SIZE))
        ensembles[label] = {"normalized": normalized, "physical": physical}
        _atomic_torch_save(
            {
                "normalized": normalized,
                "physical": physical,
                "fixed_inputs_sha256": _sha256(output / "fixed_inputs.pt"),
                "checkpoint_sha256": source["sha256"][source["checkpoint"]],
            },
            output / f"raw_{label}.pt",
        )
        metrics[label] = score_ensemble(
            normalized,
            physical,
            truth_normalized,
            truth_physical,
            persistence_physical,
            active,
            fraction,
        )
        del sampler
        torch.cuda.empty_cache()

    visual_dir = output / "matched_raw_panels"
    visual_dir.mkdir()
    for case, case_id in enumerate(case_ids):
        all_physical = torch.cat(
            [ensembles[label]["physical"][case] for label in SOURCE_LABELS], dim=0
        )
        limits = {
            "sic": (
                float(min(0.0, all_physical[:, 0::2].min().item())),
                float(max(1.0, all_physical[:, 0::2].max().item())),
            ),
            "sit": (
                float(min(0.0, all_physical[:, 1::2].min().item())),
                float(max(truth_physical[case, 1::2].max().item(), all_physical[:, 1::2].max().item())),
            ),
        }
        for label in SOURCE_LABELS:
            path = visual_dir / f"case{case:02d}_{label}_all8.png"
            _save_contact_sheet(
                path,
                label,
                case_id,
                ensembles[label]["physical"][case],
                truth_physical[case],
                persistence_physical[case],
                active[case],
                limits,
            )
            tracker.report_image("paired_raw_members", f"case{case:02d}/{label}", path, 0)
    result = {
        "status": "complete_pending_astra_and_visual_review",
        "publication_claim_permitted": False,
        "exploratory_checkpoint": "standardized_1024",
        "causal_whitening_comparison": "standardized_2048_vs_unwhitened_2048",
        "contract_sha256": _sha256(output / "contract.json"),
        "fixed_inputs_sha256": _sha256(output / "fixed_inputs.pt"),
        "metrics": metrics,
        "clearml_task_id": str(tracker.task.id),
    }
    persistence_energy, persistence_case_energy = _joint_point_mass_energy(
        persistence_normalized, truth_normalized, fraction
    )
    result["persistence_joint_energy_common_normalized_coordinate"] = persistence_energy
    result["persistence_case_joint_energy"] = persistence_case_energy
    for label, scored in metrics.items():
        for output_key, values in scored["outputs"].items():
            for metric in (
                "case_equal_ensemble_mean_rmse",
                "case_equal_fair_crps",
                "spread_skill_ratio",
                "rank_tv_to_uniform",
            ):
                tracker.report_scalar(f"paired/{metric}", f"{label}/{output_key}", values[metric], 0)
    _require_finite_scalars(result)
    _atomic_json(output / "paired_evaluation.json", result)
    tracker.upload_artifact("paired_evaluation", output / "paired_evaluation.json")
    tracker.close()
    lifecycle["tracker"] = None
    _launch_status(
        "complete_pending_astra_and_visual_review",
        code_commit=code_identity["git_commit"],
        output_dir=str(output),
        clearml_task_id=result["clearml_task_id"],
    )
    return result


def run(config_path: Path, output: Path) -> dict[str, Any]:
    lifecycle: dict[str, Any] = {"output": None, "tracker": None}
    try:
        return _run_impl(config_path, output, lifecycle)
    except BaseException as error:
        failure = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "cleanup_errors": [],
        }
        created = lifecycle.get("output")
        if isinstance(created, Path) and created.is_dir():
            try:
                _atomic_json(created / "failure.json", failure)
            except Exception as durability_error:
                failure["cleanup_errors"].append(
                    f"initial_failure_artifact: {durability_error}"
                )
        try:
            _launch_status(**failure)
        except Exception as durability_error:
            failure["cleanup_errors"].append(f"launch_status: {durability_error}")
        tracker = lifecycle.get("tracker")
        if tracker is not None:
            try:
                tracker.task.mark_failed(
                    status_reason=type(error).__name__, status_message=str(error)[:1000]
                )
            except Exception as cleanup_error:
                failure["cleanup_errors"].append(f"clearml_mark_failed: {cleanup_error}")
            try:
                tracker.close()
            except Exception as cleanup_error:
                failure["cleanup_errors"].append(f"clearml_close: {cleanup_error}")
        if isinstance(created, Path) and created.is_dir():
            try:
                _atomic_json(created / "failure.json", failure)
            except Exception:
                pass
        raise


def main() -> None:
    def terminate(signum, _frame):
        raise TimeoutError(f"paired coarse evaluation received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.config.resolve(), arguments.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
