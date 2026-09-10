"""Frozen paired raw/EMA evaluation for the full coarse dynamics run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from .clearml_tracking import ClearMLTracker
from .config import load_json
from .data import build_dataset
from .direct_dynamics_cascade_coarse import (
    coarse_target,
    load_coarse_cascade_sampler,
)
from .direct_dynamics_cascade_fine_training import _clean_code_identity, _fine_collate
from .direct_dynamics_cascade_paired_evaluation import (
    _member_seed,
    _require_finite_scalars,
    _save_contact_sheet,
    score_ensemble,
)
from .direct_dynamics_training import DIRECT_LEADS, _repeat_field_stats, validate_direct_dataset
from .trainer import _atomic_json
from .transforms import channel_denormalize


EXPECTED_LABELS = ("raw_9711", "ema_9711", "raw_19422", "ema_19422")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _even_positions(length: int, count: int) -> list[int]:
    if count < 2 or length < count:
        raise ValueError("seasonal selection requires at least two distinct cases")
    result = [round(index * (length - 1) / (count - 1)) for index in range(count)]
    if len(set(result)) != count:
        raise RuntimeError("seasonal selection produced duplicate positions")
    return result


def _validate_experiment(experiment: dict[str, Any]) -> list[dict[str, str]]:
    checkpoints = experiment.get("checkpoints")
    if not isinstance(checkpoints, list) or [item.get("label") for item in checkpoints] != list(
        EXPECTED_LABELS
    ):
        raise ValueError(f"checkpoints must be ordered as {EXPECTED_LABELS}")
    if int(experiment.get("cases", 0)) != 12 or int(experiment.get("members", 0)) != 8:
        raise ValueError("paired checkpoint evaluation requires 12 cases and 8 members")
    if int(experiment.get("rk4_timepoints", 0)) != 17:
        raise ValueError("paired checkpoint evaluation requires RK4 with 17 timepoints")
    for item in checkpoints:
        if not isinstance(item.get("checkpoint"), str) or not isinstance(item.get("sha256"), str):
            raise ValueError("each checkpoint requires an explicit path and SHA256")
        expected_ema = item["label"].startswith("ema_")
        if item["checkpoint"].startswith("ema_") != expected_ema:
            raise ValueError("checkpoint filename and raw/EMA label disagree")
    return checkpoints


def _predicted_open_water_sit(
    sic: torch.Tensor,
    sit: torch.Tensor,
    fraction: torch.Tensor,
    *,
    sic_threshold: float = 0.01,
    material_sit_threshold: float = 0.01,
) -> dict[str, Any]:
    """Measure positive SIT where the same generated member predicts open water."""
    if sic.shape != sit.shape or sic.ndim != 5 or fraction.shape != sic[:, 0].shape:
        raise ValueError("open-water consistency expects [B,M,1,H,W] members and [B,1,H,W] weights")
    case_values = []
    for case in range(sic.shape[0]):
        weight = fraction[case : case + 1].expand(sic.shape[1], -1, -1, -1).to(torch.float64)
        selected = (sic[case] < sic_threshold) & (weight > 0)
        positive = sit[case].to(torch.float64).clamp_min(0)
        selected_weight = torch.where(selected, weight, torch.zeros_like(weight))
        denominator = selected_weight.sum().clamp_min(1.0)
        values = positive[selected]
        case_values.append(
            {
                "weighted_mean_positive_sit": float((positive * selected_weight).sum().item() / denominator.item()),
                "positive_sit_p95": float(torch.quantile(values, 0.95).item()) if values.numel() else 0.0,
                "positive_sit_max": float(values.max().item()) if values.numel() else 0.0,
                "weighted_fraction_above_0p01m": float(
                    (((positive > material_sit_threshold) & selected) * weight).sum().item()
                    / denominator.item()
                ),
            }
        )
    return {
        "sic_threshold": sic_threshold,
        "material_sit_threshold_m": material_sit_threshold,
        "case_equal_mean_positive_sit": float(
            sum(item["weighted_mean_positive_sit"] for item in case_values) / len(case_values)
        ),
        "case_equal_mean_positive_sit_p95": float(
            sum(item["positive_sit_p95"] for item in case_values) / len(case_values)
        ),
        "global_positive_sit_max": max(item["positive_sit_max"] for item in case_values),
        "case_equal_fraction_above_0p01m": float(
            sum(item["weighted_fraction_above_0p01m"] for item in case_values) / len(case_values)
        ),
        "case_values": case_values,
    }


def _save_rank_histograms(path: Path, label: str, metrics: dict[str, Any]) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(12, 6), sharey=True)
    for index, (lead, field) in enumerate(
        ([(lead, field) for lead in DIRECT_LEADS for field in ("sic", "sit")])
    ):
        values = metrics["outputs"][f"d{lead}_{field}"]["case_equal_fractional_rank_frequencies"]
        axis = axes[index % 2, index // 2]
        axis.bar(np.arange(len(values)), values, color="#276dc3")
        axis.axhline(1.0 / len(values), color="#e67e22", linestyle="--", linewidth=1.5)
        axis.set_title(f"d+{lead} {field.upper()}")
        axis.set_xlabel("rank")
        if index % 2 == 0:
            axis.set_ylabel("frequency")
    figure.suptitle(f"{label}: fractional rank histograms, 8 members")
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _launch_status(status: str, **details: Any) -> None:
    raw = os.environ.get("COARSE_CHECKPOINT_EVAL_STATUS_PATH", "").strip()
    if not raw:
        return
    path = Path(raw)
    if not path.is_absolute() or path.name != "status.json":
        raise ValueError("COARSE_CHECKPOINT_EVAL_STATUS_PATH must be an absolute status.json")
    _atomic_json(path, {"status": status, **details})


@torch.no_grad()
def run(config_path: Path, output: Path) -> dict[str, Any]:
    if torch.cuda.device_count() != 1:
        raise RuntimeError("paired checkpoint evaluation requires exactly one visible GPU")
    if os.environ.get("CLEARML_REQUIRE_ONLINE") != "1":
        raise RuntimeError("paired checkpoint evaluation requires online ClearML")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to reuse output {output}")
    experiment = load_json(config_path)
    checkpoints = _validate_experiment(experiment)
    source = Path(experiment["source_run"])
    required = experiment["source_sha256"]
    for relative, expected in required.items():
        path = source / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"frozen source mismatch: {relative}")
    for item in checkpoints:
        path = source / item["checkpoint"]
        if not path.is_file() or _sha256(path) != item["sha256"]:
            raise ValueError(f"frozen checkpoint mismatch: {item['label']}")

    manifest = load_json(source / "coarse_cascade_manifest.json")
    if manifest.get("code_commit") != experiment["source_code_commit"]:
        raise ValueError("source code commit differs from evaluation contract")
    model_config = load_json(source / "config.json")
    metadata = load_json(source / "metadata.json")
    dataset = build_dataset(metadata["data_config"], split="valid")
    validate_direct_dataset(dataset)
    sentinel = load_json(source / "coarse_cascade_dataset_sentinel.json")["pilot_subset"]
    pool = sentinel["validation_indices"]
    positions = _even_positions(len(pool), int(experiment["cases"]))
    indices = [pool[position] for position in positions]
    batch = _fine_collate([dataset[index] for index in indices])
    case_ids = list(batch["meta"]["case_id"])
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("validation case identities are not unique")

    valid = batch["valid_mask"][:, :1].float()
    truth_normalized, active, fraction = coarse_target(batch["truth"], valid)
    persistence_normalized, persistence_active, persistence_fraction = coarse_target(
        batch["background"], valid
    )
    if not torch.equal(active, persistence_active) or not torch.equal(fraction, persistence_fraction):
        raise ValueError("truth and persistence coarse support differ")
    means, stds = _repeat_field_stats(dataset.means), _repeat_field_stats(dataset.stds)
    truth_physical = channel_denormalize(truth_normalized, means, stds)
    persistence_physical = channel_denormalize(persistence_normalized, means, stds)

    device = torch.device("cuda:0")
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    condition_device, valid_device = batch["structured_conditioning"].float().to(device), valid.to(device)
    noise_members, seeds = [], []
    for member in range(int(experiment["members"])):
        member_noise, member_seeds = [], []
        for case_id in case_ids:
            seed = _member_seed(case_id, member)
            generator = torch.Generator(device=device).manual_seed(seed)
            member_noise.append(torch.randn((1, 6, 160, 128), generator=generator, device=device))
            member_seeds.append(seed)
        noise_members.append(torch.cat(member_noise).cpu())
        seeds.append(member_seeds)
    noise = torch.stack(noise_members, dim=1)

    output.mkdir(parents=True, exist_ok=False)
    code_identity = _clean_code_identity(Path(__file__).resolve().parents[1])
    contract = {
        "schema_version": "coarse_raw_ema_paired_evaluation_v1",
        "code_identity": code_identity,
        "source_run": str(source),
        "source_code_commit": experiment["source_code_commit"],
        "source_sha256": required,
        "checkpoints": checkpoints,
        "split": "valid",
        "case_indices": indices,
        "case_ids": case_ids,
        "members": int(experiment["members"]),
        "rk4_timepoints": int(experiment["rk4_timepoints"]),
        "raw_unclipped_scoring": True,
        "network_and_ode_precision": "float32",
        "tf32": False,
        "fixed_display_limits": {"sic": [0.0, 1.0], "sit_m": [0.0, 3.5]},
        "member_noise_seeds": seeds,
        "optimizer_steps": 0,
    }
    _atomic_json(output / "contract.json", contract)
    tracker = ClearMLTracker(
        experiment["project_name"],
        experiment["task_name"],
        tags=experiment["clearml"]["tags"],
        env_path=experiment["clearml"]["env_path"],
    )
    tracker.connect("evaluation_contract", contract)
    _launch_status(
        "sampling",
        output_dir=str(output),
        code_commit=code_identity["git_commit"],
        clearml_task_id=str(tracker.task.id),
    )

    visual_positions = _even_positions(len(case_ids), 4)
    result: dict[str, Any] = {
        "status": "complete_pending_visual_review",
        "publication_claim_permitted": False,
        "contract_sha256": _sha256(output / "contract.json"),
        "checkpoints": {},
        "clearml_task_id": str(tracker.task.id),
    }
    visual_root = output / "fixed_scale_members"
    visual_root.mkdir()
    rank_root = output / "rank_histograms"
    rank_root.mkdir()
    for checkpoint in checkpoints:
        label = checkpoint["label"]
        sampler = load_coarse_cascade_sampler(
            str(source),
            checkpoint["checkpoint"],
            model_config,
            checkpoint["sha256"],
            experiment["source_code_commit"],
            manifest["forecast_contract_sha256"],
            device=device,
        )
        members = []
        for member in range(int(experiment["members"])):
            members.append(
                sampler.sample_conditioned(
                    structured_conditioning=condition_device,
                    valid_mask=valid_device,
                    initial_noise=noise[:, member].to(device),
                    num_timesteps=int(experiment["rk4_timepoints"]),
                    device=device,
                    method="rk4",
                    end_time=0.0,
                ).cpu()
            )
        normalized = torch.stack(members, dim=1)
        physical = channel_denormalize(normalized.flatten(0, 1), means, stds).unflatten(
            0, (len(case_ids), int(experiment["members"]))
        )
        metrics = score_ensemble(
            normalized,
            physical,
            truth_normalized,
            truth_physical,
            persistence_physical,
            active,
            fraction,
        )
        for lead_index, lead in enumerate(DIRECT_LEADS):
            metrics["outputs"][f"d{lead}_sit"]["predicted_open_water_sit"] = (
                _predicted_open_water_sit(
                    physical[:, :, 2 * lead_index : 2 * lead_index + 1],
                    physical[:, :, 2 * lead_index + 1 : 2 * lead_index + 2],
                    fraction,
                )
            )
        _require_finite_scalars(metrics)
        sample_path = output / f"{label}_fixed_visual_samples.pt"
        torch.save(
            {
                "case_ids": [case_ids[position] for position in visual_positions],
                "ensemble_physical": physical[visual_positions],
                "truth_physical": truth_physical[visual_positions],
                "persistence_physical": persistence_physical[visual_positions],
                "active": active[visual_positions],
                "checkpoint_sha256": checkpoint["sha256"],
            },
            sample_path,
        )
        for visual_index, position in enumerate(visual_positions):
            path = visual_root / f"{label}_case{visual_index:02d}_all8_fixed.png"
            _save_contact_sheet(
                path,
                label,
                case_ids[position],
                physical[position],
                truth_physical[position],
                persistence_physical[position],
                active[position],
                {"sic": (0.0, 1.0), "sit": (0.0, 3.5)},
            )
            tracker.report_image("fixed_scale_members", f"{label}/case{visual_index:02d}", path, 0)
        rank_path = rank_root / f"{label}_rank_histograms.png"
        _save_rank_histograms(rank_path, label, metrics)
        tracker.report_image("rank_histograms", label, rank_path, 0)
        result["checkpoints"][label] = {
            "checkpoint": checkpoint["checkpoint"],
            "checkpoint_sha256": checkpoint["sha256"],
            "visual_samples_sha256": _sha256(sample_path),
            "metrics": metrics,
        }
        for output_key, values in metrics["outputs"].items():
            for metric in (
                "case_equal_ensemble_mean_rmse",
                "case_equal_fair_crps",
                "spread_skill_ratio",
                "rank_tv_to_uniform",
            ):
                tracker.report_scalar(f"paired/{metric}", f"{label}/{output_key}", values[metric], 0)
        del sampler, normalized, physical, members
        torch.cuda.empty_cache()

    _require_finite_scalars(result)
    _atomic_json(output / "paired_raw_ema_evaluation.json", result)
    tracker.upload_artifact("paired_raw_ema_evaluation", output / "paired_raw_ema_evaluation.json")
    tracker.close()
    _launch_status(
        "complete_pending_visual_review",
        output_dir=str(output),
        code_commit=code_identity["git_commit"],
        clearml_task_id=result["clearml_task_id"],
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = run(args.config.resolve(), args.output.resolve())
    except BaseException as error:
        try:
            _launch_status(
                "failed",
                error_type=type(error).__name__,
                error=str(error),
            )
        except Exception:
            pass
        raise
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
