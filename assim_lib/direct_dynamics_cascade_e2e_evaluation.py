"""Paired full-resolution evaluation of generated-coarse dynamics cascades."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from .clearml_tracking import ClearMLTracker
from .config import load_json
from .data import build_dataset
from .direct_dynamics_cascade import masked_block_average, smooth_right_inverse
from .direct_dynamics_cascade_checkpoint_evaluation import (
    _predicted_open_water_sit,
    _save_rank_histograms,
)
from .direct_dynamics_cascade_end_to_end import load_cascade_predictor
from .direct_dynamics_cascade_fine_training import _clean_code_identity, _fine_collate
from .direct_dynamics_cascade_paired_evaluation import (
    _require_finite_scalars,
    _save_contact_sheet,
    score_ensemble,
)
from .direct_dynamics_training import DIRECT_LEADS, _repeat_field_stats, validate_direct_dataset
from .trainer import _atomic_json
from .transforms import channel_denormalize


EXPECTED_FINE_LABELS = ("raw_fine_512", "ema_fine_512")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_inventory(root: Path, inventory: dict[str, str], label: str) -> None:
    for relative, expected in inventory.items():
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts or len(expected) != 64:
            raise ValueError(f"unsafe {label} inventory entry: {relative}")
        path = root / candidate
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"frozen {label} source mismatch: {relative}")


def _validate_experiment(experiment: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    if int(experiment.get("cases", 0)) != 12 or int(experiment.get("members", 0)) != 8:
        raise ValueError("cascade evaluation requires exactly 12 cases and 8 members")
    if int(experiment.get("coarse_rk4_timepoints", 0)) != 17:
        raise ValueError("cascade evaluation requires 17 coarse RK4 timepoints")
    if int(experiment.get("fine_rk4_timepoints", 0)) != 17:
        raise ValueError("cascade evaluation requires 17 fine RK4 timepoints")
    case_indices = experiment.get("case_indices")
    case_ids = experiment.get("case_ids")
    if (
        not isinstance(case_indices, list)
        or len(case_indices) != 12
        or len(set(case_indices)) != 12
        or not all(isinstance(value, int) and value >= 0 for value in case_indices)
        or not isinstance(case_ids, list)
        or len(case_ids) != 12
        or len(set(case_ids)) != 12
        or not all(isinstance(value, str) and value for value in case_ids)
    ):
        raise ValueError("cascade evaluation requires twelve frozen unique cases")
    coarse = experiment.get("coarse")
    if not isinstance(coarse, dict) or coarse.get("checkpoint") != "ema_coarse_update_9711.pth":
        raise ValueError("cascade evaluation requires the frozen EMA9711 coarse source")
    fine = experiment.get("fine_checkpoints")
    if not isinstance(fine, list) or [row.get("label") for row in fine] != list(EXPECTED_FINE_LABELS):
        raise ValueError(f"fine checkpoints must be ordered as {EXPECTED_FINE_LABELS}")
    for row in fine:
        if not isinstance(row.get("checkpoint"), str) or len(str(row.get("sha256", ""))) != 64:
            raise ValueError("each fine checkpoint requires a path and SHA256")
    return coarse, fine


def _launch_status(status: str, **details: Any) -> None:
    raw = os.environ.get("CASCADE_E2E_STATUS_PATH", "").strip()
    if not raw:
        return
    path = Path(raw)
    if not path.is_absolute() or path.name != "status.json":
        raise ValueError("CASCADE_E2E_STATUS_PATH must be an absolute status.json")
    _atomic_json(path, {"status": status, **details})


def _coarse_lift_ensemble(
    coarse: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, float]:
    if coarse.ndim != 5 or valid.ndim != 4:
        raise ValueError("coarse lift expects [B,M,C,h,w] and [B,1,H,W]")
    batch, members, channels, _, _ = coarse.shape
    mask = valid[:, None].expand(-1, members, -1, -1, -1).flatten(0, 1)
    lifted = smooth_right_inverse(coarse.flatten(0, 1), mask).unflatten(0, (batch, members))
    recovered, fraction = masked_block_average(lifted.flatten(0, 1), mask)
    active = fraction > 0
    error = float((recovered - coarse.flatten(0, 1))[active.expand_as(recovered)].abs().max())
    if error > 3e-6:
        raise RuntimeError("coarse-only lift does not preserve its 2x2 means")
    if lifted.shape[2] != channels:
        raise RuntimeError("coarse lift changed the trajectory channel count")
    return lifted, error


def _add_joint_consistency(metrics: dict[str, Any], physical: torch.Tensor, valid: torch.Tensor) -> None:
    for lead_index, lead in enumerate(DIRECT_LEADS):
        metrics["outputs"][f"d{lead}_sit"]["predicted_open_water_sit"] = (
            _predicted_open_water_sit(
                physical[:, :, 2 * lead_index : 2 * lead_index + 1],
                physical[:, :, 2 * lead_index + 1 : 2 * lead_index + 2],
                valid,
            )
        )


def _summary(metrics: dict[str, Any]) -> dict[str, float]:
    outputs = list(metrics["outputs"].values())
    return {
        "mean_rmse_skill_over_persistence": float(
            sum(row["skill_over_persistence"] for row in outputs)
            / len(outputs)
        ),
        "mean_fair_crps_ratio_to_persistence": float(
            sum(
                row["case_equal_fair_crps"]
                / row["case_equal_persistence_point_mass_crps"]
                for row in outputs
            )
            / len(outputs)
        ),
        "mean_spread_skill_ratio": float(sum(row["spread_skill_ratio"] for row in outputs) / len(outputs)),
        "mean_rank_tv_to_uniform": float(sum(row["rank_tv_to_uniform"] for row in outputs) / len(outputs)),
        "mean_normalized_rank": float(sum(row["normalized_mean_rank"] for row in outputs) / len(outputs)),
        "max_member_to_truth_roughness_ratio": float(
            max(row["member_roughness"] / row["truth_roughness"] for row in outputs)
        ),
        "joint_energy_score": float(metrics["joint_energy_score_common_normalized_coordinate"]),
    }


@torch.no_grad()
def run(config_path: Path, output: Path) -> dict[str, Any]:
    if torch.cuda.device_count() != 1:
        raise RuntimeError("cascade evaluation requires exactly one visible GPU")
    if os.environ.get("CLEARML_REQUIRE_ONLINE") != "1":
        raise RuntimeError("cascade evaluation requires online ClearML")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to reuse output {output}")
    experiment = load_json(config_path)
    coarse_spec, fine_specs = _validate_experiment(experiment)
    coarse_root = Path(coarse_spec["run_dir"])
    fine_root = Path(experiment["fine_run_dir"])
    _verify_inventory(coarse_root, coarse_spec["sha256"], "coarse")
    _verify_inventory(fine_root, experiment["fine_sha256"], "fine")
    for row in fine_specs:
        path = fine_root / row["checkpoint"]
        if not path.is_file() or _sha256(path) != row["sha256"]:
            raise ValueError(f"frozen fine checkpoint mismatch: {row['label']}")

    coarse_config = load_json(coarse_root / "config.json")
    fine_config = load_json(fine_root / "config.json")
    coarse_metadata = load_json(coarse_root / "metadata.json")
    fine_metadata = load_json(fine_root / "metadata.json")
    if coarse_metadata["data_config"] != fine_metadata["data_config"]:
        raise ValueError("coarse and fine checkpoints use different data laws")
    dataset = build_dataset(fine_metadata["data_config"], split="valid")
    validate_direct_dataset(dataset)
    indices = list(experiment["case_indices"])
    batch = _fine_collate([dataset[index] for index in indices])
    case_ids = tuple(batch["meta"]["case_id"])
    if list(case_ids) != experiment["case_ids"]:
        raise ValueError("frozen case identities differ from the current archive")

    valid = batch["valid_mask"][:, :1].float()
    truth_normalized = batch["truth"].float()
    persistence_normalized = batch["background"].float()
    means = _repeat_field_stats(dataset.means)
    stds = _repeat_field_stats(dataset.stds)
    truth_physical = channel_denormalize(truth_normalized, means, stds)
    persistence_physical = channel_denormalize(persistence_normalized, means, stds)
    device = torch.device("cuda:0")
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    condition_device = batch["structured_conditioning"].float().to(device)
    valid_device = valid.to(device)

    output.mkdir(parents=True, exist_ok=False)
    visuals = output / "fixed_scale_members"
    ranks = output / "rank_histograms"
    visuals.mkdir()
    ranks.mkdir()
    code_identity = _clean_code_identity(Path(__file__).resolve().parents[1])
    contract = {
        "schema_version": "generated_coarse_fine_e2e_evaluation_v1",
        "code_identity": code_identity,
        "coarse": coarse_spec,
        "fine_run_dir": str(fine_root),
        "fine_sha256": experiment["fine_sha256"],
        "fine_checkpoints": fine_specs,
        "split": "valid",
        "case_indices": indices,
        "case_ids": list(case_ids),
        "members": int(experiment["members"]),
        "coarse_rk4_timepoints": int(experiment["coarse_rk4_timepoints"]),
        "fine_rk4_timepoints": int(experiment["fine_rk4_timepoints"]),
        "raw_unclipped_scoring": True,
        "display_only_projection": True,
        "tf32": False,
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
        "sampling", output_dir=str(output), code_commit=code_identity["git_commit"],
        clearml_task_id=str(tracker.task.id),
    )
    result: dict[str, Any] = {
        "status": "complete_pending_visual_review",
        "publication_claim_permitted": False,
        "clearml_task_id": str(tracker.task.id),
        "candidates": {},
    }
    coarse_ensemble = None
    for fine_spec in fine_specs:
        label = fine_spec["label"]
        predictor = load_cascade_predictor(
            coarse_run_dir=str(coarse_root),
            coarse_checkpoint_name=coarse_spec["checkpoint"],
            coarse_model_config=coarse_config,
            coarse_checkpoint_sha256=coarse_spec["sha256"][coarse_spec["checkpoint"]],
            fine_run_dir=str(fine_root),
            fine_checkpoint_name=fine_spec["checkpoint"],
            fine_model_config=fine_config,
            fine_checkpoint_sha256=fine_spec["sha256"],
            expected_coarse_code_commit=coarse_spec["code_commit"],
            expected_fine_code_commit=experiment["fine_code_commit"],
            replay_code_commit=code_identity["git_commit"],
            expected_forecast_contract_sha256=experiment["forecast_contract_sha256"],
            device=device,
        )
        ensemble = predictor.sample_ensemble(
            member_indices=tuple(range(int(experiment["members"]))),
            structured_conditioning=condition_device,
            valid_mask=valid_device,
            case_ids=case_ids,
            coarse_num_timesteps=int(experiment["coarse_rk4_timepoints"]),
            fine_num_timesteps=int(experiment["fine_rk4_timepoints"]),
            device=device,
            method="rk4",
            rtol=1e-5,
            atol=1e-6,
            end_time=0.0,
        )
        normalized = ensemble["forecast"].cpu()
        coarse = ensemble["coarse"].cpu()
        if coarse_ensemble is None:
            coarse_ensemble = coarse
        elif not torch.equal(coarse_ensemble, coarse):
            raise RuntimeError("paired fine candidates did not reuse identical coarse members")
        recovered, fraction = masked_block_average(
            normalized.flatten(0, 1),
            valid[:, None].expand(-1, normalized.shape[1], -1, -1, -1).flatten(0, 1),
        )
        exact_error = float((recovered - coarse.flatten(0, 1))[fraction.expand_as(recovered) > 0].abs().max())
        if exact_error > 3e-6:
            raise RuntimeError("fine forecast changed generated coarse 2x2 means")
        physical = channel_denormalize(normalized.flatten(0, 1), means, stds).unflatten(
            0, (len(case_ids), int(experiment["members"]))
        )
        metrics = score_ensemble(
            normalized, physical, truth_normalized, truth_physical,
            persistence_physical, valid, valid,
        )
        _add_joint_consistency(metrics, physical, valid)
        metrics["exact_coarse_2x2_mean_error_max"] = exact_error
        metrics["summary"] = _summary(metrics)
        _require_finite_scalars(metrics)
        rank_path = ranks / f"{label}_rank_histograms.png"
        _save_rank_histograms(rank_path, label, metrics)
        tracker.report_image("rank_histograms", label, rank_path, 0)
        for visual_index, position in enumerate((0, 4, 7, 11)):
            path = visuals / f"{label}_case{visual_index:02d}_all8_fixed.png"
            _save_contact_sheet(
                path, label, case_ids[position], physical[position], truth_physical[position],
                persistence_physical[position], valid[position], {"sic": (0.0, 1.0), "sit": (0.0, 3.5)},
            )
            tracker.report_image("fixed_scale_members", f"{label}/case{visual_index:02d}", path, 0)
        result["candidates"][label] = {"metrics": metrics}
        del predictor, ensemble, normalized, physical
        torch.cuda.empty_cache()

    if coarse_ensemble is None:
        raise RuntimeError("cascade evaluation produced no coarse ensemble")
    lifted, lift_error = _coarse_lift_ensemble(coarse_ensemble, valid)
    lifted_physical = channel_denormalize(lifted.flatten(0, 1), means, stds).unflatten(
        0, (len(case_ids), int(experiment["members"]))
    )
    coarse_metrics = score_ensemble(
        lifted, lifted_physical, truth_normalized, truth_physical,
        persistence_physical, valid, valid,
    )
    _add_joint_consistency(coarse_metrics, lifted_physical, valid)
    coarse_metrics["exact_coarse_2x2_mean_error_max"] = lift_error
    coarse_metrics["summary"] = _summary(coarse_metrics)
    _require_finite_scalars(coarse_metrics)
    result["candidates"]["coarse_ema9711_lift"] = {"metrics": coarse_metrics}
    rank_path = ranks / "coarse_ema9711_lift_rank_histograms.png"
    _save_rank_histograms(rank_path, "coarse_ema9711_lift", coarse_metrics)
    tracker.report_image("rank_histograms", "coarse_ema9711_lift", rank_path, 0)
    _require_finite_scalars(result)
    _atomic_json(output / "cascade_e2e_evaluation.json", result)
    tracker.upload_artifact("cascade_e2e_evaluation", output / "cascade_e2e_evaluation.json")
    tracker.close()
    _launch_status(
        "complete_pending_visual_review", output_dir=str(output),
        code_commit=code_identity["git_commit"], clearml_task_id=result["clearml_task_id"],
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        result = run(arguments.config.resolve(), arguments.output.resolve())
    except BaseException as error:
        try:
            _launch_status("failed", error_type=type(error).__name__, error=str(error)[:1000])
        except Exception:
            pass
        raise
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
