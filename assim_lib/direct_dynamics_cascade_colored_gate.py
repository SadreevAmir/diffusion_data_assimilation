"""Frozen generated-coarse admission gate for the colored fine reference law.

The gate compares the raw update-512 white and colored fine checkpoints with
common random numbers on validation-2022.  RK4-33 is primary and RK4-65 is a
solver-sensitivity replication.  It never selects an EMA checkpoint and never
touches the sealed test split.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from .clearml_tracking import ClearMLTracker
from .config import load_json
from .data import build_dataset
from .direct_dynamics_cascade_checkpoint_evaluation import (
    _predicted_open_water_sit,
    _save_rank_histograms,
)
from .direct_dynamics_cascade_e2e_evaluation import (
    _sha256,
    _summary,
    _verify_inventory,
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


EXPECTED_LABELS = ("white_raw512", "colored_raw512")
EXPECTED_SOLVERS = (33, 65)


def _validate_experiment(experiment: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if experiment.get("split") != "valid" or experiment.get("year") != 2022:
        raise ValueError("colored gate is restricted to validation-2022")
    if int(experiment.get("cases", 0)) != 12 or int(experiment.get("members", 0)) != 8:
        raise ValueError("colored gate requires exactly 12 cases and 8 members")
    if tuple(experiment.get("rk4_timepoints", ())) != EXPECTED_SOLVERS:
        raise ValueError("colored gate requires ordered RK4-33 and RK4-65")
    indices = experiment.get("case_indices")
    case_ids = experiment.get("case_ids")
    if (
        not isinstance(indices, list)
        or len(indices) != 12
        or len(set(indices)) != 12
        or not all(isinstance(value, int) and value >= 0 for value in indices)
        or not isinstance(case_ids, list)
        or len(case_ids) != 12
        or len(set(case_ids)) != 12
        or not all(isinstance(value, str) and value for value in case_ids)
    ):
        raise ValueError("colored gate requires twelve frozen unique validation cases")
    coarse = experiment.get("coarse")
    if not isinstance(coarse, dict) or coarse.get("checkpoint") != "ema_coarse_update_9711.pth":
        raise ValueError("colored gate requires the frozen EMA9711 coarse checkpoint")
    candidates = experiment.get("fine_candidates")
    if not isinstance(candidates, list) or tuple(row.get("label") for row in candidates) != EXPECTED_LABELS:
        raise ValueError(f"fine candidates must be ordered as {EXPECTED_LABELS}")
    for row in candidates:
        if row.get("checkpoint") != "mechanics_update_0512.pth":
            raise ValueError("colored gate is raw-update-512 only")
        if len(str(row.get("checkpoint_sha256", ""))) != 64:
            raise ValueError("each fine candidate requires a checkpoint SHA256")
        if not isinstance(row.get("run_sha256"), dict):
            raise ValueError("each fine candidate requires a frozen run inventory")
    gate = experiment.get("gate")
    expected_gate = {
        "primary_standardized_crps_ratio_max": 1.01,
        "per_output_crps_ratio_max": 1.01,
        "per_output_rmse_ratio_max": 1.01,
        "per_output_rank_tv_increase_max": 0.01,
        "per_output_adjusted_ssr_error_increase_max": 0.02,
    }
    if gate != expected_gate:
        raise ValueError("colored gate thresholds differ from the frozen scientific review")
    return coarse, candidates


def _launch_status(status: str, **details: Any) -> None:
    raw = os.environ.get("CASCADE_COLORED_GATE_STATUS_PATH", "").strip()
    if not raw:
        return
    path = Path(raw)
    if not path.is_absolute() or path.name != "status.json":
        raise ValueError("CASCADE_COLORED_GATE_STATUS_PATH must be an absolute status.json")
    _atomic_json(path, {"status": status, **details})


def _add_joint_consistency(metrics: dict[str, Any], physical: torch.Tensor, valid: torch.Tensor) -> None:
    for lead_index, lead in enumerate(DIRECT_LEADS):
        metrics["outputs"][f"d{lead}_sit"]["predicted_open_water_sit"] = (
            _predicted_open_water_sit(
                physical[:, :, 2 * lead_index : 2 * lead_index + 1],
                physical[:, :, 2 * lead_index + 1 : 2 * lead_index + 2],
                valid,
            )
        )


def _candidate_summary(metrics: dict[str, Any], field_stds: tuple[float, ...]) -> dict[str, Any]:
    outputs = metrics["outputs"]
    per_output: dict[str, Any] = {}
    standardized_crps = []
    for channel, (lead, field) in enumerate(
        ([(lead, field) for lead in DIRECT_LEADS for field in ("sic", "sit")])
    ):
        key = f"d{lead}_{field}"
        row = outputs[key]
        std = float(field_stds[channel])
        if not math.isfinite(std) or std <= 0:
            raise ValueError(f"invalid frozen train standard deviation for {key}")
        adjusted_ssr = math.sqrt(9.0 / 8.0) * float(row["spread_skill_ratio"])
        standardized = float(row["case_equal_fair_crps"]) / std
        standardized_crps.append(standardized)
        per_output[key] = {
            "standardized_fair_crps": standardized,
            "fair_crps": float(row["case_equal_fair_crps"]),
            "rmse": float(row["case_equal_ensemble_mean_rmse"]),
            "rank_tv": float(row["rank_tv_to_uniform"]),
            "adjusted_spread_skill_ratio": adjusted_ssr,
            "adjusted_spread_skill_error": abs(adjusted_ssr - 1.0),
        }
    return {
        "primary_mean_standardized_fair_crps": sum(standardized_crps) / len(standardized_crps),
        "per_output": per_output,
        "descriptive": _summary(metrics),
    }


def _positive_ratio(candidate: float, control: float, name: str) -> float:
    if not all(math.isfinite(value) and value > 0 for value in (candidate, control)):
        raise ValueError(f"{name} has a non-positive or non-finite denominator/value")
    return candidate / control


def _compare(
    control: dict[str, Any], candidate: dict[str, Any], thresholds: dict[str, float]
) -> dict[str, Any]:
    primary_ratio = _positive_ratio(
        candidate["primary_mean_standardized_fair_crps"],
        control["primary_mean_standardized_fair_crps"],
        "primary standardized fair CRPS",
    )
    checks: dict[str, bool] = {
        "primary_standardized_crps": primary_ratio
        <= thresholds["primary_standardized_crps_ratio_max"]
    }
    outputs: dict[str, Any] = {}
    for key in control["per_output"]:
        base = control["per_output"][key]
        trial = candidate["per_output"][key]
        crps_ratio = _positive_ratio(trial["fair_crps"], base["fair_crps"], f"{key} CRPS")
        rmse_ratio = _positive_ratio(trial["rmse"], base["rmse"], f"{key} RMSE")
        rank_increase = trial["rank_tv"] - base["rank_tv"]
        ssr_error_increase = (
            trial["adjusted_spread_skill_error"] - base["adjusted_spread_skill_error"]
        )
        output_checks = {
            "crps": crps_ratio <= thresholds["per_output_crps_ratio_max"],
            "rmse": rmse_ratio <= thresholds["per_output_rmse_ratio_max"],
            "rank_tv": rank_increase <= thresholds["per_output_rank_tv_increase_max"],
            "adjusted_ssr_error": ssr_error_increase
            <= thresholds["per_output_adjusted_ssr_error_increase_max"],
        }
        checks.update({f"{key}_{name}": passed for name, passed in output_checks.items()})
        outputs[key] = {
            "fair_crps_ratio": crps_ratio,
            "rmse_ratio": rmse_ratio,
            "rank_tv_increase": rank_increase,
            "adjusted_ssr_error_increase": ssr_error_increase,
            "checks": output_checks,
        }
    return {
        "pass": all(checks.values()),
        "primary_standardized_crps_ratio": primary_ratio,
        "outputs": outputs,
        "checks": checks,
    }


@torch.no_grad()
def run(config_path: Path, output: Path) -> dict[str, Any]:
    if torch.cuda.device_count() != 1:
        raise RuntimeError("colored gate requires exactly one visible GPU")
    if os.environ.get("CLEARML_REQUIRE_ONLINE") != "1":
        raise RuntimeError("colored gate requires online ClearML")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to reuse colored gate output {output}")
    experiment = load_json(config_path)
    coarse_spec, candidates = _validate_experiment(experiment)
    coarse_root = Path(coarse_spec["run_dir"])
    _verify_inventory(coarse_root, coarse_spec["sha256"], "coarse")
    for spec in candidates:
        root = Path(spec["run_dir"])
        _verify_inventory(root, spec["run_sha256"], spec["label"])
        checkpoint = root / spec["checkpoint"]
        if not checkpoint.is_file() or _sha256(checkpoint) != spec["checkpoint_sha256"]:
            raise ValueError(f"frozen checkpoint differs for {spec['label']}")

    reference_metadata = load_json(Path(candidates[0]["run_dir"]) / "metadata.json")
    for spec in candidates[1:]:
        metadata = load_json(Path(spec["run_dir"]) / "metadata.json")
        if metadata["data_config"] != reference_metadata["data_config"]:
            raise ValueError("fine candidates use different data laws")
    dataset = build_dataset(reference_metadata["data_config"], split="valid")
    validate_direct_dataset(dataset)
    batch = _fine_collate([dataset[index] for index in experiment["case_indices"]])
    case_ids = tuple(batch["meta"]["case_id"])
    if list(case_ids) != experiment["case_ids"]:
        raise ValueError("frozen validation case identities differ from the archive")

    valid = batch["valid_mask"][:, :1].float()
    truth_normalized = batch["truth"].float()
    persistence_normalized = batch["background"].float()
    means = _repeat_field_stats(dataset.means)
    stds = _repeat_field_stats(dataset.stds)
    truth_physical = channel_denormalize(truth_normalized, means, stds)
    persistence_physical = channel_denormalize(persistence_normalized, means, stds)
    condition_device = batch["structured_conditioning"].float().cuda()
    valid_device = valid.cuda()
    device = torch.device("cuda:0")
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    output.mkdir(parents=True, exist_ok=False)
    visual_dir = output / "fixed_scale_members"
    rank_dir = output / "rank_histograms"
    visual_dir.mkdir()
    rank_dir.mkdir()
    code_identity = _clean_code_identity(Path(__file__).resolve().parents[1])
    contract = {
        "schema_version": "colored_generated_coarse_admission_gate_v1",
        "code_identity": code_identity,
        "split": "valid",
        "year": 2022,
        "case_indices": experiment["case_indices"],
        "case_ids": list(case_ids),
        "members": 8,
        "rk4_timepoints": list(EXPECTED_SOLVERS),
        "coarse": coarse_spec,
        "fine_candidates": candidates,
        "common_random_numbers": True,
        "raw_weights_only": True,
        "raw_unclipped_scoring": True,
        "tf32": False,
        "optimizer_steps": 0,
        "gate": experiment["gate"],
        "sealed_test_2023_accessed": False,
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
        "sampling_rk4_33",
        output_dir=str(output),
        code_commit=code_identity["git_commit"],
        clearml_task_id=str(tracker.task.id),
    )

    predictors = {}
    coarse_config = load_json(coarse_root / "config.json")
    for spec in candidates:
        fine_root = Path(spec["run_dir"])
        predictors[spec["label"]] = load_cascade_predictor(
            coarse_run_dir=str(coarse_root),
            coarse_checkpoint_name=coarse_spec["checkpoint"],
            coarse_model_config=coarse_config,
            coarse_checkpoint_sha256=coarse_spec["sha256"][coarse_spec["checkpoint"]],
            fine_run_dir=str(fine_root),
            fine_checkpoint_name=spec["checkpoint"],
            fine_model_config=load_json(fine_root / "config.json"),
            fine_checkpoint_sha256=spec["checkpoint_sha256"],
            expected_coarse_code_commit=coarse_spec["code_commit"],
            expected_fine_code_commit=spec["code_commit"],
            replay_code_commit=code_identity["git_commit"],
            expected_forecast_contract_sha256=experiment["forecast_contract_sha256"],
            device=device,
        )

    result: dict[str, Any] = {
        "status": "complete_pending_visual_review",
        "admit_2048": False,
        "numerical_gate_pass": False,
        "admission_requires_visual_review": True,
        "publication_claim_permitted": False,
        "clearml_task_id": str(tracker.task.id),
        "solvers": {},
    }
    for timepoints in EXPECTED_SOLVERS:
        _launch_status(
            f"sampling_rk4_{timepoints}",
            output_dir=str(output),
            code_commit=code_identity["git_commit"],
            clearml_task_id=str(tracker.task.id),
        )
        solver_result: dict[str, Any] = {"candidates": {}}
        common_coarse = None
        common_raw_coarse_noise = None
        common_raw_fine_noise = None
        for spec in candidates:
            label = spec["label"]
            sampled = predictors[label].sample_ensemble(
                member_indices=tuple(range(8)),
                structured_conditioning=condition_device,
                valid_mask=valid_device,
                case_ids=case_ids,
                coarse_num_timesteps=timepoints,
                fine_num_timesteps=timepoints,
                device=device,
                method="rk4",
                rtol=1e-5,
                atol=1e-6,
                end_time=0.0,
            )
            coarse = sampled["coarse"]
            raw_coarse_noise = sampled["raw_coarse_noise"]
            raw_fine_noise = sampled["raw_fine_noise"]
            if common_coarse is None:
                common_coarse = coarse
                common_raw_coarse_noise = raw_coarse_noise
                common_raw_fine_noise = raw_fine_noise
            else:
                if not torch.equal(common_coarse, coarse):
                    raise RuntimeError("fine candidates did not reuse identical generated coarse members")
                if not torch.equal(common_raw_coarse_noise, raw_coarse_noise):
                    raise RuntimeError("fine candidates did not reuse identical coarse noise")
                if not torch.equal(common_raw_fine_noise, raw_fine_noise):
                    raise RuntimeError("fine candidates did not reuse identical fine white noise")
            normalized = sampled["forecast"]
            physical = channel_denormalize(normalized.flatten(0, 1), means, stds).unflatten(
                0, (len(case_ids), 8)
            )
            metrics = score_ensemble(
                normalized,
                physical,
                truth_normalized,
                truth_physical,
                persistence_physical,
                valid,
                valid,
            )
            _add_joint_consistency(metrics, physical, valid)
            summary = _candidate_summary(metrics, stds)
            _require_finite_scalars(summary)
            solver_result["candidates"][label] = {"summary": summary, "metrics": metrics}
            rank_path = rank_dir / f"rk4_{timepoints}_{label}.png"
            _save_rank_histograms(rank_path, f"RK4-{timepoints} {label}", metrics)
            tracker.report_image("rank_histograms", f"rk4_{timepoints}/{label}", rank_path, 0)
            for visual_index, position in enumerate((0, 4, 7, 11)):
                visual_path = visual_dir / f"rk4_{timepoints}_{label}_case{visual_index:02d}_all8.png"
                _save_contact_sheet(
                    visual_path,
                    f"RK4-{timepoints} {label}",
                    case_ids[position],
                    physical[position],
                    truth_physical[position],
                    persistence_physical[position],
                    valid[position],
                    {"sic": (0.0, 1.0), "sit": (0.0, 3.5)},
                )
                tracker.report_image(
                    "fixed_scale_members",
                    f"rk4_{timepoints}/{label}/case{visual_index:02d}",
                    visual_path,
                    0,
                )
            del sampled, normalized, physical
            torch.cuda.empty_cache()
        solver_result["comparison"] = _compare(
            solver_result["candidates"]["white_raw512"]["summary"],
            solver_result["candidates"]["colored_raw512"]["summary"],
            experiment["gate"],
        )
        result["solvers"][f"rk4_{timepoints}"] = solver_result
        _atomic_json(output / "colored_gate.partial.json", result)
        tracker.upload_artifact("colored_gate_partial", output / "colored_gate.partial.json")

    result["numerical_decision"] = {
        "rk4_33_pass": result["solvers"]["rk4_33"]["comparison"]["pass"],
        "rk4_65_pass": result["solvers"]["rk4_65"]["comparison"]["pass"],
    }
    result["numerical_decision"]["pass"] = all(result["numerical_decision"].values())
    result["numerical_gate_pass"] = bool(result["numerical_decision"]["pass"])
    _require_finite_scalars(result)
    _atomic_json(output / "colored_gate.json", result)
    tracker.upload_artifact("colored_gate", output / "colored_gate.json")
    tracker.close()
    _launch_status(
        "complete_pending_visual_review",
        output_dir=str(output),
        code_commit=code_identity["git_commit"],
        clearml_task_id=result["clearml_task_id"],
        numerical_pass=result["numerical_gate_pass"],
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
