"""Frozen FP32 solver/denoiser audit for bounded clean-state dynamics.

This diagnostic uses the four train-only overfit anchors and the exact saved
noise from the audited 512-update mechanics gate.  Teacher-forced calls use the
known target explicitly and are labelled diagnostic-only; they are never used
as forecasts or validation evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import default_collate

from .bounded_clean_state_flow import (
    BoundedCleanStateSampler,
    bounded_clean_prediction,
)
from .bounded_clean_state_overfit import (
    ENDPOINT_EPSILON,
    ENSEMBLE_MEMBERS,
    SIT_SCALE_METERS,
    diagnostic_indices,
)
from .clearml_tracking import ClearMLTracker
from .config import TrainingConfig, load_json, merge_config_overrides, resolve_path
from .data import build_dataset
from .direct_dynamics_training import (
    DIRECT_CONDITION_CHANNELS,
    DIRECT_LEADS,
    DIRECT_OUTPUT_CHANNELS,
    DirectDynamicsTrainer,
    _repeat_field_stats,
    validate_direct_dataset,
)
from .model_io import build_unet
from .runtime import make_normalized_xy_grid, seed_everything
from .structured_trajectory_evaluation import make_structured_trajectory_figure
from .trainer import _atomic_json
from .transforms import channel_denormalize


SOURCE_RUN = Path(
    "/home/autoresearch_results/direct_dynamics_all_hours_v1/"
    "bounded_overfit/bounded_x0_overfit_20260909_v1"
)
SOURCE_TASK_ID = "34b0045dd1274ab7b804006124331e44"
SOURCE_RESULT = SOURCE_RUN / "result.json"
SOURCE_RESULT_SHA256 = "83cbfada611d5e8fbb0d914623129d3c02c2a79ed41785c86893c0e2aa48eafe"
SOURCE_CONTRACT = SOURCE_RUN / "contract.json"
SOURCE_CONTRACT_SHA256 = "682a522d2028bace385ddf4007b9aa2fba7a3cc2670905471f6792dc1136269f"
SOURCE_CHECKPOINT = SOURCE_RUN / "model_step_0512.pth"
SOURCE_CHECKPOINT_SHA256 = "0e794282f62268f080b6d5471ac8bbdc47d21e5e5b3ffe9ed23c21f7f676d9f4"
SOURCE_SAMPLES = SOURCE_RUN / "diagnostics/update_0512/samples.pt"
SOURCE_SAMPLES_SHA256 = "df1f23f1046749c6f6af8c4d6bb16607089acb353a2795d0f78daa901f86441e"
GENERATIVE_CONFIGS = (
    ("fp32_rk4_17_eps001", 17, 0.01),
    ("fp32_rk4_33_eps001", 33, 0.01),
    ("fp32_rk4_65_eps001", 65, 0.01),
    ("fp32_rk4_65_eps002", 65, 0.02),
)
TEACHER_TIMES = (0.01, 0.02)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_source(
    *,
    config_path: Path,
    data_path: Path,
    model_path: Path,
    data_config: dict[str, Any],
    model_config: dict[str, Any],
) -> dict[str, Any]:
    artifacts = {
        SOURCE_RESULT: SOURCE_RESULT_SHA256,
        SOURCE_CONTRACT: SOURCE_CONTRACT_SHA256,
        SOURCE_CHECKPOINT: SOURCE_CHECKPOINT_SHA256,
        SOURCE_SAMPLES: SOURCE_SAMPLES_SHA256,
    }
    if not SOURCE_RUN.is_dir() or not all(path.is_file() for path in artifacts):
        raise FileNotFoundError("frozen bounded overfit source artifacts are incomplete")
    for path, expected_sha256 in artifacts.items():
        if _sha256(path) != expected_sha256:
            raise ValueError(f"frozen source SHA256 differs: {path}")
    result = load_json(SOURCE_RESULT)
    source_contract = load_json(SOURCE_CONTRACT)
    if result.get("contract") != source_contract:
        raise ValueError("frozen result and contract artifacts disagree")
    if result.get("clearml_task_id") != SOURCE_TASK_ID:
        raise ValueError("frozen bounded overfit source task differs")
    if result.get("numeric_gate", {}).get("status") != "passed_numeric_pending_solver_and_visual_review":
        raise ValueError("frozen bounded overfit source did not pass its numeric gate")
    for name, path in (
        ("experiment", config_path),
        ("data", data_path),
        ("model", model_path),
    ):
        expected = source_contract.get("source_configs", {}).get(name, {}).get("sha256")
        if expected is None or _sha256(path) != expected:
            raise ValueError(f"current {name} config differs from frozen source contract")
    if data_config != source_contract.get("effective_data_config"):
        raise ValueError("effective data config differs from frozen source contract")
    if model_config != source_contract.get("source_model_config"):
        raise ValueError("model config differs from frozen source contract")
    return result


def _require_finite_scalars(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _require_finite_scalars(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _require_finite_scalars(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError(f"{path} is NaN/Inf")


def _require_finite(tensor: torch.Tensor, valid: torch.Tensor, label: str) -> None:
    mask = valid.expand_as(tensor) > 0
    if not torch.any(mask) or not torch.all(torch.isfinite(tensor[mask])):
        raise FloatingPointError(f"{label} is empty or contains NaN/Inf")


def _field_metrics(
    ensemble: torch.Tensor,
    terminal: torch.Tensor,
    truth: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, Any]:
    _require_finite(ensemble, valid[:, None], "frozen endpoint ensemble")
    _require_finite(terminal, valid[:, None], "frozen terminal ensemble")
    result: dict[str, Any] = {"cases": [], "aggregate": {}}
    for case in range(truth.shape[0]):
        case_result: dict[str, Any] = {"case": case, "leads": {}}
        ocean = valid[case, 0] > 0
        for lead_index, lead_day in enumerate(DIRECT_LEADS):
            sic_truth = truth[case, 2 * lead_index]
            open_water = ocean & (sic_truth < 0.01)
            if not torch.any(open_water):
                raise ValueError("frozen audit case has no open-water points")
            lead_result = {}
            for offset, field_name in enumerate(("sic", "sit")):
                channel = 2 * lead_index + offset
                values = ensemble[case, :, channel][:, open_water]
                target = truth[case, channel][ocean]
                prediction = ensemble[case, :, channel][:, ocean]
                member_rmse = torch.sqrt((prediction - target[None]).square().mean(dim=1))
                roughness = DirectDynamicsTrainer._roughness(
                    ensemble[case, 0, channel : channel + 1],
                    valid[case],
                )
                lead_result[field_name] = {
                    "open_water_mean": float(values.mean()),
                    "open_water_p95": float(torch.quantile(values, 0.95)),
                    "open_water_gt_0p01_fraction": float((values > 0.01).float().mean()),
                    "member_mean_rmse": float(member_rmse.mean()),
                    "member0_roughness": float(roughness),
                }
                if field_name == "sit":
                    zero_truth = ocean & (truth[case, channel] == 0)
                    if not torch.any(zero_truth):
                        raise ValueError("frozen audit case/lead has no exactly-zero SIT points")
                    zero_values = ensemble[case, :, channel][:, zero_truth]
                    lead_result[field_name].update(
                        {
                            "truth_zero_count": int(zero_truth.sum()),
                            "truth_zero_mean": float(zero_values.mean()),
                            "truth_zero_p95": float(torch.quantile(zero_values, 0.95)),
                            "truth_zero_gt_0p01_fraction": float(
                                (zero_values > 0.01).float().mean()
                            ),
                        }
                    )
            case_result["leads"][f"d{lead_day}"] = lead_result
        result["cases"].append(case_result)

    for lead_index, lead_day in enumerate(DIRECT_LEADS):
        result["aggregate"][f"d{lead_day}"] = {}
        for offset, field_name in enumerate(("sic", "sit")):
            records = [
                case["leads"][f"d{lead_day}"][field_name]
                for case in result["cases"]
            ]
            result["aggregate"][f"d{lead_day}"][field_name] = {
                key: float(sum(record[key] for record in records) / len(records))
                for key in records[0]
            }
    result["support"] = {
        "sic_below_zero": int(((ensemble[:, :, 0::2] < 0) & (valid[:, None] > 0)).sum()),
        "sic_above_one": int(((ensemble[:, :, 0::2] > 1) & (valid[:, None] > 0)).sum()),
        "sit_below_zero": int(((ensemble[:, :, 1::2] < 0) & (valid[:, None] > 0)).sum()),
    }
    result["endpoint_terminal_mean_abs_difference"] = float(
        ((ensemble - terminal).abs() * valid[:, None]).sum()
        / valid[:, None].expand_as(ensemble).sum().clamp(min=1)
    )
    _require_finite_scalars(result, "field_metrics")
    return result


@torch.no_grad()
def _teacher_forced(
    model,
    *,
    truth_normalized: torch.Tensor,
    initial_noise: torch.Tensor,
    conditioning: torch.Tensor,
    valid: torch.Tensor,
    means: tuple[float, ...],
    stds: tuple[float, ...],
    time_value: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    case_count = truth_normalized.shape[0]
    truth = truth_normalized.repeat_interleave(ENSEMBLE_MEMBERS, dim=0)
    repeated_valid = valid.repeat_interleave(ENSEMBLE_MEMBERS, dim=0)
    repeated_conditioning = conditioning.repeat_interleave(ENSEMBLE_MEMBERS, dim=0)
    time = torch.full(
        (truth.shape[0],), float(time_value), dtype=torch.float32, device=truth.device
    )
    active = repeated_valid.expand_as(truth) > 0
    clean = torch.where(active, truth, torch.zeros_like(truth))
    noise = torch.where(active, initial_noise, torch.zeros_like(initial_noise))
    state = (1.0 - time[:, None, None, None]) * clean + time[:, None, None, None] * noise
    grid = make_normalized_xy_grid(
        *truth.shape[-2:], device=truth.device, dtype=truth.dtype
    ).expand(truth.shape[0], -1, -1, -1)
    logits = model(
        torch.cat((state, grid, repeated_conditioning), dim=1),
        time * 1000.0,
        return_dict=False,
    )[0]
    physical, normalized = bounded_clean_prediction(
        logits.float(),
        repeated_valid,
        means=means,
        stds=stds,
        sit_scale=SIT_SCALE_METERS,
    )
    return (
        physical.unflatten(0, (case_count, ENSEMBLE_MEMBERS)),
        normalized.unflatten(0, (case_count, ENSEMBLE_MEMBERS)),
        state.unflatten(0, (case_count, ENSEMBLE_MEMBERS)),
    )


def _background_average(metrics: dict[str, Any], field: str) -> float:
    values = [
        metrics["aggregate"][f"d{lead}"][field]["open_water_mean"]
        for lead in DIRECT_LEADS
    ]
    return float(sum(values) / len(values))


def _casewise_open_water_differences(
    candidate: dict[str, Any], reference: dict[str, Any]
) -> list[dict[str, Any]]:
    if len(candidate["cases"]) != len(reference["cases"]):
        raise ValueError("solver diagnostics have different case counts")
    differences = []
    for case_index, (candidate_case, reference_case) in enumerate(
        zip(candidate["cases"], reference["cases"], strict=True)
    ):
        for lead_day in DIRECT_LEADS:
            lead = f"d{lead_day}"
            for field in ("sic", "sit"):
                delta = abs(
                    candidate_case["leads"][lead][field]["open_water_mean"]
                    - reference_case["leads"][lead][field]["open_water_mean"]
                )
                differences.append(
                    {
                        "case": case_index,
                        "lead": lead,
                        "field": field,
                        "absolute_difference": float(delta),
                    }
                )
    return differences


def _decision(
    generative: dict[str, dict[str, Any]], teacher: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    _require_finite_scalars(generative, "generative")
    _require_finite_scalars(teacher, "teacher")
    reference = generative["fp32_rk4_65_eps001"]
    solver: dict[str, Any] = {}
    for name in ("fp32_rk4_17_eps001", "fp32_rk4_33_eps001"):
        records = _casewise_open_water_differences(generative[name], reference)
        solver[name] = {
            "case_lead_field_differences": records,
            "maximum_absolute_difference": max(
                record["absolute_difference"] for record in records
            ),
        }
    # RK4-33 versus RK4-65 is the pre-registered convergence decision.  RK4-17
    # is retained as a useful secondary comparison but does not redefine it.
    solver_stable = all(
        record["absolute_difference"] < 0.001
        for record in solver["fp32_rk4_33_eps001"]["case_lead_field_differences"]
    )
    teacher_zero_sit = []
    for name, metrics in teacher.items():
        for case in metrics["cases"]:
            for lead_day in DIRECT_LEADS:
                lead = f"d{lead_day}"
                sit = case["leads"][lead]["sit"]
                teacher_zero_sit.append(
                    {
                        "configuration": name,
                        "case": case["case"],
                        "lead": lead,
                        "truth_zero_count": sit["truth_zero_count"],
                        "truth_zero_mean": sit["truth_zero_mean"],
                    }
                )
    if not teacher_zero_sit:
        raise ValueError("teacher-forced audit has no exactly-zero SIT records")
    teacher_floor = all(record["truth_zero_mean"] >= 0.01 for record in teacher_zero_sit)
    if solver_stable and teacher_floor:
        diagnosis = "denoiser_objective_or_finite_optimization_not_solver"
    elif solver_stable and not teacher_floor:
        diagnosis = "inconclusive_teacher_endpoint_can_represent_zero_sit"
    else:
        diagnosis = "solver_sensitivity_requires_revision"
    decision = {
        "solver_casewise_differences_vs_fp32_rk4_65_eps001": solver,
        "solver_rk4_33_vs_65_stable_below_0p001_each_case_lead_field": solver_stable,
        "teacher_forced_exact_zero_sit_records": teacher_zero_sit,
        "teacher_forced_centimeter_sit_floor": teacher_floor,
        "diagnosis": diagnosis,
        "permits_training": False,
    }
    _require_finite_scalars(decision, "decision")
    return decision


def _save_panels(
    *,
    ensemble: torch.Tensor,
    truth: torch.Tensor,
    persistence: torch.Tensor,
    valid: torch.Tensor,
    label: str,
    output_dir: Path,
    tracker: ClearMLTracker,
) -> None:
    panel_dir = output_dir / "panels" / label
    panel_dir.mkdir(parents=True, exist_ok=False)
    for case in range(truth.shape[0]):
        figure = make_structured_trajectory_figure(
            truth[case],
            persistence[case],
            ensemble[case, 0],
            valid[case],
            title=f"bounded frozen audit {label}; anchor {case}, member 0",
            origin="upper",
            lead_days=DIRECT_LEADS,
        )
        path = panel_dir / f"anchor_{case:02d}_member0.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        import matplotlib.pyplot as plt

        plt.close(figure)
        tracker.report_image(
            "bounded_frozen_audit/samples", f"{label}_anchor_{case:02d}", path, 0
        )


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    if torch.cuda.device_count() != 1:
        raise RuntimeError("bounded frozen audit requires exactly one visible GPU")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to reuse bounded frozen audit output: {output_dir}")
    experiment = load_json(config_path)
    config_dir = config_path.resolve().parent
    data_path = resolve_path(experiment["data_config"], config_dir)
    model_path = resolve_path(experiment["model_config"], config_dir)
    data_config = merge_config_overrides(
        load_json(data_path), experiment.get("data_overrides")
    )
    raw_model_config = load_json(model_path)
    source_result = _require_source(
        config_path=config_path,
        data_path=data_path,
        model_path=model_path,
        data_config=data_config,
        model_config=raw_model_config,
    )
    model_config = TrainingConfig.from_dict(raw_model_config)
    if model_config.in_channels != 23 or model_config.out_channels != DIRECT_OUTPUT_CHANNELS:
        raise ValueError("bounded frozen audit model contract differs")
    seed_everything(model_config.seed)
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dataset = build_dataset(data_config, split="train")
    validate_direct_dataset(dataset)
    indices = diagnostic_indices(len(dataset))
    batch = default_collate([dataset[index] for index in indices])
    source = torch.load(SOURCE_SAMPLES, map_location="cpu", weights_only=True)
    for key, current in (
        ("truth", batch["structured_physical_truth"]),
        ("persistence", batch["structured_physical_background"]),
        ("valid_mask", batch["valid_mask"][:, :1]),
    ):
        if not torch.equal(source[key], current):
            raise ValueError(f"frozen audit {key} differs from source diagnostic")

    output_dir.mkdir(parents=True, exist_ok=False)
    tracker = ClearMLTracker(
        "sea_ice_two_stage",
        f"bounded_clean_state_frozen_audit_{output_dir.name}",
        tags=[
            "bounded-clean-state",
            "frozen-checkpoint",
            "strict-fp32",
            "solver-audit",
            "teacher-forced-diagnostic",
            "train-anchors-only",
            "zero-optimizer-steps",
            "one-gpu",
        ],
        env_path=experiment.get("clearml", {}).get("env_path"),
    )
    contract = {
        "source_task_id": SOURCE_TASK_ID,
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "source_result_sha256": SOURCE_RESULT_SHA256,
        "source_contract_sha256": SOURCE_CONTRACT_SHA256,
        "source_result_status": source_result["numeric_gate"]["status"],
        "source_samples_sha256": SOURCE_SAMPLES_SHA256,
        "dataset_split": "train_mechanics_only",
        "diagnostic_indices": indices,
        "generative_configs": GENERATIVE_CONFIGS,
        "teacher_forced_times": TEACHER_TIMES,
        "teacher_forced_uses_target": True,
        "teacher_forced_forecast_claim_permitted": False,
        "optimizer_steps": 0,
        "network_precision": "fp32",
        "ode_precision": "fp32",
        "tf32": False,
        "clipping": False,
    }
    tracker.connect("bounded_frozen_audit_contract", contract)
    _atomic_json(output_dir / "contract.json", contract)
    device = torch.device("cuda:0")
    model = build_unet(model_config).to(device=device, dtype=torch.float32)
    state_dict = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    del state_dict
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    initial_noise = source["initial_noise"].to(device=device, dtype=torch.float32)
    case_count = batch["truth"].shape[0]
    condition = batch["structured_conditioning"].float()
    valid = batch["valid_mask"][:, :1].float()
    repeated_condition = condition.repeat_interleave(ENSEMBLE_MEMBERS, dim=0)
    repeated_valid = valid.repeat_interleave(ENSEMBLE_MEMBERS, dim=0)
    means = _repeat_field_stats(dataset.means)
    stds = _repeat_field_stats(dataset.stds)
    truth = batch["structured_physical_truth"].float()
    persistence = batch["structured_physical_background"].float()
    truth_normalized = batch["truth"].float()
    generative: dict[str, dict[str, Any]] = {}
    teacher: dict[str, dict[str, Any]] = {}
    try:
        source_ensemble = source["ensemble"].to(device=device, dtype=torch.float32)
        source_terminal = source["terminal_ode_physical"].to(
            device=device, dtype=torch.float32
        )
        source_metrics = _field_metrics(
            source_ensemble, source_terminal, truth, valid
        )
        _require_finite_scalars(source_metrics, "source_bf16_rk4_17_eps001")
        _save_panels(
            ensemble=source_ensemble,
            truth=truth,
            persistence=persistence,
            valid=valid,
            label="source_bf16_rk4_17_eps001",
            output_dir=output_dir,
            tracker=tracker,
        )
        for name, steps, epsilon in GENERATIVE_CONFIGS:
            sampled = BoundedCleanStateSampler(
                model,
                means=means,
                stds=stds,
                sit_scale=SIT_SCALE_METERS,
                epsilon=epsilon,
            ).sample(
                initial_noise,
                repeated_condition,
                repeated_valid,
                num_steps=steps,
            )
            ensemble = sampled.physical.unflatten(
                0, (case_count, ENSEMBLE_MEMBERS)
            ).float()
            terminal = channel_denormalize(
                sampled.terminal_ode_state.float(), means, stds
            ).unflatten(0, (case_count, ENSEMBLE_MEMBERS))
            metrics = _field_metrics(ensemble, terminal, truth, valid)
            generative[name] = metrics
            stage = output_dir / "raw" / name
            stage.mkdir(parents=True, exist_ok=False)
            torch.save(
                {
                    "endpoint_physical": ensemble.cpu(),
                    "terminal_physical": terminal.cpu(),
                    "initial_noise": initial_noise.cpu(),
                    "truth": truth.cpu(),
                    "valid_mask": valid.cpu(),
                },
                stage / "samples.pt",
            )
            _atomic_json(stage / "metrics.json", metrics)
            _save_panels(
                ensemble=ensemble,
                truth=truth,
                persistence=persistence,
                valid=valid,
                label=name,
                output_dir=output_dir,
                tracker=tracker,
            )

        for time_value in TEACHER_TIMES:
            name = f"teacher_forced_t{str(time_value).replace('.', 'p')}"
            endpoint, normalized, teacher_state = _teacher_forced(
                model,
                truth_normalized=truth_normalized,
                initial_noise=initial_noise,
                conditioning=condition,
                valid=valid,
                means=means,
                stds=stds,
                time_value=time_value,
            )
            terminal = channel_denormalize(
                teacher_state.flatten(0, 1),
                means,
                stds,
            ).unflatten(0, (case_count, ENSEMBLE_MEMBERS))
            metrics = _field_metrics(endpoint, terminal, truth, valid)
            teacher[name] = metrics
            stage = output_dir / "raw" / name
            stage.mkdir(parents=True, exist_ok=False)
            torch.save(
                {
                    "endpoint_physical": endpoint.cpu(),
                    "endpoint_normalized": normalized.cpu(),
                    "teacher_forced_state_physical": terminal.cpu(),
                    "truth": truth.cpu(),
                    "valid_mask": valid.cpu(),
                    "uses_target": True,
                },
                stage / "samples.pt",
            )
            _atomic_json(stage / "metrics.json", metrics)
            _save_panels(
                ensemble=endpoint,
                truth=truth,
                persistence=persistence,
                valid=valid,
                label=name,
                output_dir=output_dir,
                tracker=tracker,
            )

        decision = _decision(generative, teacher)
        result = {
            "status": "complete",
            "clearml_task_id": str(tracker.task.id),
            "contract": contract,
            "source_bf16_rk4_17_eps001": source_metrics,
            "generative": generative,
            "teacher_forced": teacher,
            "decision": decision,
        }
        for section in (generative, teacher):
            for name, metrics in section.items():
                for field in ("sic", "sit"):
                    tracker.report_scalar(
                        "bounded_frozen_audit/open_water_mean",
                        f"{name}_{field}",
                        _background_average(metrics, field),
                        0,
                    )
        _require_finite_scalars(result, "result")
        _atomic_json(output_dir / "result.json", result)
        tracker.connect("bounded_frozen_audit_result", result)
        tracker.upload_artifact("bounded_frozen_audit_result", output_dir / "result.json")
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
