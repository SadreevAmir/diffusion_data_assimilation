"""Run the corrected, architecture-matched joint-state retraining pilot.

This runner intentionally contains no new generative method.  It verifies the
data/sign/mask contract, trains the unchanged joint-state flow architecture,
evaluates the new EMA-final checkpoint, then samples the accepted legacy
checkpoint on the same fixed dates and stops at a mandatory visual-review gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from argparse import Namespace
from datetime import date
from pathlib import Path
from typing import Any

import torch

from .clean_baseline_diagnostics import analyze
from .clearml_tracking import ClearMLTracker, load_clearml_env, validate_clearml_env
from .config import TrainingConfig, load_json, merge_config_overrides, resolve_path
from .data import build_dataset, previous_calendar_date, validate_temporal_split_protocol
from .evaluate import evaluate
from .main import main as train


EXPECTED_NORMALIZATION = {
    "means": [0.7343208614260409, 0.7205874274506255],
    "stds": [0.3458938161358195, 0.5776655110213467],
}
EXPECTED_ARCHITECTURE = {
    "in_channels": 13,
    "out_channels": 2,
    "block_out_channels": [96, 192, 384, 384],
    "layers_per_block": 1,
    "dropout": 0.1,
    "down_block_types": ["DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"],
    "up_block_types": ["UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"],
    "norm_num_groups": 32,
}
EXPECTED_SPLITS = {
    "train": ("2015-01-01", "2020-12-31", "2016-01-01", "2021-12-31"),
    "valid": ("2021-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    "test": ("2022-01-01", "2022-07-19", "2023-01-01", "2023-07-19"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} differs: expected {expected!r}, got {actual!r}")


def _validate_static_contract(
    pipeline: dict[str, Any],
    training_config: dict[str, Any],
    evaluation_config: dict[str, Any],
    data_config: dict[str, Any],
    model_config: dict[str, Any],
) -> None:
    _require_equal(data_config.get("fields"), ["siconc", "sithic"], "data fields")
    _require_equal(data_config.get("indices"), [0, 1], "data indices")
    _require_equal(data_config.get("means"), EXPECTED_NORMALIZATION["means"], "normalization means")
    _require_equal(data_config.get("stds"), EXPECTED_NORMALIZATION["stds"], "normalization stds")
    _require_equal(data_config.get("hour_mode"), "all", "hour mode")
    _require_equal(data_config.get("target_hour_index"), 23, "target hour")
    _require_equal(data_config.get("background_strategy"), "calendar_year_ago", "background pairing")
    _require_equal(data_config.get("calendar_features"), [], "calendar feature ablation")
    _require_equal(data_config.get("observed_channels"), [0, 1], "observed channels")
    _require_equal(data_config.get("assimilation_range"), 3, "assimilation range")
    _require_equal(data_config.get("split_protocol"), "3dvar_main_200d", "split protocol")
    _require_equal(data_config["observation_mask"].get("sral_transform_index"), 11, "base SRAL transform")
    _require_equal(data_config["observation_mask"].get("synthetic_probability"), 0.0, "valid synthetic rate")
    train_observation = data_config["train"]["observation_mask"]
    _require_equal(train_observation.get("sral_transform_index"), 11, "train SRAL transform")
    _require_equal(train_observation.get("synthetic_probability"), 0.5, "train synthetic rate")
    validate_temporal_split_protocol(data_config)
    for split, values in EXPECTED_SPLITS.items():
        keys = ("back_start_day", "back_end_day", "obs_start_day", "obs_end_day")
        _require_equal(tuple(data_config[split][key] for key in keys), values, f"{split} dates")

    for key, expected in EXPECTED_ARCHITECTURE.items():
        _require_equal(model_config.get(key), expected, f"model {key}")
    _require_equal(model_config.get("training_objective"), "flow", "training objective")
    _require_equal(model_config.get("timestep_sampler"), "beta", "timestep sampler")
    _require_equal(model_config.get("timestep_beta_params"), [1.0, 1.5], "timestep beta")
    _require_equal(model_config.get("num_epochs"), 3, "pilot epochs")
    _require_equal(model_config.get("train_batch_size"), 16, "training batch size")
    _require_equal(model_config.get("num_workers_train"), 6, "train data workers")
    _require_equal(model_config.get("num_workers_val"), 4, "validation data workers")
    _require_equal(model_config.get("learning_rate"), 0.0001, "learning rate")
    _require_equal(model_config.get("lr_warmup_steps"), 500, "warmup")
    _require_equal(model_config.get("mixed_precision"), "bf16", "training precision")
    _require_equal(model_config.get("seed"), 1701, "training seed")
    _require_equal(model_config.get("ema_decay"), 0.999, "EMA decay")
    _require_equal(
        model_config.get("conditioning_mode_probabilities"),
        {
            "no_background": 0.1666666667,
            "no_track": 0.1666666667,
            "no_conditioning": 0.5,
            "both": 0.1666666666,
        },
        "conditioning mixture",
    )
    _require_equal(model_config.get("sample_start_mode"), "noise", "sampling source")
    _require_equal(model_config.get("sample_enforce_observations"), False, "hard observation enforcement")
    _require_equal(model_config.get("smoothness_loss_weight"), 0.0, "smoothness loss")
    _require_equal(model_config.get("obs_loss_weight"), 0.0, "observation auxiliary loss")
    _require_equal(model_config.get("sample_every_n_epochs"), 0, "in-training dashboard sampling")
    _require_equal(model_config.get("metric_every_n_epochs"), 0, "in-training ensemble metrics")
    _require_equal(model_config.get("metric_save_ensemble_samples"), False, "in-training ensemble saving")
    if "loss_domain" in model_config:
        raise ValueError("clean baseline must use the framework default full-domain loss")
    parsed = TrainingConfig.from_dict(model_config)
    _require_equal(parsed.loss_domain, "full", "resolved loss domain")

    for name, config in (("training", training_config), ("evaluation", evaluation_config)):
        if config.get("clearml", {}).get("enabled") is not True:
            raise ValueError(f"{name} ClearML must be explicitly enabled and fail closed")
        _require_equal(config["clearml"].get("env_path"), "/home/.env", f"{name} ClearML env path")
    evaluation = evaluation_config.get("evaluation", {})
    overrides = evaluation_config.get("data_overrides", {})
    _require_equal(overrides.get("hour_mode"), "fixed", "strict evaluation hour mode")
    _require_equal(overrides.get("target_hour_index"), 23, "strict evaluation hour")
    _require_equal(overrides.get("observed_channels"), [0], "strict evaluation observed channels")
    _require_equal(
        overrides.get("observation_mask", {}).get("sral_transform_index"),
        1,
        "strict evaluation SRAL transform",
    )
    expected_evaluation = {
        "checkpoint_name": "ema_last_model.pth",
        "split": "valid",
        "stride_days": 45,
        "max_cases": 8,
        "ensemble_size": 10,
        "sample_batch_size": 10,
        "num_timesteps": 25,
        "method": "dopri5",
        "rtol": 0.00001,
        "atol": 0.000001,
        "inference_precision": "float32",
        "seed": 1234,
        "save_ensembles": True,
        "cfg_mode": "none",
        "cfg_background_scale": 1.0,
        "cfg_observation_scale": 1.0,
    }
    for key, expected in expected_evaluation.items():
        _require_equal(evaluation.get(key), expected, f"evaluation {key}")

    reference = pipeline.get("accepted_reference", {})
    _require_equal(
        reference.get("checkpoint_name"),
        evaluation.get("checkpoint_name"),
        "accepted-reference evaluation checkpoint name",
    )
    _require_equal(reference.get("training_steps"), 48825, "accepted reference step count")
    _require_equal(
        reference.get("checkpoint_sha256"),
        "cd73cedc97a9f19d15c70ba31d28d3248a77dc6ef568edf8793326763810fdfe",
        "accepted reference checkpoint",
    )
    diagnostics = pipeline.get("diagnostics", {})
    _require_equal(diagnostics.get("fixed_case_orders"), [0, 2, 4, 6], "fixed visual cases")
    _require_equal(diagnostics.get("fixed_member_ids"), [0, 3, 6, 9], "fixed visual members")
    _require_equal(diagnostics.get("siconc_scale"), [0.0, 1.0], "visual SIC scale")
    if diagnostics.get("human_visual_review_required") is not True:
        raise ValueError("human visual review must remain a mandatory stop/go gate")


def _physical_siconc(normalized: torch.Tensor, data_config: dict[str, Any]) -> torch.Tensor:
    return normalized * float(data_config["stds"][0]) + float(data_config["means"][0])


def _validate_dataset_samples(dataset, data_config: dict[str, Any], *, split: str) -> list[dict[str, Any]]:
    base_indices = dataset.strided_case_indices(8, stride_days=45)
    if not base_indices:
        raise ValueError(f"no fixed preflight cases selected for {split}")
    if dataset.hour_mode == "all":
        target_hour = int(data_config["target_hour_index"])
        indices = [
            (index // dataset.hours_per_day) * dataset.hours_per_day + hour
            for case_order, index in enumerate(base_indices)
            for hour in ((0, 6, 12, 18, target_hour) if case_order == 0 else (target_hour,))
        ]
    else:
        indices = base_indices
    evidence = []
    for index in indices:
        item = dataset[index]
        for key in ("truth", "background", "obs_values", "obs_mask", "valid_mask", "water_mask"):
            if not torch.all(torch.isfinite(item[key])):
                raise ValueError(f"{split} {key} contains non-finite values at index {index}")
        target = date.fromisoformat(item["meta"]["target_date"])
        background = date.fromisoformat(item["meta"]["background_date"])
        if previous_calendar_date(target) != background:
            raise ValueError(
                f"calendar pairing mismatch: {background} is not the prior calendar date of {target}"
            )
        obs_mask = item["obs_mask"]
        valid_mask = item["valid_mask"]
        if not torch.all((obs_mask == 0.0) | (obs_mask == 1.0)):
            raise ValueError("observation mask must be binary")
        if torch.any(obs_mask > valid_mask):
            raise ValueError("observation mask leaks outside the valid domain")
        if torch.any(item["obs_values"][obs_mask == 0.0] != 0.0):
            raise ValueError("masked observation values must be exactly zero")
        truth_sic = _physical_siconc(item["truth"][0], data_config)
        background_sic = _physical_siconc(item["background"][0], data_config)
        domain = valid_mask[0] > 0.0
        if torch.any((truth_sic[domain] < 0.0) | (truth_sic[domain] > 1.0)):
            raise ValueError("training SIC target lies outside physical [0,1]")
        if torch.any((background_sic[domain] < 0.0) | (background_sic[domain] > 1.0)):
            raise ValueError("background SIC lies outside physical [0,1]")
        evidence.append(
            {
                "dataset_index": int(index),
                "target_date": target.isoformat(),
                "background_date": background.isoformat(),
                "hour": int(item["meta"]["hour"]),
                "mask_kind": item["meta"]["mask_kind"],
                "obs_count": int(item["meta"]["obs_count"]),
                "sic_truth_range": [float(truth_sic[domain].min()), float(truth_sic[domain].max())],
            }
        )
    return evidence


def _validate_flow_sign() -> dict[str, float]:
    truth = torch.tensor([[[[-0.8, 0.4], [1.1, -0.2]]]], dtype=torch.float64)
    source = torch.tensor([[[[0.6, -1.2], [0.3, 0.9]]]], dtype=torch.float64)
    time = torch.tensor([0.37], dtype=torch.float64).view(1, 1, 1, 1)
    state = (1.0 - time) * truth + time * source
    velocity = source - truth
    recovered_truth = state - time * velocity
    recovered_source = state + (1.0 - time) * velocity
    truth_error = float(torch.max(torch.abs(recovered_truth - truth)))
    source_error = float(torch.max(torch.abs(recovered_source - source)))
    if truth_error > 1e-12 or source_error > 1e-12:
        raise RuntimeError("flow sign check failed: reverse integration does not recover the data endpoint")
    return {
        "interpolation": "x_t=(1-t)*data+t*gaussian",
        "training_velocity": "gaussian-data",
        "sampling_direction": "t=1_to_0",
        "truth_reconstruction_max_error": truth_error,
        "source_reconstruction_max_error": source_error,
    }


def preflight(pipeline_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    pipeline = load_json(pipeline_path)
    training_path = resolve_path(pipeline["training_config"], pipeline_path.parent).resolve()
    evaluation_path = resolve_path(pipeline["evaluation_config"], pipeline_path.parent).resolve()
    training_config = load_json(training_path)
    evaluation_config = load_json(evaluation_path)
    data_path = resolve_path(training_config["data_config"], training_path.parent).resolve()
    model_path = resolve_path(training_config["model_config"], training_path.parent).resolve()
    data_config = load_json(data_path)
    model_config = load_json(model_path)
    _require_equal(
        resolve_path(evaluation_config["data_config"], evaluation_path.parent).resolve(),
        data_path,
        "train/evaluation data config path",
    )
    _require_equal(
        resolve_path(evaluation_config["model_config"], evaluation_path.parent).resolve(),
        model_path,
        "train/evaluation model config path",
    )
    _validate_static_contract(pipeline, training_config, evaluation_config, data_config, model_config)

    load_clearml_env(training_config.get("clearml", {}).get("env_path"))
    validate_clearml_env()
    reference = pipeline["accepted_reference"]
    checkpoint = Path(reference["run_dir"]) / reference["checkpoint_name"]
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise FileNotFoundError(f"accepted reference checkpoint is missing or unsafe: {checkpoint}")
    checkpoint_hash = _sha256(checkpoint)
    _require_equal(checkpoint_hash, reference["checkpoint_sha256"], "accepted checkpoint digest")

    evaluation_data_config = merge_config_overrides(
        data_config, evaluation_config.get("data_overrides")
    )
    train_dataset = build_dataset(data_config, split="train")
    valid_dataset = build_dataset(data_config, split="valid")
    strict_valid_dataset = build_dataset(evaluation_data_config, split="valid")
    if train_dataset.conditioned_input_channels != int(model_config["in_channels"]):
        raise ValueError("dataset/model input-channel contract differs")
    if len(train_dataset.indices) != int(model_config["out_channels"]):
        raise ValueError("dataset/model output-channel contract differs")
    updates_per_epoch = len(train_dataset) // int(model_config["train_batch_size"])
    planned_updates = updates_per_epoch * int(model_config["num_epochs"])
    if planned_updates <= 8100:
        raise ValueError(
            f"pilot update budget {planned_updates} does not exceed the 8100-step residual run"
        )
    train_evidence = _validate_dataset_samples(train_dataset, data_config, split="train")
    valid_evidence = _validate_dataset_samples(valid_dataset, data_config, split="valid")
    strict_valid_evidence = _validate_dataset_samples(
        strict_valid_dataset, evaluation_data_config, split="strict_valid"
    )
    if any(row["mask_kind"] != "sral_tracks" for row in valid_evidence):
        raise ValueError("validation must use real SRAL footprints only")
    if any(row["mask_kind"] != "sral_tracks" for row in strict_valid_evidence):
        raise ValueError("strict evaluation must use real SRAL footprints only")
    for index in strict_valid_dataset.strided_case_indices(8, stride_days=45):
        item = strict_valid_dataset[index]
        if int(item["meta"]["hour"]) != 23:
            raise ValueError("strict evaluation did not resolve to the frozen hour 23")
        if torch.any(item["obs_mask"][1] != 0.0) or torch.any(item["obs_values"][1] != 0.0):
            raise ValueError("strict evaluation leaks the unobserved sithic channel")
    evidence = {
        "schema_version": "clean-baseline-preflight-v1",
        "status": "passed",
        "reference_checkpoint_sha256": checkpoint_hash,
        "flow_sign": _validate_flow_sign(),
        "data": {
            "normalization": EXPECTED_NORMALIZATION,
            "train_provenance": train_dataset.provenance(),
            "valid_provenance": valid_dataset.provenance(),
            "train_size_all_hours": len(train_dataset),
            "valid_size_all_hours": len(valid_dataset),
            "updates_per_epoch_drop_last": updates_per_epoch,
            "planned_optimizer_updates": planned_updates,
            "train_fixed_checks": train_evidence,
            "valid_fixed_checks": valid_evidence,
            "strict_valid_fixed_checks": strict_valid_evidence,
        },
        "source_files": {
            str(path): _sha256(path)
            for path in (pipeline_path, training_path, evaluation_path, data_path, model_path)
        },
    }
    resolved = {
        "pipeline": pipeline,
        "training_path": training_path,
        "evaluation_path": evaluation_path,
        "training_config": training_config,
    }
    return evidence, resolved


def _evaluation_args(config: Path, run_dir: Path, output_dir: Path) -> Namespace:
    return Namespace(
        config=str(config),
        run_dir=str(run_dir),
        output_dir=str(output_dir),
        checkpoint_name=None,
        split=None,
        stride_days=None,
        ensemble_size=None,
        sample_batch_size=None,
        max_cases=None,
        num_timesteps=None,
        method=None,
        rtol=None,
        atol=None,
        device=None,
        inference_precision=None,
        seed=None,
        cfg_mode=None,
        cfg_background_scale=None,
        cfg_observation_scale=None,
        save_ensembles=None,
        clearml_enabled=None,
    )


def _reference_evaluation_config(source: Path, destination: Path) -> Path:
    config = load_json(source)
    config["data_config"] = str(resolve_path(config["data_config"], source.parent).resolve())
    config["model_config"] = str(resolve_path(config["model_config"], source.parent).resolve())
    config["task_name"] = "accepted_joint_state_reference_same_clean_pilot_dates"
    config["clearml"]["tags"] = [*config["clearml"].get("tags", []), "accepted_reference"]
    _atomic_json(destination, config)
    return destination


def run(pipeline_path: Path, *, output_dir: Path | None = None, preflight_only: bool = False) -> dict:
    evidence, resolved = preflight(pipeline_path.resolve())
    pipeline = resolved["pipeline"]
    root = (output_dir or Path(pipeline["status_dir"])).resolve()
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "run_status.json"
    if status_path.exists() and not preflight_only:
        raise FileExistsError(f"refusing to overwrite an existing pilot status: {status_path}")
    _atomic_json(root / "preflight.json", evidence)
    if preflight_only:
        return {"status": "preflight_passed", "preflight": str(root / "preflight.json")}

    _atomic_json(
        status_path,
        {"status": "running", "stage": "candidate_training", "experiment_id": pipeline["experiment_id"]},
    )
    try:
        training_result = train(
            resolved["training_config"],
            resolved["training_path"].parent,
        )
        run_dir = Path(training_result["output_dir"])
        candidate_checkpoint = run_dir / "ema_last_model.pth"
        if not candidate_checkpoint.is_file() or candidate_checkpoint.is_symlink():
            raise FileNotFoundError(
                f"candidate EMA-final checkpoint is missing or unsafe: {candidate_checkpoint}"
            )
        candidate_checkpoint_hash = _sha256(candidate_checkpoint)
        _atomic_json(
            status_path,
            {"status": "running", "stage": "candidate_sampling", "experiment_id": pipeline["experiment_id"]},
        )
        candidate_result = evaluate(
            _evaluation_args(
                resolved["evaluation_path"],
                run_dir,
                root / "candidate_evaluation",
            )
        )
        _atomic_json(
            status_path,
            {
                "status": "running",
                "stage": "accepted_reference_sampling",
                "experiment_id": pipeline["experiment_id"],
            },
        )
        reference_config = _reference_evaluation_config(
            resolved["evaluation_path"], root / "accepted_reference_evaluation_config.json"
        )
        reference_result = evaluate(
            _evaluation_args(
                reference_config,
                Path(pipeline["accepted_reference"]["run_dir"]),
                root / "accepted_reference_evaluation",
            )
        )
        diagnostics = analyze(
            Path(candidate_result["output_dir"]) / "samples",
            Path(reference_result["output_dir"]) / "samples",
            root / "diagnostics",
            case_orders=tuple(pipeline["diagnostics"]["fixed_case_orders"]),
            member_ids=tuple(pipeline["diagnostics"]["fixed_member_ids"]),
        )
        tracker = ClearMLTracker(
            project_name="m2m_distribution_calibration",
            task_name="clean_joint_state_calendar_pilot_diagnostics",
            tags=["clean_retrain", "visual_gate", "tie_aware_ranks", "accepted_reference"],
            env_path="/home/.env",
        )
        tracker.connect("pipeline_config", pipeline)
        for variant in ("candidate", "accepted_reference"):
            metrics = diagnostics[variant]
            for key in ("fair_crps", "ordinary_crps", "ensemble_mean_rmse", "ensemble_mean_bias"):
                tracker.report_scalar("pilot_diagnostics", f"{variant}/{key}", metrics[key], 0)
            endpoint = metrics["endpoint_mass_calibration"]
            rank = metrics["rank_histogram"]
            tracker.report_scalar(
                "rank_histogram", f"{variant}/tv_from_uniform", rank["total_variation_from_uniform"], 0
            )
            tracker.report_scalar(
                "rank_histogram", f"{variant}/normalized_mean_rank", rank["normalized_mean_rank"], 0
            )
            for atom in ("zero", "one"):
                tracker.report_scalar(
                    "endpoint_mass_probability_bias",
                    f"{variant}/{atom}",
                    endpoint[atom]["probability_bias"],
                    0,
                )
        diagnostics_path = root / "diagnostics" / "diagnostics.json"
        tracker.upload_artifact("diagnostics", diagnostics_path)
        for index, panel in enumerate(diagnostics["visual_gate"]["panels"]):
            tracker.report_image("visual_gate/fixed_comparison", f"case_{index:02d}", panel, 0)
        tracker.report_image(
            "probabilistic_gate/tie_aware_rank_and_endpoint_mass",
            "candidate_vs_reference",
            diagnostics["visual_gate"]["rank_and_endpoint_figure"],
            0,
        )
        tracker.close()
        result = {
            "status": "AWAITING_VISUAL_REVIEW",
            "experiment_id": pipeline["experiment_id"],
            "training": training_result,
            "candidate_evaluation": candidate_result,
            "accepted_reference_evaluation": reference_result,
            "candidate_checkpoint_sha256": candidate_checkpoint_hash,
            "accepted_reference_checkpoint_sha256": pipeline["accepted_reference"][
                "checkpoint_sha256"
            ],
            "diagnostics": str(diagnostics_path),
            "continuation_authorized": False,
        }
        _atomic_json(status_path, result)
        return result
    except Exception as error:
        _atomic_json(
            status_path,
            {"status": "failed", "experiment_id": pipeline["experiment_id"], "detail": str(error)},
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    result = run(args.config, output_dir=args.output_dir, preflight_only=args.preflight_only)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
