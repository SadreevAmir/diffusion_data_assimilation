"""Paired 12x8 full-resolution gate for terminal proper-score refinement."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import signal
from pathlib import Path
from typing import Any, Callable

import torch

from .clearml_tracking import ClearMLTracker
from .config import load_json
from .data import build_dataset
from .direct_dynamics_cascade import masked_block_average
from .direct_dynamics_cascade_checkpoint_evaluation import _save_rank_histograms
from .direct_dynamics_cascade_coarse_proper_refinement import (
    REVIEWED_PROTOCOL,
    TerminalRefinedCoarseSampler,
)
from .direct_dynamics_cascade_e2e_evaluation import _add_joint_consistency, _summary
from .direct_dynamics_cascade_end_to_end import CascadePredictor, load_cascade_predictor
from .direct_dynamics_cascade_fine_training import _clean_code_identity, _fine_collate
from .direct_dynamics_cascade_paired_evaluation import (
    _require_finite_scalars,
    _save_contact_sheet,
    score_ensemble,
)
from .direct_dynamics_training import _repeat_field_stats, validate_direct_dataset
from .trainer import _atomic_json
from .transforms import channel_denormalize


_ACTIVE_TRACKER: ClearMLTracker | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or _sha256(path) != expected:
        raise ValueError(f"frozen {label} mismatch: {path}")


def _atomic_torch_save(payload: Any, path: Path) -> str:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(path)


def _sample_with_reviewed_precision(predictor: Any, **kwargs: Any) -> dict[str, Any]:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return predictor.sample_ensemble(**kwargs)


def _persist_evidence_then_score(
    path: Path,
    payload: dict[str, Any],
    register_sha256: Callable[[str], None],
    score: Callable[[], dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    sha256 = _atomic_torch_save(payload, path)
    register_sha256(sha256)
    return sha256, score()


def _launch_status(status: str, **details: Any) -> None:
    raw = os.environ.get("CASCADE_E2E_STATUS_PATH", "").strip()
    if not raw:
        return
    path = Path(raw)
    if not path.is_absolute() or path.name != "status.json":
        raise ValueError("CASCADE_E2E_STATUS_PATH must be an absolute status.json")
    _atomic_json(path, {"status": status, **details})


def _validate(experiment: dict[str, Any]) -> None:
    if experiment.get("cases") != 12 or experiment.get("members") != 8:
        raise ValueError("proper-refinement evaluation requires exactly 12x8")
    if experiment.get("coarse_rk4_timepoints") != 17 or experiment.get(
        "fine_rk4_timepoints"
    ) != 17:
        raise ValueError("proper-refinement evaluation requires 16 RK4 intervals per stage")
    if len(experiment.get("case_indices", [])) != 12 or len(
        set(experiment["case_indices"])
    ) != 12:
        raise ValueError("proper-refinement evaluation requires 12 unique cases")
    if len(experiment.get("case_ids", [])) != 12 or len(set(experiment["case_ids"])) != 12:
        raise ValueError("proper-refinement evaluation requires 12 unique case ids")
    if experiment["coarse"]["checkpoint"] != "ema_coarse_update_9711.pth":
        raise ValueError("raw comparator must be EMA9711")
    if experiment["fine"]["checkpoint"] != "ema_mechanics_update_4096.pth":
        raise ValueError("paired gate must keep EMA4096 fine fixed")
    if experiment["refinement"]["completed_updates"] != 64:
        raise ValueError("paired gate requires the approved 64-update candidate")


@torch.no_grad()
def _run_impl(config_path: Path, output: Path) -> dict[str, Any]:
    global _ACTIVE_TRACKER
    if torch.cuda.device_count() != 1:
        raise RuntimeError("proper-refinement evaluation requires one visible GPU")
    if os.environ.get("CLEARML_REQUIRE_ONLINE") != "1":
        raise RuntimeError("proper-refinement evaluation requires online ClearML")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to reuse output {output}")
    experiment = load_json(config_path)
    _validate(experiment)
    coarse = experiment["coarse"]
    fine = experiment["fine"]
    refinement = experiment["refinement"]
    coarse_root = Path(coarse["run_dir"])
    fine_root = Path(fine["run_dir"])
    for relative, expected in coarse["sha256"].items():
        _verify_file(coarse_root / relative, expected, f"coarse/{relative}")
    for relative, expected in fine["sha256"].items():
        _verify_file(fine_root / relative, expected, f"fine/{relative}")
    refinement_path = Path(refinement["checkpoint"])
    _verify_file(refinement_path, refinement["sha256"], "proper refinement checkpoint")
    _verify_file(
        Path(refinement["training_record"]),
        refinement["training_record_sha256"],
        "proper refinement training record",
    )

    coarse_config = load_json(coarse_root / "config.json")
    fine_config = load_json(fine_root / "config.json")
    coarse_metadata = load_json(coarse_root / "metadata.json")
    fine_metadata = load_json(fine_root / "metadata.json")
    if coarse_metadata["data_config"] != fine_metadata["data_config"]:
        raise ValueError("coarse and fine data laws differ")
    dataset = build_dataset(fine_metadata["data_config"], split="valid")
    validate_direct_dataset(dataset)
    indices = list(experiment["case_indices"])
    batch = _fine_collate([dataset[index] for index in indices])
    case_ids = tuple(batch["meta"]["case_id"])
    if list(case_ids) != experiment["case_ids"]:
        raise ValueError("frozen validation cases differ from the archive")

    payload = torch.load(refinement_path, map_location="cpu", weights_only=True)
    if payload.get("completed_updates") != 64 or payload.get("protocol") != REVIEWED_PROTOCOL:
        raise ValueError("refinement checkpoint differs from the reviewed protocol")
    if payload.get("source", {}).get("checkpoint_sha256") != coarse["sha256"][
        coarse["checkpoint"]
    ]:
        raise ValueError("refinement base differs from paired raw EMA9711")
    if payload.get("effective_forecast_contract_sha256") != experiment[
        "forecast_contract_sha256"
    ]:
        raise ValueError("refinement forecast contract differs")
    if payload.get("code_identity", {}).get("git_commit") != refinement["code_commit"]:
        raise ValueError("refinement training commit differs")

    output.mkdir(parents=True)
    visuals_dir = output / "fixed_scale_members"
    ranks_dir = output / "rank_histograms"
    visuals_dir.mkdir()
    ranks_dir.mkdir()
    code_identity = _clean_code_identity(Path(__file__).resolve().parents[1])
    contract = {
        "schema_version": "terminal_proper_refinement_paired_e2e_v1",
        "code_identity": code_identity,
        "split": "valid",
        "case_indices": indices,
        "case_ids": list(case_ids),
        "members": 8,
        "common_random_numbers": True,
        "precision": {
            "coarse_network": "bf16",
            "fine_network": "bf16",
            "ode_state": "fp32",
            "score": "fp32_input_with_float64_reductions",
        },
        "raw_unclipped_scoring": True,
        "display_only_projection": True,
        "coarse": coarse,
        "fine": fine,
        "refinement": refinement,
        "evidence_sha256": {},
    }
    fixed_inputs_path = output / "fixed_inputs.pt"
    contract["evidence_sha256"][fixed_inputs_path.name] = _atomic_torch_save(
        {
            "structured_conditioning": batch["structured_conditioning"].float(),
            "valid_mask": batch["valid_mask"][:, :1].float(),
            "truth_normalized": batch["truth"].float(),
            "persistence_normalized": batch["background"].float(),
            "case_indices": indices,
            "case_ids": case_ids,
            "member_indices": tuple(range(8)),
            "coarse": coarse,
            "fine": fine,
            "refinement": refinement,
        },
        fixed_inputs_path,
    )
    _atomic_json(output / "contract.json", contract)
    tracker = ClearMLTracker(
        experiment["project_name"],
        f"{experiment['task_name']}-{output.name}",
        tags=experiment["clearml"]["tags"],
        env_path=experiment["clearml"]["env_path"],
    )
    _ACTIVE_TRACKER = tracker
    tracker.connect("evaluation_contract", contract)
    _launch_status(
        "sampling", output_dir=str(output), code_commit=code_identity["git_commit"],
        clearml_task_id=str(tracker.task.id),
    )

    device = torch.device("cuda:0")
    condition = batch["structured_conditioning"].float().to(device)
    valid = batch["valid_mask"][:, :1].float()
    valid_device = valid.to(device)
    truth_normalized = batch["truth"].float()
    persistence_normalized = batch["background"].float()
    means = _repeat_field_stats(dataset.means)
    stds = _repeat_field_stats(dataset.stds)
    truth_physical = channel_denormalize(truth_normalized, means, stds)
    persistence_physical = channel_denormalize(persistence_normalized, means, stds)

    raw_predictor = load_cascade_predictor(
        coarse_run_dir=str(coarse_root),
        coarse_checkpoint_name=coarse["checkpoint"],
        coarse_model_config=coarse_config,
        coarse_checkpoint_sha256=coarse["sha256"][coarse["checkpoint"]],
        fine_run_dir=str(fine_root),
        fine_checkpoint_name=fine["checkpoint"],
        fine_model_config=fine_config,
        fine_checkpoint_sha256=fine["sha256"][fine["checkpoint"]],
        expected_coarse_code_commit=coarse["code_commit"],
        expected_fine_code_commit=fine["code_commit"],
        replay_code_commit=code_identity["git_commit"],
        expected_forecast_contract_sha256=experiment["forecast_contract_sha256"],
        device=device,
    )
    terminal_model = copy.deepcopy(raw_predictor.coarse_sampler.sampler.model)
    terminal_model.load_state_dict(payload["model"], strict=True)
    terminal_model.to(device=device, dtype=torch.float32).eval()
    if not all(torch.isfinite(parameter).all() for parameter in terminal_model.parameters()):
        raise FloatingPointError("proper-refinement checkpoint has non-finite weights")
    parameter_delta_square = torch.zeros((), device=device, dtype=torch.float64)
    base_square = torch.zeros((), device=device, dtype=torch.float64)
    parameter_delta_max_abs = 0.0
    for base_parameter, candidate_parameter in zip(
        raw_predictor.coarse_sampler.sampler.model.parameters(),
        terminal_model.parameters(),
        strict=True,
    ):
        difference = candidate_parameter.double() - base_parameter.double()
        parameter_delta_square += difference.square().sum()
        base_square += base_parameter.double().square().sum()
        parameter_delta_max_abs = max(
            parameter_delta_max_abs, float(difference.abs().max().cpu())
        )
    if parameter_delta_max_abs == 0.0:
        raise RuntimeError("64-update checkpoint is identical to raw EMA9711")
    refined_sampler = TerminalRefinedCoarseSampler(
        raw_predictor.coarse_sampler, terminal_model
    )
    refined_predictor = CascadePredictor(
        refined_sampler,
        raw_predictor.fine_sampler,
        replay_identity={
            **(raw_predictor.replay_identity or {}),
            "proper_refinement_checkpoint_sha256": refinement["sha256"],
            "proper_refinement_training_commit": refinement["code_commit"],
        },
    )

    result: dict[str, Any] = {
        "status": "complete_pending_visual_review",
        "publication_claim_permitted": False,
        "clearml_task_id": str(tracker.task.id),
        "refinement_parameter_change": {
            "l2": float(torch.sqrt(parameter_delta_square).cpu()),
            "relative_l2": float(torch.sqrt(parameter_delta_square / base_square).cpu()),
            "max_abs": parameter_delta_max_abs,
        },
        "candidates": {},
    }
    reference_randomness = None
    local_images: list[tuple[str, str, Path]] = []
    for label, predictor in (
        ("raw_ema9711_fine4096", raw_predictor),
        ("proper64_ema9711_fine4096", refined_predictor),
    ):
        ensemble = _sample_with_reviewed_precision(
            predictor,
            member_indices=tuple(range(8)),
            structured_conditioning=condition,
            valid_mask=valid_device,
            case_ids=case_ids,
            coarse_num_timesteps=17,
            fine_num_timesteps=17,
            device=device,
            method="rk4",
            rtol=1e-5,
            atol=1e-6,
            end_time=0.0,
        )
        randomness = {
            key: ensemble[key]
            for key in (
                "raw_coarse_noise", "raw_fine_noise", "projected_fine_noise",
                "coarse_seeds", "fine_seeds", "member_indices", "case_ids",
            )
        }
        if reference_randomness is None:
            reference_randomness = {
                key: value.clone() if isinstance(value, torch.Tensor) else value
                for key, value in randomness.items()
            }
        else:
            for key, value in randomness.items():
                reference = reference_randomness[key]
                equal = torch.equal(reference, value) if isinstance(value, torch.Tensor) else reference == value
                if not equal:
                    raise RuntimeError(f"paired candidates differ in {key}")
        normalized = ensemble["forecast"].float()
        recovered, fraction = masked_block_average(
            normalized.flatten(0, 1),
            valid[:, None].expand(-1, 8, -1, -1, -1).flatten(0, 1),
        )
        exact_error = float(
            (recovered - ensemble["coarse"].flatten(0, 1))[
                fraction.expand_as(recovered) > 0
            ].abs().max()
        )
        if exact_error > 3e-6:
            raise RuntimeError("fine forecast changed generated coarse 2x2 means")
        physical = channel_denormalize(normalized.flatten(0, 1), means, stds).unflatten(
            0, (12, 8)
        )
        evidence_path = output / f"{label}_raw.pt"
        evidence_payload = {
            "forecast_normalized": ensemble["forecast"],
            "coarse_normalized": ensemble["coarse"],
            "residual_normalized": ensemble["residual"],
            "raw_coarse_noise": ensemble["raw_coarse_noise"],
            "raw_fine_noise": ensemble["raw_fine_noise"],
            "projected_fine_noise": ensemble["projected_fine_noise"],
            "valid_mask": valid,
            "truth_normalized": truth_normalized,
            "persistence_normalized": persistence_normalized,
            "case_indices": indices,
            "case_ids": ensemble["case_ids"],
            "member_indices": ensemble["member_indices"],
            "coarse_seeds": ensemble["coarse_seeds"],
            "fine_seeds": ensemble["fine_seeds"],
            "solver": ensemble["solver"],
            "replay_identity": ensemble["replay_identity"],
            "coarse": coarse,
            "fine": fine,
            "refinement": refinement if label.startswith("proper64") else None,
        }

        def register_evidence(sha256: str) -> None:
            contract["evidence_sha256"][evidence_path.name] = sha256
            _atomic_json(output / "contract.json", contract)

        evidence_sha256, metrics = _persist_evidence_then_score(
            evidence_path,
            evidence_payload,
            register_evidence,
            lambda: score_ensemble(
                normalized, physical, truth_normalized, truth_physical,
                persistence_physical, valid, valid,
            ),
        )
        _add_joint_consistency(metrics, physical, valid)
        metrics["exact_coarse_2x2_mean_error_max"] = exact_error
        metrics["summary"] = _summary(metrics)
        _require_finite_scalars(metrics)
        result["candidates"][label] = {
            "metrics": metrics,
            "raw_evidence": evidence_path.name,
            "raw_evidence_sha256": evidence_sha256,
        }

        rank_path = ranks_dir / f"{label}.png"
        _save_rank_histograms(rank_path, label, metrics)
        local_images.append(("rank_histograms", label, rank_path))
        for position in (0, 4, 7, 11):
            visual_path = visuals_dir / f"{label}_case{position:02d}_all8_fixed.png"
            _save_contact_sheet(
                visual_path, label, case_ids[position], physical[position],
                truth_physical[position], persistence_physical[position], valid[position],
                {"sic": (0.0, 1.0), "sit": (0.0, 3.5)},
            )
            local_images.append(("fixed_scale_members", f"{label}/case{position:02d}", visual_path))
        del ensemble, normalized, physical
        torch.cuda.empty_cache()

    raw_summary = result["candidates"]["raw_ema9711_fine4096"]["metrics"]["summary"]
    refined_summary = result["candidates"]["proper64_ema9711_fine4096"]["metrics"]["summary"]
    result["paired_summary_delta_refined_minus_raw"] = {
        key: float(refined_summary[key] - raw_summary[key]) for key in raw_summary
    }
    raw_metrics = result["candidates"]["raw_ema9711_fine4096"]["metrics"]
    refined_metrics = result["candidates"]["proper64_ema9711_fine4096"]["metrics"]
    result["paired_output_deltas_refined_minus_raw"] = {}
    for output_key in raw_metrics["outputs"]:
        raw_row = raw_metrics["outputs"][output_key]
        refined_row = refined_metrics["outputs"][output_key]
        result["paired_output_deltas_refined_minus_raw"][output_key] = {
            "ensemble_mean_rmse": float(
                refined_row["case_equal_ensemble_mean_rmse"]
                - raw_row["case_equal_ensemble_mean_rmse"]
            ),
            "fair_crps": float(
                refined_row["case_equal_fair_crps"] - raw_row["case_equal_fair_crps"]
            ),
            "spread_skill_ratio": float(
                refined_row["spread_skill_ratio"] - raw_row["spread_skill_ratio"]
            ),
            "rank_tv_to_uniform": float(
                refined_row["rank_tv_to_uniform"] - raw_row["rank_tv_to_uniform"]
            ),
            "case_rmse": [
                float(candidate - baseline)
                for candidate, baseline in zip(
                    refined_row["case_rmse"], raw_row["case_rmse"], strict=True
                )
            ],
            "case_fair_crps": [
                float(candidate - baseline)
                for candidate, baseline in zip(
                    refined_row["case_fair_crps"], raw_row["case_fair_crps"], strict=True
                )
            ],
        }
    result["paired_case_joint_energy_delta_refined_minus_raw"] = [
        float(candidate - baseline)
        for candidate, baseline in zip(
            refined_metrics["case_joint_energy_score"],
            raw_metrics["case_joint_energy_score"],
            strict=True,
        )
    ]
    _require_finite_scalars(result)
    _atomic_json(output / "proper_refinement_e2e_evaluation.json", result)
    for title, series, path in local_images:
        tracker.report_image(title, series, path, 0)
    tracker.upload_artifact(
        "proper_refinement_e2e_evaluation",
        output / "proper_refinement_e2e_evaluation.json",
    )
    tracker.close()
    _ACTIVE_TRACKER = None
    _launch_status(
        "complete_pending_visual_review", output_dir=str(output),
        code_commit=code_identity["git_commit"], clearml_task_id=result["clearml_task_id"],
    )
    return result


def _record_failure_and_close(output: Path, error: BaseException) -> None:
    global _ACTIVE_TRACKER
    failure = {
        "status": "failed",
        "error_type": type(error).__name__,
        "error": str(error)[:2000],
    }
    try:
        if output.is_dir():
            _atomic_json(output / "failure.json", failure)
    except Exception:
        pass
    try:
        _launch_status(
            "failed", error_type=failure["error_type"], error=failure["error"]
        )
    except Exception:
        pass
    if _ACTIVE_TRACKER is not None:
        try:
            _ACTIVE_TRACKER.close()
        except Exception:
            pass
        _ACTIVE_TRACKER = None


def run(config_path: Path, output: Path) -> dict[str, Any]:
    global _ACTIVE_TRACKER
    try:
        return _run_impl(config_path, output)
    except BaseException as error:
        _record_failure_and_close(output, error)
        raise
    finally:
        if _ACTIVE_TRACKER is not None:
            try:
                _ACTIVE_TRACKER.close()
            except Exception:
                pass
            _ACTIVE_TRACKER = None


def main() -> None:
    def terminate(signum, _frame):
        raise TimeoutError(f"proper-refinement evaluation received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    result = run(arguments.config.resolve(), arguments.output.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
