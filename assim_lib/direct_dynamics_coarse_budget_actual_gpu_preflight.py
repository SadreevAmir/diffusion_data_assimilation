"""Zero-update actual-checkpoint preflight for coarse-budget refinement.

This runner samples the frozen production cascade once, freezes its fine
allocation, and verifies that a copied final coarse RK4 interval receives the
proper-score gradient through the reviewed physical coarse-budget chart.
It never constructs an optimizer and never opens test-2023.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import signal
from pathlib import Path
from typing import Any

import torch

from .clearml_tracking import ClearMLTracker
from .config import load_json, merge_config_overrides, resolve_path
from .data import build_dataset
from .direct_dynamics_cascade_coarse import lossless_coarse_condition
from .direct_dynamics_cascade_coarse_fine_boundary_audit import (
    _finish_success,
    _load_normalization,
    _sha256,
    _strict_atomic_json,
    _terminate,
)
from .direct_dynamics_cascade_coarse_proper_refinement import (
    _atomic_torch_save,
    frozen_prefix,
    hybrid_terminal_sample,
    proper_objective,
)
from .direct_dynamics_cascade_contract import (
    forecast_contract_from_data_config,
    forecast_contract_sha256,
)
from .direct_dynamics_cascade_end_to_end import load_cascade_predictor
from .direct_dynamics_cascade_fine_training import (
    EXPECTED_SPLIT_LENGTHS,
    _calendar_inventory,
    _canonical_sha256,
    _clean_code_identity,
    _sha256_file,
)
from .direct_dynamics_coarse_budget_refinement_integration import (
    anchored_canonical_physical_coarse,
    frozen_allocation_physical_law,
)
from .direct_dynamics_fine_support_proper_admission import apply_training_sic_decoder
from .direct_dynamics_sic_support_decoder_scoring import canonical_physical_decode
from .direct_dynamics_training import validate_direct_dataset
from .runtime import make_normalized_xy_grid, seed_everything


EXPECTED_PROTOCOL = {
    "split": "train",
    "case_index": 0,
    "members": 4,
    "coarse_rk4_timepoints": 17,
    "fine_rk4_timepoints": 33,
    "network_precision": "bf16",
    "state_precision": "fp32",
    "score_precision": "fp64",
    "optimizer_steps": 0,
    "seed": 73127,
    "test_2023": "closed",
    "max_gpu_minutes": 30,
}


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != "coarse_budget_actual_gpu_preflight_v1":
        raise ValueError("unreviewed coarse-budget actual preflight schema")
    if config.get("protocol") != EXPECTED_PROTOCOL:
        raise ValueError("coarse-budget actual preflight protocol differs from review")
    if config.get("optimizer_steps") != 0 or config.get("test_2023") != "closed":
        raise ValueError("preflight must keep optimizer and test-2023 closed")
    source = config.get("source", {})
    if source.get("coarse", {}).get("checkpoint") != "ema_coarse_update_9711.pth":
        raise ValueError("preflight must use coarse EMA9711")
    if source.get("coarse", {}).get("checkpoint_sha256") != (
        "125f520c33cbea548a3de7d131220edbfc80900c120786202cdb34e8d0a4dcaa"
    ):
        raise ValueError("preflight coarse checkpoint differs from the proved chain")
    if source.get("fine", {}).get("checkpoint") != "mechanics_update_2048.pth":
        raise ValueError("preflight must use raw colored fine2048")
    if source.get("fine", {}).get("checkpoint_sha256") != (
        "f55fae2d18c4483c7115998c576a14a4928be151807aff02cd27ea9782faf7cb"
    ):
        raise ValueError("preflight fine checkpoint differs from the proved chain")


def _validate_files(spec: dict[str, Any], label: str) -> dict[str, str]:
    root = Path(spec["run_dir"])
    if not root.is_dir():
        raise ValueError(f"{label} source run directory is missing")
    result: dict[str, str] = {}
    for name, expected in spec["files_sha256"].items():
        path = root / name
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"{label} source SHA mismatch: {name}")
        result[name] = expected
    return result


def _load_normalization_binding(
    config: dict[str, Any], repo: Path
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    spec = config["normalization_source_config"]
    path = repo / spec["path"] if spec.get("repo_relative") else Path(spec["path"])
    if not path.is_file() or _sha256(path) != spec["sha256"]:
        raise ValueError("normalization source config SHA mismatch")
    means, stds, binding = _load_normalization(load_json(path))
    binding["source_config_path"] = str(path)
    binding["source_config_sha256"] = spec["sha256"]
    return means, stds, binding


def _validate_parent(config: dict[str, Any]) -> dict[str, Any]:
    spec = config["parent_cpu_integration"]
    path = Path(spec["metrics_path"])
    if not path.is_file() or _sha256(path) != spec["metrics_sha256"]:
        raise ValueError("parent CPU integration SHA mismatch")
    payload = load_json(path)
    if (
        payload.get("optimizer_steps") != 0
        or payload.get("gpu_used") is not False
        or payload.get("test_2023_used") is not False
        or payload.get("integration_check", {}).get("step0_replay_max_abs") != 0.0
    ):
        raise ValueError("parent CPU integration contract differs")
    return {"path": str(path), "sha256": spec["metrics_sha256"]}


def _validate_proved_contract(config: dict[str, Any]) -> dict[str, Any]:
    spec = config["proved_source_contract"]
    path = Path(spec["path"])
    if not path.is_file() or _sha256(path) != spec["sha256"]:
        raise ValueError("proved source contract SHA mismatch")
    payload = load_json(path)
    coarse = config["source"]["coarse"]
    fine = config["source"]["fine"]
    if (
        payload.get("coarse", {}).get("checkpoint") != coarse["checkpoint"]
        or payload.get("coarse", {}).get("sha256", {}).get(coarse["checkpoint"])
        != coarse["checkpoint_sha256"]
        or payload.get("fine", {}).get("checkpoint") != fine["checkpoint"]
        or payload.get("fine", {}).get("sha256", {}).get(fine["checkpoint"])
        != fine["checkpoint_sha256"]
    ):
        raise ValueError("actual preflight source differs from the proved contract")
    return {"path": str(path), "sha256": spec["sha256"]}


def _repeat_case(value: torch.Tensor, members: int) -> torch.Tensor:
    return value[:, None].expand(-1, members, *value.shape[1:]).flatten(0, 1)


def _require_exact_coarse_replay(
    control: torch.Tensor, production: torch.Tensor
) -> None:
    if not torch.equal(control, production):
        error = float((control - production).abs().max())
        raise RuntimeError(
            f"custom terminal coarse path is not exact production EMA9711: {error}"
        )


def _finalize_failure(
    status_path: Path,
    reservation: dict[str, Any],
    error: BaseException,
    tracker: ClearMLTracker | None,
    evidence_sha256: dict[str, str],
) -> None:
    """Preserve the primary failure while independently closing each lifecycle edge."""
    failure = {
        **reservation,
        "status": "failed",
        "error_type": type(error).__name__,
        "error": str(error)[:2000],
        "evidence_sha256": dict(evidence_sha256),
        "cleanup_errors": [],
    }
    try:
        _strict_atomic_json(status_path, failure)
    except Exception as cleanup_error:
        failure["cleanup_errors"].append(f"status_write: {cleanup_error}")
    if tracker is not None:
        try:
            tracker.task.mark_failed(status_reason=str(error)[:1000])
        except Exception as cleanup_error:
            failure["cleanup_errors"].append(f"clearml_mark_failed: {cleanup_error}")
        try:
            tracker.close()
        except Exception as cleanup_error:
            failure["cleanup_errors"].append(f"clearml_close: {cleanup_error}")
    try:
        _strict_atomic_json(status_path, failure)
    except Exception:
        pass


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = load_json(config_path)
    _validate_config(config)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("actual preflight requires exactly one visible GPU")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to reuse output directory {output_dir}")
    output_dir.mkdir(parents=True)
    status_path = output_dir / "status.json"
    metrics_path = output_dir / "preflight.json"
    reservation = {
        "status": "reserved",
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "optimizer_steps": 0,
    }
    _strict_atomic_json(status_path, reservation)
    tracker: ClearMLTracker | None = None
    evidence_sha256: dict[str, str] = {}
    try:
        repo = Path(__file__).resolve().parents[1]
        code_identity = _clean_code_identity(repo)
        parent = _validate_parent(config)
        proved_contract = _validate_proved_contract(config)
        coarse_spec = config["source"]["coarse"]
        fine_spec = config["source"]["fine"]
        coarse_files = _validate_files(coarse_spec, "coarse")
        fine_files = _validate_files(fine_spec, "fine")
        means, stds, normalization = _load_normalization_binding(config, repo)
        config_dir = config_path.resolve().parent
        data_path = resolve_path(config["data_config"], config_dir)
        if _sha256_file(data_path) != config["data_config_sha256"]:
            raise ValueError("data config SHA mismatch")
        data_config = merge_config_overrides(
            load_json(data_path), config.get("data_overrides")
        )
        contract = forecast_contract_from_data_config(data_config)
        contract_sha = forecast_contract_sha256(contract)
        if contract_sha != config["source"]["forecast_contract_sha256"]:
            raise ValueError("effective forecast contract differs from source")
        seed_everything(int(config["protocol"]["seed"]))
        dataset = build_dataset(data_config, split="train")
        if len(dataset) != EXPECTED_SPLIT_LENGTHS["train"]:
            raise ValueError("train archive length differs from audited inventory")
        sentinel = {
            "dataset": validate_direct_dataset(dataset),
            "calendar": _calendar_inventory(dataset, split="train"),
            "data_config_sha256": _canonical_sha256(data_config),
        }
        index = int(config["protocol"]["case_index"])
        raw = dataset[index]
        case_id = str(raw.get("meta", {}).get("case_id", f"train_index_{index}"))
        condition = raw["structured_conditioning"].unsqueeze(0).float()
        valid = raw["valid_mask"].unsqueeze(0)[:, :1].float()
        truth = raw["truth"].unsqueeze(0).float()
        if tuple(truth.shape[-2:]) != (320, 256):
            raise ValueError("actual preflight requires the full 320x256 grid")
        device = torch.device("cuda:0")
        tracker = ClearMLTracker(
            config["project_name"],
            f"{config['task_name']}-{output_dir.name}",
            tags=config["clearml"]["tags"],
            env_path=config["clearml"]["env_path"],
        )
        tracker.connect("preflight_contract", config)
        _strict_atomic_json(
            status_path,
            {
                **reservation,
                "status": "running",
                "code_identity": code_identity,
                "clearml_task_id": str(tracker.task.id),
            },
        )
        fixed_inputs_path = output_dir / "fixed_inputs.pth"
        evidence_sha256["fixed_inputs.pth"] = _atomic_torch_save(
            {
                "structured_conditioning": condition,
                "valid_mask": valid,
                "truth_normalized": truth,
                "case_index": index,
                "case_id": case_id,
                "source": config["source"],
                "proved_source_contract": proved_contract,
                "code_identity": code_identity,
            },
            fixed_inputs_path,
        )
        coarse_root = Path(coarse_spec["run_dir"])
        fine_root = Path(fine_spec["run_dir"])
        predictor = load_cascade_predictor(
            coarse_run_dir=str(coarse_root),
            coarse_checkpoint_name=coarse_spec["checkpoint"],
            coarse_model_config=load_json(coarse_root / "config.json"),
            coarse_checkpoint_sha256=coarse_spec["checkpoint_sha256"],
            fine_run_dir=str(fine_root),
            fine_checkpoint_name=fine_spec["checkpoint"],
            fine_model_config=load_json(fine_root / "config.json"),
            fine_checkpoint_sha256=fine_spec["checkpoint_sha256"],
            expected_coarse_code_commit=coarse_spec["code_commit"],
            expected_fine_code_commit=fine_spec["code_commit"],
            replay_code_commit=code_identity["git_commit"],
            expected_forecast_contract_sha256=contract_sha,
            device=device,
            expected_fine_conditioning_implementation_sha256=fine_spec[
                "implementation_sha256"
            ]["fine_conditioning"],
            expected_fine_preconditioning_implementation_sha256=fine_spec[
                "implementation_sha256"
            ]["fine_preconditioning"],
            expected_fine_colored_implementation_sha256=fine_spec[
                "implementation_sha256"
            ]["fine_colored"],
        )
        for parameter in predictor.coarse_sampler.sampler.model.parameters():
            parameter.requires_grad_(False)
        for parameter in predictor.fine_sampler.sampler.model.parameters():
            parameter.requires_grad_(False)
        members = int(config["protocol"]["members"])
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            production = predictor.sample_ensemble(
                member_indices=tuple(range(members)),
                storage_device="cpu",
                structured_conditioning=condition.to(device),
                valid_mask=valid.to(device),
                case_ids=(case_id,),
                coarse_num_timesteps=17,
                fine_num_timesteps=33,
                device=device,
                method="rk4",
                rtol=1e-5,
                atol=1e-6,
                end_time=0.0,
            )
        production_path = output_dir / "production_samples.pth"
        evidence_sha256["production_samples.pth"] = _atomic_torch_save(
            {
                "production": production,
                "case_index": index,
                "case_id": case_id,
                "code_identity": code_identity,
            },
            production_path,
        )
        frozen_coarse = predictor.coarse_sampler.sampler.model.eval()
        terminal_coarse = copy.deepcopy(frozen_coarse).train()
        for parameter in terminal_coarse.parameters():
            parameter.requires_grad_(True)
        condition_m = _repeat_case(condition.to(device), members)
        valid_m = _repeat_case(valid.to(device), members)
        encoded, active, _ = lossless_coarse_condition(condition_m, valid_m)
        raw_noise = production["raw_coarse_noise"].flatten(0, 1).to(device)
        grid = make_normalized_xy_grid(
            *encoded.shape[-2:], device=device, dtype=torch.float32
        )
        prefix = frozen_prefix(frozen_coarse, raw_noise, encoded, active, grid)
        candidate_normalized = hybrid_terminal_sample(
            terminal_coarse, prefix, encoded, active, grid
        ).unflatten(0, (1, members))
        with torch.no_grad():
            control_normalized = hybrid_terminal_sample(
                frozen_coarse, prefix, encoded, active, grid
            ).unflatten(0, (1, members))
        production_coarse = production["coarse"].to(device)
        candidate_control = float(
            (candidate_normalized.detach() - control_normalized).abs().max()
        )
        candidate_control_path = output_dir / "candidate_control_coarse.pth"
        evidence_sha256["candidate_control_coarse.pth"] = _atomic_torch_save(
            {
                "candidate_coarse_normalized": candidate_normalized.detach().cpu(),
                "control_coarse_normalized": control_normalized.cpu(),
                "production_coarse_normalized": production["coarse"],
                "prefix_state": prefix.cpu(),
                "code_identity": code_identity,
            },
            candidate_control_path,
        )
        production_replay = float((control_normalized - production_coarse).abs().max())
        if candidate_control > 1e-6:
            raise RuntimeError("terminal model copy differs from frozen EMA9711")
        _require_exact_coarse_replay(control_normalized, production_coarse)
        means_gpu = means.to(device)
        stds_gpu = stds.to(device)
        anchored_candidate_normalized = production_coarse.detach() + (
            candidate_normalized - control_normalized.detach()
        )
        base_coarse, candidate_coarse = anchored_canonical_physical_coarse(
            production_coarse.detach(),
            anchored_candidate_normalized,
            means_gpu,
            stds_gpu,
        )
        base_physical = canonical_physical_decode(
            production["forecast"].to(device), means_gpu, stds_gpu
        ).detach()
        truth_physical = canonical_physical_decode(
            truth.to(device), means_gpu, stds_gpu
        )
        candidate_physical = frozen_allocation_physical_law(
            base_physical, base_coarse, candidate_coarse, valid.to(device)
        )
        with torch.no_grad():
            control_physical = apply_training_sic_decoder(
                base_physical, base_coarse, valid.to(device)
            )
        scored_replay = float(
            (candidate_physical.detach() - control_physical).abs().max()
        )
        if scored_replay != 0.0:
            raise RuntimeError("actual step zero does not replay frozen-allocation law")
        scale = stds_gpu.double().reshape(1, 1, 6, 1, 1)
        objective, crps, energy = proper_objective(
            candidate_physical / scale,
            truth_physical / scale[:, 0],
            valid.to(device).double(),
        )
        objective.backward()
        gradients = [
            parameter.grad
            for parameter in terminal_coarse.parameters()
            if parameter.grad is not None
        ]
        if not gradients or not all(torch.isfinite(value).all() for value in gradients):
            raise FloatingPointError("terminal coarse parameters lack finite gradients")
        gradient_norm = float(
            torch.sqrt(sum(value.double().square().sum() for value in gradients)).cpu()
        )
        if not math.isfinite(gradient_norm) or gradient_norm <= 0:
            raise FloatingPointError("terminal coarse parameter gradient is dead")
        if any(parameter.grad is not None for parameter in frozen_coarse.parameters()):
            raise RuntimeError("gradient leaked into frozen coarse model")
        if any(
            parameter.grad is not None
            for parameter in predictor.fine_sampler.sampler.model.parameters()
        ):
            raise RuntimeError("gradient leaked into frozen fine model")
        evidence_path = output_dir / "step0_samples.pth"
        evidence_sha = _atomic_torch_save(
            {
                "candidate_coarse_normalized": candidate_normalized.detach().cpu(),
                "control_coarse_normalized": control_normalized.cpu(),
                "production": production,
                "candidate_physical": candidate_physical.detach().cpu(),
                "control_physical": control_physical.cpu(),
                "truth_normalized": truth,
                "valid_mask": valid,
                "case_id": case_id,
                "code_identity": code_identity,
            },
            evidence_path,
        )
        evidence_sha256["step0_samples.pth"] = evidence_sha
        result = {
            "status": "preflight_passed_pending_astra_review",
            "scientific_role": "actual_checkpoint_zero_update_preflight_not_training",
            "training_performed": False,
            "optimizer_created": False,
            "optimizer_steps": 0,
            "test_2023_used": False,
            "fine_resampled_for_candidate": False,
            "code_identity": code_identity,
            "parent_cpu_integration": parent,
            "proved_source_contract": proved_contract,
            "source_files_sha256": {"coarse": coarse_files, "fine": fine_files},
            "normalization": normalization,
            "dataset_sentinel": sentinel,
            "effective_forecast_contract_sha256": contract_sha,
            "case_index": index,
            "case_id": case_id,
            "protocol": config["protocol"],
            "step0": {
                "candidate_control_max_abs": candidate_control,
                "production_coarse_replay_max_abs": production_replay,
                "scored_law_replay_max_abs": scored_replay,
                "objective": float(objective.detach().cpu()),
                "fair_crps": float(crps.detach().cpu()),
                "joint_energy": float(energy.detach().cpu()),
                "terminal_parameter_gradient_norm": gradient_norm,
                "frozen_prefix_has_no_graph": not prefix.requires_grad,
                "frozen_coarse_gradients_absent": True,
                "frozen_fine_gradients_absent": True,
                "peak_gpu_memory_mib": float(
                    torch.cuda.max_memory_allocated(device) / 2**20
                ),
                "evidence_sha256": evidence_sha,
                "fixed_inputs_sha256": evidence_sha256["fixed_inputs.pth"],
            },
            "clearml_task_id": str(tracker.task.id),
        }
        _strict_atomic_json(metrics_path, result)
        tracker.connect("actual_coarse_budget_preflight", result)
        tracker.upload_artifact("actual_coarse_budget_step0", evidence_path)
        tracker.upload_artifact("actual_coarse_budget_preflight", metrics_path)
        terminal = {
            **reservation,
            "code_identity": code_identity,
            "clearml_task_id": str(tracker.task.id),
            "scientific_role": result["scientific_role"],
            "evidence_path": str(evidence_path),
            "evidence_sha256": evidence_sha,
            "fixed_inputs_path": str(fixed_inputs_path),
            "fixed_inputs_sha256": evidence_sha256["fixed_inputs.pth"],
        }
        _finish_success(tracker, status_path, metrics_path, terminal)
        tracker = None
        return result
    except BaseException as error:
        _finalize_failure(
            status_path, reservation, error, tracker, evidence_sha256
        )
        tracker = None
        raise
    finally:
        if tracker is not None:
            try:
                tracker.close()
            except Exception:
                pass


def main() -> None:
    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
