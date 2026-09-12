"""Paired 12x8 full-resolution gate for terminal proper-score refinement."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import signal
from datetime import date
from pathlib import Path
from typing import Any, Callable

import torch

from .clearml_tracking import ClearMLTracker
from .config import load_json
from .data import build_dataset
from .direct_dynamics_cascade import masked_block_average
from .direct_dynamics_cascade_checkpoint_evaluation import _save_rank_histograms
from .direct_dynamics_cascade_coarse_proper_refinement import (
    TerminalRefinedCoarseSampler,
    _validate_reviewed_protocol,
    standardized_fair_crps,
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
from .direct_dynamics_affine_calibration import boundary_event_metrics
from .direct_dynamics_cascade_memberwise_affine import compare
from .direct_dynamics_sit_left_censor_audit import decode_with_exact_sit_zero
from .direct_dynamics_sit_support_decoder_scoring import (
    _score as score_support_decoder,
    _stage_gate as support_decoder_stage_gate,
    apply_support_decoder_physical,
)
from .trainer import _atomic_json
from .transforms import channel_denormalize


_ACTIVE_TRACKER: ClearMLTracker | None = None


def _case_anchor(case_id: str) -> date:
    if len(case_id) != 18 or case_id[10:16] != "_slice" or not case_id[16:].isdigit():
        raise ValueError(f"invalid frozen case id: {case_id}")
    slice_index = int(case_id[16:])
    if not 0 <= slice_index < 24:
        raise ValueError(f"invalid archive slice in frozen case id: {case_id}")
    return date.fromisoformat(case_id[:10])


def _validate_confirmation_panel(experiment: dict[str, Any]) -> None:
    panel = experiment.get("panel")
    if not isinstance(panel, dict) or panel.get("role") != "frozen_confirmation":
        return
    if panel.get("forecast_window_days") != 10 or panel.get(
        "minimum_anchor_separation_days"
    ) != 10:
        raise ValueError("confirmation requires disjoint d0..d9 forecast windows")
    development = panel.get("development_case_ids")
    if not isinstance(development, list) or len(development) != 12:
        raise ValueError("confirmation requires the frozen 12-case development panel")
    confirmation = experiment["case_ids"]
    anchors = [
        ("development", case_id, _case_anchor(case_id)) for case_id in development
    ] + [("confirmation", case_id, _case_anchor(case_id)) for case_id in confirmation]
    for left in range(len(anchors)):
        for right in range(left + 1, len(anchors)):
            separation = abs((anchors[left][2] - anchors[right][2]).days)
            if separation < 10:
                raise ValueError(
                    "confirmation/development forecast windows overlap: "
                    f"{anchors[left][1]} vs {anchors[right][1]}"
                )
    gate = experiment.get("decision_gate")
    if not isinstance(gate, dict) or gate.get("bootstrap_draws") != 100000:
        raise ValueError("confirmation decision gate must be frozen before sampling")
    if gate.get("primary") != (
        "case_equal_six_channel_train_standardized_fair_crps"
    ):
        raise ValueError("confirmation primary score differs from reviewed protocol")


def _primary_standardized_fair_crps(
    ensemble: torch.Tensor, truth: torch.Tensor, valid: torch.Tensor
) -> dict[str, Any]:
    case_values = [
        float(
            standardized_fair_crps(
                ensemble[index : index + 1].double(),
                truth[index : index + 1].double(),
                valid[index : index + 1].double(),
            ).item()
        )
        for index in range(ensemble.shape[0])
    ]
    return {
        "case_equal_mean": float(sum(case_values) / len(case_values)),
        "case_values": case_values,
    }


def _paired_primary_confirmation(
    raw: dict[str, Any], candidate: dict[str, Any], gate: dict[str, Any]
) -> dict[str, Any]:
    raw_values = torch.tensor(raw["case_values"], dtype=torch.float64)
    candidate_values = torch.tensor(candidate["case_values"], dtype=torch.float64)
    if raw_values.shape != (12,) or candidate_values.shape != (12,):
        raise ValueError("confirmation primary requires exactly twelve paired dates")
    delta = candidate_values - raw_values
    generator = torch.Generator().manual_seed(int(gate["bootstrap_seed"]))
    indices = torch.randint(
        12,
        (int(gate["bootstrap_draws"]), 12),
        generator=generator,
    )
    bootstrap = delta[indices].mean(dim=1)
    estimate = float(delta.mean().item())
    ci_low = float(torch.quantile(bootstrap, 0.025).item())
    ci_high = float(torch.quantile(bootstrap, 0.975).item())
    return {
        "delta_refined_minus_raw": estimate,
        "relative_change_of_aggregate": float(
            candidate["case_equal_mean"] / raw["case_equal_mean"] - 1.0
        ),
        "paired_date_bootstrap_95_ci_low": ci_low,
        "paired_date_bootstrap_95_ci_high": ci_high,
        "bootstrap_draws": int(gate["bootstrap_draws"]),
        "bootstrap_seed": int(gate["bootstrap_seed"]),
        "primary_passed": estimate < 0.0 and ci_high < 0.0,
    }


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


def _apply_frozen_support_decoder(
    forecast_normalized: torch.Tensor,
    coarse_normalized: torch.Tensor,
    valid: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Decode, project SIT support, and re-encode without fitting any parameter."""
    means = torch.as_tensor(means, dtype=torch.float32)
    stds = torch.as_tensor(stds, dtype=torch.float32)
    physical = decode_with_exact_sit_zero(forecast_normalized.float(), means, stds)
    coarse_physical = decode_with_exact_sit_zero(
        coarse_normalized.float(), means, stds
    )
    decoded = apply_support_decoder_physical(physical, coarse_physical, valid)
    means5 = means.reshape(1, 1, 6, 1, 1).to(decoded)
    stds5 = stds.reshape(1, 1, 6, 1, 1).to(decoded)
    normalized = (decoded - means5) / stds5
    members = forecast_normalized.shape[1]
    recovered, fraction = masked_block_average(
        decoded.flatten(0, 1),
        valid[:, None].expand(-1, members, -1, -1, -1).flatten(0, 1),
    )
    target = coarse_physical.flatten(0, 1).clone()
    target[:, 1::2] = target[:, 1::2].clamp_min(0)
    error = float(
        (recovered - target)[fraction.expand_as(recovered) > 0].abs().max()
    )
    if error > 3e-6:
        raise RuntimeError("support decoder changed its prescribed physical coarse means")
    return normalized, decoded, error


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
    candidate_label = experiment.get(
        "candidate_label", "proper64_ema9711_fine4096"
    )
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", candidate_label) is None:
        raise ValueError("unsafe paired candidate label")
    _validate_confirmation_panel(experiment)
    support_decoder = experiment.get("support_decoder")
    if support_decoder is not None:
        if experiment.get("panel", {}).get("role") != "frozen_confirmation":
            raise ValueError("support decoder may only run on a frozen confirmation panel")
        if support_decoder.get("law") != "sit_nonnegative_block_simplex_projection":
            raise ValueError("unreviewed support decoder law")
        if support_decoder.get("fit_on_confirmation") is not False:
            raise ValueError("support decoder confirmation must not fit on confirmation data")
        inventory = experiment["panel"].get("prior_date_level_inventory_search", {})
        if inventory.get("selected_date_matches") != 0 or not inventory.get(
            "checked_before_manifest_commit"
        ):
            raise ValueError("confirmation dates were not frozen against prior inventory")


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
    support_decoder = experiment.get("support_decoder")
    coarse_root = Path(coarse["run_dir"])
    fine_root = Path(fine["run_dir"])
    for relative, expected in coarse["sha256"].items():
        _verify_file(coarse_root / relative, expected, f"coarse/{relative}")
    for relative, expected in fine["sha256"].items():
        _verify_file(fine_root / relative, expected, f"fine/{relative}")
    refinement_path = Path(refinement["checkpoint"])
    _verify_file(refinement_path, refinement["sha256"], "proper refinement checkpoint")
    training_record_path = Path(refinement["training_record"])
    _verify_file(
        training_record_path,
        refinement["training_record_sha256"],
        "proper refinement training record",
    )
    training_record = load_json(training_record_path)
    if training_record.get("checkpoint_sha256") != refinement["sha256"]:
        raise ValueError("training record and refinement checkpoint SHA differ")
    if training_record.get("code_identity", {}).get("git_commit") != refinement[
        "code_commit"
    ]:
        raise ValueError("training record commit differs from paired refinement")
    if support_decoder is not None:
        repository_root = Path(__file__).resolve().parents[1]
        inventory = experiment["panel"]["prior_date_level_inventory_search"]
        inventory_path = Path(inventory["inventory_path"])
        if inventory_path.is_absolute() or ".." in inventory_path.parts:
            raise ValueError("unsafe prior-date inventory path")
        _verify_file(
            repository_root / inventory_path,
            inventory["sorted_date_inventory_sha256"],
            "prior-date inventory",
        )
        inventory_dates = (repository_root / inventory_path).read_text(
            encoding="utf-8"
        ).splitlines()
        if (
            inventory_dates != sorted(set(inventory_dates))
            or len(inventory_dates) != inventory["observed_2022_date_count"]
        ):
            raise ValueError("prior-date inventory is not the frozen sorted unique list")
        selected_dates = {case_id[:10] for case_id in experiment["case_ids"]}
        if selected_dates.intersection(inventory_dates):
            raise ValueError("confirmation date occurs in the prior-result inventory")
        for key, label in (
            ("implementation", "support decoder implementation"),
            ("scoring", "support decoder scoring"),
        ):
            spec = support_decoder[key]
            relative = Path(spec["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe {label} path")
            _verify_file(repository_root / relative, spec["sha256"], label)

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
    if payload.get("completed_updates") != 64:
        raise ValueError("refinement checkpoint differs from the reviewed protocol")
    _validate_reviewed_protocol(payload.get("protocol", {}))
    if training_record.get("protocol") != payload.get("protocol"):
        raise ValueError("training record and checkpoint protocols differ")
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
        "raw_unclipped_scoring": support_decoder is None,
        "display_only_projection": support_decoder is None,
        "variant_scoring_semantics": {
            "raw_ema9711_fine4096": "source raw, unclipped",
            experiment.get("candidate_label", "proper64_ema9711_fine4096"): (
                "fixed SIT support decoder is part of the scored candidate law; "
                "no additional clipping or display-only projection"
                if support_decoder is not None
                else "source raw, unclipped"
            ),
            "stored_coarse_and_residual": (
                "source/predecoder tensors; candidate residual does not reconstruct "
                "the post-decoder forecast"
                if support_decoder is not None
                else "source tensors"
            ),
        },
        "optimizer_steps": 0,
        "panel": experiment.get("panel"),
        "decision_gate": experiment.get("decision_gate"),
        "coarse": coarse,
        "fine": fine,
        "refinement": refinement,
        "support_decoder": support_decoder,
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
    if support_decoder is not None:
        means = torch.as_tensor(means, dtype=torch.float32)
        stds = torch.as_tensor(stds, dtype=torch.float32)
    if support_decoder is None:
        truth_physical = channel_denormalize(truth_normalized, means, stds)
        persistence_physical = channel_denormalize(
            persistence_normalized, means, stds
        )
    else:
        truth_physical = decode_with_exact_sit_zero(truth_normalized, means, stds)
        persistence_physical = decode_with_exact_sit_zero(
            persistence_normalized, means, stds
        )

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
    deferred_visuals: list[tuple[str, int, torch.Tensor]] = []
    candidate_label = experiment.get(
        "candidate_label", "proper64_ema9711_fine4096"
    )
    for label, predictor in (
        ("raw_ema9711_fine4096", raw_predictor),
        (candidate_label, refined_predictor),
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
        predecoder_normalized = ensemble["forecast"].float()
        recovered, fraction = masked_block_average(
            predecoder_normalized.flatten(0, 1),
            valid[:, None].expand(-1, 8, -1, -1, -1).flatten(0, 1),
        )
        predecoder_exact_error = float(
            (recovered - ensemble["coarse"].flatten(0, 1))[
                fraction.expand_as(recovered) > 0
            ].abs().max()
        )
        if predecoder_exact_error > 3e-6:
            raise RuntimeError("fine forecast changed generated coarse 2x2 means")
        if support_decoder is None:
            predecoder_physical = channel_denormalize(
                predecoder_normalized.flatten(0, 1), means, stds
            ).unflatten(0, (12, 8))
        else:
            predecoder_physical = decode_with_exact_sit_zero(
                predecoder_normalized, means, stds
            )
        physical = predecoder_physical
        normalized = predecoder_normalized
        support_decoder_applied = label == candidate_label and support_decoder is not None
        support_coarse_error = None
        if support_decoder_applied:
            normalized, physical, support_coarse_error = (
                _apply_frozen_support_decoder(
                    predecoder_normalized,
                    ensemble["coarse"],
                    valid,
                    means,
                    stds,
                )
            )
        evidence_path = output / f"{label}_raw.pt"
        evidence_payload = {
            "forecast_normalized": normalized,
            "pre_support_decoder_forecast_normalized": (
                predecoder_normalized if support_decoder_applied else None
            ),
            "coarse_normalized": ensemble["coarse"],
            "residual_normalized": ensemble["residual"],
            "residual_semantics": (
                "source_pre_support_decoder"
                if support_decoder_applied
                else "source_forecast_residual"
            ),
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
            "refinement": refinement if label == candidate_label else None,
            "support_decoder": support_decoder if support_decoder_applied else None,
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
        metrics["boundary_events"] = boundary_event_metrics(
            physical, truth_physical, valid
        )
        metrics["exact_coarse_2x2_mean_error_max"] = predecoder_exact_error
        metrics["support_decoder_physical_coarse_error_max"] = support_coarse_error
        metrics["primary_standardized_fair_crps"] = _primary_standardized_fair_crps(
            normalized, truth_normalized, valid
        )
        if support_decoder is not None:
            metrics["support_decoder_scoring"] = score_support_decoder(
                physical, truth_physical, valid, stds.float()
            )
        metrics["summary"] = _summary(metrics)
        _require_finite_scalars(metrics)
        result["candidates"][label] = {
            "metrics": metrics,
            "raw_evidence": evidence_path.name,
            "raw_evidence_sha256": evidence_sha256,
        }

        for position in (0, 4, 7, 11):
            deferred_visuals.append((label, position, physical[position].clone()))
        del ensemble, normalized, predecoder_normalized, predecoder_physical, physical
        torch.cuda.empty_cache()

    raw_summary = result["candidates"]["raw_ema9711_fine4096"]["metrics"]["summary"]
    refined_summary = result["candidates"][candidate_label]["metrics"]["summary"]
    result["paired_summary_delta_refined_minus_raw"] = {
        key: float(refined_summary[key] - raw_summary[key]) for key in raw_summary
    }
    raw_metrics = result["candidates"]["raw_ema9711_fine4096"]["metrics"]
    refined_metrics = result["candidates"][candidate_label]["metrics"]
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
    if experiment.get("panel", {}).get("role") == "frozen_confirmation":
        result["frozen_confirmation"] = _paired_primary_confirmation(
            raw_metrics["primary_standardized_fair_crps"],
            refined_metrics["primary_standardized_fair_crps"],
            experiment["decision_gate"],
        )
        if support_decoder is not None:
            raw_support = raw_metrics["support_decoder_scoring"]
            candidate_support = refined_metrics["support_decoder_scoring"]
            comparison = compare(
                candidate_support,
                raw_support,
                experiment["decision_gate"],
                0,
            )
            support_gate = support_decoder_stage_gate(
                candidate_support,
                raw_support,
                comparison,
                experiment["decision_gate"],
            )
            result["support_decoder_confirmation"] = {
                "candidate_law": support_decoder["law"],
                "comparison_versus_champion": comparison,
                "stage_gate": support_gate,
            }
            secondary_passed = support_gate["passed_pending_visual_review"]
            result["frozen_confirmation"]["secondary_passed"] = secondary_passed
            result["frozen_confirmation"]["numerical_decision"] = (
                "GO_PENDING_VISUAL_REVIEW"
                if result["frozen_confirmation"]["primary_passed"]
                and secondary_passed
                else "HOLD"
            )
        else:
            result["frozen_confirmation"]["numerical_decision"] = (
                "GO_PENDING_SECONDARY_AND_VISUAL_REVIEW"
                if result["frozen_confirmation"]["primary_passed"]
                else "HOLD"
            )
    elif experiment.get("panel", {}).get("role") == "development_reuse":
        result["paired_development_diagnostic"] = _paired_primary_confirmation(
            raw_metrics["primary_standardized_fair_crps"],
            refined_metrics["primary_standardized_fair_crps"],
            experiment["decision_gate"],
        )
        result["paired_development_diagnostic"]["claim_status"] = (
            "DEVELOPMENT_ONLY_NOT_INDEPENDENT_CONFIRMATION"
        )
    _require_finite_scalars(result)
    _atomic_json(output / "proper_refinement_e2e_evaluation.json", result)
    for label in ("raw_ema9711_fine4096", candidate_label):
        rank_path = ranks_dir / f"{label}.png"
        _save_rank_histograms(
            rank_path, label, result["candidates"][label]["metrics"]
        )
        tracker.report_image("rank_histograms", label, rank_path, 0)
    for label, position, values in deferred_visuals:
        visual_path = visuals_dir / f"{label}_case{position:02d}_all8_fixed.png"
        _save_contact_sheet(
            visual_path, label, case_ids[position], values,
            truth_physical[position], persistence_physical[position], valid[position],
            {"sic": (0.0, 1.0), "sit": (0.0, 3.5)},
        )
        tracker.report_image(
            "fixed_scale_members", f"{label}/case{position:02d}", visual_path, 0
        )
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
