"""Paired generated-coarse validation of terminal fine support refinement."""

from __future__ import annotations

import argparse
import copy
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
from .config import load_json
from .data import build_dataset
from .direct_dynamics_affine_calibration import boundary_event_metrics
from .direct_dynamics_cascade import masked_block_average
from .direct_dynamics_cascade_checkpoint_evaluation import _save_rank_histograms
from .direct_dynamics_cascade_e2e_evaluation import _add_joint_consistency, _summary
from .direct_dynamics_cascade_end_to_end import CascadePredictor, load_cascade_predictor
from .direct_dynamics_cascade_fine import validate_fine_condition
from .direct_dynamics_cascade_fine_training import _clean_code_identity, _fine_collate
from .direct_dynamics_cascade_memberwise_affine import compare
from .direct_dynamics_cascade_paired_evaluation import (
    _require_finite_scalars,
    _save_contact_sheet,
    score_ensemble,
)
from .direct_dynamics_cascade_proper_refinement_evaluation import (
    _atomic_torch_save,
    _paired_primary_confirmation,
    _primary_standardized_fair_crps,
    _sample_with_reviewed_precision,
)
from .direct_dynamics_fine_support_proper_admission import apply_training_sic_decoder
from .direct_dynamics_fine_support_proper_integration import (
    fine_frozen_prefix,
    terminal_fine_residual,
)
from .direct_dynamics_fine_support_proper_training import EXPECTED_PROTOCOL
from .direct_dynamics_sic_support_decoder_scoring import (
    _stage_gate as support_decoder_stage_gate,
    canonical_physical_decode,
    diagnostics as score_support_decoder,
)
from .direct_dynamics_training import _repeat_field_stats, validate_direct_dataset
from .runtime import make_normalized_xy_grid
from .trainer import _atomic_json


_ACTIVE_TRACKER: ClearMLTracker | None = None


class TerminalRefinedFineSampler:
    """Use the source fine model for 31 RK4 intervals and a terminal copy for one."""

    def __init__(self, source_sampler: Any, terminal_model: torch.nn.Module):
        self.source_sampler = source_sampler
        self.sampler = source_sampler.sampler
        self.terminal_model = terminal_model.eval()

    def project_initial_noise(
        self, raw_noise: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.source_sampler.project_initial_noise(raw_noise, valid_mask)

    @torch.no_grad()
    def sample_conditioned(self, **kwargs: Any) -> torch.Tensor:
        if (
            kwargs.get("num_timesteps") != 33
            or kwargs.get("method") != "rk4"
            or float(kwargs.get("end_time", math.nan)) != 0.0
        ):
            raise ValueError("terminal fine refinement requires exact RK4-33 to t=0")
        condition = kwargs.get("model_conditioning")
        white = kwargs.get("initial_noise")
        valid = kwargs.get("valid_mask")
        if condition is None or white is None or valid is None:
            raise ValueError("terminal fine refinement requires condition, noise, and mask")
        valid = valid[:, :1].to(device=white.device, dtype=torch.float32)
        condition = condition.to(device=white.device, dtype=torch.float32)
        checked_mask, lift = validate_fine_condition(condition, valid)
        colored = self.project_initial_noise(white.float(), checked_mask)
        grid = make_normalized_xy_grid(
            *white.shape[-2:], device=white.device, dtype=torch.float32
        )
        prefix = fine_frozen_prefix(
            self.sampler.model, colored, condition, checked_mask, grid
        )
        residual = terminal_fine_residual(
            self.terminal_model, prefix, condition, checked_mask, grid
        )
        return lift + residual


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or _sha256(path) != expected:
        raise ValueError(f"frozen {label} SHA mismatch: {path}")


def _launch_status(status: str, **details: Any) -> None:
    raw = os.environ.get("FINE_SUPPORT_EVAL_STATUS_PATH", "").strip()
    if not raw:
        return
    path = Path(raw)
    if not path.is_absolute() or path.name != "status.json":
        raise ValueError("FINE_SUPPORT_EVAL_STATUS_PATH must be an absolute status.json")
    _atomic_json(path, {"status": status, **details})


def _persist_and_register(
    path: Path,
    payload: dict[str, Any],
    contract_path: Path,
    contract: dict[str, Any],
) -> str:
    """Make evidence and its contract binding durable before fallible analysis."""
    sha256 = _atomic_torch_save(payload, path)
    contract.setdefault("evidence_sha256", {})[path.name] = sha256
    _atomic_json(contract_path, contract)
    return sha256


def _validate_config(experiment: dict[str, Any]) -> None:
    if experiment.get("schema_version") != "fine_support_terminal_paired_evaluation_v1":
        raise ValueError("unreviewed fine support evaluation schema")
    if experiment.get("split") != "valid" or experiment.get("panel", {}).get("role") != "development_reuse":
        raise ValueError("fine support evaluation is restricted to development validation-2022")
    if experiment.get("cases") != 12 or experiment.get("members") != 8:
        raise ValueError("fine support evaluation requires exactly 12 cases x 8 members")
    if experiment.get("coarse_rk4_timepoints") != 17 or experiment.get("fine_rk4_timepoints") != 33:
        raise ValueError("fine support evaluation requires coarse RK4-17 and fine RK4-33")
    if len(experiment.get("case_indices", [])) != 12 or len(set(experiment["case_indices"])) != 12:
        raise ValueError("fine support evaluation requires 12 frozen unique indices")
    if len(experiment.get("case_ids", [])) != 12 or len(set(experiment["case_ids"])) != 12:
        raise ValueError("fine support evaluation requires 12 frozen unique identities")
    gate = experiment.get("decision_gate", {})
    if (
        gate.get("primary") != "case_equal_six_channel_train_standardized_fair_crps"
        or gate.get("bootstrap_draws") != 100000
        or gate.get("bootstrap_seed") != 20260912
        or gate.get("per_output_score_tolerance") != 0.01
        or gate.get("per_output_rank_tv_tolerance") != 0.01
        or gate.get("roughness_tolerance") != 0.02
        or gate.get("temporal_tolerance") != 0.01
    ):
        raise ValueError("fine support decision gate differs from Astra review")
    if experiment.get("scoring_law") != {
        "primary": "canonical_physical_then_identical_fixed_budget_sic_decoder_both_branches",
        "secondary": "canonical_physical_raw_predecoder_both_branches",
        "sit_transform": "none",
    }:
        raise ValueError("fine support scored law differs from Astra review")
    if experiment.get("precision") != {
        "network": "bf16",
        "ode_state": "fp32",
        "score": "fp64",
    }:
        raise ValueError("fine support evaluation precision differs from preflight")
    if experiment.get("test_2023") != "closed" or experiment.get("optimizer_steps") != 0:
        raise ValueError("evaluation must keep test-2023 closed and perform zero optimization")


def _load_refinement(
    spec: dict[str, Any], source_model: torch.nn.Module, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any], dict[str, Any]]:
    run = Path(spec["run_dir"])
    status_path = run / spec["status"]
    record_path = run / spec["training_record"]
    checkpoint_path = run / spec["checkpoint"]
    _verify(status_path, spec["status_sha256"], "training status")
    _verify(record_path, spec["training_record_sha256"], "training record")
    _verify(checkpoint_path, spec["checkpoint_sha256"], "terminal fine checkpoint")
    status = load_json(status_path)
    record = load_json(record_path)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    counters = (64, 64, 64, 64)
    for source in (status, record):
        observed = (
            source.get("completed_updates"),
            source.get("optimizer_steps"),
            source.get("attempted_optimizer_steps"),
            source.get("executed_optimizer_steps"),
        )
        if observed != counters:
            raise ValueError("training provenance does not prove exact 64 committed updates")
        if source.get("code_identity", {}).get("git_commit") != spec["code_commit"]:
            raise ValueError("training provenance commit mismatch")
        if source.get("test_2023_used") not in (False, None):
            raise ValueError("training provenance unexpectedly used test-2023")
    if (
        status.get("status") != "complete"
        or record.get("status") != "training_complete_pending_generated_coarse_validation"
        or status.get("terminal_checkpoint_sha256") != spec["checkpoint_sha256"]
        or record.get("terminal_checkpoint_sha256") != spec["checkpoint_sha256"]
        or payload.get("schema_version") != "terminal_fine_support_model_v1"
        or payload.get("completed_updates") != 64
        or payload.get("optimizer_steps") != 64
        or payload.get("protocol") != EXPECTED_PROTOCOL
        or record.get("protocol") != EXPECTED_PROTOCOL
        or payload.get("config_sha256") != spec["training_config_sha256"]
        or payload.get("source_checkpoint_sha256") != spec["source_fine_checkpoint_sha256"]
        or status.get("config_sha256") != spec["training_config_sha256"]
    ):
        raise ValueError("terminal fine checkpoint differs from reviewed training contract")
    terminal = copy.deepcopy(source_model)
    terminal.load_state_dict(payload["terminal_model_state"], strict=True)
    terminal.to(device=device, dtype=torch.float32).eval()
    if not all(torch.isfinite(parameter).all() for parameter in terminal.parameters()):
        raise FloatingPointError("terminal fine checkpoint contains NaN/Inf")
    return terminal, status, record, payload


def _decode_sic_law(
    forecast: torch.Tensor,
    coarse: torch.Tensor,
    valid: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    physical = canonical_physical_decode(forecast, means, stds)
    coarse_physical = canonical_physical_decode(coarse, means, stds)
    decoded = apply_training_sic_decoder(
        physical.double(), coarse_physical.double(), valid.double()
    )
    if not torch.equal(decoded[:, :, 1::2], physical[:, :, 1::2].double()):
        raise RuntimeError("scored SIC decoder changed SIT")
    members = forecast.shape[1]
    member_mask = valid[:, None].expand(-1, members, -1, -1, -1).flatten(0, 1)
    recovered, fraction = masked_block_average(decoded.flatten(0, 1)[:, 0::2], member_mask)
    target = coarse_physical.flatten(0, 1)[:, 0::2].clamp(0, 1)
    error = float((recovered - target)[fraction.expand_as(recovered) > 0].abs().max())
    if error > 3e-14:
        raise RuntimeError("scored SIC law lost its generated coarse budget")
    active = valid[:, None].expand(-1, members, -1, -1, -1) > 0
    sic = decoded[:, :, 0::2]
    sic_active = active.expand(-1, -1, 3, -1, -1)
    if torch.any(sic[sic_active] < 0) or torch.any(sic[sic_active] > 1):
        raise RuntimeError("scored SIC law violates exact [0,1] support")
    means5 = means.double().reshape(1, 1, 6, 1, 1)
    stds5 = stds.double().reshape(1, 1, 6, 1, 1)
    return (decoded - means5) / stds5, decoded, error


def _save_sic_member_delta_panel(
    path: Path,
    case_id: str,
    lead_days: int,
    channel: int,
    control: torch.Tensor,
    candidate: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    ocean = valid[0].numpy() > 0
    figure, axes = plt.subplots(8, 3, figsize=(12, 24))
    for member in range(8):
        left = control[member, channel].numpy()
        right = candidate[member, channel].numpy()
        delta = right - left
        for column, (value, title, limits, cmap) in enumerate(
            (
                (left, "control mean", (0.0, 1.0), "Blues"),
                (right, "candidate mean", (0.0, 1.0), "Blues"),
                (delta, "candidate-control", (-0.15, 0.15), "RdBu_r"),
            )
        ):
            axes[member, column].imshow(
                np.where(ocean, value, np.nan), cmap=cmap, vmin=limits[0], vmax=limits[1]
            )
            axes[member, column].set_title(f"member {member}: {title}")
            axes[member, column].axis("off")
    figure.suptitle(f"{case_id}: d{lead_days} SIC matched raw members and delta")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


@torch.no_grad()
def run(config_path: Path, output: Path) -> dict[str, Any]:
    global _ACTIVE_TRACKER
    if torch.cuda.device_count() != 1:
        raise RuntimeError("fine support evaluation requires exactly one visible GPU")
    if os.environ.get("CLEARML_REQUIRE_ONLINE") != "1":
        raise RuntimeError("fine support evaluation requires online ClearML")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to reuse evaluation output {output}")
    experiment = load_json(config_path)
    _validate_config(experiment)
    coarse = experiment["coarse"]
    fine = experiment["fine"]
    refinement = experiment["refinement"]
    coarse_root = Path(coarse["run_dir"])
    fine_root = Path(fine["run_dir"])
    for relative, sha in coarse["sha256"].items():
        _verify(coarse_root / relative, sha, f"coarse/{relative}")
    for relative, sha in fine["sha256"].items():
        _verify(fine_root / relative, sha, f"fine/{relative}")
    repository = Path(__file__).resolve().parents[1]
    for label, spec in experiment["implementation_bindings"].items():
        _verify(repository / spec["path"], spec["sha256"], label)

    coarse_config = load_json(coarse_root / "config.json")
    fine_config = load_json(fine_root / "config.json")
    coarse_metadata = load_json(coarse_root / "metadata.json")
    fine_metadata = load_json(fine_root / "metadata.json")
    if coarse_metadata["data_config"] != fine_metadata["data_config"]:
        raise ValueError("coarse and fine sources use different target laws")
    dataset = build_dataset(fine_metadata["data_config"], split="valid")
    validate_direct_dataset(dataset)
    indices = list(experiment["case_indices"])
    batch = _fine_collate([dataset[index] for index in indices])
    case_ids = tuple(batch["meta"]["case_id"])
    if list(case_ids) != experiment["case_ids"]:
        raise ValueError("frozen validation identities differ from the archive")
    code_identity = _clean_code_identity(repository)
    output.mkdir(parents=True)
    ranks = output / "rank_histograms"
    visuals = output / "fixed_scale_members"
    deltas = output / "sic_deltas"
    ranks.mkdir(); visuals.mkdir(); deltas.mkdir()

    valid = batch["valid_mask"][:, :1].float()
    truth_source = batch["truth"].float()
    persistence_source = batch["background"].float()
    means = torch.as_tensor(_repeat_field_stats(dataset.means), dtype=torch.float32)
    stds = torch.as_tensor(_repeat_field_stats(dataset.stds), dtype=torch.float32)
    truth_physical = canonical_physical_decode(truth_source, means, stds)
    persistence_physical = canonical_physical_decode(persistence_source, means, stds)
    means4 = means.double().reshape(1, 6, 1, 1)
    stds4 = stds.double().reshape(1, 6, 1, 1)
    truth_normalized = (truth_physical - means4) / stds4
    persistence_normalized = (persistence_physical - means4) / stds4
    contract = {
        "schema_version": experiment["schema_version"],
        "code_identity": code_identity,
        "optimizer_steps": 0,
        "test_2023_used": False,
        "split": "valid",
        "panel": experiment["panel"],
        "decision_gate": experiment["decision_gate"],
        "scoring_law": experiment["scoring_law"],
        "precision": experiment["precision"],
        "case_indices": indices,
        "case_ids": list(case_ids),
        "members": 8,
        "coarse_rk4_timepoints": 17,
        "fine_rk4_timepoints": 33,
        "coarse": coarse,
        "fine": fine,
        "refinement": refinement,
        "evidence_sha256": {},
    }
    contract_path = output / "contract.json"
    _atomic_json(contract_path, contract)
    fixed_inputs = output / "fixed_inputs.pt"
    fixed_inputs_sha = _persist_and_register(
        fixed_inputs,
        {
            "structured_conditioning": batch["structured_conditioning"].float(),
            "valid_mask": valid,
            "truth_normalized": truth_normalized,
            "persistence_normalized": persistence_normalized,
            "truth_physical": truth_physical,
            "persistence_physical": persistence_physical,
            "normalization_means": means,
            "normalization_stds": stds,
            "case_indices": indices,
            "case_ids": case_ids,
            "member_indices": tuple(range(8)),
        },
        contract_path,
        contract,
    )
    contract["fixed_inputs_sha256"] = fixed_inputs_sha
    _atomic_json(contract_path, contract)
    tracker = ClearMLTracker(
        experiment["project_name"], f"{experiment['task_name']}-{output.name}",
        tags=experiment["clearml"]["tags"], env_path=experiment["clearml"]["env_path"],
    )
    _ACTIVE_TRACKER = tracker
    tracker.connect("evaluation_contract", contract)
    _launch_status("sampling", clearml_task_id=str(tracker.task.id), code_commit=code_identity["git_commit"])

    device = torch.device("cuda:0")
    condition = batch["structured_conditioning"].float().to(device)
    valid_device = valid.to(device)
    source_predictor = load_cascade_predictor(
        coarse_run_dir=str(coarse_root), coarse_checkpoint_name=coarse["checkpoint"],
        coarse_model_config=coarse_config, coarse_checkpoint_sha256=coarse["sha256"][coarse["checkpoint"]],
        fine_run_dir=str(fine_root), fine_checkpoint_name=fine["checkpoint"],
        fine_model_config=fine_config, fine_checkpoint_sha256=fine["sha256"][fine["checkpoint"]],
        expected_coarse_code_commit=coarse["code_commit"], expected_fine_code_commit=fine["code_commit"],
        replay_code_commit=code_identity["git_commit"],
        expected_forecast_contract_sha256=experiment["forecast_contract_sha256"], device=device,
        expected_fine_conditioning_implementation_sha256=fine["implementation_sha256"]["fine_conditioning"],
        expected_fine_preconditioning_implementation_sha256=fine["implementation_sha256"]["fine_preconditioning"],
        expected_fine_colored_implementation_sha256=fine["implementation_sha256"]["fine_colored"],
    )
    terminal, training_status, training_record, terminal_payload = _load_refinement(
        refinement, source_predictor.fine_sampler.sampler.model, device
    )
    expected_source = {
        "run_dir": fine["run_dir"],
        "checkpoint": fine["checkpoint"],
        "code_commit": fine["code_commit"],
        "forecast_contract_sha256": experiment["forecast_contract_sha256"],
        "files_sha256": fine["sha256"],
        "implementation_sha256": fine["implementation_sha256"],
    }
    if any(training_record["source"].get(key) != value for key, value in expected_source.items()):
        raise ValueError("training record source differs from paired fine control")
    normalization = training_record.get("normalization", {})
    if (
        normalization.get("sha256") != fine["sha256"]["metadata.json"]
        or normalization.get("fields") != ["siconc", "sithic"]
        or normalization.get("means") != [float(value) for value in dataset.means]
        or normalization.get("stds") != [float(value) for value in dataset.stds]
        or training_record.get("effective_forecast_contract_sha256")
        != experiment["forecast_contract_sha256"]
    ):
        raise ValueError("evaluation normalization/forecast law differs from training")
    delta_square = torch.zeros((), dtype=torch.float64, device=device)
    base_square = torch.zeros((), dtype=torch.float64, device=device)
    delta_max = 0.0
    for base, candidate in zip(
        source_predictor.fine_sampler.sampler.model.parameters(), terminal.parameters(), strict=True
    ):
        difference = candidate.double() - base.double()
        delta_square += difference.square().sum(); base_square += base.double().square().sum()
        delta_max = max(delta_max, float(difference.abs().max().cpu()))
    if delta_max == 0.0:
        raise RuntimeError("terminal fine refinement is identical to its source")
    candidate_predictor = CascadePredictor(
        source_predictor.coarse_sampler,
        TerminalRefinedFineSampler(source_predictor.fine_sampler, terminal),
        replay_identity={
            **(source_predictor.replay_identity or {}),
            "terminal_fine_checkpoint_sha256": refinement["checkpoint_sha256"],
            "terminal_fine_training_commit": refinement["code_commit"],
        },
    )
    control_terminal = copy.deepcopy(source_predictor.fine_sampler.sampler.model).eval()
    hybrid_control = CascadePredictor(
        source_predictor.coarse_sampler,
        TerminalRefinedFineSampler(source_predictor.fine_sampler, control_terminal),
    )
    solver = dict(
        coarse_num_timesteps=17, fine_num_timesteps=33, method="rk4",
        rtol=1e-5, atol=1e-6, end_time=0.0, device=device,
    )
    parity_source = _sample_with_reviewed_precision(
        source_predictor, member_indices=(0,), structured_conditioning=condition[:1],
        valid_mask=valid_device[:1], case_ids=case_ids[:1], **solver,
    )
    parity_hybrid = _sample_with_reviewed_precision(
        hybrid_control, member_indices=(0,), structured_conditioning=condition[:1],
        valid_mask=valid_device[:1], case_ids=case_ids[:1], **solver,
    )
    parity_path = output / "unrefined_hybrid_parity.pt"
    _persist_and_register(
        parity_path,
        {
            "production": parity_source,
            "unrefined_hybrid": parity_hybrid,
            "fixed_inputs_sha256": fixed_inputs_sha,
        },
        contract_path,
        contract,
    )
    parity_error = float((parity_source["forecast"] - parity_hybrid["forecast"]).abs().max())
    if parity_error > 2e-5:
        raise RuntimeError("unrefined terminal fine hybrid does not replay production")

    result: dict[str, Any] = {
        "status": "numerical_complete_pending_visual_review",
        "publication_claim_permitted": False,
        "clearml_task_id": str(tracker.task.id),
        "optimizer_steps": 0,
        "test_2023_used": False,
        "unrefined_hybrid_production_max_abs": parity_error,
        "parameter_change": {
            "l2": float(torch.sqrt(delta_square).cpu()),
            "relative_l2": float(torch.sqrt(delta_square / base_square).cpu()),
            "max_abs": delta_max,
        },
        "candidates": {},
    }
    reference_randomness = None
    visual_values: dict[str, torch.Tensor] = {}
    raw_visual_values: dict[str, torch.Tensor] = {}
    labels = (experiment["baseline_label"], experiment["candidate_label"])
    for label, predictor in zip(labels, (source_predictor, candidate_predictor), strict=True):
        ensemble = _sample_with_reviewed_precision(
            predictor, member_indices=tuple(range(8)), structured_conditioning=condition,
            valid_mask=valid_device, case_ids=case_ids, **solver,
        )
        raw_evidence_path = output / f"{label}_raw_ensemble.pt"
        raw_evidence_sha = _persist_and_register(
            raw_evidence_path,
            {
                "forecast_normalized": ensemble["forecast"],
                "coarse_normalized": ensemble["coarse"],
                "residual_normalized": ensemble["residual"],
                "raw_coarse_noise": ensemble["raw_coarse_noise"],
                "raw_fine_noise": ensemble["raw_fine_noise"],
                "projected_fine_noise": ensemble["projected_fine_noise"],
                "coarse_seeds": ensemble["coarse_seeds"],
                "fine_seeds": ensemble["fine_seeds"],
                "member_indices": ensemble["member_indices"],
                "case_ids": ensemble["case_ids"],
                "solver": ensemble["solver"],
                "replay_identity": ensemble["replay_identity"],
                "fixed_inputs_sha256": fixed_inputs_sha,
            },
            contract_path,
            contract,
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
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in randomness.items()
            }
            reference_coarse = ensemble["coarse"].clone()
        else:
            for key, value in randomness.items():
                reference = reference_randomness[key]
                equal = torch.equal(reference, value) if torch.is_tensor(value) else reference == value
                if not equal:
                    raise RuntimeError(f"paired branches differ in {key}")
            if not torch.equal(reference_coarse, ensemble["coarse"]):
                raise RuntimeError("paired branches differ in generated coarse members")
        raw_normalized = ensemble["forecast"].float()
        member_mask = valid[:, None].expand(-1, 8, -1, -1, -1).flatten(0, 1)
        recovered, fraction = masked_block_average(raw_normalized.flatten(0, 1), member_mask)
        predecoder_error = float(
            (recovered - ensemble["coarse"].flatten(0, 1))[
                fraction.expand_as(recovered) > 0
            ].abs().max()
        )
        if predecoder_error > 3e-6:
            raise RuntimeError("fine branch changed its generated coarse member")
        raw_physical = canonical_physical_decode(raw_normalized, means, stds)
        scored_normalized, scored_physical, decoded_error = _decode_sic_law(
            raw_normalized, ensemble["coarse"], valid, means, stds
        )
        evidence_path = output / f"{label}_scored_law.pt"
        evidence_sha = _persist_and_register(
            evidence_path,
            {
                "raw_forecast_normalized": raw_normalized,
                "scored_forecast_normalized": scored_normalized,
                "raw_forecast_physical": raw_physical,
                "scored_forecast_physical": scored_physical,
                "truth_physical": truth_physical,
                "persistence_physical": persistence_physical,
                "truth_normalized": truth_normalized,
                "persistence_normalized": persistence_normalized,
                "normalization_means": means,
                "normalization_stds": stds,
                "fixed_inputs_sha256": fixed_inputs_sha,
                "coarse_normalized": ensemble["coarse"],
                "residual_normalized": ensemble["residual"],
                **randomness,
                "solver": ensemble["solver"],
                "replay_identity": ensemble["replay_identity"],
            },
            contract_path,
            contract,
        )
        primary = score_ensemble(
            scored_normalized, scored_physical, truth_normalized, truth_physical,
            persistence_physical, valid, valid,
        )
        raw_secondary = score_ensemble(
            raw_normalized, raw_physical, truth_normalized, truth_physical,
            persistence_physical, valid, valid,
        )
        for metrics, physical in ((primary, scored_physical), (raw_secondary, raw_physical)):
            _add_joint_consistency(metrics, physical, valid)
            metrics["boundary_events"] = boundary_event_metrics(physical, truth_physical, valid)
            metrics["summary"] = _summary(metrics)
        primary["support_decoder_scoring"] = score_support_decoder(
            scored_physical, truth_physical, valid, stds
        )
        primary["primary_standardized_fair_crps"] = _primary_standardized_fair_crps(
            scored_normalized, truth_normalized, valid
        )
        primary["predecoder_exact_coarse_2x2_mean_error_max"] = predecoder_error
        primary["decoded_sic_generated_coarse_error_max"] = decoded_error
        _require_finite_scalars(primary); _require_finite_scalars(raw_secondary)
        result["candidates"][label] = {
            "primary_metrics": primary,
            "raw_secondary_metrics": raw_secondary,
            "forecast_evidence": evidence_path.name,
            "forecast_evidence_sha256": evidence_sha,
            "raw_ensemble_evidence": raw_evidence_path.name,
            "raw_ensemble_evidence_sha256": raw_evidence_sha,
        }
        visual_values[label] = scored_physical.clone()
        raw_visual_values[label] = raw_physical.clone()
        del ensemble, raw_normalized, raw_physical, scored_normalized, scored_physical
        torch.cuda.empty_cache()

    result["paired_generated_coarse_bitwise_equal"] = True
    result["paired_random_streams_bitwise_equal"] = {
        "raw_coarse_noise": True,
        "raw_fine_noise": True,
        "projected_colored_fine_noise": True,
        "coarse_seeds": True,
        "fine_seeds": True,
        "member_indices": True,
        "case_ids": True,
    }
    baseline = result["candidates"][labels[0]]["primary_metrics"]
    candidate = result["candidates"][labels[1]]["primary_metrics"]
    decision = _paired_primary_confirmation(
        baseline["primary_standardized_fair_crps"],
        candidate["primary_standardized_fair_crps"],
        experiment["decision_gate"],
    )
    comparison = compare(
        candidate["support_decoder_scoring"], baseline["support_decoder_scoring"],
        experiment["decision_gate"], 0,
    )
    secondary = support_decoder_stage_gate(
        candidate["support_decoder_scoring"], baseline["support_decoder_scoring"],
        comparison, experiment["decision_gate"],
    )
    result["paired_development_diagnostic"] = {
        **decision,
        "secondary_gate": secondary,
        "comparison": comparison,
        "numerical_decision": (
            "GO_PENDING_VISUAL_REVIEW"
            if decision["primary_passed"] and secondary["passed_pending_visual_review"]
            else "HOLD"
        ),
        "claim_status": "DEVELOPMENT_ONLY_NOT_INDEPENDENT_CONFIRMATION",
    }
    _require_finite_scalars(result)
    numerical_path = output / "fine_support_paired_evaluation.json"
    _atomic_json(numerical_path, result)

    for label in labels:
        for role, metrics in (
            ("primary", result["candidates"][label]["primary_metrics"]),
            ("raw_secondary", result["candidates"][label]["raw_secondary_metrics"]),
        ):
            path = ranks / f"{label}_{role}.png"
            _save_rank_histograms(path, f"{label} {role}", metrics)
            tracker.report_image("rank_histograms", f"{label}/{role}", path, 0)
        for position in (0, 4, 7, 11):
            path = visuals / f"{label}_case{position:02d}_all8_scored_fixed.png"
            _save_contact_sheet(
                path, label, case_ids[position], visual_values[label][position],
                truth_physical[position], persistence_physical[position], valid[position],
                {"sic": (0.0, 1.0), "sit": (0.0, 3.5)},
            )
            tracker.report_image("fixed_scale_members", f"{label}/case{position:02d}", path, 0)
    for position in (0, 4, 7, 11):
        for lead_days, channel in ((3, 0), (6, 2), (9, 4)):
            path = deltas / f"case{position:02d}_d{lead_days}_sic_raw_member_deltas.png"
            _save_sic_member_delta_panel(
                path, case_ids[position], lead_days, channel,
                raw_visual_values[labels[0]][position],
                raw_visual_values[labels[1]][position], valid[position],
            )
            tracker.report_image(
                "sic_raw_member_delta", f"case{position:02d}/d{lead_days}", path, 0
            )
    tracker.upload_artifact("fine_support_paired_evaluation", numerical_path)
    tracker.close(); _ACTIVE_TRACKER = None
    _launch_status(
        "complete_pending_visual_review", clearml_task_id=result["clearml_task_id"],
        code_commit=code_identity["git_commit"], numerical_decision=result["paired_development_diagnostic"]["numerical_decision"],
    )
    return result


def _terminate(signum: int, _frame: Any) -> None:
    raise TimeoutError(f"fine support evaluation received signal {signum}")


def main() -> None:
    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        value = run(arguments.config.resolve(), arguments.output.resolve())
    except BaseException as error:
        try:
            if arguments.output.is_dir():
                contract_path = arguments.output / "contract.json"
                contract_binding = None
                if contract_path.is_file():
                    persisted_contract = load_json(contract_path)
                    contract_binding = {
                        "sha256": _sha256(contract_path),
                        "evidence_sha256": persisted_contract.get("evidence_sha256", {}),
                    }
                _atomic_json(
                    arguments.output / "failure.json",
                    {
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error)[:2000],
                        "contract": contract_binding,
                    },
                )
        except BaseException:
            pass
        try:
            _launch_status("failed", error_type=type(error).__name__, error=str(error)[:2000])
        except BaseException:
            pass
        if _ACTIVE_TRACKER is not None:
            try:
                _ACTIVE_TRACKER.task.mark_failed(status_reason=str(error)[:1000])
            except BaseException:
                pass
            try:
                _ACTIVE_TRACKER.close()
            except BaseException:
                pass
        raise
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
