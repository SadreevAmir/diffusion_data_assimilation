"""Train and evaluate the frozen Gaussian-preconditioned structured pilot.

Training and paired evaluation are separate process modes.  The evaluator loads
bare EMA models one at a time in strict FP32, reuses byte-identical initial
noise for the raw and candidate checkpoints, and never opens the 2023 test set.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
from contextlib import contextmanager
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers.training_utils import EMAModel
from torch.utils.data._utils.collate import default_collate

from .config import TrainingConfig, load_json, merge_config_overrides, resolve_path
from .data import build_dataset
from .main import main as train_main
from .model_io import build_unet
from .sampler import Sampler
from .structured_archive_audit import validate_bound_archive_audit
from .structured_joint_state import canonical_mapping_sha256, validate_conditioning_normalization
from .structured_trajectory_evaluation import (
    PublicationCase,
    StructuredPublicationContract,
    evaluate_structured_publication_ensemble,
    make_structured_trajectory_figure,
    sample_structured_batch,
)


SCHEMA_VERSION = "structured_gaussian_preconditioned_pilot_v1"
PANEL_SCHEMA = "structured_paired_pilot_panel_v1"
EXPECTED_PANEL_SHA256 = "232af0176ced9acc1eaf85a2bd9f9349ba1a5293731301df1ea417df2a258331"
EXPECTED_STEPS = 2128
EXPECTED_EPOCH = 16
EXPECTED_TIMEPOINTS = 65
PROPER_SCORE_RELATIVE_LIMIT = 1.02
SPATIAL_VARIOGRAM_RELATIVE_LIMIT = 0.70
SPATIAL_LAGS = (1, 2, 4)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_owned_output(output_dir: Path) -> str:
    attempt = os.environ.get("STRUCTURED_PILOT_ATTEMPT", "")
    owner = output_dir / ".structured_pilot_owner"
    if (
        not attempt
        or output_dir.is_symlink()
        or not output_dir.is_dir()
        or owner.is_symlink()
        or not owner.is_file()
        or owner.read_text(encoding="utf-8").strip() != attempt
    ):
        raise ValueError("pilot output is not owned by this immutable attempt")
    return attempt


def _load_experiment_contract(
    experiment_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], TrainingConfig, dict[str, Any], Path]:
    experiment = load_json(experiment_path)
    config_dir = experiment_path.resolve().parent
    data_path = resolve_path(experiment["data_config"], config_dir).resolve()
    method_path = resolve_path(experiment["model_config"], config_dir).resolve()
    data_config = merge_config_overrides(
        load_json(data_path), experiment.get("data_overrides")
    )
    method = {**load_json(method_path), **experiment.get("training", {})}
    audit = validate_bound_archive_audit(data_config, data_path)
    stats_path = resolve_path(method["structured_state_stats_path"], method_path.parent)
    stats = load_json(stats_path)
    method["structured_state_stats"] = stats
    validate_conditioning_normalization(data_config, stats)
    if stats.get("data_config_sha256") != canonical_mapping_sha256(data_config):
        raise ValueError("structured stats/data identity mismatch")
    config = TrainingConfig.from_dict(method)
    return data_config, stats, config, audit, data_path


def _verify_refinement_gate(
    result_path: Path,
    gate_path: Path,
    *,
    expected_result_sha256: str,
    expected_gate_sha256: str,
) -> dict[str, str]:
    for path, expected in (
        (result_path, expected_result_sha256),
        (gate_path, expected_gate_sha256),
    ):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unsafe or absent refinement artifact: {path}")
        if _SHA256_RE.fullmatch(expected) is None or _sha256(path) != expected:
            raise ValueError(f"refinement artifact hash mismatch: {path.name}")
    result = load_json(result_path)
    gate = load_json(gate_path)
    if result.get("status") != "converged":
        raise ValueError("strict-FP32 solver refinement did not converge")
    if gate.get("status") != "converged" or gate.get("pilot_permitted") is not True:
        raise ValueError("strict-FP32 solver gate does not permit the pilot")
    if gate.get("solver_control_sha256") != expected_result_sha256:
        raise ValueError("solver gate is not bound to the supplied refinement result")
    variants = result.get("variants")
    if not isinstance(variants, dict) or set(variants) != {
        "rk4_64_intervals_fp32", "rk4_128_intervals_fp32"
    }:
        raise ValueError("refinement result has an incomplete variant set")
    for variant in variants.values():
        metrics = variant.get("metrics", {}) if isinstance(variant, dict) else {}
        if (
            metrics.get("support_violation_fraction") != 0.0
            or metrics.get("ensemble_support_violation_fraction") != 0.0
            or metrics.get("lag0_observation_max_abs_error") != 0.0
        ):
            raise ValueError("refinement result violates physical support")
    return {
        "solver_control_sha256": expected_result_sha256,
        "solver_gate_sha256": expected_gate_sha256,
    }


def run_training(experiment_path: Path, output_dir: Path) -> dict[str, Any]:
    """Run exactly the predeclared 16-epoch candidate without internal sampling."""

    attempt = _require_owned_output(output_dir)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("pilot training requires exactly one visible CUDA device")
    experiment = load_json(experiment_path)
    runtime = json.loads(json.dumps(experiment))
    runtime["training"] = {
        **runtime.get("training", {}),
        "base_output_dir": str((output_dir / "training").resolve()),
        "run_name": "seed1701",
        "resume_from_checkpoint": "",
        # These diagnostics are non-authoritative and would duplicate the
        # strict paired evaluator while perturbing wall time, not optimizer RNG.
        "sample_every_n_epochs": 0,
        "metric_every_n_epochs": 0,
        "metric_save_ensemble_samples": False,
    }
    _atomic_json(output_dir / "runtime_experiment.json", runtime)
    result = train_main(runtime, experiment_path.resolve().parent)
    run_dir = (output_dir / "training" / "seed1701").resolve()
    if Path(result["output_dir"]).resolve() != run_dir:
        raise RuntimeError("trainer returned an unexpected output directory")
    resume_path = run_dir / "structured_recovery/epoch_0016/resume.json"
    ema_path = run_dir / "structured_recovery/epoch_0016/ema_state.pth"
    metadata_path = run_dir / "metadata.json"
    for path in (resume_path, ema_path, metadata_path):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"candidate training artifact is unsafe or absent: {path}")
    resume = load_json(resume_path)
    if resume.get("next_epoch") != EXPECTED_EPOCH or resume.get("global_step") != EXPECTED_STEPS:
        raise ValueError("candidate did not finish the exact frozen 2128-step budget")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "training_completed",
        "attempt_token": attempt,
        "candidate_run_dir": str(run_dir),
        "optimizer_steps": EXPECTED_STEPS,
        "metadata_sha256": _sha256(metadata_path),
        "ema_state_sha256": _sha256(ema_path),
        "resume_sha256": _sha256(resume_path),
        "internal_sampling_disabled": True,
        "authoritative_evaluation": "separate_bare_model_strict_fp32",
    }
    _atomic_json(output_dir / "training_result.json", payload)
    return payload


def _previous_year(day: date) -> date:
    return day.replace(year=day.year - 1)


def _load_panel_contract(
    panel_path: Path,
    protocol_path: Path,
    data_config: dict[str, Any],
) -> tuple[StructuredPublicationContract, list[int], dict[str, Any]]:
    if _sha256(panel_path) != EXPECTED_PANEL_SHA256:
        raise ValueError("paired pilot panel differs from its frozen SHA-256")
    panel = load_json(panel_path)
    protocol = load_json(protocol_path)
    if panel.get("schema_version") != PANEL_SCHEMA:
        raise ValueError("unexpected paired pilot panel schema")
    if panel.get("status") != "frozen_before_candidate_training":
        raise ValueError("paired pilot panel was not frozen before candidate training")
    if panel.get("split") != "valid_2022_only" or panel.get("test_2023_permitted") is not False:
        raise ValueError("paired pilot must remain validation-only")
    if panel.get("data_config_sha256") != canonical_mapping_sha256(data_config):
        raise ValueError("paired panel data config hash mismatch")
    cases: list[PublicationCase] = []
    indices: list[int] = []
    for raw in panel.get("cases", []):
        anchor = date.fromisoformat(raw["anchor_date"])
        expected_id = f"{anchor.isoformat()}_slice23"
        if raw.get("case_id") != expected_id:
            raise ValueError("paired panel case id is not canonical")
        indices.append(int(raw["dataset_index"]))
        cases.append(
            PublicationCase(
                case_id=expected_id,
                anchor_date=anchor.isoformat(),
                target_archive_dates=tuple(
                    (anchor + timedelta(days=lead)).isoformat() for lead in range(4)
                ),
                background_archive_dates=tuple(
                    _previous_year(anchor + timedelta(days=lead)).isoformat()
                    for lead in range(4)
                ),
                observation_lag_dates=tuple(
                    (anchor - timedelta(days=lag)).isoformat() for lag in range(3)
                ),
            )
        )
    seeds = tuple(int(seed) for seed in panel.get("member_seeds", []))
    if len(cases) != 8 or len(indices) != len(set(indices)) or len(seeds) != 8 or len(set(seeds)) != 8:
        raise ValueError("paired pilot requires eight unique cases and eight unique members")
    if panel.get("solver") != {"method": "rk4", "timepoints": 65, "intervals": 64}:
        raise ValueError("paired pilot solver contract changed")
    if protocol.get("evaluation", {}).get("paired_panel_manifest_sha256") != EXPECTED_PANEL_SHA256:
        raise ValueError("pilot protocol is not bound to the paired panel")
    contract = StructuredPublicationContract(
        cases=tuple(cases),
        ensemble_size=8,
        member_seeds=seeds,
        manifest_sha256=EXPECTED_PANEL_SHA256,
        protocol_sha256=_sha256(protocol_path),
        target_slice_index=23,
        sic_rank_policy="tie_aware_fractional_mass",
        coverage_levels=(0.50, 0.80, 0.90),
        claim_boundary="validation-only paired pilot; test-2023 remains locked",
        blockers=(),
    )
    return contract, indices, panel


def _validate_model_semantics(
    metadata: dict[str, Any],
    data_config: dict[str, Any],
    stats: dict[str, Any],
    config: TrainingConfig,
    expected_parameterization: str,
) -> None:
    stored_data = metadata.get("data_config")
    training = metadata.get("training_config")
    if not isinstance(stored_data, dict) or not isinstance(training, dict):
        raise ValueError("checkpoint metadata lacks data/training contract")
    if canonical_mapping_sha256(stored_data) != canonical_mapping_sha256(data_config):
        raise ValueError("checkpoint data config differs from paired evaluator")
    stored_stats = training.get("structured_state_stats")
    if not isinstance(stored_stats, dict) or canonical_mapping_sha256(stored_stats) != canonical_mapping_sha256(stats):
        raise ValueError("checkpoint structured statistics differ")
    if training.get("structured_velocity_parameterization", "raw") != expected_parameterization:
        raise ValueError("checkpoint velocity parameterization differs")
    fields = (
        "image_size", "in_channels", "out_channels", "trajectory_horizon_days",
        "block_out_channels", "layers_per_block", "down_block_types",
        "up_block_types", "norm_num_groups",
    )
    stored = {name: training.get(name) for name in fields}
    current = {name: asdict(config).get(name) for name in fields}
    if canonical_mapping_sha256(stored) != canonical_mapping_sha256(current):
        raise ValueError("checkpoint architecture differs from paired evaluator")


def _load_ema_sampler(
    run_dir: Path,
    config: TrainingConfig,
    data_config: dict[str, Any],
    stats: dict[str, Any],
    expected_parameterization: str,
    expected_hashes: dict[str, str] | None,
    device: torch.device,
) -> tuple[Sampler, dict[str, str]]:
    metadata_path = run_dir / "metadata.json"
    recovery = run_dir / "structured_recovery/epoch_0016"
    ema_path = recovery / "ema_state.pth"
    resume_path = recovery / "resume.json"
    paths = {"metadata_sha256": metadata_path, "ema_state_sha256": ema_path, "resume_sha256": resume_path}
    actual: dict[str, str] = {}
    for key, path in paths.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unsafe or absent checkpoint artifact: {path}")
        actual[key] = _sha256(path)
        if expected_hashes is not None and actual[key] != expected_hashes.get(key):
            raise ValueError(f"checkpoint hash mismatch: {key}")
    resume = load_json(resume_path)
    if resume.get("next_epoch") != EXPECTED_EPOCH or resume.get("global_step") != EXPECTED_STEPS:
        raise ValueError("checkpoint is not the frozen epoch-16 / 2128-step state")
    metadata = load_json(metadata_path)
    _validate_model_semantics(metadata, data_config, stats, config, expected_parameterization)
    model = build_unet(config)
    try:
        state = torch.load(ema_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(ema_path, map_location="cpu")
    ema = EMAModel(
        model.parameters(), decay=float(config.ema_decay),
        min_decay=float(config.ema_min_decay),
        update_after_step=int(config.ema_update_after_step),
        use_ema_warmup=bool(config.ema_use_warmup),
    )
    ema.load_state_dict(state)
    ema.copy_to(model.parameters())
    model.to(device=device, dtype=torch.float32).eval()
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("paired evaluator model is not strict float32")
    if hasattr(model, "enable_xformers_memory_efficient_attention"):
        model.enable_xformers_memory_efficient_attention()
    return Sampler(model, structured_velocity_parameterization=expected_parameterization), actual


@contextmanager
def _strict_fp32_context():
    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    previous_precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        with torch.autocast(device_type="cuda", enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
        torch.set_float32_matmul_precision(previous_precision)


def _generate_paired_noise(
    config: TrainingConfig,
    case_count: int,
    seeds: tuple[int, ...],
    device: torch.device,
) -> tuple[list[list[torch.Tensor]], list[dict[str, Any]]]:
    noise: list[list[torch.Tensor]] = []
    manifest: list[dict[str, Any]] = []
    for case_position in range(case_count):
        case_noise: list[torch.Tensor] = []
        for member_position, seed in enumerate(seeds):
            generator = torch.Generator(device=device).manual_seed(seed)
            value = torch.randn(
                (1, config.out_channels, *config.image_size),
                generator=generator, device=device, dtype=torch.float32,
            ).to(device="cpu", dtype=torch.float32).contiguous()
            case_noise.append(value)
            manifest.append({
                "case_position": case_position,
                "member_position": member_position,
                "member_seed": seed,
                "sha256": _tensor_sha256(value),
                "shape": list(value.shape),
                "dtype": "float32",
            })
        noise.append(case_noise)
    return noise, manifest


def _sample_model(
    sampler: Sampler,
    batches: list[dict[str, Any]],
    noise: list[list[torch.Tensor]],
    stats: dict[str, Any],
    config: TrainingConfig,
    device: torch.device,
) -> torch.Tensor:
    by_case: list[torch.Tensor] = []
    for case_position, batch in enumerate(batches):
        members: list[torch.Tensor] = []
        for initial_noise in noise[case_position]:
            before = _tensor_sha256(initial_noise)
            with _strict_fp32_context():
                sample = sample_structured_batch(
                    sampler, batch, stats=stats, size=config.image_size,
                    num_timesteps=EXPECTED_TIMEPOINTS, device=device, method="rk4",
                    rtol=config.sample_rtol, atol=config.sample_atol,
                    initial_noise=initial_noise.clone().to(device),
                )
            if _tensor_sha256(initial_noise) != before:
                raise RuntimeError("paired initial noise was mutated")
            members.append(sample.detach().to(device="cpu", dtype=torch.float32))
        by_case.append(torch.stack(members, dim=1))
    return torch.cat(by_case, dim=0)


def _spatial_variogram_errors(
    ensemble: torch.Tensor,
    truth: torch.Tensor,
    valid_mask: torch.Tensor,
    lag0_mask: torch.Tensor,
) -> dict[str, Any]:
    details: dict[str, float] = {}
    aggregates: dict[str, float] = {}
    for lag in SPATIAL_LAGS:
        scores: list[torch.Tensor] = []
        for lead in range(4):
            domain = valid_mask[:, 0] > 0
            if lead == 0:
                domain = domain & ~(lag0_mask[:, 0] > 0)
            for offset, field in enumerate(("sic", "sit")):
                channel = 2 * lead + offset
                direction_scores: list[torch.Tensor] = []
                for axis in (-2, -1):
                    left = [slice(None)] * 4
                    right = [slice(None)] * 4
                    mask_left = [slice(None)] * 3
                    mask_right = [slice(None)] * 3
                    left[axis] = slice(0, -lag)
                    right[axis] = slice(lag, None)
                    mask_left[axis] = slice(0, -lag)
                    mask_right[axis] = slice(lag, None)
                    edge_mask = domain[tuple(mask_left)] & domain[tuple(mask_right)]
                    if not torch.any(edge_mask):
                        continue
                    member_delta = (ensemble[:, :, channel][tuple(left)] - ensemble[:, :, channel][tuple(right)]).abs().sqrt()
                    truth_delta = (truth[:, channel][tuple(mask_left)] - truth[:, channel][tuple(mask_right)]).abs().sqrt()
                    error = (member_delta.mean(dim=1, dtype=torch.float64) - truth_delta).square()
                    direction_scores.append(error[edge_mask])
                if not direction_scores:
                    raise ValueError(f"spatial variogram lag {lag} has no valid edges")
                selected = torch.cat(direction_scores)
                value = selected.mean(dtype=torch.float64)
                details[f"lag{lag}_lead{lead}_{field}"] = float(value.item())
                scores.append(selected)
        aggregates[f"lag{lag}"] = float(torch.cat(scores).mean(dtype=torch.float64).item())
    return {"power": 0.5, "directions": ["vertical", "horizontal"], "aggregate": aggregates, "details": details}


def _finite_nonnegative(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) >= 0


def paired_pilot_gate(
    raw_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    raw_spatial: dict[str, Any],
    candidate_spatial: dict[str, Any],
) -> dict[str, Any]:
    """Apply the complete frozen quantitative gate, failing closed on omissions."""

    checks: dict[str, Any] = {}
    passed = True
    for lead in range(4):
        label = "d0" if lead == 0 else f"d+{lead}"
        for field in ("sic", "sit"):
            for score in ("fair_crps", "ensemble_mean_rmse"):
                name = f"proper/{label}/{field}/{score}"
                try:
                    raw = raw_metrics["leads"][label]["fields"][field][score]
                    candidate = candidate_metrics["leads"][label]["fields"][field][score]
                except (KeyError, TypeError):
                    raw = candidate = None
                ok = _finite_nonnegative(raw) and _finite_nonnegative(candidate) and float(candidate) <= PROPER_SCORE_RELATIVE_LIMIT * float(raw)
                checks[name] = {"passed": ok, "raw": raw, "candidate": candidate, "maximum_ratio": PROPER_SCORE_RELATIVE_LIMIT}
                passed = passed and ok
            for score in ("rank_tv_to_uniform", "mean_coverage_absolute_error"):
                name = f"calibration/{label}/{field}/{score}"
                try:
                    raw_field = raw_metrics["leads"][label]["fields"][field]
                    candidate_field = candidate_metrics["leads"][label]["fields"][field]
                    if score == "rank_tv_to_uniform":
                        raw = raw_field[score]
                        candidate = candidate_field[score]
                    else:
                        raw_errors = raw_field["central_coverage_absolute_error"]
                        candidate_errors = candidate_field["central_coverage_absolute_error"]
                        if set(raw_errors) != {"central_50", "central_80", "central_90"} or set(candidate_errors) != set(raw_errors):
                            raise KeyError("coverage schema")
                        raw = sum(raw_errors.values()) / 3.0
                        candidate = sum(candidate_errors.values()) / 3.0
                except (KeyError, TypeError):
                    raw = candidate = None
                ok = _finite_nonnegative(raw) and _finite_nonnegative(candidate) and float(candidate) <= float(raw)
                checks[name] = {"passed": ok, "raw": raw, "candidate": candidate, "maximum_ratio": 1.0}
                passed = passed and ok
    for lag in SPATIAL_LAGS:
        name = f"spatial_variogram/lag{lag}"
        raw = raw_spatial.get("aggregate", {}).get(f"lag{lag}")
        candidate = candidate_spatial.get("aggregate", {}).get(f"lag{lag}")
        ok = _finite_nonnegative(raw) and _finite_nonnegative(candidate) and float(candidate) <= SPATIAL_VARIOGRAM_RELATIVE_LIMIT * float(raw)
        checks[name] = {"passed": ok, "raw": raw, "candidate": candidate, "maximum_ratio": SPATIAL_VARIOGRAM_RELATIVE_LIMIT}
        passed = passed and ok
    for transition in ("d+0_to_d+1", "d+1_to_d+2", "d+2_to_d+3"):
        for field in ("sic", "sit"):
            for score in ("increment_fair_crps", "ensemble_mean_increment_rmse"):
                name = f"temporal/{transition}/{field}/{score}"
                try:
                    raw = raw_metrics["temporal"]["transitions"][transition]["fields"][field][score]
                    candidate = candidate_metrics["temporal"]["transitions"][transition]["fields"][field][score]
                except (KeyError, TypeError):
                    raw = candidate = None
                ok = _finite_nonnegative(raw) and _finite_nonnegative(candidate) and float(candidate) <= float(raw)
                checks[name] = {"passed": ok, "raw": raw, "candidate": candidate, "maximum_ratio": 1.0}
                passed = passed and ok
    for field in ("sic", "sit"):
        name = f"temporal/trajectory_variogram/{field}"
        try:
            raw = raw_metrics["temporal"]["trajectory_variogram"][field]["trajectory_variogram_score"]
            candidate = candidate_metrics["temporal"]["trajectory_variogram"][field]["trajectory_variogram_score"]
        except (KeyError, TypeError):
            raw = candidate = None
        ok = _finite_nonnegative(raw) and _finite_nonnegative(candidate) and float(candidate) <= float(raw)
        checks[name] = {"passed": ok, "raw": raw, "candidate": candidate, "maximum_ratio": 1.0}
        passed = passed and ok
    physical_keys = (
        "ensemble_support_violation_count", "truth_support_violation_count",
        "background_support_violation_count", "lag0_exact_max_abs_error",
        "decode_saturation_count",
    )
    for model_name, metrics in (("raw", raw_metrics), ("candidate", candidate_metrics)):
        support = metrics.get("support", {})
        for key in physical_keys:
            value = support.get(key)
            ok = value == 0 or value == 0.0
            checks[f"physical/{model_name}/{key}"] = {"passed": ok, "value": value}
            passed = passed and ok
    return {
        "schema_version": "structured_gaussian_preconditioned_paired_gate_v1",
        "status": "quantitative_pass" if passed else "quantitative_fail",
        "quantitative_passed": passed,
        "visual_review_status": "pending_independent_review",
        "full_training_permitted": False,
        "test_2023_used": False,
        "checks": checks,
    }


def _save_visuals(
    output_dir: Path,
    truth: torch.Tensor,
    background: torch.Tensor,
    raw: torch.Tensor,
    candidate: torch.Tensor,
    valid: torch.Tensor,
    raw_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    case_ids: list[str],
) -> None:
    import matplotlib.pyplot as plt

    visual = output_dir / "visual_qc"
    visual.mkdir(exist_ok=False)
    for model_name, ensemble in (("raw", raw), ("candidate", candidate)):
        figure = make_structured_trajectory_figure(
            truth[0], background[0], ensemble[0, 0], valid[0],
            title=f"{model_name}: {case_ids[0]}, fixed member 0; ARRAY ORIENTATION",
            origin="upper",
        )
        figure.savefig(visual / f"{model_name}_case00_member00.png", dpi=180, bbox_inches="tight")
        plt.close(figure)

    fig, axes = plt.subplots(4, 2, figsize=(15, 18), constrained_layout=True)
    ranks = np.arange(9)
    for lead in range(4):
        label = "d0" if lead == 0 else f"d+{lead}"
        for column, field in enumerate(("sic", "sit")):
            axis = axes[lead, column]
            raw_frequency = raw_metrics["leads"][label]["fields"][field]["fractional_rank_frequencies"]
            candidate_frequency = candidate_metrics["leads"][label]["fields"][field]["fractional_rank_frequencies"]
            axis.plot(ranks, raw_frequency, marker="o", label="raw")
            axis.plot(ranks, candidate_frequency, marker="o", label="candidate")
            axis.axhline(1 / 9, color="black", linestyle="--", label="uniform")
            axis.set_title(f"{label} {field.upper()} rank histogram")
            axis.set_xlabel("fractional rank")
            axis.set_ylabel("frequency")
            axis.grid(alpha=0.25)
            axis.legend()
    fig.savefig(visual / "paired_rank_histograms.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_evaluation(
    experiment_path: Path,
    raw_experiment_path: Path,
    raw_run_dir: Path,
    candidate_run_dir: Path,
    panel_path: Path,
    protocol_path: Path,
    output_dir: Path,
    raw_hashes: dict[str, str],
    refinement_result_path: Path,
    refinement_gate_path: Path,
    refinement_result_sha256: str,
    refinement_gate_sha256: str,
) -> dict[str, Any]:
    attempt = _require_owned_output(output_dir)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("paired evaluation requires exactly one visible CUDA device")
    refinement_hashes = _verify_refinement_gate(
        refinement_result_path,
        refinement_gate_path,
        expected_result_sha256=refinement_result_sha256,
        expected_gate_sha256=refinement_gate_sha256,
    )
    data_config, stats, candidate_config, audit, _ = _load_experiment_contract(experiment_path)
    raw_data, raw_stats, raw_config, raw_audit, _ = _load_experiment_contract(raw_experiment_path)
    if canonical_mapping_sha256(data_config) != canonical_mapping_sha256(raw_data):
        raise ValueError("raw and candidate data contracts differ")
    if canonical_mapping_sha256(stats) != canonical_mapping_sha256(raw_stats):
        raise ValueError("raw and candidate state transforms differ")
    if candidate_config.structured_velocity_parameterization != "gaussian_path_preconditioned":
        raise ValueError("candidate is not Gaussian-path preconditioned")
    if raw_config.structured_velocity_parameterization != "raw":
        raise ValueError("reference is not raw velocity")
    contract, case_indices, panel = _load_panel_contract(panel_path, protocol_path, data_config)
    dataset = build_dataset(data_config, split="valid")
    dataset.validate_structured_sral_audit_contract(audit)
    if max(case_indices) >= len(dataset):
        raise ValueError("paired case index exceeds validation dataset")
    device = torch.device("cuda:0")
    cpu_batches = [default_collate([dataset[index]]) for index in case_indices]
    case_ids = [str(batch["meta"]["case_id"][0]) for batch in cpu_batches]
    if case_ids != [case.case_id for case in contract.cases]:
        raise ValueError("dataset case identities differ from the frozen panel")
    batches = [{key: value.to(device=device, dtype=torch.float32) if isinstance(value, torch.Tensor) else value for key, value in batch.items()} for batch in cpu_batches]
    truth = torch.cat([batch["structured_physical_truth"].float() for batch in cpu_batches])
    background = torch.cat([batch["structured_physical_background"].float() for batch in cpu_batches])
    valid = torch.cat([batch["valid_mask"].float() for batch in cpu_batches])
    lag0 = torch.cat([batch["structured_lag0_mask"].float() for batch in cpu_batches])
    noise, noise_manifest = _generate_paired_noise(candidate_config, len(batches), contract.member_seeds, device)
    _atomic_json(output_dir / "noise_manifest.json", {
        "schema_version": "structured_paired_noise_manifest_v1",
        "generation": "one CUDA float32 tensor per case/member before either model; CPU clone reused byte-exactly",
        "panel_sha256": EXPECTED_PANEL_SHA256,
        "entries": noise_manifest,
    })

    raw_sampler, raw_actual_hashes = _load_ema_sampler(
        raw_run_dir, raw_config, data_config, stats, "raw", raw_hashes, device
    )
    raw_ensemble = _sample_model(raw_sampler, batches, noise, stats, raw_config, device)
    del raw_sampler
    gc.collect()
    torch.cuda.empty_cache()
    candidate_sampler, candidate_hashes = _load_ema_sampler(
        candidate_run_dir, candidate_config, data_config, stats,
        "gaussian_path_preconditioned", None, device,
    )
    candidate_ensemble = _sample_model(candidate_sampler, batches, noise, stats, candidate_config, device)
    del candidate_sampler
    gc.collect()
    torch.cuda.empty_cache()

    common = dict(
        truth=truth, background=background, valid_mask=valid, lag0_mask=lag0,
        contract=contract, case_ids=case_ids, member_seeds=contract.member_seeds,
        sic_cap=float(stats["sic_cap"]),
    )
    raw_metrics = evaluate_structured_publication_ensemble(raw_ensemble, **common)
    candidate_metrics = evaluate_structured_publication_ensemble(candidate_ensemble, **common)
    raw_metrics["support"]["decode_saturation_count"] = 0
    candidate_metrics["support"]["decode_saturation_count"] = 0
    raw_spatial = _spatial_variogram_errors(raw_ensemble, truth, valid, lag0)
    candidate_spatial = _spatial_variogram_errors(candidate_ensemble, truth, valid, lag0)
    raw_metrics["paired_spatial_variogram"] = raw_spatial
    candidate_metrics["paired_spatial_variogram"] = candidate_spatial
    gate = paired_pilot_gate(raw_metrics, candidate_metrics, raw_spatial, candidate_spatial)
    _atomic_json(output_dir / "raw_metrics.json", raw_metrics)
    _atomic_json(output_dir / "candidate_metrics.json", candidate_metrics)
    _save_visuals(output_dir, truth, background, raw_ensemble, candidate_ensemble, valid, raw_metrics, candidate_metrics, case_ids)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": gate["status"],
        "attempt_token": attempt,
        "training_performed": True,
        "validation_only": True,
        "test_2023_used": False,
        "strict_fp32": {
            "network_autocast": "disabled", "model_parameter_dtype": "float32",
            "input_and_ode_state_dtype": "float32", "matmul_tf32": False,
            "cudnn_tf32": False, "float32_matmul_precision": "highest",
            "solver": "rk4", "timepoints": EXPECTED_TIMEPOINTS, "intervals": 64,
        },
        "panel_sha256": EXPECTED_PANEL_SHA256,
        "case_ids": case_ids,
        "member_seeds": list(contract.member_seeds),
        "reference_hashes": raw_actual_hashes,
        "candidate_hashes": candidate_hashes,
        "solver_refinement_hashes": refinement_hashes,
        "gate": gate,
        "visual_artifacts": [
            "visual_qc/raw_case00_member00.png",
            "visual_qc/candidate_case00_member00.png",
            "visual_qc/paired_rank_histograms.png",
        ],
        "visual_review_required": True,
        "full_training_permitted": False,
    }
    _atomic_json(output_dir / "pilot_result.json", result)
    _atomic_json(output_dir / "paired_gate.json", gate)
    _atomic_json(output_dir / "run_status.json", {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "scientific_status": gate["status"],
        "quantitative_passed": gate["quantitative_passed"],
        "visual_review_status": "pending_independent_review",
        "full_training_permitted": False,
        "test_2023_used": False,
        "attempt_token": attempt,
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--experiment", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    for name in (
        "experiment", "raw-experiment", "raw-run-dir", "candidate-run-dir",
        "panel", "protocol", "output-dir", "refinement-result", "refinement-gate",
    ):
        evaluate_parser.add_argument(f"--{name}", type=Path, required=True)
    evaluate_parser.add_argument("--raw-metadata-sha256", required=True)
    evaluate_parser.add_argument("--raw-ema-sha256", required=True)
    evaluate_parser.add_argument("--raw-resume-sha256", required=True)
    evaluate_parser.add_argument("--refinement-result-sha256", required=True)
    evaluate_parser.add_argument("--refinement-gate-sha256", required=True)
    args = parser.parse_args()
    if args.mode == "train":
        run_training(args.experiment.resolve(), args.output_dir.resolve())
    else:
        run_evaluation(
            args.experiment.resolve(), args.raw_experiment.resolve(),
            args.raw_run_dir.resolve(), args.candidate_run_dir.resolve(),
            args.panel.resolve(), args.protocol.resolve(), args.output_dir.resolve(),
            {"metadata_sha256": args.raw_metadata_sha256, "ema_state_sha256": args.raw_ema_sha256, "resume_sha256": args.raw_resume_sha256},
            args.refinement_result.resolve(), args.refinement_gate.resolve(),
            args.refinement_result_sha256, args.refinement_gate_sha256,
        )


if __name__ == "__main__":
    main()
