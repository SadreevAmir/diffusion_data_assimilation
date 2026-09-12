"""CPU-only audit of coarse-training versus fine-grid calibration events.

The audit consumes frozen paired evidence.  It never samples a model and never
uses CUDA.  Its purpose is to test whether a score optimized on ``D(Y)`` is a
faithful proxy for the full-resolution events used by the publication gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
from pathlib import Path
from typing import Any

import torch

from .clearml_tracking import ClearMLTracker
from .config import load_json
from .direct_dynamics_cascade import masked_block_average, smooth_right_inverse
from .direct_dynamics_cascade_paired_evaluation import (
    _weighted_case_fair_crps,
)
from .direct_dynamics_cascade_fine_training import _clean_code_identity
from .direct_dynamics_threshold_weighted_score import boundary_emphasis_transform


OUTPUTS = ("d3_sic", "d3_sit", "d6_sic", "d6_sit", "d9_sic", "d9_sit")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_frozen(spec: dict[str, Any]) -> dict[str, Any]:
    path = Path(spec["path"])
    if not path.is_file() or _sha256(path) != spec["sha256"]:
        raise ValueError(f"frozen evidence SHA mismatch: {path}")
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def _strict_atomic_json(path: Path, payload: dict[str, Any]) -> None:
    def validate(value: Any) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise FloatingPointError("audit result contains NaN/Inf")
        if isinstance(value, dict):
            for item in value.values():
                validate(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                validate(item)

    validate(payload)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _record_failure(path: Path, reservation: dict[str, Any], error: BaseException) -> None:
    _strict_atomic_json(
        path,
        {
            **reservation,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error)[:2000],
        },
    )


def _finish_success(
    tracker: ClearMLTracker,
    status_path: Path,
    metrics_path: Path,
    terminal: dict[str, Any],
) -> None:
    """Mark complete only after the online tracker closes successfully."""
    tracker.close()
    _strict_atomic_json(
        status_path,
        {
            **terminal,
            "status": "complete",
            "metrics_path": str(metrics_path),
            "metrics_sha256": _sha256(metrics_path),
        },
    )


def _terminate(signum: int, _frame: Any) -> None:
    raise TimeoutError(f"coarse/fine audit received signal {signum}")


def _require_equal(left: dict[str, Any], right: dict[str, Any], label: str) -> None:
    for key in (
        "truth_normalized", "persistence_normalized", "valid_mask", "case_indices", "case_ids",
        "member_indices", "coarse_seeds", "fine_seeds", "raw_coarse_noise",
        "raw_fine_noise", "projected_fine_noise",
    ):
        a, b = left[key], right[key]
        equal = torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b
        if not equal:
            raise ValueError(f"paired evidence differs in {key}: {label}")


def _validate_payload(name: str, payload: dict[str, Any], spec: dict[str, Any], config: dict[str, Any]) -> None:
    shapes = {
        "forecast_normalized": (12, 8, 6, 320, 256),
        "coarse_normalized": (12, 8, 6, 160, 128),
        "residual_normalized": (12, 8, 6, 320, 256),
        "truth_normalized": (12, 6, 320, 256),
        "persistence_normalized": (12, 6, 320, 256),
        "valid_mask": (12, 1, 320, 256),
        "raw_coarse_noise": (12, 8, 6, 160, 128),
        "raw_fine_noise": (12, 8, 6, 320, 256),
        "projected_fine_noise": (12, 8, 6, 320, 256),
    }
    for key, shape in shapes.items():
        value = payload.get(key)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"{name}/{key} shape differs from frozen 12x8x6 contract")
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"{name}/{key} contains NaN/Inf")
    mask = payload["valid_mask"]
    if torch.any((mask != 0) & (mask != 1)) or torch.any(mask.sum(dim=(1, 2, 3)) == 0):
        raise ValueError(f"{name} mask must be binary and nonempty in every case")
    if payload.get("case_indices") != config["case_indices"]:
        raise ValueError(f"{name} case indices differ from frozen contract")
    if list(payload.get("case_ids", ())) != config["case_ids"]:
        raise ValueError(f"{name} case ids differ from frozen contract")
    if tuple(payload.get("member_indices", ())) != tuple(range(8)):
        raise ValueError(f"{name} member ids differ from frozen contract")
    if payload.get("solver") != config["solver"]:
        raise ValueError(f"{name} solver differs from frozen contract")
    if payload.get("coarse") != config["coarse"] or payload.get("fine") != config["fine"]:
        raise ValueError(f"{name} source model identity differs from frozen contract")
    replay = payload.get("replay_identity")
    if not isinstance(replay, dict):
        raise ValueError(f"{name} has no replay identity")
    for key, expected in config["common_replay_identity"].items():
        if replay.get(key) != expected:
            raise ValueError(f"{name} replay identity differs in {key}")
    if replay.get("replay_code_commit") != spec["replay_code_commit"]:
        raise ValueError(f"{name} replay commit differs from frozen evidence")
    refinement_present = bool(spec.get("refinement_present", False))
    if not refinement_present:
        if payload.get("refinement") is not None or "proper_refinement_checkpoint_sha256" in replay:
            raise ValueError(f"{name} unexpectedly contains a refinement")
    else:
        refinement = payload.get("refinement")
        if not isinstance(refinement, dict):
            raise ValueError(f"{name} is missing its expected refinement")
        if refinement.get("sha256") != spec["refinement_checkpoint_sha256"]:
            raise ValueError(f"{name} refinement checkpoint differs from frozen contract")
        if refinement.get("code_commit") != spec["refinement_training_commit"]:
            raise ValueError(f"{name} refinement commit differs from frozen contract")
        for key, expected in (
            ("proper_refinement_checkpoint_sha256", spec["refinement_checkpoint_sha256"]),
            ("proper_refinement_training_commit", spec["refinement_training_commit"]),
        ):
            if replay.get(key) != expected:
                raise ValueError(f"{name} refinement identity differs in {key}")


def _repeat_stats(values: list[float]) -> torch.Tensor:
    if len(values) != 2:
        raise ValueError("audit requires SIC/SIT means and stds")
    return torch.tensor(values * 3, dtype=torch.float64)


def _load_normalization(config: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    source = config["normalization_source"]
    path = Path(source["path"])
    if not path.is_file() or _sha256(path) != source["sha256"]:
        raise ValueError("normalization source metadata SHA mismatch")
    metadata = load_json(path)
    data_config = metadata.get("data_config")
    if not isinstance(data_config, dict) or data_config.get("fields") != ["siconc", "sithic"]:
        raise ValueError("normalization source is not the frozen SIC/SIT data law")
    means, stds = _repeat_stats(data_config["means"]), _repeat_stats(data_config["stds"])
    return means, stds, {
        "path": str(path), "sha256": source["sha256"],
        "fields": data_config["fields"], "means": data_config["means"], "stds": data_config["stds"],
    }


def _to_physical(value: torch.Tensor, means: torch.Tensor, stds: torch.Tensor) -> torch.Tensor:
    shape = (1,) * (value.ndim - 3) + (6, 1, 1)
    return value.double() * stds.reshape(shape) + means.reshape(shape)


def _per_case_bias(
    members: torch.Tensor, truth: torch.Tensor, fraction: torch.Tensor
) -> list[float]:
    error = members.double().mean(dim=1) - truth.double()
    weight = fraction.double().expand(-1, error.shape[1], -1, -1)
    numerator = (error * weight).sum(dim=(2, 3))
    denominator = weight.sum(dim=(2, 3))
    return (numerator / denominator).tolist()


def _boundary_event(
    members: torch.Tensor,
    truth: torch.Tensor,
    fraction: torch.Tensor,
    *,
    comparison: str,
    threshold: float,
) -> dict[str, Any]:
    if comparison == "le":
        member_event, truth_event = members <= threshold, truth <= threshold
    elif comparison == "ge":
        member_event, truth_event = members >= threshold, truth >= threshold
    else:
        raise ValueError(f"unknown comparison: {comparison}")
    probability = member_event.double().mean(dim=1)
    target = truth_event.double()
    weight = fraction.double()
    per_case_brier = (
        ((probability - target).square() * weight).sum(dim=(1, 2, 3))
        / weight.sum(dim=(1, 2, 3))
    )
    per_case_ece = []
    members_count = members.shape[1]
    for case in range(members.shape[0]):
        denominator = weight[case].sum()
        error = torch.zeros((), dtype=torch.float64)
        for index in range(members_count + 1):
            forecast_probability = index / members_count
            selected = probability[case] == forecast_probability
            mass = (weight[case] * selected).sum()
            if mass > 0:
                observed = (weight[case] * selected * target[case]).sum() / mass
                error += mass / denominator * abs(observed - forecast_probability)
        per_case_ece.append(float(error))
    return {
        "comparison": comparison,
        "threshold": threshold,
        "brier": float(per_case_brier.mean()),
        "per_case_brier": per_case_brier.tolist(),
        "reliability_ece": sum(per_case_ece) / len(per_case_ece),
        "per_case_reliability_ece": per_case_ece,
        "forecast_rate": float((probability * weight).sum() / weight.sum()),
        "truth_rate": float((target * weight).sum() / weight.sum()),
    }


def _rank_with_cases(
    members: torch.Tensor, truth: torch.Tensor, fraction: torch.Tensor
) -> dict[str, Any]:
    count = members.shape[1]
    less = (members < truth[:, None]).sum(dim=1)
    equal = (members == truth[:, None]).sum(dim=1)
    case_probabilities = []
    for case in range(members.shape[0]):
        mass = torch.zeros(count + 1, dtype=torch.float64)
        weight = fraction[case].double()
        for rank in range(count + 1):
            selected = (rank >= less[case]) & (rank <= less[case] + equal[case])
            mass[rank] = torch.where(
                selected, weight / (equal[case].double() + 1), torch.zeros_like(weight)
            ).sum()
        case_probabilities.append(mass / mass.sum())
    per_case = torch.stack(case_probabilities)
    probabilities = per_case.mean(dim=0)
    uniform = torch.full_like(probabilities, 1 / (count + 1))
    coordinates = torch.arange(count + 1, dtype=torch.float64)
    return {
        "case_equal_fractional_rank_frequencies": probabilities.tolist(),
        "per_case_fractional_rank_frequencies": per_case.tolist(),
        "per_case_normalized_mean_rank": ((per_case * coordinates).sum(dim=1) / count).tolist(),
        "rank_tv_to_uniform": float(0.5 * (probabilities - uniform).abs().sum()),
        "normalized_mean_rank": float((probabilities * coordinates).sum() / count),
    }


def _score_stage(
    members_normalized: torch.Tensor,
    truth_normalized: torch.Tensor,
    fraction: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor,
) -> dict[str, Any]:
    transformed_members = boundary_emphasis_transform(
        members_normalized.double(), means, stds
    )
    transformed_truth = boundary_emphasis_transform(truth_normalized.double(), means, stds)
    physical_members = _to_physical(members_normalized, means, stds)
    physical_truth = _to_physical(truth_normalized, means, stds)
    biases = _per_case_bias(physical_members, physical_truth, fraction)
    outputs: dict[str, Any] = {}
    ordinary_values, weighted_values = [], []
    for channel, name in enumerate(OUTPUTS):
        ordinary, ordinary_case = _weighted_case_fair_crps(
            members_normalized[:, :, channel : channel + 1],
            truth_normalized[:, channel : channel + 1], fraction,
        )
        weighted, weighted_case = _weighted_case_fair_crps(
            transformed_members[:, :, channel : channel + 1],
            transformed_truth[:, channel : channel + 1], fraction,
        )
        rank = _rank_with_cases(
            members_normalized[:, :, channel : channel + 1],
            truth_normalized[:, channel : channel + 1], fraction,
        )
        field = "sic" if channel % 2 == 0 else "sit"
        events = {
            f"{field}_le_0p01": _boundary_event(
                physical_members[:, :, channel : channel + 1],
                physical_truth[:, channel : channel + 1], fraction,
                comparison="le", threshold=0.01,
            )
        }
        if field == "sic":
            events["sic_ge_0p99"] = _boundary_event(
                physical_members[:, :, channel : channel + 1],
                physical_truth[:, channel : channel + 1], fraction,
                comparison="ge", threshold=0.99,
            )
        outputs[name] = {
            "ordinary_fair_crps": ordinary,
            "per_case_ordinary_fair_crps": ordinary_case,
            "threshold_weighted_fair_crps": weighted,
            "per_case_threshold_weighted_fair_crps": weighted_case,
            "signed_bias": sum(row[channel] for row in biases) / len(biases),
            "per_case_signed_bias": [row[channel] for row in biases],
            "rank": rank,
            "boundary_events": events,
        }
        ordinary_values.extend(ordinary_case)
        weighted_values.extend(weighted_case)
    return {
        "ordinary_fair_crps": sum(ordinary_values) / len(ordinary_values),
        "threshold_weighted_fair_crps": sum(weighted_values) / len(weighted_values),
        "outputs": outputs,
    }


def _weighted_case_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.double().expand(-1, value.shape[1], -1, -1)
    return (value.double() * weight).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))


def _decomposition(
    raw: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    mask = raw["valid_mask"].double()
    raw_mean = raw["forecast_normalized"].double().mean(dim=1)
    candidate_mean = candidate["forecast_normalized"].double().mean(dim=1)
    truth = raw["truth_normalized"].double()
    delta_coarse = (
        candidate["coarse_normalized"].double().mean(dim=1)
        - raw["coarse_normalized"].double().mean(dim=1)
    )
    lifted = smooth_right_inverse(delta_coarse, mask)
    delta_residual = (
        candidate["residual_normalized"].double().mean(dim=1)
        - raw["residual_normalized"].double().mean(dim=1)
    )
    delta = candidate_mean - raw_mean
    reconstruction_error = float((delta - lifted - delta_residual)[mask.expand_as(delta) > 0].abs().max())
    residual_coarse, residual_fraction = masked_block_average(delta_residual, mask)
    nullspace_error = float(residual_coarse[residual_fraction > 0].abs().max())
    error = raw_mean - truth
    terms = {
        "linear_coarse": 2 * error * lifted,
        "linear_residual": 2 * error * delta_residual,
        "square_coarse": lifted.square(),
        "cross_coarse_residual": 2 * lifted * delta_residual,
        "square_residual": delta_residual.square(),
    }
    direct = (error + delta).square() - error.square()
    per_case_direct = _weighted_case_mean(direct, mask)
    per_case_terms = {name: _weighted_case_mean(value, mask) for name, value in terms.items()}
    decomposed = sum(per_case_terms.values())
    identity_error = float((per_case_direct - decomposed).abs().max())
    scale = max(
        float(delta[mask.expand_as(delta) > 0].abs().max()),
        float(error[mask.expand_as(error) > 0].abs().max()),
        1.0,
    )
    # Evidence was persisted in FP32.  The audit reduces in FP64, but identities
    # cannot legitimately be gated below the quantization error of their source.
    source_epsilon = torch.finfo(torch.float32).eps
    reconstruction_tolerance = 128 * source_epsilon * scale
    nullspace_tolerance = 128 * source_epsilon * scale
    identity_tolerance = 512 * source_epsilon * scale * scale
    if reconstruction_error > reconstruction_tolerance:
        raise ValueError("candidate delta does not equal U(delta coarse) + delta residual")
    if nullspace_error > nullspace_tolerance:
        raise ValueError("candidate residual delta is not in the nullspace of D")
    if identity_error > identity_tolerance:
        raise ValueError("direct and decomposed MSE changes differ")
    return {
        "reconstruction_max_abs": reconstruction_error,
        "residual_coarse_max_abs": nullspace_error,
        "mse_change_identity_max_abs": identity_error,
        "tolerances": {
            "reconstruction_max_abs": reconstruction_tolerance,
            "residual_coarse_max_abs": nullspace_tolerance,
            "mse_change_identity_max_abs": identity_tolerance,
        },
        "per_case_output_direct_mse_change": per_case_direct.tolist(),
        "per_case_output_terms": {name: value.tolist() for name, value in per_case_terms.items()},
        "case_output_mean": {
            "direct_mse_change": float(per_case_direct.mean()),
            **{name: float(value.mean()) for name, value in per_case_terms.items()},
        },
    }


def _d6_sit_selectors(
    raw: dict[str, Any], candidate: dict[str, Any], means: torch.Tensor, stds: torch.Tensor
) -> dict[str, Any]:
    channel = 3
    mask = raw["valid_mask"].bool()
    truth_normalized = raw["truth_normalized"][:, channel : channel + 1]
    truth = _to_physical(raw["truth_normalized"], means, stds)[:, channel : channel + 1]
    raw_mean = _to_physical(raw["forecast_normalized"].mean(dim=1), means, stds)[:, channel : channel + 1]
    candidate_mean = _to_physical(candidate["forecast_normalized"].mean(dim=1), means, stds)[:, channel : channel + 1]
    result = {}
    mean = torch.as_tensor(means[channel], dtype=truth_normalized.dtype)
    std = torch.as_tensor(stds[channel], dtype=truth_normalized.dtype)
    encoded_zero = (torch.zeros((), dtype=truth_normalized.dtype) - mean) / std
    exact_zero = truth_normalized == encoded_zero
    selectors = {
        "exact_zero_truth": mask & exact_zero,
        "positive_truth": mask & (truth_normalized > encoded_zero),
    }
    for label, selected in selectors.items():
        raw_case, candidate_case = [], []
        for case in range(truth.shape[0]):
            if not torch.any(selected[case]):
                raw_case.append(None); candidate_case.append(None)
            else:
                raw_case.append(float((raw_mean[case] - truth[case])[selected[case]].mean()))
                candidate_case.append(float((candidate_mean[case] - truth[case])[selected[case]].mean()))
        pairs = [(a, b) for a, b in zip(raw_case, candidate_case, strict=True) if a is not None]
        result[label] = {
            "selector_definition": (
                f"truth_normalized == encoded_zero ({float(encoded_zero)})"
                if label == "exact_zero_truth" else
                f"valid ocean and truth_normalized != encoded_zero ({float(encoded_zero)})"
            ),
            "selected_point_count": int(selected.sum()),
            "case_count": len(pairs),
            "status": "complete" if pairs else "inconclusive_empty_selector",
            "raw_signed_error": sum(a for a, _ in pairs) / len(pairs) if pairs else None,
            "candidate_signed_error": sum(b for _, b in pairs) / len(pairs) if pairs else None,
            "delta": sum(b - a for a, b in pairs) / len(pairs) if pairs else None,
            "per_case_raw": raw_case,
            "per_case_candidate": candidate_case,
        }
    return result


def _heterogeneous_block_counterexample() -> dict[str, Any]:
    value = torch.tensor([0.0, 0.0, 0.0, 0.08], dtype=torch.float64).reshape(1, 1, 2, 2)
    mask = torch.ones_like(value)
    coarse, _ = masked_block_average(value, mask)
    return {
        "fine_fraction_sit_le_0p01": float((value <= 0.01).double().mean()),
        "coarse_mean_sit_metres": float(coarse.item()),
        "coarse_event_sit_le_0p01": bool(coarse.item() <= 0.01),
    }


def run(config_path: Path, output: Path) -> dict[str, Any]:
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    if torch.get_num_threads() != 6 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("failed to enforce the reviewed 6/1 CPU thread envelope")
    if torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise RuntimeError("coarse/fine audit must run with CUDA_VISIBLE_DEVICES empty")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    config = load_json(config_path)
    if config.get("schema_version") != "coarse_fine_boundary_audit_v1":
        raise ValueError("unreviewed coarse/fine audit schema")
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = output.with_name(f"{output.stem}.metrics.json")
    if metrics_path.exists() or metrics_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {metrics_path}")
    reservation = {
        "status": "reserved", "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
    }
    with output.open("x") as handle:
        json.dump(reservation, handle, indent=2, allow_nan=False); handle.write("\n")
    tracker = None
    try:
        code_identity = _clean_code_identity(Path(__file__).resolve().parents[1])
        tracker = ClearMLTracker(
            config["project_name"], f'{config["task_name"]}-{output.parent.name}',
            tags=config["clearml"]["tags"], env_path=config["clearml"]["env_path"],
        )
        tracker.connect("audit_contract", config)
        _strict_atomic_json(output, {**reservation, "status": "running", "code_identity": code_identity, "clearml_task_id": str(tracker.task.id)})
        evidence = {name: _load_frozen(spec) for name, spec in config["evidence"].items()}
        canonical = evidence["raw"]
        for name, payload in evidence.items():
            _validate_payload(name, payload, config["evidence"][name], config)
            _require_equal(canonical, payload, name)
        if not torch.equal(canonical["forecast_normalized"], evidence["ordinary_raw"]["forecast_normalized"]):
            raise ValueError("raw replay forecasts differ across saved evaluations")
        means, stds, normalization = _load_normalization(config)
        if torch.any(stds <= 0):
            raise ValueError("field standard deviations must be positive")
        mask = canonical["valid_mask"].double()
        truth = canonical["truth_normalized"].double()
        truth_coarse, fraction = masked_block_average(truth, mask)
        result: dict[str, Any] = {
            "status": "numerical_results_complete_pending_finalization",
            "scientific_role": "development_only_diagnostic_not_champion_selection",
            "config_path": str(config_path), "config_sha256": reservation["config_sha256"],
            "code_identity": code_identity, "clearml_task_id": str(tracker.task.id),
            "normalization_source": normalization,
            "counterexample": _heterogeneous_block_counterexample(), "evidence": config["evidence"], "candidates": {},
        }
        for name in ("raw", "ordinary64", "threshold64"):
            payload = evidence[name]
            result["candidates"][name] = {
                "coarse": _score_stage(payload["coarse_normalized"].double(), truth_coarse, fraction, means, stds),
                "fine": _score_stage(payload["forecast_normalized"].double(), truth, mask, means, stds),
            }
        for name in ("ordinary64", "threshold64"):
            result["candidates"][name]["decomposition_from_raw"] = _decomposition(canonical, evidence[name])
            result["candidates"][name]["d6_sit_signed_error_selectors"] = _d6_sit_selectors(canonical, evidence[name], means, stds)
        _strict_atomic_json(metrics_path, result)
        tracker.upload_artifact("coarse_fine_boundary_audit", metrics_path)
        _finish_success(
            tracker,
            output,
            metrics_path,
            {
                **reservation,
                "code_identity": code_identity,
                "clearml_task_id": str(tracker.task.id),
                "scientific_role": result["scientific_role"],
            },
        )
        tracker = None
        return result
    except BaseException as error:
        try:
            _record_failure(output, reservation, error)
        except Exception:
            pass
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
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.config.resolve(), arguments.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
