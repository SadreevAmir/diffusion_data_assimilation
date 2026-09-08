"""Sampling, metrics, and visual diagnostics for structured SIC/SIT trajectories."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as torch_functional

from .sampler import Sampler


PUBLICATION_MANIFEST_SCHEMA = "structured_joint_publication_case_manifest_v1"
PUBLICATION_PROTOCOL_SCHEMA = "structured_joint_publication_protocol_v1"
PUBLICATION_RESULT_SCHEMA = "structured_joint_publication_evaluation_v1"
PUBLICATION_QC_SCHEMA = "structured_joint_publication_visual_qc_v1"
PUBLICATION_COVERAGE_LEVELS = (0.50, 0.80, 0.90)
PUBLICATION_RELIABILITY_BIN_EDGES = tuple(index / 10.0 for index in range(11))
PUBLICATION_EDGE_PROBABILITY_THRESHOLD = 0.50
PUBLICATION_TRAJECTORY_VARIOGRAM_POWER = 0.50
PUBLICATION_SPATIAL_VARIOGRAM_LAGS = ((0, 1), (1, 0), (1, 1), (1, -1))
PUBLICATION_FSS_SCALES = (1, 3, 5, 9)
PUBLICATION_CPU_CASE_CHUNK = 8
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

_DETERMINISTIC_BASELINE_POLICIES = {
    "exact_previous_calendar_year_background": (
        "frozen manifest background_archive_dates only"
    ),
    "deterministic_background_persistence": (
        "lead-0 previous-calendar-year background repeated through d0..d3"
    ),
}


@dataclass(frozen=True)
class PublicationCase:
    """One immutable, paired archive-hindcast case from the frozen manifest."""

    case_id: str
    anchor_date: str
    target_archive_dates: tuple[str, ...]
    background_archive_dates: tuple[str, ...]
    observation_lag_dates: tuple[str, ...]


@dataclass(frozen=True)
class StructuredPublicationContract:
    """Validated publication inputs that contain no model or accelerator state."""

    cases: tuple[PublicationCase, ...]
    ensemble_size: int
    member_seeds: tuple[int, ...]
    manifest_sha256: str
    protocol_sha256: str
    target_slice_index: int
    sic_rank_policy: str
    coverage_levels: tuple[float, ...]
    claim_boundary: str
    blockers: tuple[str, ...]

    def require_ready(self) -> None:
        if self.blockers:
            raise ValueError(
                "publication protocol is not frozen: " + "; ".join(self.blockers)
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _iso_date(value: Any, label: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO date string")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{label} is not an ISO date: {value!r}") from error


def _calendar_year_ago(value: date) -> date:
    try:
        return value.replace(year=value.year - 1)
    except ValueError as error:
        raise ValueError(
            f"{value.isoformat()} has no exact previous-calendar-year counterpart"
        ) from error


def load_structured_publication_contract(
    case_manifest_path: str | Path,
    protocol_path: str | Path,
    *,
    require_frozen: bool = True,
) -> StructuredPublicationContract:
    """Load and cross-check the fixed cases and publication protocol.

    The checked-in protocol deliberately still carries a pending case-manifest
    hash.  Callers can inspect that state with ``require_frozen=False``; scoring
    remains fail-closed until a trusted controller records the exact hash.
    """

    manifest_path = Path(case_manifest_path)
    publication_protocol_path = Path(protocol_path)
    manifest = _read_json_object(manifest_path, "case manifest")
    protocol = _read_json_object(publication_protocol_path, "publication protocol")
    if manifest.get("schema_version") != PUBLICATION_MANIFEST_SCHEMA:
        raise ValueError("unexpected structured publication case-manifest schema")
    if protocol.get("schema_version") != PUBLICATION_PROTOCOL_SCHEMA:
        raise ValueError("unexpected structured publication protocol schema")
    if manifest.get("split") != "test":
        raise ValueError("publication case manifest must be bound to the test split")
    if manifest.get("trajectory_semantics") != "consecutive_daily_archive_snapshots":
        raise ValueError("publication cases must use consecutive archive snapshots")

    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("publication case manifest must contain a non-empty cases list")
    if manifest.get("case_count") != len(raw_cases):
        raise ValueError("publication case_count does not match the cases list")
    first_anchor = _iso_date(manifest.get("first_anchor"), "first_anchor")
    last_anchor = _iso_date(manifest.get("last_anchor"), "last_anchor")
    protocol_range = protocol.get("test_anchor_range")
    if protocol_range != [first_anchor.isoformat(), last_anchor.isoformat()]:
        raise ValueError("protocol test_anchor_range differs from the case manifest")

    cases: list[PublicationCase] = []
    case_ids: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise ValueError(f"case {index} must be a JSON object")
        anchor = _iso_date(raw_case.get("anchor_date"), f"case {index} anchor_date")
        expected_anchor = first_anchor + timedelta(days=index)
        if anchor != expected_anchor:
            raise ValueError("publication anchors must be consecutive and manifest-ordered")
        case_id = raw_case.get("case_id")
        expected_case_id = f"{anchor.isoformat()}_slice{manifest.get('target_slice_index')}_d0_d3"
        if not isinstance(case_id, str) or case_id != expected_case_id:
            raise ValueError(f"case {index} has a non-canonical case_id")
        if case_id in case_ids:
            raise ValueError(f"duplicate publication case_id {case_id!r}")
        case_ids.add(case_id)

        targets = raw_case.get("target_archive_dates")
        backgrounds = raw_case.get("background_archive_dates")
        observations = raw_case.get("observation_lag_dates")
        expected_targets = [(anchor + timedelta(days=lead)).isoformat() for lead in range(4)]
        expected_backgrounds = [
            _calendar_year_ago(anchor + timedelta(days=lead)).isoformat()
            for lead in range(4)
        ]
        expected_observations = [
            (anchor - timedelta(days=lag)).isoformat() for lag in range(3)
        ]
        if targets != expected_targets:
            raise ValueError(f"case {case_id} target dates are not d..d+3")
        if backgrounds != expected_backgrounds:
            raise ValueError(f"case {case_id} backgrounds are not calendar-year paired")
        if observations != expected_observations:
            raise ValueError(f"case {case_id} observation lags are not d,d-1,d-2")
        cases.append(
            PublicationCase(
                case_id=case_id,
                anchor_date=anchor.isoformat(),
                target_archive_dates=tuple(targets),
                background_archive_dates=tuple(backgrounds),
                observation_lag_dates=tuple(observations),
            )
        )
    if cases[-1].anchor_date != last_anchor.isoformat():
        raise ValueError("last_anchor does not match the final publication case")

    ensemble_size = protocol.get("ensemble_size")
    member_seeds = protocol.get("member_seeds")
    if not isinstance(ensemble_size, int) or ensemble_size < 2:
        raise ValueError("publication ensemble_size must be an integer of at least two")
    if (
        not isinstance(member_seeds, list)
        or len(member_seeds) != ensemble_size
        or any(not isinstance(seed, int) for seed in member_seeds)
        or len(set(member_seeds)) != ensemble_size
    ):
        raise ValueError("member_seeds must contain one unique integer per member")
    marginal_scores = protocol.get("required_scores", {}).get("marginal_probabilistic", [])
    required_marginal = {
        "fair_CRPS",
        "tie_aware_fractional_rank_histogram",
        "rank_TV",
        "central_50_80_90_coverage",
    }
    if not required_marginal.issubset(set(marginal_scores)):
        raise ValueError("publication protocol is missing required marginal scores")

    manifest_sha256 = _sha256_file(manifest_path)
    recorded_manifest_sha = protocol.get("case_manifest_sha256")
    blockers: list[str] = []
    if (
        not isinstance(recorded_manifest_sha, str)
        or _SHA256_RE.fullmatch(recorded_manifest_sha) is None
    ):
        blockers.append("case_manifest_sha256 is unresolved")
    elif recorded_manifest_sha != manifest_sha256:
        raise ValueError("case manifest SHA-256 differs from the frozen protocol")
    status = protocol.get("status")
    if not isinstance(status, str):
        raise ValueError("publication protocol status must be a string")
    if status.startswith("blocked_"):
        blockers.append(f"protocol status is {status}")

    contract = StructuredPublicationContract(
        cases=tuple(cases),
        ensemble_size=ensemble_size,
        member_seeds=tuple(member_seeds),
        manifest_sha256=manifest_sha256,
        protocol_sha256=_sha256_file(publication_protocol_path),
        target_slice_index=int(manifest["target_slice_index"]),
        sic_rank_policy="tie_aware_fractional_mass",
        coverage_levels=PUBLICATION_COVERAGE_LEVELS,
        claim_boundary=str(protocol.get("claim_boundary", "")),
        blockers=tuple(blockers),
    )
    if require_frozen:
        contract.require_ready()
    return contract


@torch.no_grad()
def sample_structured_batch(
    sampler: Sampler,
    batch: dict[str, torch.Tensor],
    *,
    stats: dict,
    size: tuple[int, int],
    num_timesteps: int,
    device,
    method: str,
    rtol: float,
    atol: float,
    initial_noise: torch.Tensor | None = None,
    physical_dtype: torch.dtype | None = None,
    return_latent: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    required = (
        "background",
        "structured_conditioning",
        "valid_mask",
        "structured_flow_mask",
        "structured_lag0_physical_values",
        "structured_lag0_mask",
    )
    missing = [name for name in required if name not in batch]
    if missing:
        raise KeyError(f"structured trajectory batch is missing {missing}")
    return sampler.sample_structured_trajectory(
        background_trajectory=batch["background"],
        model_conditioning=batch["structured_conditioning"],
        valid_mask=batch["valid_mask"],
        flow_mask=batch["structured_flow_mask"],
        lag0_physical_values=batch["structured_lag0_physical_values"],
        lag0_mask=batch["structured_lag0_mask"],
        stats=stats,
        size=size,
        num_timesteps=num_timesteps,
        device=device,
        method=method,
        rtol=rtol,
        atol=atol,
        initial_noise=initial_noise,
        physical_dtype=physical_dtype,
        return_latent=return_latent,
    )


def fractional_rank_counts(
    members: torch.Tensor,
    truth: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Tie-correct rank counts with exact fractional mass conservation."""
    if members.ndim != truth.ndim + 1:
        raise ValueError("members must add one ensemble dimension after batch")
    ensemble_size = members.shape[1]
    expanded_truth = truth.unsqueeze(1)
    less = (members < expanded_truth).sum(dim=1)
    equal = (members == expanded_truth).sum(dim=1)
    valid = mask.expand_as(truth) > 0
    counts = torch.zeros(ensemble_size + 1, dtype=torch.float64, device=members.device)
    tie_denominator = (equal + 1).to(torch.float64)
    for rank in range(ensemble_size + 1):
        in_tie_block = (rank >= less) & (rank <= less + equal) & valid
        counts[rank] = (
            in_tie_block.to(torch.float64) / tie_denominator
        ).sum(dtype=torch.float64)
    expected = valid.sum(dtype=torch.float64)
    if not torch.allclose(counts.sum(), expected, atol=1e-7, rtol=1e-12):
        raise RuntimeError("fractional rank mass invariant failed")
    return counts


def _fair_crps(
    members: torch.Tensor,
    truth: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Finite-ensemble unbiased CRPS without an O(M**2) pairwise tensor."""
    ensemble_size = members.shape[1]
    valid = mask.expand_as(truth) > 0
    if not torch.any(valid):
        raise ValueError("fair CRPS requires at least one valid point")
    selected = members.movedim(1, 0)[:, valid]
    target = truth[valid]
    work_dtype = (
        torch.float32
        if selected.dtype in (torch.float16, torch.bfloat16)
        else selected.dtype
    )
    selected = selected.to(dtype=work_dtype)
    target = target.to(dtype=work_dtype)
    first = (selected - target.unsqueeze(0)).abs().mean(dim=0)
    if ensemble_size > 1:
        ordered = selected.sort(dim=0).values
        coefficients = (
            2 * torch.arange(ensemble_size, device=members.device, dtype=work_dtype)
            - ensemble_size
            + 1
        ).reshape(ensemble_size, 1)
        # Sum_{i<j} |x_i-x_j| / (M(M-1)) is the fair spread correction.
        second = (coefficients * ordered).sum(dim=0) / (
            ensemble_size * (ensemble_size - 1)
        )
    else:
        second = torch.zeros_like(first)
    return float((first - second).mean(dtype=torch.float64).item())


def _require_cpu_tensor(value: torch.Tensor, label: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if value.device.type != "cpu":
        raise ValueError(f"publication evaluation is CPU-only; {label} is on {value.device}")
    if value.layout != torch.strided:
        raise ValueError(f"{label} must use strided dense storage")


def _require_binary_mask(mask: torch.Tensor, label: str) -> None:
    _require_cpu_tensor(mask, label)
    if not torch.all(torch.isfinite(mask)):
        raise ValueError(f"{label} must be finite")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError(f"{label} must be exactly binary")


def _support_violation_count(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    sic_cap: float,
    ensemble: bool,
) -> int:
    sic = values[:, :, 0::2] if ensemble else values[:, 0::2]
    sit = values[:, :, 1::2] if ensemble else values[:, 1::2]
    mask = valid_mask[:, None] if ensemble else valid_mask
    mask = mask.expand_as(sic) > 0
    invalid = (
        ~torch.isfinite(sic)
        | ~torch.isfinite(sit)
        | (sic < 0)
        | (sic > float(sic_cap))
        | (sit < 0)
        | ((sic > 0) != (sit > 0))
    ) & mask
    return int(invalid.sum().item())


def validate_structured_publication_tensors(
    ensemble: torch.Tensor,
    truth: torch.Tensor,
    background: torch.Tensor,
    valid_mask: torch.Tensor,
    lag0_mask: torch.Tensor,
    *,
    contract: StructuredPublicationContract,
    case_ids: Sequence[str],
    member_seeds: Sequence[int],
    sic_cap: float,
) -> dict[str, int]:
    """Fail closed on pairing, ensemble identity, shape, masks, and support."""

    contract.require_ready()
    for label, value in (
        ("ensemble", ensemble),
        ("truth", truth),
        ("background", background),
    ):
        _require_cpu_tensor(value, label)
        if not value.is_floating_point():
            raise ValueError(f"{label} must use a floating-point physical dtype")
    _require_binary_mask(valid_mask, "valid_mask")
    _require_binary_mask(lag0_mask, "lag0_mask")
    if not math.isfinite(float(sic_cap)) or float(sic_cap) <= 0:
        raise ValueError("sic_cap must be finite and positive")
    if ensemble.ndim != 5:
        raise ValueError("ensemble must have shape [case,member,2*lead,height,width]")
    if truth.ndim != 4 or background.shape != truth.shape:
        raise ValueError("truth/background must be paired [case,2*lead,height,width]")
    if truth.shape[1] != 8 or ensemble.shape[2] != 8:
        raise ValueError("d0..d3 publication tensors require eight paired SIC/SIT channels")
    if ensemble.shape[0] != truth.shape[0] or ensemble.shape[3:] != truth.shape[2:]:
        raise ValueError("ensemble and paired truth have incompatible event shapes")
    expected_mask_shape = (truth.shape[0], 1, *truth.shape[2:])
    if (
        tuple(valid_mask.shape) != expected_mask_shape
        or tuple(lag0_mask.shape) != expected_mask_shape
    ):
        raise ValueError("valid_mask and lag0_mask must have shape [case,1,height,width]")
    if ensemble.shape[0] != len(contract.cases):
        raise ValueError("tensor case count differs from the frozen publication manifest")
    if ensemble.shape[1] != contract.ensemble_size:
        raise ValueError("tensor member count differs from the frozen publication protocol")
    expected_case_ids = tuple(case.case_id for case in contract.cases)
    if tuple(case_ids) != expected_case_ids:
        raise ValueError("case_ids do not exactly match frozen manifest order")
    if tuple(member_seeds) != contract.member_seeds:
        raise ValueError("member_seeds do not exactly match frozen protocol order")
    if not torch.any(valid_mask > 0):
        raise ValueError("publication evaluation has no valid ocean points")
    if torch.any((lag0_mask > 0) & (valid_mask == 0)):
        raise ValueError("lag0_mask must be a subset of valid_mask")

    support_counts: dict[str, int] = {}
    for label, values, is_ensemble in (
        ("ensemble", ensemble, True),
        ("truth", truth, False),
        ("background", background, False),
    ):
        violations = _support_violation_count(
            values, valid_mask, sic_cap=float(sic_cap), ensemble=is_ensemble
        )
        support_counts[f"{label}_support_violation_count"] = violations
        if violations:
            raise ValueError(
                f"{label} violates finite joint SIC/SIT support at {violations} valid points"
            )
    exact = (lag0_mask[:, None] > 0).expand_as(ensemble[:, :, :2])
    exact_errors = (ensemble[:, :, :2] - truth[:, None, :2]).abs()
    exact_max_abs_error = (
        float(exact_errors[exact].max().item()) if torch.any(exact) else 0.0
    )
    if exact_max_abs_error != 0.0:
        raise ValueError(
            "ensemble violates exact paired day-0 observation support: "
            f"max_abs_error={exact_max_abs_error}"
        )
    support_counts["valid_ocean_points"] = int(valid_mask.sum().item())
    support_counts["lag0_exact_points"] = int(lag0_mask.sum().item())
    support_counts["lag0_exact_max_abs_error"] = exact_max_abs_error
    return support_counts


def _central_coverage(
    members: torch.Tensor,
    truth: torch.Tensor,
    mask: torch.Tensor,
    levels: Sequence[float],
) -> dict[str, float]:
    valid = mask.expand_as(truth) > 0
    selected = members.movedim(1, 0)[:, valid]
    target = truth[valid]
    if target.numel() == 0:
        raise ValueError("central coverage requires at least one valid point")
    if selected.dtype in (torch.float16, torch.bfloat16):
        selected = selected.to(dtype=torch.float32)
        target = target.to(dtype=torch.float32)
    result: dict[str, float] = {}
    for level in levels:
        if not 0 < float(level) < 1:
            raise ValueError("central coverage levels must lie strictly within (0,1)")
        tail = (1.0 - float(level)) / 2.0
        bounds = torch.quantile(
            selected,
            torch.tensor([tail, 1.0 - tail], dtype=selected.dtype),
            dim=0,
        )
        covered = (target >= bounds[0]) & (target <= bounds[1])
        key = f"central_{round(100 * float(level)):d}"
        result[key] = float(covered.to(torch.float64).mean().item())
    return result


def _brier_reliability(
    member_event: torch.Tensor,
    truth_event: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, Any]:
    """Empirical-probability Brier score and fixed-bin reliability table."""

    valid = mask.expand_as(truth_event) > 0
    if not torch.any(valid):
        raise ValueError("event diagnostics require at least one valid point")
    probability = (
        member_event.sum(dim=1, dtype=torch.int32).to(torch.float64)
        / member_event.shape[1]
    )
    outcome = truth_event.to(torch.float64)
    selected_probability = probability[valid]
    selected_outcome = outcome[valid]
    boundaries = torch.tensor(
        PUBLICATION_RELIABILITY_BIN_EDGES[1:-1], dtype=torch.float64
    )
    bin_index = torch.bucketize(selected_probability, boundaries, right=True)
    bins: list[dict[str, Any]] = []
    reliability_l1 = 0.0
    total = int(selected_outcome.numel())
    for index, (lower, upper) in enumerate(
        zip(
            PUBLICATION_RELIABILITY_BIN_EDGES[:-1],
            PUBLICATION_RELIABILITY_BIN_EDGES[1:],
        )
    ):
        selected = bin_index == index
        count = int(selected.sum().item())
        mean_probability = (
            float(selected_probability[selected].mean().item()) if count else None
        )
        observed_frequency = (
            float(selected_outcome[selected].mean().item()) if count else None
        )
        if count:
            reliability_l1 += (
                count
                * abs(float(mean_probability) - float(observed_frequency))
                / total
            )
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "upper_inclusive": index == len(PUBLICATION_RELIABILITY_BIN_EDGES) - 2,
                "count": count,
                "forecast_probability_mean": mean_probability,
                "observed_frequency": observed_frequency,
            }
        )
    return {
        "evaluated_points": total,
        "brier_score": float(
            (selected_probability - selected_outcome).square().mean().item()
        ),
        "forecast_frequency": float(selected_probability.mean().item()),
        "observed_frequency": float(selected_outcome.mean().item()),
        "reliability_l1": reliability_l1,
        "reliability_bins": bins,
    }


def _internal_binary_edge(binary: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Pixels adjacent to a unlike valid 4-neighbor on either side of the edge."""

    state = np.asarray(binary, dtype=bool)
    domain = np.asarray(valid, dtype=bool)
    if state.ndim != 2 or state.shape != domain.shape:
        raise ValueError("edge inputs must be co-registered 2-D arrays")
    edge = np.zeros_like(state)
    horizontal = domain[:, 1:] & domain[:, :-1] & (state[:, 1:] != state[:, :-1])
    vertical = domain[1:, :] & domain[:-1, :] & (state[1:, :] != state[:-1, :])
    edge[:, 1:] |= horizontal
    edge[:, :-1] |= horizontal
    edge[1:, :] |= vertical
    edge[:-1, :] |= vertical
    return edge & domain


def _l1_distance_to_set(points: np.ndarray) -> np.ndarray:
    """Exact city-block distance transform in O(HW) time and O(HW) memory."""

    selected = np.asarray(points, dtype=bool)
    if selected.ndim != 2 or not np.any(selected):
        raise ValueError("distance transform requires a non-empty 2-D point set")
    height, width = selected.shape
    distance = np.full((height, width), height + width, dtype=np.int32)
    distance[selected] = 0
    for row in range(1, height):
        distance[row] = np.minimum(distance[row], distance[row - 1] + 1)
    for row in range(height - 2, -1, -1):
        distance[row] = np.minimum(distance[row], distance[row + 1] + 1)
    for column in range(1, width):
        distance[:, column] = np.minimum(
            distance[:, column], distance[:, column - 1] + 1
        )
    for column in range(width - 2, -1, -1):
        distance[:, column] = np.minimum(
            distance[:, column], distance[:, column + 1] + 1
        )
    return distance


def _symmetric_edge_displacement_l1(
    forecast: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray,
) -> tuple[float | None, str]:
    """Symmetric mean nearest-edge city-block distance in grid cells."""

    forecast_edge = _internal_binary_edge(forecast, valid)
    truth_edge = _internal_binary_edge(truth, valid)
    has_forecast = bool(np.any(forecast_edge))
    has_truth = bool(np.any(truth_edge))
    if not has_forecast and not has_truth:
        return 0.0, "both_absent"
    if not has_forecast or not has_truth:
        return None, "one_absent"
    distance_to_truth = _l1_distance_to_set(truth_edge)
    distance_to_forecast = _l1_distance_to_set(forecast_edge)
    displacement = 0.5 * (
        float(distance_to_truth[forecast_edge].mean())
        + float(distance_to_forecast[truth_edge].mean())
    )
    return displacement, "both_present"


def _mean_or_none(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _spatial_occurrence_metrics(
    member_sic: torch.Tensor,
    truth_sic: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, Any]:
    """Equal-case extent, expected IIEE, and consensus-edge summaries."""

    valid = valid_mask > 0
    extent_errors: list[float] = []
    extent_biases: list[float] = []
    truth_extents: list[float] = []
    forecast_extents: list[float] = []
    expected_iiee: list[float] = []
    edge_displacements: list[float] = []
    both_present = 0
    both_absent = 0
    one_absent = 0
    for case_index in range(truth_sic.shape[0]):
        case_valid = valid[case_index]
        valid_count = int(case_valid.sum().item())
        if valid_count == 0:
            continue
        case_member_ice = member_sic[case_index] > 0
        case_truth = truth_sic[case_index] > 0
        probability = (
            case_member_ice.sum(dim=0, dtype=torch.int32).to(torch.float64)
            / member_sic.shape[1]
        )
        consensus = probability >= PUBLICATION_EDGE_PROBABILITY_THRESHOLD
        truth_extent = float(case_truth[case_valid].sum().item())
        member_extent = case_member_ice[:, case_valid].sum(dim=1)
        forecast_extent = float(member_extent.to(torch.float64).mean().item())
        truth_extents.append(truth_extent)
        forecast_extents.append(forecast_extent)
        extent_errors.append(abs(forecast_extent - truth_extent))
        extent_biases.append(forecast_extent - truth_extent)
        expected_iiee.append(
            float(
                (
                    probability[case_valid]
                    - case_truth[case_valid].to(torch.float64)
                )
                .abs()
                .mean()
                .item()
            )
        )
        displacement, edge_state = _symmetric_edge_displacement_l1(
            consensus.numpy(),
            case_truth.numpy(),
            case_valid.numpy(),
        )
        if edge_state == "both_present":
            both_present += 1
        elif edge_state == "both_absent":
            both_absent += 1
        else:
            one_absent += 1
        if displacement is not None:
            edge_displacements.append(displacement)
    return {
        "case_aggregation": "equal_case",
        "occurrence_definition": "SIC>0",
        "extent_unit": "valid_grid_cells",
        "truth_extent_mean": _mean_or_none(truth_extents),
        "ensemble_expected_extent_mean": _mean_or_none(forecast_extents),
        "ensemble_expected_extent_mae": _mean_or_none(extent_errors),
        "ensemble_expected_extent_bias": _mean_or_none(extent_biases),
        "expected_iiee_fraction": _mean_or_none(expected_iiee),
        "consensus_probability_threshold": PUBLICATION_EDGE_PROBABILITY_THRESHOLD,
        "edge_definition": "unlike valid 4-neighbors; both adjacent pixels retained",
        "edge_distance": "symmetric_mean_nearest_edge_L1_grid_cells",
        "edge_displacement_mean": _mean_or_none(edge_displacements),
        "edge_cases_both_present": both_present,
        "edge_cases_both_absent": both_absent,
        "edge_cases_one_absent": one_absent,
    }


def _lagged_views(
    values: torch.Tensor,
    delta_row: int,
    delta_column: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = values.shape[-2:]
    if abs(delta_row) >= height or abs(delta_column) >= width:
        return values[..., :0, :0], values[..., :0, :0]
    left_rows = slice(max(0, -delta_row), min(height, height - delta_row))
    right_rows = slice(max(0, delta_row), min(height, height + delta_row))
    left_columns = slice(max(0, -delta_column), min(width, width - delta_column))
    right_columns = slice(max(0, delta_column), min(width, width + delta_column))
    return (
        values[..., left_rows, left_columns],
        values[..., right_rows, right_columns],
    )


def _local_spatial_variogram_score(
    members: torch.Tensor,
    truth: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, Any]:
    """Local variogram score at fixed grid lags, pooled over valid pairs."""

    lag_results: list[dict[str, Any]] = []
    available_scores: list[float] = []
    for delta_row, delta_column in PUBLICATION_SPATIAL_VARIOGRAM_LAGS:
        pair_count = 0
        score_sum = 0.0
        for case_index in range(truth.shape[0]):
            member_left, member_right = _lagged_views(
                members[case_index], delta_row, delta_column
            )
            truth_left, truth_right = _lagged_views(
                truth[case_index], delta_row, delta_column
            )
            valid_left, valid_right = _lagged_views(
                valid_mask[case_index], delta_row, delta_column
            )
            pair_valid = (valid_left > 0) & (valid_right > 0)
            case_pair_count = int(pair_valid.sum().item())
            if not case_pair_count:
                continue
            member_increment = (member_right - member_left).abs().pow(
                PUBLICATION_TRAJECTORY_VARIOGRAM_POWER
            )
            truth_increment = (truth_right - truth_left).abs().pow(
                PUBLICATION_TRAJECTORY_VARIOGRAM_POWER
            )
            expectation = member_increment.mean(dim=0, dtype=torch.float64)
            score_sum += float(
                (expectation - truth_increment)[pair_valid]
                .square()
                .sum(dtype=torch.float64)
                .item()
            )
            pair_count += case_pair_count
        score: float | None = None
        if pair_count:
            score = score_sum / pair_count
            available_scores.append(score)
        lag_results.append(
            {
                "delta_row": delta_row,
                "delta_column": delta_column,
                "valid_pairs": pair_count,
                "score": score,
            }
        )
    return {
        "definition": "mean((E|X(s+h)-X(s)|^p-|y(s+h)-y(s)|^p)^2)",
        "power": PUBLICATION_TRAJECTORY_VARIOGRAM_POWER,
        "lags": lag_results,
        "equal_lag_mean": _mean_or_none(available_scores),
    }


def _fraction_skill_score(
    member_sic: torch.Tensor,
    truth_sic: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, Any]:
    """Occurrence FSS from valid-neighborhood fractions at fixed odd scales."""

    scale_results: list[dict[str, Any]] = []
    for scale in PUBLICATION_FSS_SCALES:
        kernel = torch.ones((1, 1, scale, scale), dtype=torch.float64)
        padding = scale // 2
        numerator = 0.0
        denominator = 0.0
        evaluated_centers = 0
        for start in range(0, truth_sic.shape[0], PUBLICATION_CPU_CASE_CHUNK):
            stop = min(start + PUBLICATION_CPU_CASE_CHUNK, truth_sic.shape[0])
            valid = (valid_mask[start:stop] > 0).to(torch.float64)[:, None]
            probability = (
                (member_sic[start:stop] > 0)
                .sum(dim=1, keepdim=True, dtype=torch.int32)
                .to(torch.float64)
                / member_sic.shape[1]
            )
            outcome = (truth_sic[start:stop] > 0).to(torch.float64)[:, None]
            valid_count = torch_functional.conv2d(valid, kernel, padding=padding)
            forecast_fraction = torch_functional.conv2d(
                probability * valid, kernel, padding=padding
            ) / valid_count.clamp(min=1)
            observed_fraction = torch_functional.conv2d(
                outcome * valid, kernel, padding=padding
            ) / valid_count.clamp(min=1)
            selector = (valid > 0) & (valid_count > 0)
            forecast_selected = forecast_fraction[selector]
            observed_selected = observed_fraction[selector]
            numerator += float(
                (forecast_selected - observed_selected).square().sum().item()
            )
            denominator += float(
                (forecast_selected.square() + observed_selected.square()).sum().item()
            )
            evaluated_centers += int(selector.sum().item())
        fss = (
            1.0
            if denominator == 0.0
            else 1.0 - numerator / denominator
        )
        scale_results.append(
            {
                "scale_grid_cells": scale,
                "evaluated_centers": evaluated_centers,
                "fss": fss,
            }
        )
    return {
        "event_definition": "SIC>0",
        "neighborhood": "square_valid_cell_normalized",
        "aggregation": "pooled_valid_centers",
        "scales": scale_results,
    }


def _spatial_correlation(
    left: torch.Tensor,
    right: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-case/member Pearson correlation over valid spatial pixels."""

    valid = (valid_mask > 0).to(torch.float64)
    left64 = left.to(torch.float64)
    right64 = right.to(torch.float64)
    count = valid.sum(dim=(-2, -1), keepdim=True)
    safe_count = count.clamp(min=1)
    left_mean = (left64 * valid).sum(dim=(-2, -1), keepdim=True) / safe_count
    right_mean = (right64 * valid).sum(dim=(-2, -1), keepdim=True) / safe_count
    left_centered = (left64 - left_mean) * valid
    right_centered = (right64 - right_mean) * valid
    covariance = (left_centered * right_centered).sum(dim=(-2, -1))
    left_sum_square = left_centered.square().sum(dim=(-2, -1))
    right_sum_square = right_centered.square().sum(dim=(-2, -1))
    denominator = (left_sum_square * right_sum_square).sqrt()
    defined = (count.squeeze(-1).squeeze(-1) >= 2) & (denominator > 0)
    correlation = torch.zeros_like(covariance)
    correlation[defined] = covariance[defined] / denominator[defined]
    return correlation, defined


def _lagged_spatial_correlation_summary(
    member_left: torch.Tensor,
    member_right: torch.Tensor,
    truth_left: torch.Tensor,
    truth_right: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, Any]:
    case_errors: list[float] = []
    forecast_values: list[float] = []
    truth_values: list[float] = []
    defined_member_case_pairs = 0
    for case_index in range(truth_left.shape[0]):
        member_correlation, member_defined = _spatial_correlation(
            member_left[case_index : case_index + 1],
            member_right[case_index : case_index + 1],
            valid_mask[case_index : case_index + 1, None],
        )
        truth_correlation, truth_defined = _spatial_correlation(
            truth_left[case_index : case_index + 1, None],
            truth_right[case_index : case_index + 1, None],
            valid_mask[case_index : case_index + 1, None],
        )
        if not bool(truth_defined[0, 0]):
            continue
        selected = member_defined[0]
        if not torch.any(selected):
            continue
        defined_member_case_pairs += int(selected.sum().item())
        forecast_value = float(
            member_correlation[0, selected].mean(dtype=torch.float64).item()
        )
        truth_value = float(truth_correlation[0, 0].item())
        forecast_values.append(forecast_value)
        truth_values.append(truth_value)
        case_errors.append(abs(forecast_value - truth_value))
    return {
        "definition": "Pearson correlation across valid spatial pixels",
        "case_aggregation": "equal_case",
        "defined_cases": len(case_errors),
        "defined_member_case_pairs": defined_member_case_pairs,
        "forecast_member_mean_correlation": _mean_or_none(forecast_values),
        "truth_correlation": _mean_or_none(truth_values),
        "mean_absolute_error": _mean_or_none(case_errors),
    }


def _temporal_metrics(
    ensemble: torch.Tensor,
    truth: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, Any]:
    """Member-aligned adjacent increments and all-pairs trajectory variogram."""

    valid = valid_mask > 0
    transitions: dict[str, Any] = {}
    for lead in range(3):
        transition_label = f"d+{lead}_to_d+{lead + 1}"
        transition_fields: dict[str, Any] = {}
        for offset, field in enumerate(("sic", "sit")):
            left_channel = 2 * lead + offset
            right_channel = 2 * (lead + 1) + offset
            member_left = ensemble[:, :, left_channel]
            member_right = ensemble[:, :, right_channel]
            truth_left = truth[:, left_channel]
            truth_right = truth[:, right_channel]
            member_increment = member_right - member_left
            truth_increment = truth_right - truth_left
            channel_members = member_increment.unsqueeze(2)
            channel_truth = truth_increment.unsqueeze(1)
            mean_increment = member_increment.mean(dim=1)
            error = mean_increment[valid[:, 0]] - truth_increment[valid[:, 0]]
            truth_sign = torch.sign(truth_increment)
            member_sign = torch.sign(member_increment)
            member_agreement = member_sign == truth_sign[:, None]
            mean_sign_agreement = torch.sign(mean_increment) == truth_sign
            transition_fields[field] = {
                "evaluated_points": int(error.numel()),
                "increment_fair_crps": _fair_crps(
                    channel_members, channel_truth, valid
                ),
                "ensemble_mean_increment_rmse": float(
                    error.square().mean(dtype=torch.float64).sqrt().item()
                ),
                "ensemble_mean_increment_mae": float(
                    error.abs().mean(dtype=torch.float64).item()
                ),
                "tendency_definition": "exact sign(delta): decrease=-1, steady=0, increase=1",
                "ensemble_mean_tendency_accuracy": float(
                    mean_sign_agreement[valid[:, 0]].to(torch.float64).mean().item()
                ),
                "member_tendency_agreement_probability": float(
                    member_agreement.movedim(1, 0)[:, valid[:, 0]]
                    .to(torch.float64)
                    .mean()
                    .item()
                ),
                "lagged_spatial_correlation": _lagged_spatial_correlation_summary(
                    member_left,
                    member_right,
                    truth_left,
                    truth_right,
                    valid_mask[:, 0],
                ),
            }
        transitions[transition_label] = {"fields": transition_fields}

    trajectory: dict[str, Any] = {}
    for offset, field in enumerate(("sic", "sit")):
        member_field = ensemble[:, :, offset::2]
        truth_field = truth[:, offset::2]
        point_score = torch.zeros_like(truth_field[:, 0], dtype=torch.float64)
        pairs = 0
        for left in range(4):
            for right in range(left + 1, 4):
                member_difference = (
                    member_field[:, :, right] - member_field[:, :, left]
                ).abs().pow(PUBLICATION_TRAJECTORY_VARIOGRAM_POWER)
                truth_difference = (
                    truth_field[:, right] - truth_field[:, left]
                ).abs().pow(PUBLICATION_TRAJECTORY_VARIOGRAM_POWER)
                expectation = member_difference.mean(dim=1, dtype=torch.float64)
                point_score += (expectation - truth_difference).square()
                pairs += 1
        trajectory[field] = {
            "evaluated_points": int(valid_mask.sum().item()),
            "power": PUBLICATION_TRAJECTORY_VARIOGRAM_POWER,
            "lead_pairs": pairs,
            "pair_weights": "all_one",
            "trajectory_variogram_score": float(
                point_score[valid[:, 0]].mean().item()
            ),
        }
    return {
        "increment_pairing": "same member across consecutive archive dates",
        "calibration_domain": "all_valid_ocean_including_observed_day0_origins",
        "transitions": transitions,
        "trajectory_variogram": trajectory,
    }


def _deterministic_spatial_metrics(
    prediction_sic: torch.Tensor,
    truth_sic: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, Any]:
    raw = _spatial_occurrence_metrics(
        prediction_sic[:, None], truth_sic, valid_mask
    )
    return {
        "case_aggregation": raw["case_aggregation"],
        "occurrence_definition": raw["occurrence_definition"],
        "extent_unit": raw["extent_unit"],
        "truth_extent_mean": raw["truth_extent_mean"],
        "deterministic_extent_mean": raw["ensemble_expected_extent_mean"],
        "deterministic_extent_mae": raw["ensemble_expected_extent_mae"],
        "deterministic_extent_bias": raw["ensemble_expected_extent_bias"],
        "iiee_fraction": raw["expected_iiee_fraction"],
        "edge_definition": raw["edge_definition"],
        "edge_distance": raw["edge_distance"],
        "edge_displacement_mean": raw["edge_displacement_mean"],
        "edge_cases_both_present": raw["edge_cases_both_present"],
        "edge_cases_both_absent": raw["edge_cases_both_absent"],
        "edge_cases_one_absent": raw["edge_cases_one_absent"],
    }


def _deterministic_temporal_metrics(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, Any]:
    raw = _temporal_metrics(prediction[:, None], truth, valid_mask)
    transitions: dict[str, Any] = {}
    for transition_label, transition in raw["transitions"].items():
        fields: dict[str, Any] = {}
        for field, values in transition["fields"].items():
            correlation = values["lagged_spatial_correlation"]
            fields[field] = {
                "evaluated_points": values["evaluated_points"],
                "increment_rmse": values["ensemble_mean_increment_rmse"],
                "increment_mae": values["ensemble_mean_increment_mae"],
                "tendency_definition": values["tendency_definition"],
                "tendency_accuracy": values["ensemble_mean_tendency_accuracy"],
                "lagged_spatial_correlation": {
                    "definition": correlation["definition"],
                    "case_aggregation": correlation["case_aggregation"],
                    "defined_cases": correlation["defined_cases"],
                    "prediction_correlation": correlation[
                        "forecast_member_mean_correlation"
                    ],
                    "truth_correlation": correlation["truth_correlation"],
                    "mean_absolute_error": correlation["mean_absolute_error"],
                },
            }
        transitions[transition_label] = {"fields": fields}
    return {
        "increment_pairing": "deterministic consecutive archive dates",
        "domain": raw["calibration_domain"],
        "transitions": transitions,
        "trajectory_variogram": raw["trajectory_variogram"],
    }


@torch.no_grad()
def evaluate_structured_publication_ensemble(
    ensemble: torch.Tensor,
    truth: torch.Tensor,
    background: torch.Tensor,
    valid_mask: torch.Tensor,
    lag0_mask: torch.Tensor,
    *,
    contract: StructuredPublicationContract,
    case_ids: Sequence[str],
    member_seeds: Sequence[int],
    sic_cap: float,
) -> dict[str, Any]:
    """Compute the CPU-only marginal publication core for fixed paired cases.

    Day-0 forced track points are excluded from calibration exactly as declared
    by the protocol.  Future leads use every valid ocean point.  Mixed atoms use
    deterministic fractional rank mass, never arbitrary tie breaking.
    """

    support = validate_structured_publication_tensors(
        ensemble,
        truth,
        background,
        valid_mask,
        lag0_mask,
        contract=contract,
        case_ids=case_ids,
        member_seeds=member_seeds,
        sic_cap=sic_cap,
    )
    ensemble_mean = ensemble.mean(dim=1)
    result: dict[str, Any] = {
        "schema_version": PUBLICATION_RESULT_SCHEMA,
        "execution": "cpu_only",
        "case_manifest_sha256": contract.manifest_sha256,
        "publication_protocol_sha256": contract.protocol_sha256,
        "case_count": len(contract.cases),
        "ensemble_size": contract.ensemble_size,
        "member_seeds": list(contract.member_seeds),
        "trajectory_labels": ["d0", "d+1", "d+2", "d+3"],
        "channel_order": ["SIC", "SIT"],
        "aggregation": "pooled_valid_ocean_points",
        "rank_policy": contract.sic_rank_policy,
        "coverage_policy": "linear_empirical_central_quantiles_inclusive_endpoints",
        "cpu_bounding": {
            "member_pairwise_tensor_materialized": False,
            "edge_distance_complexity": "O(height*width)_per_case",
            "spatial_case_chunk": PUBLICATION_CPU_CASE_CHUNK,
        },
        "support": support,
        "leads": {},
    }
    valid = valid_mask > 0
    for lead in range(4):
        lead_label = "d0" if lead == 0 else f"d+{lead}"
        selector = valid & ~(lag0_mask > 0) if lead == 0 else valid
        point_count = int(selector.sum().item())
        if point_count == 0:
            raise ValueError(f"{lead_label} calibration stratum has no evaluated points")
        lead_result: dict[str, Any] = {
            "stratum": "day0_unobserved_ocean" if lead == 0 else "future_days",
            "fields": {},
        }
        for offset, field in enumerate(("sic", "sit")):
            channel = 2 * lead + offset
            channel_truth = truth[:, channel : channel + 1]
            channel_members = ensemble[:, :, channel : channel + 1]
            channel_mean = ensemble_mean[:, channel : channel + 1]
            error = channel_mean[selector] - channel_truth[selector]
            ranks = fractional_rank_counts(channel_members, channel_truth, selector)
            rank_total = ranks.sum()
            rank_frequencies = ranks / rank_total
            uniform = 1.0 / ranks.numel()
            coverage = _central_coverage(
                channel_members,
                channel_truth,
                selector,
                contract.coverage_levels,
            )
            field_result: dict[str, Any] = {
                "evaluated_points": point_count,
                "fair_crps": _fair_crps(channel_members, channel_truth, selector),
                "ensemble_mean_rmse": float(
                    error.square().mean(dtype=torch.float64).sqrt().item()
                ),
                "ensemble_mean_mae": float(error.abs().mean(dtype=torch.float64).item()),
                "fractional_rank_counts": ranks.tolist(),
                "fractional_rank_frequencies": rank_frequencies.tolist(),
                "rank_tv_to_uniform": float(
                    (rank_frequencies - uniform).abs().sum().mul(0.5).item()
                ),
                "central_coverage": coverage,
                "central_coverage_absolute_error": {
                    f"central_{round(100 * level):d}": abs(
                        coverage[f"central_{round(100 * level):d}"] - level
                    )
                    for level in contract.coverage_levels
                },
            }
            lead_result["fields"][field] = field_result
        sic_members = ensemble[:, :, 2 * lead]
        sic_truth = truth[:, 2 * lead]
        spatial_valid = selector[:, 0]
        lead_result["events"] = {
            "ice_occurrence": {
                "definition": "SIC>0",
                **_brier_reliability(
                    sic_members[:, :, None] > 0,
                    sic_truth[:, None] > 0,
                    selector,
                ),
            },
            "sic_archive_cap": {
                "definition": f"SIC=={float(sic_cap):.17g}",
                **_brier_reliability(
                    sic_members[:, :, None] == float(sic_cap),
                    sic_truth[:, None] == float(sic_cap),
                    selector,
                ),
            },
        }
        lead_result["spatial_occurrence"] = _spatial_occurrence_metrics(
            sic_members,
            sic_truth,
            spatial_valid,
        )
        lead_result["spatial_occurrence"]["fraction_skill_score"] = (
            _fraction_skill_score(sic_members, sic_truth, spatial_valid)
        )
        lead_result["spatial_variogram"] = {
            field: _local_spatial_variogram_score(
                ensemble[:, :, 2 * lead + offset],
                truth[:, 2 * lead + offset],
                spatial_valid,
            )
            for offset, field in enumerate(("sic", "sit"))
        }
        result["leads"][lead_label] = lead_result

    exact = (lag0_mask[:, None] > 0).expand_as(ensemble[:, :, :2])
    exact_error = (ensemble[:, :, :2] - truth[:, None, :2]).abs()
    result["day0_exact_track"] = {
        "evaluated_points_per_member": int(lag0_mask.sum().item()),
        "max_abs_error": float(exact_error[exact].max().item()) if torch.any(exact) else 0.0,
        "excluded_from_calibration": True,
    }
    result["temporal"] = _temporal_metrics(ensemble, truth, valid_mask)
    return result


@torch.no_grad()
def evaluate_structured_deterministic_baseline(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    valid_mask: torch.Tensor,
    lag0_mask: torch.Tensor,
    reference_background: torch.Tensor,
    *,
    baseline_name: str,
    conditioning_policy: str,
    contract: StructuredPublicationContract,
    case_ids: Sequence[str],
    sic_cap: float,
) -> dict[str, Any]:
    """Score one deterministic baseline without ensemble-only diagnostics."""

    contract.require_ready()
    if not isinstance(baseline_name, str) or not baseline_name.strip():
        raise ValueError("baseline_name must be a non-empty string")
    required_policy = _DETERMINISTIC_BASELINE_POLICIES.get(baseline_name)
    if required_policy is None:
        raise ValueError(
            "baseline_name is not an approved equal-information deterministic baseline"
        )
    if conditioning_policy != required_policy:
        raise ValueError(
            "conditioning_policy differs from the approved equal-information contract"
        )
    _require_cpu_tensor(prediction, "prediction")
    _require_cpu_tensor(truth, "truth")
    _require_cpu_tensor(reference_background, "reference_background")
    if (
        not prediction.is_floating_point()
        or not truth.is_floating_point()
        or not reference_background.is_floating_point()
    ):
        raise ValueError(
            "deterministic prediction/truth/background must use floating-point dtypes"
        )
    _require_binary_mask(valid_mask, "valid_mask")
    _require_binary_mask(lag0_mask, "lag0_mask")
    if prediction.ndim != 4 or prediction.shape != truth.shape or prediction.shape[1] != 8:
        raise ValueError("deterministic prediction/truth must be paired d0..d3 tensors")
    if reference_background.shape != truth.shape:
        raise ValueError("reference_background must be co-registered with truth")
    if tuple(valid_mask.shape) != (truth.shape[0], 1, *truth.shape[2:]):
        raise ValueError("valid_mask must have shape [case,1,height,width]")
    if lag0_mask.shape != valid_mask.shape:
        raise ValueError("lag0_mask must be co-registered with valid_mask")
    if torch.any((lag0_mask > 0) & (valid_mask == 0)):
        raise ValueError("lag0_mask must be a subset of valid_mask")
    if prediction.shape[0] != len(contract.cases):
        raise ValueError("deterministic tensor case count differs from frozen cases")
    if tuple(case_ids) != tuple(case.case_id for case in contract.cases):
        raise ValueError("deterministic baseline case_ids differ from frozen manifest order")
    if not torch.any(valid_mask > 0):
        raise ValueError("deterministic baseline has no valid ocean points")
    if baseline_name == "exact_previous_calendar_year_background":
        expected_prediction = reference_background
    else:
        expected_prediction = torch.cat(
            [reference_background[:, :2]] * 4, dim=1
        )
    if not torch.equal(prediction, expected_prediction):
        raise ValueError(
            f"{baseline_name} prediction differs from its approved source construction"
        )
    for label, values in (
        ("prediction", prediction),
        ("truth", truth),
        ("reference_background", reference_background),
    ):
        violations = _support_violation_count(
            values, valid_mask, sic_cap=float(sic_cap), ensemble=False
        )
        if violations:
            raise ValueError(
                f"{label} violates finite joint SIC/SIT support at {violations} valid points"
            )

    valid = valid_mask > 0
    result: dict[str, Any] = {
        "schema_version": "structured_joint_deterministic_baseline_v1",
        "baseline_name": baseline_name,
        "conditioning_policy": conditioning_policy,
        "score_family": "deterministic_only",
        "probabilistic_scores_forbidden": True,
        "case_manifest_sha256": contract.manifest_sha256,
        "case_count": len(contract.cases),
        "leads": {},
    }
    for lead in range(4):
        lead_label = "d0" if lead == 0 else f"d+{lead}"
        score_domain = valid & ~(lag0_mask > 0) if lead == 0 else valid
        if not torch.any(score_domain):
            raise ValueError(f"{lead_label} deterministic score domain is empty")
        lead_fields: dict[str, Any] = {}
        for offset, field in enumerate(("sic", "sit")):
            channel = 2 * lead + offset
            error = (
                prediction[:, channel : channel + 1][score_domain]
                - truth[:, channel : channel + 1][score_domain]
            )
            lead_fields[field] = {
                "evaluated_points": int(error.numel()),
                "rmse": float(error.square().mean(dtype=torch.float64).sqrt().item()),
                "mae": float(error.abs().mean(dtype=torch.float64).item()),
            }
        predicted_sic = prediction[:, 2 * lead]
        true_sic = truth[:, 2 * lead]
        domain = score_domain[:, 0]
        spatial = _deterministic_spatial_metrics(
            predicted_sic, true_sic, domain
        )
        spatial["fraction_skill_score"] = _fraction_skill_score(
            predicted_sic[:, None], true_sic, domain
        )
        result["leads"][lead_label] = {
            "stratum": "day0_unobserved_ocean" if lead == 0 else "future_days",
            "fields": lead_fields,
            "event_classification": {
                "ice_occurrence_error_fraction": float(
                    ((predicted_sic > 0) != (true_sic > 0))[domain]
                    .to(torch.float64)
                    .mean()
                    .item()
                ),
                "sic_archive_cap_error_fraction": float(
                    (
                        (predicted_sic == float(sic_cap))
                        != (true_sic == float(sic_cap))
                    )[domain]
                    .to(torch.float64)
                    .mean()
                    .item()
                ),
            },
            "spatial_occurrence": spatial,
            "spatial_variogram": {
                field: _local_spatial_variogram_score(
                    prediction[:, None, 2 * lead + offset],
                    truth[:, 2 * lead + offset],
                    domain,
                )
                for offset, field in enumerate(("sic", "sit"))
            },
        }
    result["temporal"] = _deterministic_temporal_metrics(
        prediction, truth, valid_mask
    )
    exact_domain = lag0_mask > 0
    result["day0_exact_track"] = {
        field: {
            "evaluated_points": int(exact_domain.sum().item()),
            "rmse": float(
                (
                    prediction[:, offset : offset + 1][exact_domain]
                    - truth[:, offset : offset + 1][exact_domain]
                )
                .square()
                .mean(dtype=torch.float64)
                .sqrt()
                .item()
            )
            if torch.any(exact_domain)
            else 0.0,
        }
        for offset, field in enumerate(("sic", "sit"))
    }
    return result


def build_structured_publication_visual_qc_metadata(
    contract: StructuredPublicationContract,
    *,
    sic_cap: float,
    case_indices: Sequence[int] = (0, 28, 56, 84, 112, 140, 168, 196),
    member_indices: Sequence[int] = (0, 16, 33, 49),
) -> dict[str, Any]:
    """Return a deterministic rendering manifest for publication visual QC."""

    contract.require_ready()
    if not math.isfinite(float(sic_cap)) or float(sic_cap) <= 0:
        raise ValueError("visual QC sic_cap must be finite and positive")
    if len(set(case_indices)) != len(case_indices) or not case_indices:
        raise ValueError("visual QC case_indices must be non-empty and unique")
    if len(set(member_indices)) != len(member_indices) or not member_indices:
        raise ValueError("visual QC member_indices must be non-empty and unique")
    if any(index < 0 or index >= len(contract.cases) for index in case_indices):
        raise ValueError("visual QC case index is outside the frozen manifest")
    if any(index < 0 or index >= contract.ensemble_size for index in member_indices):
        raise ValueError("visual QC member index is outside the frozen ensemble")
    selected_cases = []
    for index in case_indices:
        case = contract.cases[index]
        selected_cases.append(
            {
                "case_index": index,
                "case_id": case.case_id,
                "anchor_date": case.anchor_date,
                "target_archive_dates": list(case.target_archive_dates),
                "background_archive_dates": list(case.background_archive_dates),
                "member_indices": list(member_indices),
                "member_seeds": [contract.member_seeds[item] for item in member_indices],
                "expected_output": f"visual_qc/{case.case_id}.png",
            }
        )
    return {
        "schema_version": PUBLICATION_QC_SCHEMA,
        "selection_policy": "fixed_manifest_indices_no_metric_based_selection",
        "case_manifest_sha256": contract.manifest_sha256,
        "publication_protocol_sha256": contract.protocol_sha256,
        "values_space": "physical",
        "trajectory_labels": ["d0", "d+1", "d+2", "d+3"],
        "panel_columns": [
            "truth_SIC",
            "background_SIC",
            "member_SIC",
            "truth_SIT",
            "background_SIT",
            "member_SIT",
        ],
        "sic_color_limits": [0.0, float(sic_cap)],
        "sit_color_policy": "shared_valid_domain_p99_with_positive_floor",
        "rank_histogram_output": "visual_qc/fractional_rank_histograms.png",
        "coverage_output": "visual_qc/central_coverage_by_lead.png",
        "cases": selected_cases,
    }


def structured_trajectory_metrics(
    ensemble: torch.Tensor,
    truth: torch.Tensor,
    background: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    lag0_mask: torch.Tensor | None = None,
    sic_cap: float,
    exclude_lag0_from_day0_scores: bool = False,
) -> dict[str, float | list[float]]:
    """Return per-lead physical errors, fair CRPS, and tie-correct ranks."""
    if ensemble.ndim != 5 or truth.ndim != 4 or background.shape != truth.shape:
        raise ValueError("expected ensemble [B,M,2*T,H,W] and paired truth/background")
    if truth.shape[1] % 2 != 0 or ensemble.shape[2] != truth.shape[1]:
        raise ValueError("trajectory channels must be paired SIC/SIT at every lead")
    if ensemble.shape[1] < 2:
        raise ValueError("fair CRPS and rank diagnostics require at least two members")
    valid = valid_mask[:, :1]
    support_diagnostics = {}
    for name, values in (
        ("ensemble", ensemble),
        ("truth", truth.unsqueeze(1)),
        ("background", background.unsqueeze(1)),
    ):
        sic = values[:, :, 0::2]
        sit = values[:, :, 1::2]
        support_valid = valid[:, None].expand_as(sic) > 0
        nonfinite = (~torch.isfinite(sic) | ~torch.isfinite(sit)) & support_valid
        invalid = (
            nonfinite
            | (sic < 0)
            | (sic > float(sic_cap))
            | (sit < 0)
            | ((sic > 0) != (sit > 0))
        ) & support_valid
        invalid_count = int(invalid.sum().item())
        support_count = int(support_valid.sum().item())
        support_diagnostics[f"{name}_support_valid_count"] = float(support_count)
        support_diagnostics[f"{name}_support_violation_fraction"] = (
            invalid_count / max(support_count, 1)
        )
        if invalid_count:
            raise ValueError(
                f"{name} violates finite joint SIC/SIT support at {invalid_count} valid points "
                f"(sic_cap={sic_cap})"
            )
    ensemble_mean = ensemble.to(dtype=torch.float64).mean(dim=1)
    result: dict[str, float | list[float]] = dict(support_diagnostics)
    fields = ("sic", "sit")
    for lead in range(truth.shape[1] // 2):
        score_valid = valid
        if lead == 0 and exclude_lag0_from_day0_scores:
            if lag0_mask is None:
                raise ValueError(
                    "excluding exact day-0 observations requires lag0_mask"
                )
            score_valid = (valid > 0) & ~(lag0_mask[:, :1] > 0)
            if not torch.any(score_valid):
                raise ValueError("day-0 unobserved solver domain is empty")
        for offset, field in enumerate(fields):
            channel = 2 * lead + offset
            channel_truth = truth[:, channel : channel + 1]
            channel_background = background[:, channel : channel + 1]
            channel_members = ensemble[:, :, channel : channel + 1]
            channel_mean = ensemble_mean[:, channel : channel + 1]
            valid_count = score_valid.sum().clamp(min=1.0)
            result[f"lead{lead}_{field}_mean_rmse"] = float(
                (
                    (
                        (channel_mean - channel_truth.to(dtype=torch.float64)).square()
                        * score_valid
                    ).sum()
                    / valid_count
                )
                .sqrt()
                .item()
            )
            result[f"lead{lead}_{field}_background_rmse"] = float(
                (
                    (
                        (
                            channel_background.to(dtype=torch.float64)
                            - channel_truth.to(dtype=torch.float64)
                        ).square()
                        * score_valid
                    ).sum()
                    / valid_count
                )
                .sqrt()
                .item()
            )
            result[f"lead{lead}_{field}_fair_crps"] = _fair_crps(
                channel_members, channel_truth, score_valid
            )
            ranks = fractional_rank_counts(
                channel_members, channel_truth, score_valid
            )
            result[f"lead{lead}_{field}_rank_counts"] = ranks.cpu().tolist()
    result["support_violation_fraction"] = result[
        "ensemble_support_violation_fraction"
    ]
    if lag0_mask is not None:
        exact = lag0_mask[:, :1].unsqueeze(1).expand_as(ensemble[:, :, :1]) > 0
        exact_pair = exact.expand_as(ensemble[:, :, :2])
        result["lag0_observation_max_abs_error"] = float(
            (ensemble[:, :, :2] - truth[:, None, :2])[exact_pair].abs().max().item()
            if torch.any(exact_pair)
            else 0.0
        )
    return result


def make_structured_trajectory_figure(
    truth: torch.Tensor,
    background: torch.Tensor,
    sample: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    title: str,
    origin: str = "lower",
    lead_days: Sequence[int] | None = None,
):
    """Large per-lead truth/background/member figure in physical units."""
    import matplotlib.pyplot as plt

    arrays = [value.detach().to(dtype=torch.float32, device="cpu").numpy() for value in (
        truth, background, sample
    )]
    valid = valid_mask[:1].detach().to(device="cpu").numpy()[0] > 0
    leads = truth.shape[0] // 2
    displayed_leads = tuple(range(leads)) if lead_days is None else tuple(lead_days)
    if len(displayed_leads) != leads:
        raise ValueError("lead_days must match the trajectory channel count")
    fig, axes = plt.subplots(leads, 6, figsize=(25, 4.2 * leads), constrained_layout=True)
    if leads == 1:
        axes = np.asarray(axes)[None, :]
    background_label = (
        "initial d0 / persistence" if displayed_leads[0] > 0 else "background"
    )
    column_labels = (
        "truth SIC",
        f"{background_label} SIC",
        "sample SIC",
        "truth SIT",
        f"{background_label} SIT",
        "sample SIT",
    )
    sic_images = []
    sit_images = []
    sit_max = max(
        float(
            np.nanpercentile(
                np.where(valid, array[channel], np.nan),
                99,
            )
        )
        for array in arrays
        for channel in range(1, truth.shape[0], 2)
    )
    sit_max = max(sit_max, 1e-6)
    for lead in range(leads):
        fields = (
            arrays[0][2 * lead], arrays[1][2 * lead], arrays[2][2 * lead],
            arrays[0][2 * lead + 1], arrays[1][2 * lead + 1], arrays[2][2 * lead + 1],
        )
        for column, field in enumerate(fields):
            display = np.where(valid, field, np.nan)
            image = axes[lead, column].imshow(
                display,
                origin=origin,
                cmap="Blues" if column < 3 else "viridis",
                vmin=0.0,
                vmax=1.0 if column < 3 else sit_max,
            )
            axes[lead, column].set_title(column_labels[column])
            day = displayed_leads[lead]
            axes[lead, column].set_ylabel("d0" if day == 0 else f"d+{day}")
            axes[lead, column].set_xticks([])
            axes[lead, column].set_yticks([])
            (sic_images if column < 3 else sit_images).append(image)
    fig.colorbar(sic_images[-1], ax=axes[:, :3], shrink=0.75, label="SIC")
    fig.colorbar(sit_images[-1], ax=axes[:, 3:], shrink=0.75, label="SIT")
    fig.suptitle(title, fontsize=16)
    return fig
