#!/usr/bin/env python3
"""Dependency-free review oracle for rank-coherent anomaly transport.

This is not an experiment entry point and reads no project data.  It makes the
frozen fold, ordering, anomaly-selection, and bounded mean-preservation rules
executable for an independent trusted-runner review.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path


EXPECTED_CASES = 40
EXPECTED_MEMBERS = 10
FEATURE_COUNT = 6
NEIGHBOR_COUNT = 10
HOLDOUT_SIZE = 8
PURGE = 3
ALPHAS = (0.0, 0.5, 0.75, 1.0, 1.25)
SOURCE_EXPERIMENT = "joint_full_condition_validation_2022"
ARTIFACT_POLICY = "summary_only"
DATASET_SPLIT = "valid"
START_DATE = "2022-01-01"
END_DATE = "2022-07-15"
CASE_STRIDE = 5
MANDATORY_GATE_FAMILIES = (
    "proper_score",
    "finite_ensemble_reliability",
    "boundary",
    "spatial_physical",
    "operational",
)
FOLD_SELECTION_KEYS = (
    "fold_index",
    "holdout_case_indices",
    "training_case_indices",
    "selected_alpha",
    "no_positive_feasible_alpha",
)
PROJECTION_DIAGNOSTIC_KEYS = (
    "changed_member_fraction",
    "lower_cap_mass",
    "upper_cap_mass",
    "maximum_mean_error",
    "member_semivariogram_distortion_pre",
    "member_semivariogram_distortion_post",
)
REQUIRED_AGGREGATE_METRICS = (
    "analysis_fair_crps",
    "analysis_crps",
    "analysis_mean_rmse",
)
METRIC_RECORD_KEYS = ("raw", "candidate", "delta")
PAIRED_RECORD_KEYS = (
    "raw",
    "candidate",
    "paired_mean_delta",
    "paired_date_95_ci",
    "four_case_block_95_ci",
)
MEMBER_SPATIAL_RECORD_KEYS = (
    "raw",
    "candidate",
    "absolute_delta",
    "maximum_allowed_absolute_delta",
    "passed",
)
ADMISSION_RECORD_KEYS = (
    "reviewed_mode", "publication_commit", "runner_sha256", "contract_sha256",
    "synthetic_result_sha256", "test_command", "test_sentinel",
    "decision_bearing_validation", "deviations",
)
FROZEN_CONTRACT_PATH = Path(__file__).with_name("NEXT_RANK_COHERENT_CONTRACT.md")


def frozen_contract_sha256() -> str:
    """Return the digest of the contract bytes reviewed with this oracle."""
    return hashlib.sha256(FROZEN_CONTRACT_PATH.read_bytes()).hexdigest()


def validate_admission_record(record: Mapping[str, object]) -> str:
    """Validate the exact controller-to-proposal admission record."""
    if set(record) != set(ADMISSION_RECORD_KEYS):
        raise ValueError("admission record keys must be exact")
    for key in ("reviewed_mode", "test_command", "test_sentinel"):
        value = record[key]
        if not isinstance(value, str) or not value.strip() or "<" in value or ">" in value:
            raise ValueError(f"{key} must be a non-placeholder string")
    if not re.fullmatch(r"[0-9a-f]{40}", str(record["publication_commit"])):
        raise ValueError("publication_commit must be lowercase 40-hex")
    for key in ("runner_sha256", "contract_sha256", "synthetic_result_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(record[key])):
            raise ValueError(f"{key} must be lowercase SHA-256")
    if record["contract_sha256"] != frozen_contract_sha256():
        raise ValueError("contract_sha256 must match the frozen local contract")
    if record["decision_bearing_validation"] != "PASS":
        raise ValueError("decision_bearing_validation must be PASS")
    if record["deviations"] != []:
        raise ValueError("deviations must be an empty list")
    return str(record["reviewed_mode"])


def _field_shape_and_finiteness(
    field: Sequence[Sequence[float]], expected_shape: tuple[int, int] | None = None
) -> tuple[int, int]:
    """Validate a finite, non-empty rectangular 2-D anomaly field."""
    rows = len(field)
    if rows == 0:
        raise ValueError("anomaly fields must be non-empty")
    columns = len(field[0])
    if columns == 0 or any(len(row) != columns for row in field):
        raise ValueError("anomaly fields must be non-empty rectangular arrays")
    try:
        finite = all(math.isfinite(value) for row in field for value in row)
    except TypeError as exc:
        raise ValueError("anomaly fields must contain scalar values") from exc
    if not finite:
        raise ValueError("anomaly fields must be finite")
    shape = (rows, columns)
    if expected_shape is not None and shape != expected_shape:
        raise ValueError("all anomaly fields must share one spatial shape")
    return shape


def purged_folds() -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Return five contiguous holdouts and their non-circular purged training sets."""
    folds = []
    for start in range(0, EXPECTED_CASES, HOLDOUT_SIZE):
        stop = start + HOLDOUT_SIZE
        holdout = tuple(range(start, stop))
        excluded_start = max(0, start - PURGE)
        excluded_stop = min(EXPECTED_CASES, stop + PURGE)
        training = tuple(
            index
            for index in range(EXPECTED_CASES)
            if not excluded_start <= index < excluded_stop
        )
        folds.append((holdout, training))
    return tuple(folds)


def validate_runner_interface(
    parameters: Mapping[str, object], artifact_policy: str
) -> None:
    """Reject runtime knobs or retrieval beyond the frozen interface."""
    if dict(parameters) != {"source_experiment": SOURCE_EXPERIMENT}:
        raise ValueError("runner must expose only the frozen source_experiment")
    if artifact_policy != ARTIFACT_POLICY:
        raise ValueError("runner retrieval must remain summary_only")


def validate_run_envelope(
    dataset_split: str,
    start_date: str,
    end_date: str,
    cases: int,
    ensemble_size: int,
    case_stride: int,
) -> None:
    """Reject any drift from the frozen development sampling envelope."""
    observed = (
        dataset_split,
        start_date,
        end_date,
        cases,
        ensemble_size,
        case_stride,
    )
    expected = (
        DATASET_SPLIT,
        START_DATE,
        END_DATE,
        EXPECTED_CASES,
        EXPECTED_MEMBERS,
        CASE_STRIDE,
    )
    if observed != expected:
        raise ValueError("runner must use the exact frozen development envelope")


def validate_compact_gate(gate: Mapping[str, object]) -> None:
    """Require a complete, boolean and logically consistent joint gate."""
    expected = {*MANDATORY_GATE_FAMILIES, "overall_eligible"}
    if set(gate) != expected:
        raise ValueError("compact gate must contain every mandatory family exactly once")
    if any(type(gate[name]) is not bool for name in expected):
        raise ValueError("compact gate flags must be JSON booleans")
    conjunction = all(gate[name] for name in MANDATORY_GATE_FAMILIES)
    if gate["overall_eligible"] is not conjunction:
        raise ValueError("overall_eligible must equal the mandatory-family conjunction")


def validate_fold_selections(folds: Sequence[Mapping[str, object]]) -> None:
    """Require complete frozen folds and logically consistent alpha decisions."""
    expected_folds = purged_folds()
    if len(folds) != len(expected_folds):
        raise ValueError("compact fold selections must contain exactly five folds")
    for position, (record, (holdout, training)) in enumerate(zip(folds, expected_folds)):
        if set(record) != set(FOLD_SELECTION_KEYS):
            raise ValueError("compact fold selection has missing or extra fields")
        if type(record["fold_index"]) is not int or record["fold_index"] != position:
            raise ValueError("compact fold indices must be ordered integers 0..4")
        if tuple(record["holdout_case_indices"]) != holdout:
            raise ValueError("compact holdout cases must match the frozen folds")
        if tuple(record["training_case_indices"]) != training:
            raise ValueError("compact training cases must match the frozen purge")
        alpha = record["selected_alpha"]
        no_positive = record["no_positive_feasible_alpha"]
        if type(alpha) not in (int, float) or type(alpha) is bool or alpha not in ALPHAS:
            raise ValueError("selected_alpha must be a number from the frozen set")
        if type(no_positive) is not bool:
            raise ValueError("no_positive_feasible_alpha must be a JSON boolean")
        if no_positive is not (alpha == 0.0):
            raise ValueError("no_positive_feasible_alpha must agree with selected_alpha")


def validate_rank_target_balance(balance: Sequence[object]) -> None:
    """Require every order-statistic target exactly once per held-out case."""
    if len(balance) != EXPECTED_MEMBERS:
        raise ValueError("rank-target balance must contain ten counts")
    if any(type(count) is not int or count != EXPECTED_CASES for count in balance):
        raise ValueError("every rank target must be used once in each held-out case")


def validate_projection_diagnostics(diagnostics: Mapping[str, object]) -> None:
    """Require finite, bounded projection accounting and the frozen mean invariant."""
    if set(diagnostics) != set(PROJECTION_DIAGNOSTIC_KEYS):
        raise ValueError("projection diagnostics have missing or extra fields")
    for name, value in diagnostics.items():
        if type(value) not in (int, float) or type(value) is bool or not math.isfinite(value):
            raise ValueError("projection diagnostics must be finite JSON numbers")
        if value < 0.0:
            raise ValueError("projection diagnostics must be non-negative")
        if name in {"changed_member_fraction", "lower_cap_mass", "upper_cap_mass"} and value > 1.0:
            raise ValueError("projection fractions and masses must lie in [0,1]")
    if diagnostics["maximum_mean_error"] > 1e-10:
        raise ValueError("projection diagnostics violate the frozen mean invariant")


def _finite_number(value: object, context: str) -> float:
    if type(value) not in (int, float) or type(value) is bool or not math.isfinite(value):
        raise ValueError(f"{context} must be a finite JSON number")
    return float(value)


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12)


def validate_aggregate_metrics(metrics: Mapping[str, object]) -> None:
    """Require complete proper-score anchors and arithmetically exact deltas."""
    if not set(REQUIRED_AGGREGATE_METRICS) <= set(metrics):
        raise ValueError("aggregate metrics omit a mandatory proper-score anchor")
    if not metrics:
        raise ValueError("aggregate metrics must not be empty")
    for name, record in metrics.items():
        if not isinstance(name, str) or not name or not isinstance(record, Mapping):
            raise ValueError("aggregate metrics must be named JSON objects")
        if set(record) != set(METRIC_RECORD_KEYS):
            raise ValueError("aggregate metric has missing or extra fields")
        raw = _finite_number(record["raw"], f"aggregate {name}.raw")
        candidate = _finite_number(record["candidate"], f"aggregate {name}.candidate")
        delta = _finite_number(record["delta"], f"aggregate {name}.delta")
        if not _close(delta, candidate - raw):
            raise ValueError("aggregate metric delta does not equal candidate minus raw")


def _validate_interval(value: object, point: float, context: str) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{context} must be a two-number interval")
    lower = _finite_number(value[0], f"{context} lower")
    upper = _finite_number(value[1], f"{context} upper")
    if lower > upper:
        raise ValueError(f"{context} must be ordered")


def validate_paired_uncertainty(
    paired: Mapping[str, object], aggregates: Mapping[str, object]
) -> None:
    """Reconcile paired summaries with the same complete aggregate metric set."""
    if set(paired) != set(aggregates):
        raise ValueError("paired uncertainty must cover exactly the aggregate metrics")
    for name, record in paired.items():
        if not isinstance(record, Mapping) or set(record) != set(PAIRED_RECORD_KEYS):
            raise ValueError("paired uncertainty record has missing or extra fields")
        aggregate = aggregates[name]
        raw = _finite_number(record["raw"], f"paired {name}.raw")
        candidate = _finite_number(record["candidate"], f"paired {name}.candidate")
        delta = _finite_number(record["paired_mean_delta"], f"paired {name}.delta")
        if not (_close(raw, aggregate["raw"]) and _close(candidate, aggregate["candidate"])):
            raise ValueError("paired and aggregate raw/candidate means disagree")
        if not _close(delta, candidate - raw):
            raise ValueError("paired mean delta does not equal candidate minus raw")
        _validate_interval(record["paired_date_95_ci"], delta, f"paired {name} date interval")
        _validate_interval(record["four_case_block_95_ci"], delta, f"paired {name} block interval")


def validate_member_spatial_deltas(spatial: Mapping[str, object]) -> None:
    """Require every member-spatial decision to follow its frozen absolute tolerance."""
    if not spatial:
        raise ValueError("member-spatial deltas must not be empty")
    for name, record in spatial.items():
        if not isinstance(name, str) or not name or not isinstance(record, Mapping):
            raise ValueError("member-spatial deltas must be named JSON objects")
        if set(record) != set(MEMBER_SPATIAL_RECORD_KEYS):
            raise ValueError("member-spatial record has missing or extra fields")
        raw = _finite_number(record["raw"], f"member-spatial {name}.raw")
        candidate = _finite_number(record["candidate"], f"member-spatial {name}.candidate")
        delta = _finite_number(record["absolute_delta"], f"member-spatial {name}.delta")
        tolerance = _finite_number(
            record["maximum_allowed_absolute_delta"], f"member-spatial {name}.tolerance"
        )
        if tolerance < 0.0 or delta < 0.0 or not _close(delta, abs(candidate - raw)):
            raise ValueError("member-spatial absolute delta or tolerance is inconsistent")
        if type(record["passed"]) is not bool or record["passed"] is not (delta <= tolerance):
            raise ValueError("member-spatial pass flag must follow its frozen tolerance")


def validate_gate_inputs(
    inputs: Mapping[str, object], gate: Mapping[str, object], spatial: Mapping[str, object]
) -> None:
    """Tie every family decision to complete criterion flags and spatial records."""
    if set(inputs) != set(MANDATORY_GATE_FAMILIES):
        raise ValueError("gate inputs must contain every mandatory family exactly once")
    for family, criteria in inputs.items():
        if not isinstance(criteria, Mapping) or not criteria:
            raise ValueError("every gate family must contain criterion flags")
        if any(not isinstance(name, str) or not name for name in criteria):
            raise ValueError("gate criteria must have non-empty names")
        if any(type(value) is not bool for value in criteria.values()):
            raise ValueError("gate criterion flags must be JSON booleans")
        if gate[family] is not all(criteria.values()):
            raise ValueError("family gate flag must equal its criterion conjunction")
    spatial_flags = inputs["spatial_physical"]
    if set(spatial_flags) != set(spatial):
        raise ValueError("spatial gate criteria must cover exactly member-spatial deltas")
    if any(spatial_flags[name] is not spatial[name]["passed"] for name in spatial):
        raise ValueError("spatial gate criteria disagree with member-spatial decisions")


def validate_compact_handoff(payload: Mapping[str, object]) -> None:
    """Validate the decision-bearing structural subset of the compact result."""
    expected = {
        "gate", "gate_inputs", "aggregate_metrics", "paired_uncertainty",
        "fold_selections", "rank_target_balance", "projection_diagnostics",
        "member_spatial_deltas",
    }
    if set(payload) != expected:
        raise ValueError("compact handoff has missing or extra structural sections")
    mapping_sections = expected - {"fold_selections", "rank_target_balance"}
    if any(not isinstance(payload[name], Mapping) for name in mapping_sections):
        raise ValueError("compact handoff mapping section has the wrong container type")
    if not isinstance(payload["fold_selections"], Sequence) or isinstance(payload["fold_selections"], (str, bytes)):
        raise ValueError("compact fold selections have the wrong container type")
    if not isinstance(payload["rank_target_balance"], Sequence) or isinstance(payload["rank_target_balance"], (str, bytes)):
        raise ValueError("rank-target balance has the wrong container type")
    validate_compact_gate(payload["gate"])
    validate_aggregate_metrics(payload["aggregate_metrics"])
    validate_paired_uncertainty(payload["paired_uncertainty"], payload["aggregate_metrics"])
    validate_fold_selections(payload["fold_selections"])
    validate_rank_target_balance(payload["rank_target_balance"])
    validate_projection_diagnostics(payload["projection_diagnostics"])
    validate_member_spatial_deltas(payload["member_spatial_deltas"])
    validate_gate_inputs(payload["gate_inputs"], payload["gate"], payload["member_spatial_deltas"])


def select_forecast_analogs(
    training_case_indices: Sequence[int],
    training_features: Sequence[Sequence[float]],
    heldout_features: Sequence[float],
) -> tuple[int, ...]:
    """Select ten analogs after training-only population standardization.

    Inputs are forecast-only features.  Distances are computed with means and
    population standard deviations derived exclusively from retained training
    rows; equal distances are resolved by the ordered case index.
    """
    if len(training_case_indices) != len(training_features):
        raise ValueError("training indices and feature rows must align")
    if len(training_case_indices) < NEIGHBOR_COUNT:
        raise ValueError("at least ten retained training cases are required")
    if len(set(training_case_indices)) != len(training_case_indices):
        raise ValueError("training case indices must be distinct")
    if any(index < 0 or index >= EXPECTED_CASES for index in training_case_indices):
        raise ValueError("training case index is outside the frozen envelope")
    if len(heldout_features) != FEATURE_COUNT:
        raise ValueError("held-out forecast must contain exactly six features")
    if any(len(row) != FEATURE_COUNT for row in training_features):
        raise ValueError("every training forecast must contain exactly six features")
    values = [value for row in training_features for value in row]
    if not all(math.isfinite(value) for value in (*values, *heldout_features)):
        raise ValueError("forecast-only features must be finite")

    means = tuple(
        sum(row[column] for row in training_features) / len(training_features)
        for column in range(FEATURE_COUNT)
    )
    standard_deviations = tuple(
        math.sqrt(
            sum((row[column] - means[column]) ** 2 for row in training_features)
            / len(training_features)
        )
        for column in range(FEATURE_COUNT)
    )
    if any(not math.isfinite(scale) or scale == 0.0 for scale in standard_deviations):
        raise ValueError("training population standard deviation must be positive and finite")

    distances = []
    for case_index, row in zip(training_case_indices, training_features):
        squared_distance = sum(
            ((row[column] - heldout_features[column]) / standard_deviations[column]) ** 2
            for column in range(FEATURE_COUNT)
        )
        distances.append((squared_distance, case_index))
    distances.sort()
    return tuple(case_index for _, case_index in distances[:NEIGHBOR_COUNT])


def select_rank_stratified_fields(
    analog_case_indices: Sequence[int],
    analog_truth_ranks: Sequence[float],
    anomaly_case_means: Sequence[Sequence[float]],
    anomaly_fields: Sequence[Sequence[Sequence[Sequence[float]]]],
) -> tuple[Sequence[Sequence[float]], ...]:
    """Take order statistic j from analog date j after scalar-rank sorting.

    Inputs contain exactly ten already-selected analog dates in neighbor order.
    Case-rank ties use the ordered case index; member-mean ties use member index.
    Complete fields are returned by reference and are never spatially shuffled.
    """
    lengths = {
        len(analog_case_indices),
        len(analog_truth_ranks),
        len(anomaly_case_means),
        len(anomaly_fields),
    }
    if lengths != {EXPECTED_MEMBERS}:
        raise ValueError("exactly ten selected analog dates are required")
    if len(set(analog_case_indices)) != EXPECTED_MEMBERS:
        raise ValueError("selected analog dates must be distinct")
    if any(index < 0 or index >= EXPECTED_CASES for index in analog_case_indices):
        raise ValueError("selected analog date is outside the frozen envelope")
    if not all(math.isfinite(rank) and 0.0 <= rank <= 1.0 for rank in analog_truth_ranks):
        raise ValueError("normalized analog truth ranks must be finite and bounded")
    if any(len(means) != EXPECTED_MEMBERS for means in anomaly_case_means):
        raise ValueError("every analog date must contain ten anomaly means")
    if any(len(fields) != EXPECTED_MEMBERS for fields in anomaly_fields):
        raise ValueError("every analog date must contain ten anomaly fields")

    spatial_shape = _field_shape_and_finiteness(anomaly_fields[0][0])
    for fields in anomaly_fields:
        for field in fields:
            _field_shape_and_finiteness(field, spatial_shape)

    date_order = sorted(
        range(EXPECTED_MEMBERS),
        key=lambda position: (
            analog_truth_ranks[position],
            analog_case_indices[position],
        ),
    )
    selected = []
    for order_statistic, date_position in enumerate(date_order):
        means = anomaly_case_means[date_position]
        if not all(math.isfinite(value) for value in means):
            raise ValueError("anomaly case means must be finite")
        member_order = sorted(
            range(EXPECTED_MEMBERS), key=lambda member: (means[member], member)
        )
        selected.append(anomaly_fields[date_position][member_order[order_statistic]])
    return tuple(selected)


def pair_with_heldout_member_order(
    heldout_anomaly_means: Sequence[float],
    selected_fields: Sequence[Sequence[Sequence[float]]],
) -> tuple[Sequence[Sequence[float]], ...]:
    """Place borrowed fields into held-out raw-member order-statistic slots."""
    if len(heldout_anomaly_means) != EXPECTED_MEMBERS:
        raise ValueError("held-out ensemble must contain ten anomaly means")
    if len(selected_fields) != EXPECTED_MEMBERS:
        raise ValueError("exactly ten selected fields are required")
    if not all(math.isfinite(value) for value in heldout_anomaly_means):
        raise ValueError("held-out anomaly means must be finite")
    member_order = sorted(
        range(EXPECTED_MEMBERS),
        key=lambda member: (heldout_anomaly_means[member], member),
    )
    paired: list[Sequence[Sequence[float]] | None] = [None] * EXPECTED_MEMBERS
    for order_statistic, member in enumerate(member_order):
        paired[member] = selected_fields[order_statistic]
    if any(field is None for field in paired):
        raise AssertionError("incomplete held-out member pairing")
    return tuple(field for field in paired if field is not None)


def capped_simplex_projection(
    provisional: Sequence[float], target_mean: float, tolerance: float = 1e-13
) -> tuple[float, ...]:
    """Euclidean projection onto [0,1]^M with an exact target member mean."""
    if len(provisional) != EXPECTED_MEMBERS:
        raise ValueError("projection requires ten members")
    if not all(math.isfinite(value) for value in provisional):
        raise ValueError("projection input must be finite")
    if not math.isfinite(target_mean) or not 0.0 <= target_mean <= 1.0:
        raise ValueError("target mean must be finite and bounded")
    target_sum = EXPECTED_MEMBERS * target_mean
    lower = min(provisional) - 1.0
    upper = max(provisional)
    for _ in range(100):
        shift = 0.5 * (lower + upper)
        current = sum(min(1.0, max(0.0, value - shift)) for value in provisional)
        if current > target_sum:
            lower = shift
        else:
            upper = shift
        if upper - lower <= tolerance:
            break
    shift = 0.5 * (lower + upper)
    projected = tuple(min(1.0, max(0.0, value - shift)) for value in provisional)
    if abs(sum(projected) / EXPECTED_MEMBERS - target_mean) > 1e-10:
        raise ValueError("projection failed the frozen mean-preservation tolerance")
    return projected


def construct_projected_candidate(
    heldout_mean: Sequence[Sequence[float]],
    paired_anomaly_fields: Sequence[Sequence[Sequence[float]]],
    alpha: float,
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    """Apply one frozen alpha and project every pixel to the held-out mean."""
    mean_shape = _field_shape_and_finiteness(heldout_mean)
    if len(paired_anomaly_fields) != EXPECTED_MEMBERS:
        raise ValueError("candidate construction requires ten paired fields")
    for field in paired_anomaly_fields:
        _field_shape_and_finiteness(field, mean_shape)
    if alpha not in ALPHAS:
        raise ValueError("candidate alpha must come from the frozen set")

    rows, columns = mean_shape
    candidate = [
        [[0.0 for _ in range(columns)] for _ in range(rows)]
        for _ in range(EXPECTED_MEMBERS)
    ]
    for row in range(rows):
        for column in range(columns):
            target_mean = heldout_mean[row][column]
            provisional = tuple(
                target_mean + alpha * paired_anomaly_fields[member][row][column]
                for member in range(EXPECTED_MEMBERS)
            )
            projected = capped_simplex_projection(provisional, target_mean)
            for member, value in enumerate(projected):
                candidate[member][row][column] = value

    frozen = tuple(
        tuple(tuple(row) for row in member_field) for member_field in candidate
    )
    for row in range(rows):
        for column in range(columns):
            values = tuple(field[row][column] for field in frozen)
            if min(values) < 0.0 or max(values) > 1.0:
                raise AssertionError("projected candidate escaped physical bounds")
            if abs(sum(values) / EXPECTED_MEMBERS - heldout_mean[row][column]) > 1e-10:
                raise AssertionError("projected candidate changed the held-out mean")
    return frozen


def select_alpha(training_scores: Mapping[float, float], feasible: Mapping[float, bool]) -> tuple[float, bool]:
    """Select minimum fair CRPS among feasible frozen alphas, tie to smaller."""
    if set(training_scores) != set(ALPHAS) or set(feasible) != set(ALPHAS):
        raise ValueError("alpha accounting must contain the exact frozen set")
    if not all(math.isfinite(score) for score in training_scores.values()):
        raise ValueError("training fair CRPS must be finite")
    candidates = [alpha for alpha in ALPHAS if feasible[alpha]]
    if not candidates:
        raise ValueError("alpha=0.0 must provide the null feasible construction")
    positive = [alpha for alpha in candidates if alpha > 0.0]
    if not positive:
        return 0.0, True
    selected = min(candidates, key=lambda alpha: (training_scores[alpha], alpha))
    return selected, False


def _self_test() -> None:
    validate_runner_interface(
        {"source_experiment": SOURCE_EXPERIMENT}, ARTIFACT_POLICY
    )
    validate_run_envelope(
        DATASET_SPLIT,
        START_DATE,
        END_DATE,
        EXPECTED_CASES,
        EXPECTED_MEMBERS,
        CASE_STRIDE,
    )
    for changed in (
        ("external_holdout", START_DATE, END_DATE, 40, 10, 5),
        (DATASET_SPLIT, START_DATE, END_DATE, 39, 10, 5),
        (DATASET_SPLIT, START_DATE, END_DATE, 40, 9, 5),
        (DATASET_SPLIT, START_DATE, END_DATE, 40, 10, 1),
    ):
        try:
            validate_run_envelope(*changed)
        except ValueError as exc:
            assert "exact frozen" in str(exc)
        else:
            raise AssertionError("modified run envelope did not fail closed")

    passing_gate = {name: True for name in MANDATORY_GATE_FAMILIES}
    passing_gate["overall_eligible"] = True
    validate_compact_gate(passing_gate)
    for malformed_gate in (
        {name: True for name in MANDATORY_GATE_FAMILIES},
        {**passing_gate, "boundary": 1},
        {**passing_gate, "spatial_physical": False},
    ):
        try:
            validate_compact_gate(malformed_gate)
        except ValueError:
            pass
        else:
            raise AssertionError("incomplete or inconsistent compact gate did not fail closed")

    folds = purged_folds()
    assert len(folds) == 5
    assert folds[0][0] == tuple(range(8)) and folds[-1][0] == tuple(range(32, 40))
    for holdout, training in folds:
        assert not set(holdout) & set(training)
        assert all(min(abs(case - held) for held in holdout) > PURGE for case in training)

    fold_records = tuple(
        {
            "fold_index": index,
            "holdout_case_indices": holdout,
            "training_case_indices": training,
            "selected_alpha": 0.0 if index == 0 else 1.0,
            "no_positive_feasible_alpha": index == 0,
        }
        for index, (holdout, training) in enumerate(folds)
    )
    projection_diagnostics = {
        "changed_member_fraction": 0.2,
        "lower_cap_mass": 0.01,
        "upper_cap_mass": 0.02,
        "maximum_mean_error": 5e-12,
        "member_semivariogram_distortion_pre": 0.03,
        "member_semivariogram_distortion_post": 0.04,
    }
    aggregate_metrics = {
        name: {"raw": 1.0, "candidate": 0.95, "delta": -0.05}
        for name in REQUIRED_AGGREGATE_METRICS
    }
    paired_uncertainty = {
        name: {
            "raw": 1.0,
            "candidate": 0.95,
            "paired_mean_delta": -0.05,
            "paired_date_95_ci": (-0.08, -0.02),
            "four_case_block_95_ci": (-0.09, 0.01),
        }
        for name in REQUIRED_AGGREGATE_METRICS
    }
    member_spatial_deltas = {
        "member_semivariogram_lag_1": {
            "raw": 0.20,
            "candidate": 0.205,
            "absolute_delta": 0.005,
            "maximum_allowed_absolute_delta": 0.01,
            "passed": True,
        },
        "member_semivariogram_lag_4": {
            "raw": 0.30,
            "candidate": 0.32,
            "absolute_delta": 0.02,
            "maximum_allowed_absolute_delta": 0.01,
            "passed": False,
        },
    }
    gate_inputs = {
        "proper_score": {"fair_crps": True, "crps": True, "mean_rmse": True},
        "finite_ensemble_reliability": {"rank_uniformity": True, "coverage": True},
        "boundary": {"exact_zero": True, "exact_one": True},
        "spatial_physical": {
            "member_semivariogram_lag_1": True,
            "member_semivariogram_lag_4": False,
        },
        "operational": {"case_count": True, "finite_members": True},
    }
    passing_gate["spatial_physical"] = False
    passing_gate["overall_eligible"] = False
    compact_handoff = {
        "gate": passing_gate,
        "gate_inputs": gate_inputs,
        "aggregate_metrics": aggregate_metrics,
        "paired_uncertainty": paired_uncertainty,
        "fold_selections": fold_records,
        "rank_target_balance": (EXPECTED_CASES,) * EXPECTED_MEMBERS,
        "projection_diagnostics": projection_diagnostics,
        "member_spatial_deltas": member_spatial_deltas,
    }
    validate_compact_handoff(compact_handoff)
    malformed_handoffs = (
        {**compact_handoff, "fold_selections": fold_records[:-1]},
        {
            **compact_handoff,
            "fold_selections": (
                {**fold_records[0], "no_positive_feasible_alpha": False},
                *fold_records[1:],
            ),
        },
        {
            **compact_handoff,
            "rank_target_balance": (EXPECTED_CASES - 1,) + (EXPECTED_CASES,) * 9,
        },
        {
            **compact_handoff,
            "projection_diagnostics": {
                **projection_diagnostics,
                "maximum_mean_error": 1.1e-10,
            },
        },
        {
            **compact_handoff,
            "aggregate_metrics": {
                **aggregate_metrics,
                "analysis_fair_crps": {
                    **aggregate_metrics["analysis_fair_crps"], "delta": -0.04
                },
            },
        },
        {
            **compact_handoff,
            "paired_uncertainty": {
                **paired_uncertainty,
                "analysis_crps": {
                    **paired_uncertainty["analysis_crps"], "raw": 0.9
                },
            },
        },
        {
            **compact_handoff,
            "member_spatial_deltas": {
                **member_spatial_deltas,
                "member_semivariogram_lag_4": {
                    **member_spatial_deltas["member_semivariogram_lag_4"],
                    "passed": True,
                },
            },
        },
        {
            **compact_handoff,
            "gate_inputs": {
                **gate_inputs,
                "proper_score": {**gate_inputs["proper_score"], "fair_crps": False},
            },
        },
    )
    for malformed in malformed_handoffs:
        try:
            validate_compact_handoff(malformed)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("malformed compact handoff did not fail closed")

    training_indices = tuple(range(10, 22))
    training_features = tuple(
        tuple(float((case + 1) * (feature + 1)) for feature in range(FEATURE_COUNT))
        for case in range(12)
    )
    analogs = select_forecast_analogs(
        training_indices,
        training_features,
        tuple(5.5 * (feature + 1) for feature in range(FEATURE_COUNT)),
    )
    assert analogs[:2] == (14, 15)
    assert len(analogs) == NEIGHBOR_COUNT and len(set(analogs)) == NEIGHBOR_COUNT
    try:
        select_forecast_analogs(
            training_indices,
            tuple(
                tuple(float(case) if feature else 1.0 for feature in range(FEATURE_COUNT))
                for case in range(12)
            ),
            (0.0,) * FEATURE_COUNT,
        )
    except ValueError as exc:
        assert "standard deviation" in str(exc)
    else:
        raise AssertionError("zero-variance training feature did not fail closed")

    fields = [
        [[1000 * case + 10 * member + row + col for col in range(2)] for row in range(2)]
        for case in range(10)
        for member in range(10)
    ]
    nested_fields = [fields[case * 10 : (case + 1) * 10] for case in range(10)]
    means = [[float(member) for member in range(10)] for _ in range(10)]
    selected = select_rank_stratified_fields(
        list(reversed(range(10))), [0.5] * 10, means, nested_fields
    )
    assert selected[0] is nested_fields[9][0]
    assert selected[9] is nested_fields[0][9]
    paired = pair_with_heldout_member_order(list(reversed(range(10))), selected)
    assert paired[9] is selected[0] and paired[0] is selected[9]
    malformed_fields = [[field for field in case] for case in nested_fields]
    malformed_fields[0] = [field for field in malformed_fields[0]]
    malformed_fields[0][0] = [[math.nan]]
    try:
        select_rank_stratified_fields(
            list(range(10)), [0.5] * 10, means, malformed_fields
        )
    except ValueError as exc:
        assert "spatial shape" in str(exc) or "finite" in str(exc)
    else:
        raise AssertionError("malformed anomaly field did not fail closed")
    try:
        select_rank_stratified_fields(
            list(range(10)), [1.1] + [0.5] * 9, means, nested_fields
        )
    except ValueError as exc:
        assert "truth ranks" in str(exc)
    else:
        raise AssertionError("out-of-range normalized truth rank did not fail closed")

    projected = capped_simplex_projection(
        (-0.5, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5), 0.5
    )
    assert min(projected) == 0.0 and max(projected) == 1.0
    assert abs(sum(projected) / 10 - 0.5) <= 1e-10
    heldout_mean = ((0.05, 0.5), (0.95, 0.25))
    coherent_fields = tuple(
        tuple(
            tuple((member - 4.5) * (row + column + 1) / 5.0 for column in range(2))
            for row in range(2)
        )
        for member in range(EXPECTED_MEMBERS)
    )
    candidate = construct_projected_candidate(heldout_mean, coherent_fields, 1.25)
    assert len(candidate) == EXPECTED_MEMBERS
    assert any(value == 0.0 for field in candidate for row in field for value in row)
    assert any(value == 1.0 for field in candidate for row in field for value in row)
    for row in range(2):
        for column in range(2):
            assert abs(
                sum(field[row][column] for field in candidate) / EXPECTED_MEMBERS
                - heldout_mean[row][column]
            ) <= 1e-10
    try:
        construct_projected_candidate(heldout_mean, coherent_fields, 0.6)
    except ValueError as exc:
        assert "frozen set" in str(exc)
    else:
        raise AssertionError("non-frozen alpha did not fail closed")
    selected_alpha, no_positive = select_alpha(
        {alpha: abs(alpha - 1.0) for alpha in ALPHAS},
        {alpha: alpha <= 1.0 for alpha in ALPHAS},
    )
    assert selected_alpha == 1.0 and no_positive is False
    assert select_alpha(
        {alpha: alpha for alpha in ALPHAS},
        {alpha: alpha == 0.0 for alpha in ALPHAS},
    ) == (0.0, True)
    validate_runner_interface({"source_experiment": SOURCE_EXPERIMENT}, ARTIFACT_POLICY)
    admission = {
        "reviewed_mode": "validation_reviewed_rank_coherent",
        "publication_commit": "a" * 40,
        "runner_sha256": "b" * 64,
        "contract_sha256": frozen_contract_sha256(),
        "synthetic_result_sha256": "d" * 64,
        "test_command": "python3 trusted_test.py",
        "test_sentinel": "trusted rank-coherent adapter: PASS",
        "decision_bearing_validation": "PASS",
        "deviations": [],
    }
    assert validate_admission_record(admission) == admission["reviewed_mode"]
    for key, bad_value in (("reviewed_mode", "<mode>"), ("publication_commit", "A" * 40), ("runner_sha256", "b" * 63), ("decision_bearing_validation", "FAIL"), ("deviations", ["waiver"])):
        invalid = dict(admission); invalid[key] = bad_value
        try:
            validate_admission_record(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid admission field {key} did not fail closed")
    try:
        validate_runner_interface(
            {"source_experiment": SOURCE_EXPERIMENT, "alpha": 1.0},
            ARTIFACT_POLICY,
        )
    except ValueError as exc:
        assert "only" in str(exc)
    else:
        raise AssertionError("runtime alpha did not fail closed")


if __name__ == "__main__":
    _self_test()
    print("rank-coherent reference checks: PASS")
