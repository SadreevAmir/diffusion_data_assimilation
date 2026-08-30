#!/usr/bin/env python3
"""Fail-closed executor adapter for frozen score-aware raw reweighting.

The adapter consumes only the atomic payload emitted by the combined admission
validator.  It does not select or invent a production mode.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import score_aware_raw_reweighting_runner as runner


START_DATE = "2022-01-01"
END_DATE = "2022-07-15"
DATASET_SPLIT = "valid"
RESOURCE_KIND = "server_cpu"


@dataclass(frozen=True)
class FrozenRequest:
    mode: str
    parameters: dict[str, str]
    dataset_split: str
    start_date: str
    end_date: str
    cases: int
    ensemble_size: int
    resource_kind: str
    artifact_policy: str


def validate_admission(admission: object, requested_mode: str) -> None:
    if not isinstance(admission, dict) or admission.get("admission") != "GO":
        raise ValueError("adapter requires the atomic admission=GO payload")
    if admission.get("reviewed_mode") != requested_mode:
        raise ValueError("requested mode differs from the reviewed literal mode")
    identities = (
        ("publication_commit", 40),
        ("runner_sha256", 64),
        ("contract_sha256", 64),
        ("reference_sha256", 64),
        ("admission_record_sha256", 64),
        ("compact_directory_sha256", 64),
    )
    if set(admission) != {"admission", "reviewed_mode", *(key for key, _ in identities)}:
        raise ValueError("admission payload has schema drift")
    for key, length in identities:
        value = admission.get(key)
        if not isinstance(value, str) or len(value) != length or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"admission lacks bound {key}")


def validate_request(request: FrozenRequest, admission: object) -> None:
    validate_admission(admission, request.mode)
    runner.validate_interface(request.parameters, request.artifact_policy)
    if not all(
        (
            request.dataset_split == DATASET_SPLIT,
            request.start_date == START_DATE,
            request.end_date == END_DATE,
            request.cases == 40,
            request.ensemble_size == runner.MEMBERS,
            request.resource_kind == RESOURCE_KIND,
        )
    ):
        raise ValueError("request differs from the frozen server envelope")


class ReviewedModeInventory:
    """Immutable one-entry inventory populated from combined admission only."""

    def __init__(self) -> None:
        self._entry: tuple[str, dict[str, str]] | None = None

    def register(self, admission: object) -> str:
        if not isinstance(admission, dict):
            raise ValueError("admission must be one object")
        mode = admission.get("reviewed_mode")
        if not isinstance(mode, str):
            raise ValueError("admission lacks reviewed_mode")
        validate_admission(admission, mode)
        entry = (mode, dict(admission))
        if self._entry is not None and self._entry != entry:
            raise ValueError("reviewed inventory entry is immutable")
        self._entry = entry
        return mode

    def resolve(self, mode: str) -> dict[str, str]:
        if self._entry is None or self._entry[0] != mode:
            raise KeyError("mode is not present in reviewed executor inventory")
        return dict(self._entry[1])


def construct_case(
    inventory: ReviewedModeInventory,
    request: FrozenRequest,
    training_predictors,
    training_targets,
    heldout_predictors,
    raw_members,
):
    if not isinstance(inventory, ReviewedModeInventory):
        raise ValueError("dispatch requires the reviewed mode inventory")
    admission = inventory.resolve(request.mode)
    validate_request(request, admission)
    return runner.construct_case(
        training_predictors, training_targets, heldout_predictors, raw_members
    )
