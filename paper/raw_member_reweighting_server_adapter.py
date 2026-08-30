#!/usr/bin/env python3
"""Fail-closed executor adapter for frozen raw-member reweighting.

The trusted executor supplies the literal reviewed mode and sealed server data.
This module intentionally neither invents a mode nor reads files or project data.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import raw_member_reweighting_runner as runner


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
    """Bind dispatch to the literal mode in one verified admission payload."""
    if not isinstance(admission, dict) or admission.get("admission") != "GO":
        raise ValueError("adapter requires one admission=GO payload")
    if admission.get("reviewed_mode") != requested_mode:
        raise ValueError("requested mode differs from the reviewed literal mode")
    required_identities = (
        ("review_record_sha256", 64),
        ("publication_commit", 40),
        ("runner_sha256", 64),
        ("contract_sha256", 64),
        ("synthetic_result_sha256", 64),
    )
    for key, length in required_identities:
        value = admission.get(key)
        if not isinstance(value, str) or len(value) != length or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"admission lacks bound {key}")


def validate_request(request: FrozenRequest, admission: object) -> None:
    validate_admission(admission, request.mode)
    runner.validate_interface(request.parameters, request.artifact_policy)
    expected = (
        request.dataset_split == DATASET_SPLIT,
        request.start_date == START_DATE,
        request.end_date == END_DATE,
        request.cases == runner.CASES,
        request.ensemble_size == runner.MEMBERS,
        request.resource_kind == RESOURCE_KIND,
    )
    if not all(expected):
        raise ValueError("request differs from the frozen server envelope")


class ReviewedModeInventory:
    """One-entry inventory populated only from an atomic reviewed admission."""

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
    inventory: ReviewedModeInventory, request: FrozenRequest, ranks, means, fields
):
    """Resolve reviewed dispatch before delegating deterministic construction."""
    if not isinstance(inventory, ReviewedModeInventory):
        raise ValueError("dispatch requires the reviewed mode inventory")
    admission = inventory.resolve(request.mode)
    validate_request(request, admission)
    return runner.construct_case(ranks, means, fields)
