#!/usr/bin/env python3
"""Fail-closed adapter populated only by atomic observation-likelihood admission."""

from dataclasses import dataclass
from . import observation_likelihood_reweighting_runner as runner


@dataclass(frozen=True)
class FrozenRequest:
    mode: str; parameters: dict[str, str]; dataset_split: str; start_date: str
    end_date: str; cases: int; ensemble_size: int; resource_kind: str; artifact_policy: str


def validate_admission(payload, mode):
    keys = {"admission", "reviewed_mode", "publication_commit", "runner_sha256", "contract_sha256", "rank_reference_sha256", "admission_record_sha256", "compact_directory_sha256"}
    if not isinstance(payload, dict) or set(payload) != keys or payload.get("admission") != "GO" or payload.get("reviewed_mode") != mode:
        raise ValueError("adapter requires exact atomic admission=GO")
    for key in keys - {"admission", "reviewed_mode"}:
        value, length = payload[key], 40 if key == "publication_commit" else 64
        if not isinstance(value, str) or len(value) != length or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"invalid bound identity: {key}")


def validate_request(request, admission):
    validate_admission(admission, request.mode)
    runner.validate_interface(request.parameters, request.artifact_policy)
    if (request.dataset_split, request.start_date, request.end_date, request.cases, request.ensemble_size, request.resource_kind) != ("valid", "2022-01-01", "2022-07-15", 40, 10, "server_cpu"):
        raise ValueError("request differs from frozen envelope")
