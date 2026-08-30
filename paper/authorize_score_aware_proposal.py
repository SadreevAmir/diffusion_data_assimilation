#!/usr/bin/env python3
"""Atomically turn a successful score-aware admission into a frozen proposal.

The literal mode remains an out-of-band controller input.  This command never
creates or registers a mode: it emits a proposal only from the GO payload
returned by the combined independent admission validator.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

try:
    from .score_aware_raw_reweighting_runner import ARTIFACT_POLICY, SOURCE_EXPERIMENT
    from .score_aware_raw_reweighting_server_adapter import (
        DATASET_SPLIT,
        END_DATE,
        RESOURCE_KIND,
        START_DATE,
        FrozenRequest,
        validate_request,
    )
    from .validate_score_aware_raw_reweighting_admission import load_and_validate
except ImportError:  # Keep direct script execution stable.
    from score_aware_raw_reweighting_runner import ARTIFACT_POLICY, SOURCE_EXPERIMENT
    from score_aware_raw_reweighting_server_adapter import (
        DATASET_SPLIT,
        END_DATE,
        RESOURCE_KIND,
        START_DATE,
        FrozenRequest,
        validate_request,
    )
    from validate_score_aware_raw_reweighting_admission import load_and_validate


def frozen_request(mode: str) -> FrozenRequest:
    return FrozenRequest(
        mode=mode,
        parameters={"source_experiment": SOURCE_EXPERIMENT},
        dataset_split=DATASET_SPLIT,
        start_date=START_DATE,
        end_date=END_DATE,
        cases=40,
        ensemble_size=10,
        resource_kind=RESOURCE_KIND,
        artifact_policy=ARTIFACT_POLICY,
    )


def authorize(admission: object) -> dict:
    if not hasattr(admission, "reviewed_mode"):
        raise ValueError("authorization requires the combined admission result")
    payload = asdict(admission)
    request = frozen_request(admission.reviewed_mode)
    validate_request(request, payload)
    return {
        "admission": payload["admission"],
        "decision_bearing_validation": payload["decision_bearing_validation"],
        "deviations": list(payload["deviations"]),
        "proposal_authorized": True,
        "reviewed_mode": payload["reviewed_mode"],
        "request": asdict(request),
        "bound_admission": payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combined score-aware admission and proposal authorization."
    )
    parser.add_argument("record", type=Path)
    parser.add_argument("runner", type=Path)
    parser.add_argument("compact_directory", type=Path)
    parser.add_argument("--expected-publication-commit", required=True)
    parser.add_argument("--controller-visible-mode", action="append", required=True)
    args = parser.parse_args()
    admission = load_and_validate(
        args.record,
        args.runner,
        args.compact_directory,
        args.expected_publication_commit,
        frozenset(args.controller_visible_mode),
    )
    print(json.dumps(authorize(admission), sort_keys=True))


if __name__ == "__main__":
    main()
