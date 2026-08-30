#!/usr/bin/env python3
"""Atomic fail-closed admission for observation-likelihood reweighting."""

import hashlib, importlib.util, json
from dataclasses import dataclass
from pathlib import Path
from .validate_observation_likelihood_compact_outputs import directory_sha256, validate_directory

PAPER = Path(__file__).resolve().parent
CONTRACT = PAPER / "NEXT_OBSERVATION_LIKELIHOOD_REWEIGHTING_CONTRACT.md"
RANK_REFERENCE = PAPER / "weighted_rank_cell_reference.py"
REQUIRED = {"schema_version", "reviewed_mode", "publication_commit", "runner_sha256", "contract_sha256", "rank_reference_sha256", "compact_directory_sha256", "decision_bearing_validation", "deviations"}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_and_validate(record_path, runner_path, compact_directory):
    frozen = Path(record_path).read_bytes(); record = json.loads(frozen)
    if set(record) != REQUIRED or record["schema_version"] != "observation-likelihood-admission-v1" or record["decision_bearing_validation"] != "PASS" or record["deviations"] != []:
        raise ValueError("admission record fails exact schema")
    mode = record["reviewed_mode"]
    if not isinstance(mode, str) or not mode.startswith("validation_"): raise ValueError("literal reviewed_mode required")
    if not isinstance(record["publication_commit"], str) or len(record["publication_commit"]) != 40: raise ValueError("publication commit required")
    expected = {"runner_sha256": sha(runner_path), "contract_sha256": sha(CONTRACT), "rank_reference_sha256": sha(RANK_REFERENCE), "compact_directory_sha256": directory_sha256(compact_directory)}
    for key, value in expected.items():
        if record[key] != value: raise ValueError(f"{key} mismatch")
    spec = importlib.util.spec_from_file_location("observation_likelihood_candidate", runner_path)
    if spec is None or spec.loader is None: raise ValueError("runner not importable")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    if module.purged_folds() != __import__("paper.observation_likelihood_reweighting_runner", fromlist=["purged_folds"]).purged_folds(): raise ValueError("runner fold parity failed")
    weights, ess = module.likelihood_weights([[0.1*i, 0.2*i] for i in range(10)], [0.4, 0.8], 0.3)
    if len(weights) != 10 or not 1 < ess <= 10: raise ValueError("runner likelihood parity failed")
    validate_directory(compact_directory)
    if Path(record_path).read_bytes() != frozen or directory_sha256(compact_directory) != expected["compact_directory_sha256"]: raise ValueError("artifact changed during admission")
    return {"admission":"GO", "reviewed_mode":mode, "publication_commit":record["publication_commit"], "runner_sha256":record["runner_sha256"], "contract_sha256":record["contract_sha256"], "rank_reference_sha256":record["rank_reference_sha256"], "admission_record_sha256":hashlib.sha256(frozen).hexdigest(), "compact_directory_sha256":record["compact_directory_sha256"]}
