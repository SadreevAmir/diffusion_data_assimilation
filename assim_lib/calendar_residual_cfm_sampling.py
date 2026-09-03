"""Frozen validation sampler for the completed calendar-residual CFM.

This module is deliberately a thin, fail-closed wrapper around ``assim_lib.evaluate``.
The expensive arrays remain in the server result directory; only status and a
hash-bound manifest are intended for retrieval by the controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

MODE = "validation_siconc_calendar_residual_cfm_sampling"
EXPERIMENT_ID = "siconc_calendar_residual_cfm_sampling_valid"
SOURCE_EXPERIMENT = "siconc_calendar_residual_cfm_training_retry1"
CHECKPOINT_NAME = "ema_last_model.pth"
CHECKPOINT_SHA256 = "78867f38b185b37bb0e1d0fb728aa90320e0676b98a5a689d49d0732af6ccd47"
SOURCE_METADATA_SHA256 = "b5cf8cedfc23333f65659c011fb78dfa5b69f46740aeee6470151400a687bb06"
SOURCE_TRAINING_METADATA_SHA256 = (
    "c4a9ddbe73d8761caae71a06a9283b3ac6593c527c46bcd5245da678f10d3aed"
)
SOURCE_METRICS_SHA256 = "c6a74d381d1d5d7ceec0a19ecb403d316e00386c01975a9bdbf2354a00121494"
CASE_COUNT = 40
MEMBER_COUNT = 10
EXPECTED_DATES = tuple(date(2022, 1, 2) + timedelta(days=5 * index) for index in range(CASE_COUNT))
PUBLIC_FILES = {"run_status.json", "metadata.json"}


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def validate_source(source_root: Path) -> Path:
    root = source_root.resolve(strict=True)
    expected = {
        "metadata.json": SOURCE_METADATA_SHA256,
        "training/metadata.json": SOURCE_TRAINING_METADATA_SHA256,
        "training/metrics.json": SOURCE_METRICS_SHA256,
        f"training/{CHECKPOINT_NAME}": CHECKPOINT_SHA256,
    }
    for relative, digest in expected.items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or _hash(path) != digest:
            raise ValueError(f"source artifact differs from frozen inventory: {relative}")
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("experiment_id") != SOURCE_EXPERIMENT or metadata.get("test_data_used") is not False:
        raise ValueError("source metadata identity or split safety differs")
    if metadata.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("source metadata checkpoint digest differs")
    return root / "training"


def validate_evaluation(evaluation_root: Path) -> tuple[list[dict[str, Any]], str]:
    metadata = json.loads((evaluation_root / "metadata.json").read_text(encoding="utf-8"))
    expected = {
        "split": "valid",
        "num_cases": CASE_COUNT,
        "stride_days": 5,
        "ensemble_size": MEMBER_COUNT,
        "sample_batch_size": MEMBER_COUNT,
        "num_timesteps": 25,
        "sampling_method": "dopri5",
        "sampling_rtol": 1e-5,
        "sampling_atol": 1e-6,
        "inference_precision": "float32",
        "sample_target": "residual",
        "sample_cfg_mode": "none",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"evaluation metadata differs for {key}")
    cases = metadata.get("cases")
    if not isinstance(cases, list) or len(cases) != CASE_COUNT:
        raise ValueError("evaluation must contain exactly forty cases")
    paths = sorted((evaluation_root / "samples").glob("*.npz"))
    if len(paths) != CASE_COUNT:
        raise ValueError("evaluation sample inventory must contain exactly forty NPZ files")
    manifest = hashlib.sha256()
    records: list[dict[str, Any]] = []
    for index, (expected_date, case, path) in enumerate(zip(EXPECTED_DATES, cases, paths, strict=True)):
        target_date = case.get("target_date")
        if case.get("case_order") != index or target_date != expected_date.isoformat():
            raise ValueError("evaluation date schedule differs from frozen residual schedule")
        if expected_date.isoformat() not in path.name or path.is_symlink():
            raise ValueError("sample filename differs from frozen residual schedule")
        with np.load(path, allow_pickle=False) as payload:
            ensemble = np.asarray(payload["analysis_ensemble"])
            truth = np.asarray(payload["truth"])
            valid = np.asarray(payload["valid_mask"])
            if ensemble.shape != (MEMBER_COUNT, 1, 320, 256):
                raise ValueError("analysis ensemble has unexpected shape")
            if truth.shape != (1, 320, 256) or valid.shape != (1, 320, 256):
                raise ValueError("truth or valid mask has unexpected shape")
            if ensemble.dtype != np.float32 or truth.dtype != np.float32 or valid.dtype != np.bool_:
                raise ValueError("sample dtypes differ from the frozen contract")
            if not np.all(np.isfinite(ensemble)) or not np.all(np.isfinite(truth)):
                raise ValueError("sample contains non-finite concentrations")
            if np.any((ensemble < 0) | (ensemble > 1)) or np.any((truth < 0) | (truth > 1)):
                raise ValueError("sample concentrations are outside [0,1]")
        digest = _hash(path)
        manifest.update(path.name.encode())
        manifest.update(b"\0")
        manifest.update(digest.encode())
        manifest.update(b"\n")
        records.append({"case_index": index, "target_date": target_date, "sample_sha256": digest})
    return records, manifest.hexdigest()


def run(output_dir: Path, source_root: Path) -> None:
    output = output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    unexpected = {path.name for path in output.iterdir()} - PUBLIC_FILES
    if unexpected:
        raise ValueError(f"output directory is not fresh: {sorted(unexpected)}")
    _atomic_json(
        output / "run_status.json",
        {"status": "running", "experiment_id": EXPERIMENT_ID, "completed_cases": 0, "total_cases": CASE_COUNT},
    )
    internal = output / "internal" / "evaluation"
    try:
        training_dir = validate_source(source_root)
        repo = Path(__file__).resolve().parents[1]
        subprocess.run(
            [
                sys.executable,
                "-m",
                "assim_lib.evaluate",
                "--config",
                str(repo / "config/experiments/evaluate_siconc_calendar_residual_cfm.json"),
                "--run-dir",
                str(training_dir),
                "--output-dir",
                str(internal),
                "--checkpoint-name",
                CHECKPOINT_NAME,
            ],
            cwd=repo,
            check=True,
        )
        cases, sample_manifest = validate_evaluation(internal)
        _atomic_json(
            output / "metadata.json",
            {
                "status": "completed",
                "mode": MODE,
                "experiment_id": EXPERIMENT_ID,
                "source_experiment": SOURCE_EXPERIMENT,
                "checkpoint_name": CHECKPOINT_NAME,
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "dataset_split": "valid",
                "date_range": [EXPECTED_DATES[0].isoformat(), EXPECTED_DATES[-1].isoformat()],
                "case_stride_days": 5,
                "num_cases": CASE_COUNT,
                "ensemble_size": MEMBER_COUNT,
                "sample_manifest_sha256": sample_manifest,
                "cases": cases,
                "test_data_used": False,
                "raw_arrays": "server_only",
            },
        )
        _atomic_json(
            output / "run_status.json",
            {"status": "completed", "experiment_id": EXPERIMENT_ID, "completed_cases": CASE_COUNT, "total_cases": CASE_COUNT},
        )
    except Exception as error:
        _atomic_json(
            output / "run_status.json",
            {"status": "failed", "experiment_id": EXPERIMENT_ID, "detail": str(error), "completed_cases": 0, "total_cases": CASE_COUNT},
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    arguments = parser.parse_args()
    run(arguments.output_dir, arguments.source_root)


if __name__ == "__main__":
    main()
