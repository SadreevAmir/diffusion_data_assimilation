"""Sealed raw sampling for the frozen 48-day 2024 calendar confirmation."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_json, merge_config_overrides, resolve_path
from .deep_ensemble import (
    COMPACT_OUTPUTS,
    _atomic_csv,
    _atomic_json,
    _canonical_hash,
    _file_hash,
    _run_checked,
    _sample_manifest,
)
from .latent_temperature_ensemble import _validate_source

MODE = "external_2024_calendar_global_bias_raw48_primary"
EXPERIMENT_ID = "external_2024_calendar_global_bias_raw48_primary_v1"
DEPENDENT_EXPERIMENT_ID = "external_2024_calendar_global_bias_confirm48_primary_v1"
DEPENDENT_MODE = "external_2024_calendar_global_bias_confirm48_primary"
PROTOCOL_ID = "external_2024_calendar_global_bias_confirm48_primary_v1"
PRIMARY_CANDIDATE = "calendar_global_bias_mixture_v1_refit_all40"
SOURCE_EXPERIMENT = "joint_full_condition_validation_2022"
CASE_COUNT = 48
MEMBER_COUNT = 10
BASE_NOISE_SEED = 1234
EXPECTED_DATES = tuple(date(2024, 1, 1) + timedelta(days=index) for index in range(CASE_COUNT))
BACKGROUND_DATES = tuple(date(2023, 1, 1) + timedelta(days=index) for index in range(CASE_COUNT))
TRACK_DATES = tuple(
    date(2023, 12, 30) + timedelta(days=index)
    for index in range((date(2024, 2, 18) - date(2023, 12, 30)).days + 1)
)
TARGET_DATE_MANIFEST_SHA256 = "3b929bd4dfc0b064b1397ae642c047896914e1fe7cbee2037f7e70a28f49836c"
CHECKPOINT_NAME = "ema_last_model.pth"
CHECKPOINT_SHA256 = "cd73cedc97a9f19d15c70ba31d28d3248a77dc6ef568edf8793326763810fdfe"
DEVELOPMENT_SOURCE_SAMPLE_MANIFEST_SHA256 = "d69cd03b183b75bac368c2ed59af5aef84d8bec2205031e1cdfc57ceed90c907"
DEVELOPMENT_FAMILY_CONTRACT_SHA256 = "087bddd55453f4a45eaecb226a6181c4624a297cb556eee824efb95802b0ee7b"
PRIMARY_CONTRACT_SHA256 = "5d77ac9d40dfd8ccba0ba75729a5f8a7830b72bbe63368ea4eaab8a8e17abaf3"
CONFIG_PATH = "config/experiments/external_2024_calendar_global_bias_raw48_primary_v1.json"
MODEL_CONFIG_PATH = "config/methods/concat_conditioning_diffusion_balanced_2f.json"
DATA_CONFIG_PATH = "config/data/m2m_2f_1y.json"
INPUT_SEAL_NAME = "input_seal.json"
SAMPLE_SEAL_NAME = "sealed_manifest.json"


def frozen_primary_contract() -> dict[str, Any]:
    """Return the pre-registered primary contract whose hash was frozen before outcomes."""
    return {
        "protocol_id": PROTOCOL_ID,
        "primary_candidate": PRIMARY_CANDIDATE,
        "alternative_not_tested": "locked_mc_dropout_p010_final_ema_ensemble",
        "launch_rule": (
            "run_once_regardless_of_both_2022_development_gate_outcomes_after_exact_artifacts_exist"
        ),
        "development_gate_used_for_selection": False,
        "development_family_contract_sha256": DEVELOPMENT_FAMILY_CONTRACT_SHA256,
        "development_source_experiment": SOURCE_EXPERIMENT,
        "development_source_sample_manifest_sha256": (DEVELOPMENT_SOURCE_SAMPLE_MANIFEST_SHA256),
        "development_dates": {
            "start": "2022-01-01",
            "end": "2022-07-15",
            "stride_days": 5,
            "num_cases": 40,
        },
        "fit": "all_40_development_area_residuals",
        "evaluation_split": "external_chronological_holdout_2024_confirm48",
        "target_dates": {
            "start": "2024-01-01",
            "end": "2024-02-17",
            "stride_days": 1,
            "num_cases": CASE_COUNT,
            "manifest_sha256": TARGET_DATE_MANIFEST_SHA256,
        },
        "background_dates": {"start": "2023-01-01", "end": "2023-02-17"},
        "track_history_days": [0, -1, -2],
        "target_hour": 23,
        "field_protocol": "siconc-only",
        "observation_source": "M2M_values_on_real_SRAL_rule1_footprints",
        "checkpoint_name": CHECKPOINT_NAME,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "conditioning_mode": "full",
        "ensemble_size": MEMBER_COUNT,
        "sample_batch_size": MEMBER_COUNT,
        "sampler": {
            "method": "dopri5",
            "num_timesteps": 25,
            "rtol": 1e-5,
            "atol": 1e-6,
            "inference_precision": "float32",
            "seed": BASE_NOISE_SEED,
            "member_seed_formula": "1234 + case_index*10 + member_index",
        },
        "mixture_probability": 0.5,
        "calendar_bandwidth_days": 30.0,
        "calendar_distance": "cyclic_no_leap_day_of_year_reference_2001",
        "auxiliary_seed_formula": "SeedSequence([20220827,case_index,member_index])_PCG64",
        "boundary": {
            "established_threshold": 0.15,
            "near_one_threshold": 0.999,
            "zero_unchanged": True,
            "near_one_unchanged": True,
            "regime_preserving_clip": True,
        },
        "bootstrap": {
            "samples": 20_000,
            "date_seed": 20220815,
            "block_seed": 20220816,
            "blocks": "12_fixed_nonoverlapping_consecutive_four_date_blocks",
        },
        "gate": (
            "unchanged_calendar_global_bias_full_no_compensation_gate_scaled_only_to_exact_48_case_operational_count"
        ),
        "compact_outputs": [
            "aggregate_case_mean_metrics.json",
            "metadata.json",
            "per_case_metrics.csv",
            "run_status.json",
        ],
        "raw_download": False,
        "post_result_tuning": False,
        "technical_retry": ("resume_or_recompute_only_with_identical_code_data_checkpoint_and_seed_hashes"),
    }


def noise_seed(case_index: int, member_index: int) -> int:
    if not 0 <= case_index < CASE_COUNT or not 0 <= member_index < MEMBER_COUNT:
        raise ValueError("case or member index outside frozen schedule")
    return BASE_NOISE_SEED + case_index * MEMBER_COUNT + member_index


def _ordinary_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    return path


def _record(path: Path, root: Path) -> dict[str, object]:
    ordinary = _ordinary_file(path, "sealed input")
    try:
        relative = ordinary.relative_to(root)
    except ValueError as error:
        raise ValueError("sealed input escaped its trusted root") from error
    return {
        "path": relative.as_posix(),
        "size_bytes": ordinary.stat().st_size,
        "sha256": _file_hash(ordinary),
    }


def build_input_seal(
    repo: Path,
    dataset_root: Path,
    sral_root: Path,
    mask_path: Path,
) -> dict[str, Any]:
    """Hash every exact holdout input before any model or truth evaluation starts."""
    repo = repo.resolve()
    dataset_root = dataset_root.resolve()
    sral_root = sral_root.resolve()
    mask_path = mask_path.resolve()
    if dataset_root.is_symlink() or sral_root.is_symlink():
        raise ValueError("holdout input roots cannot be symlinks")
    preds = dataset_root / "preds"
    if preds.is_symlink() or not preds.is_dir() or not sral_root.is_dir():
        raise ValueError("holdout prediction or SRAL root is unavailable")
    forecast_dates = sorted(
        {*BACKGROUND_DATES, *TRACK_DATES}
    )  # TRACK_DATES includes truth/history and next-track context.
    forecast_records = [
        _record(preds / f"ocean+atmosphere_24_{target.isoformat()}.npy", dataset_root)
        for target in forecast_dates
    ]
    sral_records = [_record(sral_root / f"{target.isoformat()}.npy", sral_root) for target in TRACK_DATES]
    config_records = [
        _record(repo / CONFIG_PATH, repo),
        _record(repo / DATA_CONFIG_PATH, repo),
        _record(repo / MODEL_CONFIG_PATH, repo),
    ]
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "primary_contract_sha256": PRIMARY_CONTRACT_SHA256,
        "target_date_manifest_sha256": _canonical_hash([target.isoformat() for target in EXPECTED_DATES]),
        "forecast_records": forecast_records,
        "sral_records": sral_records,
        "mask_record": _record(mask_path, mask_path.parent),
        "config_records": config_records,
        "truth_values_read": False,
        "outcomes_read": False,
    }
    if payload["target_date_manifest_sha256"] != TARGET_DATE_MANIFEST_SHA256:
        raise RuntimeError("compiled target-date manifest differs from the signed contract")
    return {**payload, "seal_sha256": _canonical_hash(payload)}


def _effective_input_paths(repo: Path) -> tuple[Path, Path, Path]:
    experiment_path = repo / CONFIG_PATH
    experiment = load_json(experiment_path)
    data_path = resolve_path(experiment["data_config"], experiment_path.parent).resolve()
    data = merge_config_overrides(load_json(data_path), experiment.get("data_overrides"))
    if data.get("split_protocol") is not None:
        raise ValueError("external holdout config must not claim the old 200-day split")
    return (
        Path(str(data["dataset_dir"])),
        Path(str(data["sral_dir"])),
        Path(str(data["mask_path"])),
    )


def _validate_sampling_metadata(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = load_json(root / "metadata.json")
    expected = {
        "dataset_split": "test",
        "start_date": "2024-01-01",
        "end_date": "2024-02-17",
        "num_cases": CASE_COUNT,
        "case_stride_days": 1,
        "target_hour": 23,
        "ensemble_size": MEMBER_COUNT,
        "sample_batch_size": MEMBER_COUNT,
        "num_timesteps": 25,
        "method": "dopri5",
        "rtol": 1e-5,
        "atol": 1e-6,
        "inference_precision": "float32",
        "seed": BASE_NOISE_SEED,
        "initial_noise_scale": 1.0,
        "field_protocol": "siconc-only",
        "conditioning_mode": "full",
        "cfg_mode": "none",
        "cfg_background_scale": 1.0,
        "cfg_observation_scale": 1.0,
        "save_ensembles": True,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"raw sampling metadata differs for {key}")
    if Path(str(metadata.get("checkpoint", ""))).name != CHECKPOINT_NAME:
        raise ValueError("raw sampling used an unexpected checkpoint")
    if metadata.get("checkpoint_training_split_kind") != "legacy_overlapping_2023_validation":
        raise ValueError("checkpoint limitation provenance differs")
    if metadata.get("checkpoint_selection") != (
        "fixed final epoch; legacy 2023 validation was not used for checkpoint selection"
    ):
        raise ValueError("checkpoint selection provenance differs")
    cases_path = root / str(metadata.get("cases_file", ""))
    if cases_path.parent != root or cases_path.is_symlink() or not cases_path.is_file():
        raise ValueError("sampling cases_file is not the exact local ordinary file")
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or len(cases) != CASE_COUNT:
        raise ValueError("sampling cases file does not contain exactly 48 cases")
    return metadata, cases


def seal_samples(sampling_root: Path, sealed_root: Path) -> dict[str, Any]:
    """Validate and seal 48x10 raw members before any downstream truth access."""
    if sampling_root.is_symlink() or sealed_root.is_symlink() or sealed_root.exists():
        raise ValueError("sampling/sealed roots violate the new-directory contract")
    _, cases = _validate_sampling_metadata(sampling_root)
    sample_source = sampling_root / "samples"
    if sample_source.is_symlink() or not sample_source.is_dir():
        raise ValueError("sampling samples directory is unavailable")
    sealed_root.mkdir(parents=True)
    samples_dir = sealed_root / "samples"
    sample_source.replace(samples_dir)
    sample_paths: list[Path] = []
    case_records: list[dict[str, Any]] = []
    for case_index, (target, background) in enumerate(zip(EXPECTED_DATES, BACKGROUND_DATES, strict=True)):
        name = f"{case_index:03d}_{target.isoformat()}_h23.npz"
        path = _ordinary_file(samples_dir / name, "sealed sample")
        with np.load(path, allow_pickle=False) as payload:
            required = {"analysis_ensemble", "truth", "valid_mask"}
            if not required.issubset(payload.files):
                raise ValueError("sealed sample lacks required arrays")
            ensemble = np.asarray(payload["analysis_ensemble"])
            truth = np.asarray(payload["truth"])
            valid = np.asarray(payload["valid_mask"])
            if ensemble.ndim != 3 or ensemble.shape[0] != MEMBER_COUNT:
                raise ValueError("sealed sample does not contain exactly ten 2D members")
            if truth.shape != ensemble.shape[1:] or valid.shape != truth.shape:
                raise ValueError("sealed sample truth/mask shapes differ")
            if not np.all(np.isfinite(ensemble)):
                raise ValueError("sealed raw ensemble contains non-finite values")
        case = cases[case_index]
        if (
            case.get("case_order") != case_index
            or case.get("target_date") != target.isoformat()
            or case.get("background_date") != background.isoformat()
        ):
            raise ValueError("sampling case identity differs from the frozen schedule")
        base_hashes = case.get("initial_noise_sha256")
        scaled_hashes = case.get("scaled_initial_noise_sha256")
        if (
            not isinstance(base_hashes, list)
            or len(base_hashes) != MEMBER_COUNT
            or base_hashes != scaled_hashes
        ):
            raise ValueError("raw initial-noise accounting differs from the unit schedule")
        members = []
        for member_index, noise_hash in enumerate(base_hashes):
            if not isinstance(noise_hash, str) or len(noise_hash) != 64:
                raise ValueError("raw initial-noise hash is invalid")
            members.append(
                {
                    "member_index": member_index,
                    "noise_seed": noise_seed(case_index, member_index),
                    "initial_noise_sha256": noise_hash,
                    "checkpoint_sha256": CHECKPOINT_SHA256,
                }
            )
        sample_paths.append(path)
        case_records.append(
            {
                "case_index": case_index,
                "target_date": target.isoformat(),
                "background_date": background.isoformat(),
                "sample_name": name,
                "sample_sha256": _file_hash(path),
                "members": members,
                "member_manifest_sha256": _canonical_hash(members),
            }
        )
    if sorted(path.name for path in samples_dir.iterdir()) != [
        f"{index:03d}_{target.isoformat()}_h23.npz" for index, target in enumerate(EXPECTED_DATES)
    ]:
        raise ValueError("sealed samples directory differs from the exact schedule")
    payload = {
        "schema_version": 1,
        "stage": "raw_48x10_sealed_before_candidate_truth_access",
        "protocol_id": PROTOCOL_ID,
        "primary_contract_sha256": PRIMARY_CONTRACT_SHA256,
        "experiment_id": EXPERIMENT_ID,
        "candidate_transform_started": False,
        "candidate_truth_read": False,
        "case_count": CASE_COUNT,
        "member_count": MEMBER_COUNT,
        "checkpoint_name": CHECKPOINT_NAME,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "sample_manifest_sha256": _sample_manifest(sample_paths),
        "cases": case_records,
    }
    sealed = {**payload, "sealed_manifest_sha256": _canonical_hash(payload)}
    _atomic_json(sealed_root / SAMPLE_SEAL_NAME, sealed)
    return sealed


def run(output_dir: Path, source_experiment: str, run_dir: Path) -> None:
    if source_experiment != SOURCE_EXPERIMENT:
        raise ValueError("source_experiment differs from the frozen primary contract")
    if _canonical_hash(frozen_primary_contract()) != PRIMARY_CONTRACT_SHA256:
        raise RuntimeError("compiled primary contract hash differs from its signature")
    output_dir = output_dir.resolve()
    run_dir = run_dir.resolve()
    if output_dir.is_symlink() or run_dir.is_symlink():
        raise ValueError("input and output directories cannot be symlinks")
    output_dir.mkdir(parents=True, exist_ok=True)
    unexpected = {path.name for path in output_dir.iterdir()} - COMPACT_OUTPUTS - {"internal"}
    if unexpected:
        raise ValueError(f"output_dir contains unexpected entries: {sorted(unexpected)}")
    internal = output_dir / "internal"
    if internal.is_symlink() or internal.exists():
        raise ValueError("internal holdout path must be a new ordinary directory")
    internal.mkdir()
    repo = Path(__file__).resolve().parents[1]
    sampling_root = internal / "sampling_unsealed"
    sealed_root = internal / "sealed_holdout"
    try:
        checkpoint_record = _validate_source(run_dir)
        dataset_root, sral_root, mask_path = _effective_input_paths(repo)
        input_seal = build_input_seal(repo, dataset_root, sral_root, mask_path)
        _atomic_json(internal / INPUT_SEAL_NAME, input_seal)
        _run_checked(
            [
                sys.executable,
                "-m",
                "assim_lib.compare_3dvar",
                "--config",
                str(repo / CONFIG_PATH),
                "--run-dir",
                str(run_dir),
                "--output-dir",
                str(sampling_root),
            ],
            repo,
            output_dir,
            "sampling_external_2024_raw48",
            0,
            total_units=CASE_COUNT,
        )
        sealed = seal_samples(sampling_root, sealed_root)
        per_case = [
            {
                "case_index": row["case_index"],
                "target_date": row["target_date"],
                "ensemble_size": MEMBER_COUNT,
                "member_manifest_sha256": row["member_manifest_sha256"],
                "sample_sha256": row["sample_sha256"],
            }
            for row in sealed["cases"]
        ]
        _atomic_csv(output_dir / "per_case_metrics.csv", per_case)
        _atomic_json(
            output_dir / "aggregate_case_mean_metrics.json",
            {
                "schema_version": 1,
                "status": "raw_sampling_sealed_gate_pending",
                "protocol_id": PROTOCOL_ID,
                "primary_candidate": PRIMARY_CANDIDATE,
                "num_cases": CASE_COUNT,
                "ensemble_size": MEMBER_COUNT,
                "primary_contract_sha256": PRIMARY_CONTRACT_SHA256,
                "input_seal_sha256": input_seal["seal_sha256"],
                "sealed_manifest_sha256": sealed["sealed_manifest_sha256"],
            },
        )
        _atomic_json(
            output_dir / "metadata.json",
            {
                "status": "completed",
                "mode": MODE,
                "experiment_id": EXPERIMENT_ID,
                "dependent_mode": DEPENDENT_MODE,
                "dependent_experiment_id": DEPENDENT_EXPERIMENT_ID,
                "protocol_id": PROTOCOL_ID,
                "primary_candidate": PRIMARY_CANDIDATE,
                "primary_contract": frozen_primary_contract(),
                "primary_contract_sha256": PRIMARY_CONTRACT_SHA256,
                "source_experiment": SOURCE_EXPERIMENT,
                "date_range": ["2024-01-01", "2024-02-17"],
                "num_cases": CASE_COUNT,
                "ensemble_size": MEMBER_COUNT,
                "checkpoint_record": checkpoint_record,
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "input_seal_file": f"internal/{INPUT_SEAL_NAME}",
                "input_seal_sha256": input_seal["seal_sha256"],
                "sealed_manifest_file": f"internal/sealed_holdout/{SAMPLE_SEAL_NAME}",
                "sealed_manifest_sha256": sealed["sealed_manifest_sha256"],
                "sample_manifest_sha256": sealed["sample_manifest_sha256"],
                "truth_access_barrier": (
                    "dependent transform must verify both seals and all sample hashes before np.load"
                ),
                "candidate_transform_started": False,
                "candidate_truth_read": False,
                "fallback_member_count": 0,
                "raw_arrays_retrieved": False,
                "outcome_values_exposed": False,
                "gate_pending": True,
                "compact_outputs": sorted(COMPACT_OUTPUTS),
                "runner_sha256": _file_hash(Path(__file__)),
                "config_sha256": _file_hash(repo / CONFIG_PATH),
            },
        )
        _atomic_json(
            output_dir / "run_status.json",
            {
                "status": "completed",
                "experiment_id": EXPERIMENT_ID,
                "completed_cases": CASE_COUNT,
                "total_cases": CASE_COUNT,
                "progress_percent": 100.0,
                "raw_sampling_sealed": True,
                "gate_pending": True,
                "dependent_experiment_id": DEPENDENT_EXPERIMENT_ID,
            },
        )
    except Exception as error:
        for name in COMPACT_OUTPUTS - {"run_status.json"}:
            (output_dir / name).unlink(missing_ok=True)
        _atomic_json(
            output_dir / "run_status.json",
            {
                "status": "failed",
                "experiment_id": EXPERIMENT_ID,
                "error_type": type(error).__name__,
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-experiment", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.output_dir, args.source_experiment, args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
