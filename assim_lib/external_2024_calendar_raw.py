"""Sealed raw sampling for the frozen 48-day 2024 calendar confirmation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
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
    _forward_signal,
    _run_checked,
    _sample_manifest,
)
from .forecast import forecast_records, parse_date
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
SUPERSEDED_ORIGINAL_PRIMARY_CONTRACT_SHA256 = (
    "5d77ac9d40dfd8ccba0ba75729a5f8a7830b72bbe63368ea4eaab8a8e17abaf3"
)
CONFIRMATORY_GATE_CONTRACT_SHA256 = "57e8dd1859c4ac9a144be68904450926a6098b08a3b58a35e3d7dcc6a5bb9185"
PRIMARY_CONTRACT_SHA256 = "f2225da7a05cab53b14604e45bed840a0ec559aed20856ae8ef2dd72d915b8f8"
CONFIG_PATH = "config/experiments/external_2024_calendar_global_bias_raw48_primary_v1.json"
MODEL_CONFIG_PATH = "config/methods/concat_conditioning_diffusion_balanced_2f.json"
DATA_CONFIG_PATH = "config/data/m2m_2f_1y.json"
INPUT_SEAL_NAME = "input_seal.json"
SAMPLE_SEAL_NAME = "sealed_manifest.json"
TRUTH_SEAL_NAME = "truth_manifest.json"


def confirmatory_gate_contract() -> dict[str, Any]:
    return {
        "protocol": "external_2024_calendar_global_bias_confirm48_primary_v1",
        "case_weighting": "equal_weight_per_date",
        "methods": ["raw", PRIMARY_CANDIDATE],
        "proper_scores": {
            "fair_crps_candidate_le_raw_times": 0.97,
            "ordinary_crps_candidate_le_raw_times": 1.01,
            "fair_date_and_block_95ci_upper_below": 0.0,
            "energy_score_candidate_le_raw_times": 1.02,
        },
        "rank_reliability": {
            "randomized_rank_tie_rng": "default_rng(20220815+case_index)",
            "histogram_bins": 11,
            "total_variation_to_uniform_max": 0.10,
            "max_bin_deviation_from_uniform_max": 0.03,
            "normalized_mean_rank_interval": [0.45, 0.55],
            "normalized_mean_rank_block_95ci_must_contain": 0.5,
            "member_range_error_vs_9_over_11_max": 0.05,
            "inner_order_error_vs_7_over_11_max": 0.05,
        },
        "boundary": {
            "thresholds": [0.0, 0.15, 0.90, 0.95, 0.99],
            "event_definition": "member>q and truth>q for every q",
            "attainable_probabilities": 11,
            "metrics": ["brier", "reliability_l1", "marginal_frequency_error"],
            "each_brier_candidate_le_raw_times": 1.02,
            "each_brier_block_noninferiority": (
                "for d_i=candidate_i-1.02*raw_i, shared 12-block bootstrap 95% upper<=0"
            ),
            "mean_high_sic_brier_thresholds": [0.90, 0.95, 0.99],
            "mean_high_sic_brier_candidate_le_raw": True,
            "mean_reliability_l1_candidate_le_raw": True,
            "mean_marginal_frequency_error_candidate_le_raw": True,
            "exact_zero_exact_one_and_ge_0999_are_encoding_diagnostics_only": True,
            "structural_mask_equality_is_provenance_only": True,
        },
        "spatial": {
            "lags": [1, 2, 4],
            "member_mean_semivariogram_relative_error_to_truth_max": 0.20,
            "semivariogram_tolerance_source": (
                "frozen from 2022 raw errors 0.146 through 0.184 before holdout access"
            ),
            "local_variogram_score_candidate_le_raw_times": 1.02,
            "local_variogram_block_noninferiority": (
                "for each lag d_i=candidate_i-1.02*raw_i, shared 12-block bootstrap 95% upper<=0"
            ),
            "analysis_mean_rmse_and_iiee_candidate_le_raw_times": 1.02,
            "edge_error_candidate_le_raw_times": 1.02,
            "area_extent_and_analysis_mean_variogram_use_existing_absolute_or_2pct_tolerance": True,
            "worst_member_gate": False,
        },
        "bootstrap": {
            "samples": 20_000,
            "date_seed": 20220815,
            "block_seed": 20220816,
            "shared_across_metrics": True,
            "blocks": "12_fixed_noncircular_nonoverlapping_consecutive_four_date_blocks",
        },
        "operational": {
            "cases": 48,
            "members": 10,
            "finite": True,
            "sealed_before_truth_scoring": True,
            "gate_generates_and_seals_all_candidate_draws_and_arrays_before_opening_any_truth_npz": (True),
            "no_compensation_across_families": True,
        },
    }


def frozen_primary_contract() -> dict[str, Any]:
    """Return the pre-registered primary contract whose hash was frozen before outcomes."""
    return {
        "protocol_id": PROTOCOL_ID,
        "supersedes": SUPERSEDED_ORIGINAL_PRIMARY_CONTRACT_SHA256,
        "amendment_reason": (
            "domain correction before any 2024 outcome access; truth-support audit used 2022 only"
        ),
        "amendment_timestamp": "2026-08-16T22:47:56+03:00",
        "amendment_based_on_publication_commit": ("e8a5e950c56b847dd48ec8657cb9db46cecf38e9"),
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
        "confirmatory_gate_contract_sha256": CONFIRMATORY_GATE_CONTRACT_SHA256,
        "gate": confirmatory_gate_contract(),
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


def _resolved_without_symlinks(path: Path, label: str) -> Path:
    lexical = path.expanduser().absolute()
    # Reject the configured lexical target before resolve(); system prefixes
    # such as macOS /var -> /private/var are outside the configurable root.
    if lexical.is_symlink():
        raise ValueError(f"{label} is a symlink")
    return lexical.resolve(strict=True)


def build_input_seal(
    repo: Path,
    dataset_root: Path,
    sral_root: Path,
    mask_path: Path,
) -> dict[str, Any]:
    """Hash every exact holdout input before any model or truth evaluation starts."""
    repo = _resolved_without_symlinks(repo, "publication repository")
    dataset_root = _resolved_without_symlinks(dataset_root, "dataset root")
    sral_root = _resolved_without_symlinks(sral_root, "SRAL root")
    mask_path = _resolved_without_symlinks(mask_path, "valid mask")
    preds = dataset_root / "preds"
    if preds.is_symlink() or not preds.is_dir() or not sral_root.is_dir():
        raise ValueError("holdout prediction or SRAL root is unavailable")
    forecast_dates = sorted(
        {*BACKGROUND_DATES, *TRACK_DATES}
    )  # TRACK_DATES includes truth/history and next-track context.
    indexed_forecasts: dict[date, list[Path]] = {target: [] for target in forecast_dates}
    for record in forecast_records(preds, 24):
        if record.date in indexed_forecasts:
            indexed_forecasts[record.date].append(record.path)
    if any(len(paths) != 1 for paths in indexed_forecasts.values()):
        raise ValueError("holdout forecast glob inventory is not exactly one file per date")
    forecast_records_sealed = [
        _record(indexed_forecasts[target][0], dataset_root) for target in forecast_dates
    ]
    indexed_sral: dict[date, list[Path]] = {target: [] for target in TRACK_DATES}
    for path in sorted(sral_root.glob("*.npy")):
        parsed = parse_date(path.name)
        if parsed in indexed_sral:
            indexed_sral[parsed].append(path)
    if any(not paths for paths in indexed_sral.values()):
        raise ValueError("holdout SRAL glob inventory lacks a required date")
    sral_records = [_record(path, sral_root) for target in TRACK_DATES for path in indexed_sral[target]]
    config_records = [
        _record(repo / CONFIG_PATH, repo),
        _record(repo / DATA_CONFIG_PATH, repo),
        _record(repo / MODEL_CONFIG_PATH, repo),
    ]
    source_records = [
        _record(repo / relative, repo)
        for relative in (
            "assim_lib/external_2024_calendar_raw.py",
            "assim_lib/compare_3dvar.py",
            "assim_lib/data.py",
            "assim_lib/evaluate.py",
        )
    ]
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "primary_contract_sha256": PRIMARY_CONTRACT_SHA256,
        "target_date_manifest_sha256": _canonical_hash([target.isoformat() for target in EXPECTED_DATES]),
        "forecast_records": forecast_records_sealed,
        "sral_records": sral_records,
        "mask_record": _record(mask_path, mask_path.parent),
        "config_records": config_records,
        "source_records": source_records,
        "dense_target_may_be_loaded_only_to_construct_masked_model_observations": True,
        "holdout_outcomes_used_for_metrics_or_method_selection": False,
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


def _validate_runtime() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible or visible.strip() != visible:
        raise RuntimeError("exactly one explicit CUDA_VISIBLE_DEVICES token is required")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("offline model/data mode is required")
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("trusted holdout sampling requires exactly one visible CUDA GPU")


def _input_seal_unchanged(
    before: dict[str, Any], repo: Path, dataset_root: Path, sral_root: Path, mask_path: Path
) -> None:
    after = build_input_seal(repo, dataset_root, sral_root, mask_path)
    if after != before:
        raise RuntimeError("sealed holdout inputs changed during raw sampling")


def _sampling_stage_complete(sampling_root: Path) -> bool:
    if sampling_root.is_symlink() or not sampling_root.is_dir():
        return False
    for name in ("run_status.json", "metadata.json", "cases.json"):
        path = sampling_root / name
        if path.is_symlink() or not path.is_file():
            return False
    status = load_json(sampling_root / "run_status.json")
    samples = sampling_root / "samples"
    expected = [
        f"{case_index:03d}_{target.isoformat()}_h23.npz" for case_index, target in enumerate(EXPECTED_DATES)
    ]
    return (
        status.get("status") == "completed"
        and status.get("completed_cases") == CASE_COUNT
        and samples.is_dir()
        and not samples.is_symlink()
        and sorted(path.name for path in samples.iterdir()) == expected
        and all((samples / name).is_file() and not (samples / name).is_symlink() for name in expected)
    )


def _checkpoint_source_unchanged(run_dir: Path, expected: dict[str, Any]) -> None:
    if _validate_source(run_dir) != expected:
        raise RuntimeError("checkpoint source changed during holdout processing")


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
        "sealed_raw_only": True,
        "outcome_metrics_computed": False,
        "truth_saved_with_raw_samples": False,
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
    if metadata.get("temporal_split_protocol") != "external_chronological_holdout":
        raise ValueError("raw sampling reports stale temporal-split metadata")
    cases_path = root / str(metadata.get("cases_file", ""))
    if cases_path.parent != root or cases_path.is_symlink() or not cases_path.is_file():
        raise ValueError("sampling cases_file is not the exact local ordinary file")
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or len(cases) != CASE_COUNT:
        raise ValueError("sampling cases file does not contain exactly 48 cases")
    return metadata, cases


def seal_samples(
    sampling_root: Path, sealed_root: Path, input_seal: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Validate and seal 48x10 raw members before any downstream truth access."""
    if sampling_root.is_symlink() or sealed_root.is_symlink() or sealed_root.exists():
        raise ValueError("sampling/sealed roots violate the new-directory contract")
    sampling_metadata, cases = _validate_sampling_metadata(sampling_root)
    normalization_means = np.asarray(sampling_metadata.get("normalization_means"), dtype=np.float64)
    normalization_stds = np.asarray(sampling_metadata.get("normalization_stds"), dtype=np.float64)
    if (
        normalization_means.ndim != 1
        or normalization_stds.shape != normalization_means.shape
        or normalization_means.size < 1
        or not np.all(np.isfinite(normalization_means))
        or not np.all(np.isfinite(normalization_stds))
        or np.any(normalization_stds <= 0.0)
    ):
        raise ValueError("sampling normalization arrays are invalid")
    sealed_sral_by_date: dict[date, set[str]] | None = None
    if input_seal is not None:
        sealed_sral_by_date = {target: set() for target in TRACK_DATES}
        for record in input_seal.get("sral_records", []):
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                raise ValueError("input seal contains an invalid SRAL record")
            name = Path(record["path"]).name
            parsed = parse_date(name)
            if parsed not in sealed_sral_by_date:
                raise ValueError("input seal SRAL record is outside the frozen schedule")
            sealed_sral_by_date[parsed].add(name)
    sample_source = sampling_root / "samples"
    if sample_source.is_symlink() or not sample_source.is_dir():
        raise ValueError("sampling samples directory is unavailable")
    sealed_root.mkdir(parents=True)
    samples_dir = sealed_root / "samples"
    sample_source.replace(samples_dir)
    sample_paths: list[Path] = []
    case_records: list[dict[str, Any]] = []
    seen_noise_hashes: set[str] = set()
    for case_index, (target, background) in enumerate(zip(EXPECTED_DATES, BACKGROUND_DATES, strict=True)):
        name = f"{case_index:03d}_{target.isoformat()}_h23.npz"
        path = _ordinary_file(samples_dir / name, "sealed sample")
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != {"analysis_ensemble", "valid_mask"}:
                raise ValueError("sealed raw sample contains truth or an unexpected array")
            ensemble = np.asarray(payload["analysis_ensemble"])
            valid = np.asarray(payload["valid_mask"])
            if ensemble.ndim != 3 or ensemble.shape[0] != MEMBER_COUNT:
                raise ValueError("sealed sample does not contain exactly ten 2D members")
            if valid.shape != ensemble.shape[1:]:
                raise ValueError("sealed sample ensemble/mask shapes differ")
            if not np.all(np.isfinite(ensemble)):
                raise ValueError("sealed raw ensemble contains non-finite values")
        case = cases[case_index]
        if (
            case.get("case_order") != case_index
            or case.get("target_date") != target.isoformat()
            or case.get("background_date") != background.isoformat()
        ):
            raise ValueError("sampling case identity differs from the frozen schedule")
        expected_track_days = [target - timedelta(days=offset) for offset in range(3)]
        track_days = case.get("conditioning_track_days")
        if not isinstance(track_days, list) or len(track_days) != 3:
            raise ValueError("sampling case lacks the exact three conditioning track days")
        for offset, (record, expected_date) in enumerate(zip(track_days, expected_track_days)):
            if (
                not isinstance(record, dict)
                or record.get("offset") != offset
                or record.get("date") != expected_date.isoformat()
                or type(record.get("num_sral_files")) is not int
                or record["num_sral_files"] <= 0
                or not isinstance(record.get("sral_files"), list)
                or len(record["sral_files"]) != record["num_sral_files"]
                or any(parse_date(Path(str(path)).name) != expected_date for path in record["sral_files"])
            ):
                raise ValueError("sampling conditioning-track provenance differs")
        if sealed_sral_by_date is not None:
            for record, expected_date in zip(track_days, expected_track_days, strict=True):
                observed = {Path(str(path)).name for path in record["sral_files"]}
                if observed != sealed_sral_by_date[expected_date]:
                    raise ValueError("sampling conditioning SRAL set differs from the input seal")
            imitation = case.get("track_imitation_sral_files")
            if (
                not isinstance(imitation, list)
                or {Path(str(path)).name for path in imitation}
                != sealed_sral_by_date[target + timedelta(days=1)]
            ):
                raise ValueError("sampling imitation SRAL set differs from the input seal")
        if (
            case.get("track_imitation_date") != (target + timedelta(days=1)).isoformat()
            or case.get("mask_kind") != "sral_tracks"
            or case.get("sral_files_used") != sum(row["num_sral_files"] for row in track_days)
            or case.get("empty_obs_days") != 0
            or type(case.get("conditioning_obs_count")) is not int
            or case["conditioning_obs_count"] <= 0
            or type(case.get("track_imitation_count")) is not int
            or case["track_imitation_count"] <= 0
        ):
            raise ValueError("sampling SRAL/mask provenance differs")
        for key in ("conditioning_mask_sha256", "track_imitation_mask_sha256"):
            value = str(case.get(key, ""))
            try:
                decoded = bytes.fromhex(value)
            except ValueError as error:
                raise ValueError(f"sampling {key} is not hexadecimal") from error
            if len(decoded) != 32:
                raise ValueError(f"sampling {key} is not a SHA-256 digest")
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
            try:
                bytes.fromhex(noise_hash)
            except ValueError as error:
                raise ValueError("raw initial-noise hash is not hexadecimal") from error
            if noise_hash in seen_noise_hashes:
                raise ValueError("raw initial-noise hashes are not unique across 480 members")
            seen_noise_hashes.add(noise_hash)
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
    if len(seen_noise_hashes) != CASE_COUNT * MEMBER_COUNT:
        raise ValueError("raw initial-noise inventory does not contain 480 unique hashes")
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
        "sampling_metadata_sha256": _file_hash(sampling_root / "metadata.json"),
        "normalization_means": normalization_means.tolist(),
        "normalization_stds": normalization_stds.tolist(),
        "sample_manifest_sha256": _sample_manifest(sample_paths),
        "cases": case_records,
    }
    sealed = {**payload, "sealed_manifest_sha256": _canonical_hash(payload)}
    _atomic_json(sealed_root / SAMPLE_SEAL_NAME, sealed)
    return sealed


def validate_existing_sample_seal(sealed_root: Path) -> dict[str, Any]:
    """Revalidate a completed raw seal so wrapper retries do not rerun the GPU."""
    path = sealed_root / SAMPLE_SEAL_NAME
    if sealed_root.is_symlink() or path.is_symlink() or not path.is_file():
        raise ValueError("existing raw seal is missing or unsafe")
    sealed = load_json(path)
    signed = dict(sealed)
    claimed = signed.pop("sealed_manifest_sha256", None)
    if claimed != _canonical_hash(signed):
        raise ValueError("existing raw seal canonical hash differs")
    expected = {
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
    }
    for key, value in expected.items():
        if sealed.get(key) != value:
            raise ValueError(f"existing raw seal differs for {key}")
    if (
        not isinstance(sealed.get("sampling_metadata_sha256"), str)
        or len(sealed["sampling_metadata_sha256"]) != 64
        or not isinstance(sealed.get("normalization_means"), list)
        or not isinstance(sealed.get("normalization_stds"), list)
        or len(sealed["normalization_means"]) != len(sealed["normalization_stds"])
    ):
        raise ValueError("existing raw seal normalization provenance differs")
    cases = sealed.get("cases")
    samples = sealed_root / "samples"
    if not isinstance(cases, list) or len(cases) != CASE_COUNT or samples.is_symlink():
        raise ValueError("existing raw seal inventory differs")
    paths = []
    for case_index, (target, record) in enumerate(zip(EXPECTED_DATES, cases, strict=True)):
        expected_name = f"{case_index:03d}_{target.isoformat()}_h23.npz"
        sample = samples / expected_name
        members = record.get("members") if isinstance(record, dict) else None
        if (
            not isinstance(record, dict)
            or record.get("case_index") != case_index
            or record.get("target_date") != target.isoformat()
            or record.get("background_date") != BACKGROUND_DATES[case_index].isoformat()
            or record.get("sample_name") != expected_name
            or not isinstance(members, list)
            or len(members) != MEMBER_COUNT
            or _canonical_hash(members) != record.get("member_manifest_sha256")
            or sample.is_symlink()
            or not sample.is_file()
            or _file_hash(sample) != record.get("sample_sha256")
        ):
            raise ValueError("existing sealed raw sample differs")
        with np.load(sample, allow_pickle=False) as payload:
            if set(payload.files) != {"analysis_ensemble", "valid_mask"}:
                raise ValueError("existing sealed raw sample arrays differ")
            ensemble = np.asarray(payload["analysis_ensemble"])
            valid = np.asarray(payload["valid_mask"])
        if (
            ensemble.ndim != 3
            or ensemble.shape[0] != MEMBER_COUNT
            or valid.shape != ensemble.shape[1:]
            or valid.dtype != np.bool_
            or not np.all(np.isfinite(ensemble))
            or np.any((ensemble < 0.0) | (ensemble > 1.0))
        ):
            raise ValueError("existing sealed raw sample values differ")
        paths.append(sample)
    if _sample_manifest(paths) != sealed.get("sample_manifest_sha256"):
        raise ValueError("existing raw aggregate manifest differs")
    return sealed


def write_truth_bundle(
    repo: Path, sampling_root: Path, sealed_root: Path, truth_root: Path
) -> dict[str, Any]:
    """Materialize scoring truth only after the immutable raw 48x10 seal exists."""
    if not (sealed_root / SAMPLE_SEAL_NAME).is_file():
        raise ValueError("raw samples must be sealed before truth extraction")
    if truth_root.is_symlink() or truth_root.exists():
        raise ValueError("truth bundle must be a new ordinary directory")
    from .compare_3dvar import _case_indices
    from .data import build_dataset
    from .evaluate import denormalize_and_clip

    experiment_path = repo / CONFIG_PATH
    experiment = load_json(experiment_path)
    data_path = resolve_path(experiment["data_config"], experiment_path.parent).resolve()
    data_config = merge_config_overrides(load_json(data_path), experiment.get("data_overrides"))
    dataset = build_dataset(data_config, split="test")
    indices = _case_indices(dataset, EXPECTED_DATES[0], EXPECTED_DATES[-1], 23, 1)
    if len(indices) != CASE_COUNT:
        raise ValueError("truth extraction schedule differs from the signed 48 dates")
    raw_seal = load_json(sealed_root / SAMPLE_SEAL_NAME)
    if _file_hash(sampling_root / "metadata.json") != raw_seal.get("sampling_metadata_sha256"):
        raise ValueError("sampling metadata changed after the raw seal")
    means = raw_seal.get("normalization_means")
    stds = raw_seal.get("normalization_stds")
    fields = list(data_config["fields"])
    channel = fields.index("siconc")
    truth_root.mkdir()
    truth_paths: list[Path] = []
    records: list[dict[str, Any]] = []
    for case_index, (dataset_index, target) in enumerate(zip(indices, EXPECTED_DATES, strict=True)):
        item = dataset[dataset_index]
        if item["meta"].get("target_date") != target.isoformat():
            raise ValueError("truth extraction case identity differs")
        truth = denormalize_and_clip(item["truth"].unsqueeze(0), means, stds, channel)[0, channel]
        valid = item["valid_mask"][channel].detach().cpu().numpy() > 0.5
        raw_path = sealed_root / "samples" / f"{case_index:03d}_{target.isoformat()}_h23.npz"
        with np.load(raw_path, allow_pickle=False) as raw:
            raw_valid = np.asarray(raw["valid_mask"], dtype=bool)
        if truth.shape != valid.shape or not np.array_equal(valid, raw_valid):
            raise ValueError("truth valid mask differs from the sealed raw sample")
        selected = valid & np.isfinite(truth)
        if not np.any(selected) or np.any((truth[selected] < 0.0) | (truth[selected] > 1.0)):
            raise ValueError("truth bundle contains invalid SIC")
        path = truth_root / f"{case_index:03d}_{target.isoformat()}_h23_truth.npz"
        np.savez_compressed(path, truth=np.asarray(truth, dtype=np.float32))
        truth_paths.append(path)
        records.append(
            {
                "case_index": case_index,
                "target_date": target.isoformat(),
                "truth_name": path.name,
                "truth_sha256": _file_hash(path),
            }
        )
    payload = {
        "schema_version": 1,
        "stage": "truth_materialized_only_after_raw_48x10_seal",
        "protocol_id": PROTOCOL_ID,
        "primary_contract_sha256": PRIMARY_CONTRACT_SHA256,
        "raw_sealed_manifest_sha256": load_json(sealed_root / SAMPLE_SEAL_NAME)["sealed_manifest_sha256"],
        "case_count": CASE_COUNT,
        "truth_manifest_sha256": _sample_manifest(truth_paths),
        "cases": records,
    }
    sealed = {**payload, "truth_seal_sha256": _canonical_hash(payload)}
    _atomic_json(truth_root / TRUTH_SEAL_NAME, sealed)
    return sealed


def validate_existing_truth_seal(truth_root: Path, raw_seal: dict[str, Any]) -> dict[str, Any]:
    path = truth_root / TRUTH_SEAL_NAME
    if truth_root.is_symlink() or path.is_symlink() or not path.is_file():
        raise ValueError("existing truth seal is missing or unsafe")
    sealed = load_json(path)
    signed = dict(sealed)
    claimed = signed.pop("truth_seal_sha256", None)
    if claimed != _canonical_hash(signed):
        raise ValueError("existing truth seal canonical hash differs")
    if (
        sealed.get("stage") != "truth_materialized_only_after_raw_48x10_seal"
        or sealed.get("protocol_id") != PROTOCOL_ID
        or sealed.get("primary_contract_sha256") != PRIMARY_CONTRACT_SHA256
        or sealed.get("raw_sealed_manifest_sha256") != raw_seal.get("sealed_manifest_sha256")
        or sealed.get("case_count") != CASE_COUNT
    ):
        raise ValueError("existing truth seal provenance differs")
    records = sealed.get("cases")
    if not isinstance(records, list) or len(records) != CASE_COUNT:
        raise ValueError("existing truth seal inventory differs")
    paths = []
    for case_index, (target, record) in enumerate(zip(EXPECTED_DATES, records, strict=True)):
        expected_name = f"{case_index:03d}_{target.isoformat()}_h23_truth.npz"
        truth = truth_root / expected_name
        if (
            not isinstance(record, dict)
            or record.get("case_index") != case_index
            or record.get("target_date") != target.isoformat()
            or record.get("truth_name") != expected_name
            or truth.is_symlink()
            or not truth.is_file()
            or _file_hash(truth) != record.get("truth_sha256")
        ):
            raise ValueError("existing sealed truth file differs")
        paths.append(truth)
    if _sample_manifest(paths) != sealed.get("truth_manifest_sha256"):
        raise ValueError("existing truth aggregate manifest differs")
    return sealed


def run(output_dir: Path, source_experiment: str, run_dir: Path) -> None:
    if source_experiment != SOURCE_EXPERIMENT:
        raise ValueError("source_experiment differs from the frozen primary contract")
    if _canonical_hash(frozen_primary_contract()) != PRIMARY_CONTRACT_SHA256:
        raise RuntimeError("compiled primary contract hash differs from its signature")
    lexical_output = output_dir.expanduser().absolute()
    if lexical_output.is_symlink():
        raise ValueError("output directory cannot be a symlink")
    output_parent = _resolved_without_symlinks(lexical_output.parent, "output parent")
    output_dir = output_parent / lexical_output.name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = _resolved_without_symlinks(output_dir, "output directory")
    run_dir = _resolved_without_symlinks(run_dir, "checkpoint run directory")
    unexpected = {path.name for path in output_dir.iterdir()} - COMPACT_OUTPUTS - {"internal"}
    if unexpected:
        raise ValueError(f"output_dir contains unexpected entries: {sorted(unexpected)}")
    internal = output_dir / "internal"
    if internal.is_symlink():
        raise ValueError("internal holdout path cannot be a symlink")
    internal.mkdir(exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    sampling_root = internal / "sampling_unsealed"
    sealed_root = internal / "sealed_holdout"
    truth_root = internal / "truth_holdout"
    try:
        _validate_runtime()
        checkpoint_record = _validate_source(run_dir)
        dataset_root, sral_root, mask_path = _effective_input_paths(repo)
        input_seal = build_input_seal(repo, dataset_root, sral_root, mask_path)
        input_seal_path = internal / INPUT_SEAL_NAME
        if input_seal_path.exists():
            if input_seal_path.is_symlink() or load_json(input_seal_path) != input_seal:
                raise ValueError("existing input seal differs from the current exact inventory")
        else:
            _atomic_json(input_seal_path, input_seal)
        if (sealed_root / SAMPLE_SEAL_NAME).is_file():
            sealed = validate_existing_sample_seal(sealed_root)
        else:
            if sealed_root.exists():
                if sealed_root.is_symlink():
                    raise ValueError("partial sealed stage is a symlink")
                shutil.rmtree(sealed_root)
            sampling_complete = _sampling_stage_complete(sampling_root)
            if not sampling_complete:
                if sampling_root.exists():
                    if sampling_root.is_symlink():
                        raise ValueError("partial sampling stage is a symlink")
                    shutil.rmtree(sampling_root)
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
                        "--sealed-raw-only",
                    ],
                    repo,
                    output_dir,
                    "sampling_external_2024_raw48",
                    0,
                    total_units=CASE_COUNT,
                )
            sealed = seal_samples(sampling_root, sealed_root, input_seal)
        if (
            (sampling_root / "metadata.json").is_symlink()
            or not (sampling_root / "metadata.json").is_file()
            or _file_hash(sampling_root / "metadata.json") != sealed.get("sampling_metadata_sha256")
        ):
            raise ValueError("sampling metadata is missing or differs from the raw seal")
        _input_seal_unchanged(input_seal, repo, dataset_root, sral_root, mask_path)
        _checkpoint_source_unchanged(run_dir, checkpoint_record)
        if (truth_root / TRUTH_SEAL_NAME).is_file():
            truth_seal = validate_existing_truth_seal(truth_root, sealed)
        else:
            if truth_root.exists():
                if truth_root.is_symlink():
                    raise ValueError("partial truth stage is a symlink")
                shutil.rmtree(truth_root)
            truth_seal = write_truth_bundle(repo, sampling_root, sealed_root, truth_root)
        _input_seal_unchanged(input_seal, repo, dataset_root, sral_root, mask_path)
        _checkpoint_source_unchanged(run_dir, checkpoint_record)
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
                "truth_seal_sha256": truth_seal["truth_seal_sha256"],
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
                "truth_manifest_file": f"internal/truth_holdout/{TRUTH_SEAL_NAME}",
                "truth_seal_sha256": truth_seal["truth_seal_sha256"],
                "truth_manifest_sha256": truth_seal["truth_manifest_sha256"],
                "truth_access_barrier": (
                    "raw samples were sealed before truth extraction; dependent transform must "
                    "verify input/raw/truth seals before any np.load"
                ),
                "candidate_transform_started": False,
                "candidate_truth_read": False,
                "fallback_member_count": 0,
                "raw_arrays_retrieved": False,
                "outcome_values_exposed": False,
                "gate_pending": True,
                "library_downloads_disabled_by_offline_environment": True,
                "network_namespace_isolation_required_from_controller": True,
                "visible_cuda_devices": 1,
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
    signal.signal(signal.SIGTERM, _forward_signal)
    signal.signal(signal.SIGINT, _forward_signal)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-experiment", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.output_dir, args.source_experiment, args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
