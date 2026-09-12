"""Frozen paired scoring gate for the support-consistent SIT decoder."""

from __future__ import annotations

import argparse
import json
import os
import signal
from pathlib import Path
from typing import Any, Callable

import torch

from .clearml_tracking import ClearMLTracker
from .config import load_json
from .direct_dynamics_cascade_coarse_fine_boundary_audit import (
    _finish_success,
    _load_frozen,
    _load_normalization,
    _record_failure,
    _require_equal,
    _sha256,
    _strict_atomic_json,
    _terminate,
    _validate_payload,
)
from .direct_dynamics_cascade_fine_training import _clean_code_identity
from .direct_dynamics_cascade_memberwise_affine import (
    _require_finite_tree,
    compare,
    diagnostics,
)
from .direct_dynamics_cascade_paired_evaluation import _save_contact_sheet
from .direct_dynamics_sit_left_censor_audit import censor_sit, decode_with_exact_sit_zero
from .direct_dynamics_sit_support_decoder import (
    project_masked_blocks_to_nonnegative_mean,
)


SIT_CHANNELS = (1, 3, 5)
OUTPUTS = ("d3_sic", "d3_sit", "d6_sic", "d6_sit", "d9_sic", "d9_sit")


def apply_support_decoder_physical(
    physical: torch.Tensor,
    coarse_physical: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Apply the reviewed decoder to SIT and leave all SIC channels bitwise fixed."""
    if physical.ndim != 5 or physical.shape[2] != 6:
        raise ValueError("physical ensemble must be [case,member,6,y,x]")
    expected_coarse = (
        physical.shape[0],
        physical.shape[1],
        6,
        physical.shape[-2] // 2,
        physical.shape[-1] // 2,
    )
    if coarse_physical.shape != expected_coarse:
        raise ValueError("coarse ensemble shape is incompatible with physical ensemble")
    if mask.shape != (physical.shape[0], 1, *physical.shape[-2:]):
        raise ValueError("mask shape is incompatible with physical ensemble")
    if not torch.isfinite(physical).all() or not torch.isfinite(coarse_physical).all():
        raise FloatingPointError("decoder scoring inputs contain NaN/Inf")

    decoded = physical.clone()
    cases, members = physical.shape[:2]
    expanded_mask = mask[:, None].expand(-1, members, -1, -1, -1).flatten(0, 1)
    for channel in SIT_CHANNELS:
        projected = project_masked_blocks_to_nonnegative_mean(
            physical[:, :, channel : channel + 1].flatten(0, 1),
            coarse_physical[:, :, channel : channel + 1].flatten(0, 1),
            expanded_mask,
        )
        decoded[:, :, channel : channel + 1] = projected.unflatten(
            0, (cases, members)
        )
    if not torch.equal(decoded[:, :, 0::2], physical[:, :, 0::2]):
        raise RuntimeError("support decoder changed SIC")
    return decoded


def _source_is_complete(status_spec: dict[str, str], metrics_spec: dict[str, str]) -> None:
    status_path = Path(status_spec["path"])
    metrics_path = Path(metrics_spec["path"])
    if not status_path.is_file() or _sha256(status_path) != status_spec["sha256"]:
        raise ValueError("source status SHA mismatch")
    if not metrics_path.is_file() or _sha256(metrics_path) != metrics_spec["sha256"]:
        raise ValueError("source metrics SHA mismatch")
    status = load_json(status_path)
    if status.get("status") != "complete" or status.get("metrics_sha256") != metrics_spec["sha256"]:
        raise ValueError("source audit is not terminal complete")


def _score(
    ensemble: torch.Tensor,
    truth: torch.Tensor,
    mask: torch.Tensor,
    channel_stds: torch.Tensor,
) -> dict[str, Any]:
    scored = diagnostics(ensemble, truth, mask, channel_stds)
    scored["d6_sit_fractional_rank"] = {
        key: scored["outputs"]["d6_sit"][key]
        for key in (
            "case_equal_fractional_rank_frequencies",
            "rank_tv_to_uniform",
            "normalized_mean_rank",
        )
    }
    scored["d6_sit_exact_zero"] = scored["exact_atoms"]["d6_sit"]["zero"]
    return scored


def _stage_gate(
    candidate: dict[str, Any],
    champion: dict[str, Any],
    versus_champion: dict[str, Any],
    spec: dict[str, Any],
) -> dict[str, Any]:
    """Apply the frozen proper64 noninferiority contract to this SIT-only stage."""
    failures: list[str] = []
    paired = versus_champion["paired_primary_delta"]
    if paired["estimate"] >= 0 or paired["paired_date_bootstrap_95_ci_high"] >= 0:
        failures.append("primary standardized fair CRPS paired-date CI does not exclude zero")
    tolerance = 1.0 + float(spec["per_output_score_tolerance"])
    if versus_champion["joint_energy_ratio"] > tolerance:
        failures.append("joint energy worsened beyond tolerance")
    for name, row in versus_champion["outputs"].items():
        if row["fair_crps_ratio"] > tolerance:
            failures.append(f"{name} fair CRPS worsened beyond tolerance")
        if row["rmse_ratio"] > tolerance:
            failures.append(f"{name} RMSE worsened beyond tolerance")
        if row["rank_tv_difference"] > float(spec["per_output_rank_tv_tolerance"]):
            failures.append(f"{name} rank TV worsened beyond tolerance")
        if row["roughness_ratio"] > 1.0 + float(spec["roughness_tolerance"]):
            failures.append(f"{name} roughness worsened beyond tolerance")

    temporal_tolerance = 1.0 + float(spec["temporal_tolerance"])
    for key, value in candidate["temporal_increments"].items():
        if value > temporal_tolerance * champion["temporal_increments"][key] + 1e-12:
            failures.append(f"temporal diagnostic worsened beyond tolerance: {key}")

    # Boundary-event diagnostics were part of the frozen scientific checks even
    # though their tolerance is inherited from per-output noninferiority.
    for name, events in candidate["boundary_events"].items():
        for event, row in events.items():
            if event == "open_water_haze":
                continue
            reference = champion["boundary_events"][name][event]
            for metric in ("brier", "reliability_ece"):
                if row[metric] > tolerance * reference[metric] + 1e-8:
                    failures.append(f"{name} {event} {metric} worsened beyond tolerance")

    sit_support_passed = all(
        candidate["outputs"][name]["support"]["global_max_excess"] == 0
        for name in ("d3_sit", "d6_sit", "d9_sit")
    )
    if not sit_support_passed:
        failures.append("candidate violates exact SIT physical support")
    unresolved_sic_support = {
        name: candidate["outputs"][name]["support"]["global_max_excess"]
        for name in ("d3_sic", "d6_sic", "d9_sic")
    }
    return {
        "passed_pending_visual_review": not failures,
        "failures": failures,
        "formal_comparator": "champion",
        "sit_exact_support_passed": sit_support_passed,
        "unresolved_unchanged_sic_support_excess": unresolved_sic_support,
        "sic_support_is_out_of_scope_for_sit_only_stage": True,
    }


def _persist_scores_before_reporting(
    metrics_path: Path,
    result: dict[str, Any],
    report: Callable[[], None],
) -> None:
    """Durably persist finite scores before any fallible plotting/reporting."""
    _require_finite_tree(result)
    _strict_atomic_json(metrics_path, result)
    report()


def run(config_path: Path, output: Path) -> dict[str, Any]:
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise RuntimeError("support decoder scoring must run with CUDA_VISIBLE_DEVICES empty")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    config = load_json(config_path)
    if config.get("schema_version") != "sit_support_decoder_scoring_v1":
        raise ValueError("unreviewed support decoder scoring schema")

    evidence_spec = config["evidence_config"]
    evidence_path = Path(evidence_spec["path"])
    if not evidence_path.is_file() or _sha256(evidence_path) != evidence_spec["sha256"]:
        raise ValueError("evidence config SHA mismatch")
    gate_spec = config["gate_config"]
    gate_path = Path(gate_spec["path"])
    if not gate_path.is_file() or _sha256(gate_path) != gate_spec["sha256"]:
        raise ValueError("gate config SHA mismatch")
    _source_is_complete(config["source_decoder_status"], config["source_decoder_metrics"])
    _source_is_complete(config["source_censor_status"], config["source_censor_metrics"])

    output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = output.with_name(f"{output.stem}.metrics.json")
    if metrics_path.exists() or metrics_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {metrics_path}")
    reservation = {
        "status": "reserved",
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
    }
    with output.open("x") as handle:
        json.dump(reservation, handle, indent=2, allow_nan=False)
        handle.write("\n")

    tracker = None
    try:
        code_identity = _clean_code_identity(Path(__file__).resolve().parents[1])
        tracker = ClearMLTracker(
            config["project_name"],
            f'{config["task_name"]}-{output.parent.name}',
            tags=config["clearml"]["tags"],
            env_path=config["clearml"]["env_path"],
        )
        tracker.connect("audit_contract", config)
        _strict_atomic_json(
            output,
            {
                **reservation,
                "status": "running",
                "code_identity": code_identity,
                "clearml_task_id": str(tracker.task.id),
            },
        )

        evidence_config = load_json(evidence_path)
        evidence = {
            name: _load_frozen(spec)
            for name, spec in evidence_config["evidence"].items()
        }
        canonical = evidence["raw"]
        for name, payload in evidence.items():
            _validate_payload(name, payload, evidence_config["evidence"][name], evidence_config)
            _require_equal(canonical, payload, name)

        means, stds, normalization = _load_normalization(evidence_config)
        means5 = means.float().reshape(1, 1, 6, 1, 1)
        stds5 = stds.float().reshape(1, 1, 6, 1, 1)
        mask = canonical["valid_mask"].float()
        truth = decode_with_exact_sit_zero(canonical["truth_normalized"].float(), means, stds)
        persistence = decode_with_exact_sit_zero(
            canonical["persistence_normalized"].float(), means, stds
        )
        champion = decode_with_exact_sit_zero(
            canonical["forecast_normalized"].float(), means, stds
        )
        threshold_payload = evidence["threshold64"]
        threshold = decode_with_exact_sit_zero(
            threshold_payload["forecast_normalized"].float(), means, stds
        )
        threshold_left_censor = censor_sit(threshold)
        threshold_coarse = threshold_payload["coarse_normalized"].float() * stds5 + means5
        candidate = apply_support_decoder_physical(threshold, threshold_coarse, mask)

        scored = {
            "champion": _score(champion, truth, mask, stds.float()),
            "threshold64_uncensored": _score(threshold, truth, mask, stds.float()),
            "threshold64_left_censor": _score(
                threshold_left_censor, truth, mask, stds.float()
            ),
            "threshold64_support_decoder": _score(candidate, truth, mask, stds.float()),
        }
        gate_config = load_json(gate_path)["decision_gate"]
        comparisons = {
            "versus_champion": compare(
                scored["threshold64_support_decoder"], scored["champion"], gate_config, 0
            ),
            "versus_uncensored_threshold64": compare(
                scored["threshold64_support_decoder"],
                scored["threshold64_uncensored"],
                gate_config,
                1,
            ),
            "versus_threshold64_left_censor": compare(
                scored["threshold64_support_decoder"],
                scored["threshold64_left_censor"],
                gate_config,
                2,
            ),
        }
        gate = _stage_gate(
            scored["threshold64_support_decoder"],
            scored["champion"],
            comparisons["versus_champion"],
            gate_config,
        )
        result: dict[str, Any] = {
            "status": "numerical_results_complete_pending_visual_review",
            "scientific_role": "frozen_development_gate_not_champion_replacement",
            "training_performed": False,
            "sampling_performed": False,
            "code_identity": code_identity,
            "normalization_source": normalization,
            "sources": {
                key: config[key]
                for key in (
                    "evidence_config",
                    "gate_config",
                    "source_decoder_status",
                    "source_decoder_metrics",
                    "source_censor_status",
                    "source_censor_metrics",
                )
            },
            "candidate_law": "threshold64 with per-block SIT simplex projection preserving max(C,0)",
            "scores": scored,
            "comparisons": comparisons,
            "gate": {
                **gate,
                "formal_control": "champion",
                "diagnostic_controls": [
                    "threshold64_uncensored",
                    "threshold64_left_censor",
                ],
                "training_or_replacement_permitted": False,
            },
        }

        def report_visuals() -> None:
            visuals = output.parent / "matched_support_decoder_scoring_members"
            visuals.mkdir()
            variants = {
                "champion": champion,
                "threshold64_uncensored": threshold,
                "threshold64_left_censor": threshold_left_censor,
                "threshold64_support_decoder": candidate,
            }
            for position in (0, 4, 7, 11):
                for label, values in variants.items():
                    path = visuals / f"{label}_case{position:02d}_all8_fixed.png"
                    _save_contact_sheet(
                        path,
                        label,
                        canonical["case_ids"][position],
                        values[position],
                        truth[position],
                        persistence[position],
                        mask[position],
                        {"sic": (0.0, 1.0), "sit": (0.0, 3.5)},
                    )
                    tracker.report_image(
                        "sit_support_decoder_scoring/matched_members",
                        f"{label}/case{position:02d}",
                        path,
                        0,
                    )

        _persist_scores_before_reporting(metrics_path, result, report_visuals)
        tracker.upload_artifact("sit_support_decoder_scoring", metrics_path)
        _finish_success(
            tracker,
            output,
            metrics_path,
            {
                **reservation,
                "code_identity": code_identity,
                "clearml_task_id": str(tracker.task.id),
                "scientific_role": result["scientific_role"],
                "gate_passed_pending_visual_review": gate["passed_pending_visual_review"],
            },
        )
        tracker = None
        return result
    except BaseException as error:
        try:
            _record_failure(output, reservation, error)
        except Exception:
            pass
        raise
    finally:
        if tracker is not None:
            try:
                tracker.close()
            except Exception:
                pass


def main() -> None:
    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
