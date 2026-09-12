"""Frozen full-resolution counterfactual for physical left-censoring of SIT."""

from __future__ import annotations

import argparse
import json
import os
import signal
from pathlib import Path
from typing import Any

import torch

from .clearml_tracking import ClearMLTracker
from .config import load_json
from .direct_dynamics_affine_calibration import boundary_event_metrics
from .direct_dynamics_cascade import masked_block_average
from .direct_dynamics_cascade_coarse_fine_boundary_audit import (
    _finish_success, _load_frozen, _load_normalization, _record_failure,
    _require_equal, _sha256, _strict_atomic_json, _terminate, _validate_payload,
)
from .direct_dynamics_cascade_e2e_evaluation import _add_joint_consistency, _summary
from .direct_dynamics_cascade_fine_training import _clean_code_identity
from .direct_dynamics_cascade_paired_evaluation import _save_contact_sheet, score_ensemble


SIT_CHANNELS = (1, 3, 5)
OUTPUTS = ("d3_sic", "d3_sit", "d6_sic", "d6_sit", "d9_sic", "d9_sit")


def censor_sit(physical: torch.Tensor) -> torch.Tensor:
    if physical.ndim != 5 or physical.shape[2] != 6:
        raise ValueError("SIT censoring requires [case,member,6,y,x]")
    if not torch.isfinite(physical).all():
        raise FloatingPointError("SIT censoring input contains NaN/Inf")
    censored = physical.clone()
    censored[:, :, SIT_CHANNELS] = censored[:, :, SIT_CHANNELS].clamp_min(0)
    if not torch.equal(censored[:, :, 0::2], physical[:, :, 0::2]):
        raise RuntimeError("SIT censoring changed SIC")
    return censored


def decode_with_exact_sit_zero(
    normalized: torch.Tensor, means: torch.Tensor, stds: torch.Tensor
) -> torch.Tensor:
    shape = (1,) * (normalized.ndim - 3) + (6, 1, 1)
    mean = means.to(normalized).reshape(shape)
    std = stds.to(normalized).reshape(shape)
    physical = normalized * std + mean
    for channel in SIT_CHANNELS:
        encoded_zero = (
            torch.zeros((), dtype=normalized.dtype) - means[channel].to(normalized)
        ) / stds[channel].to(normalized)
        selected = normalized[..., channel : channel + 1, :, :] == encoded_zero
        physical[..., channel : channel + 1, :, :] = torch.where(
            selected,
            torch.zeros_like(physical[..., channel : channel + 1, :, :]),
            physical[..., channel : channel + 1, :, :],
        )
    return physical


def _exact_zero_metrics(
    members: torch.Tensor,
    truth_normalized: torch.Tensor,
    mask: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor,
) -> dict[str, Any]:
    result = {}
    weight = mask.double()
    for channel in SIT_CHANNELS:
        probability = (members[:, :, channel : channel + 1] == 0).double().mean(dim=1)
        mean = torch.as_tensor(means[channel], dtype=truth_normalized.dtype)
        std = torch.as_tensor(stds[channel], dtype=truth_normalized.dtype)
        encoded_zero = (torch.zeros((), dtype=truth_normalized.dtype) - mean) / std
        event = (truth_normalized[:, channel : channel + 1] == encoded_zero).double()
        per_case = (
            ((probability - event).square() * weight).sum(dim=(1, 2, 3))
            / weight.sum(dim=(1, 2, 3))
        )
        result[OUTPUTS[channel]] = {
            "brier": float(per_case.mean()), "per_case_brier": per_case.tolist(),
            "forecast_atom_mass": float((probability * weight).sum() / weight.sum()),
            "truth_atom_mass": float((event * weight).sum() / weight.sum()),
        }
    return result


def _coarse_consistency_change(
    censored_normalized: torch.Tensor, original_coarse: torch.Tensor, mask: torch.Tensor
) -> dict[str, Any]:
    cases, members = censored_normalized.shape[:2]
    expanded_mask = mask[:, None].expand(-1, members, -1, -1, -1).flatten(0, 1)
    recovered, fraction = masked_block_average(censored_normalized.flatten(0, 1), expanded_mask)
    delta = recovered.unflatten(0, (cases, members)).double() - original_coarse.double()
    active = fraction.unflatten(0, (cases, members)).double()
    result = {}
    for channel in SIT_CHANNELS:
        value = delta[:, :, channel : channel + 1]
        weight = active[:, :, channel : channel + 1]
        result[OUTPUTS[channel]] = {
            "max_abs_D_censored_minus_original_C": float(value[weight > 0].abs().max()),
            "weighted_rms_D_censored_minus_original_C": float(
                torch.sqrt((value.square() * weight).sum() / weight.sum())
            ),
        }
    return result


def _evaluate(
    normalized: torch.Tensor,
    physical: torch.Tensor,
    truth_normalized: torch.Tensor,
    truth_physical: torch.Tensor,
    persistence_physical: torch.Tensor,
    mask: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor,
) -> dict[str, Any]:
    metrics = score_ensemble(
        normalized, physical, truth_normalized, truth_physical,
        persistence_physical, mask, mask,
    )
    _add_joint_consistency(metrics, physical, mask)
    metrics["boundary_events"] = boundary_event_metrics(physical, truth_physical, mask)
    metrics["exact_zero_events"] = _exact_zero_metrics(
        physical, truth_normalized, mask, means, stds
    )
    metrics["summary"] = _summary(metrics)
    return metrics


def _validate_expected_score_behavior(original: dict[str, Any], censored: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for channel in SIT_CHANNELS:
        name = OUTPUTS[channel]
        before = original["outputs"][name]["case_equal_fair_crps"]
        after = censored["outputs"][name]["case_equal_fair_crps"]
        unchanged_event = original["boundary_events"][name]["sit_le_0p01"]
        censored_event = censored["boundary_events"][name]["sit_le_0p01"]
        result[name] = {
            "fair_crps_before": before, "fair_crps_after": after,
            "fair_crps_change": after - before,
            "aggregate_nonincrease_observed": after <= before + 1e-12,
            "sit_le_0p01_brier_bitwise_unchanged": unchanged_event["brier"] == censored_event["brier"],
            "sit_le_0p01_ece_bitwise_unchanged": unchanged_event["reliability_ece"] == censored_event["reliability_ece"],
        }
        if not result[name]["sit_le_0p01_brier_bitwise_unchanged"] or not result[name]["sit_le_0p01_ece_bitwise_unchanged"]:
            raise RuntimeError("left censoring changed the SIT<=0.01 event")
    return result


def run(config_path: Path, output: Path) -> dict[str, Any]:
    torch.set_num_threads(6); torch.set_num_interop_threads(1)
    if torch.get_num_threads() != 6 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("failed to enforce reviewed 6/1 CPU envelope")
    if torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise RuntimeError("SIT censor audit requires CUDA_VISIBLE_DEVICES empty")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    config = load_json(config_path)
    if config.get("schema_version") != "sit_left_censor_audit_v1":
        raise ValueError("unreviewed SIT censor audit schema")
    for group in ("evidence_config", "source_rank_status", "source_rank_metrics"):
        source = config[group]; path = Path(source["path"])
        if not path.is_file() or _sha256(path) != source["sha256"]:
            raise ValueError(f"{group} SHA mismatch")
    source_status = load_json(config["source_rank_status"]["path"])
    if source_status.get("status") != "complete" or source_status.get("metrics_sha256") != config["source_rank_metrics"]["sha256"]:
        raise ValueError("rank-atom source audit is not terminal complete")
    evidence_config = load_json(config["evidence_config"]["path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = output.with_name(f"{output.stem}.metrics.json")
    if metrics_path.exists() or metrics_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {metrics_path}")
    reservation = {"status": "reserved", "config_path": str(config_path), "config_sha256": _sha256(config_path)}
    with output.open("x") as handle:
        json.dump(reservation, handle, indent=2, allow_nan=False); handle.write("\n")
    tracker = None
    try:
        code_identity = _clean_code_identity(Path(__file__).resolve().parents[1])
        tracker = ClearMLTracker(
            config["project_name"], f'{config["task_name"]}-{output.parent.name}',
            tags=config["clearml"]["tags"], env_path=config["clearml"]["env_path"],
        )
        tracker.connect("audit_contract", config)
        _strict_atomic_json(output, {**reservation, "status": "running", "code_identity": code_identity, "clearml_task_id": str(tracker.task.id)})
        evidence = {name: _load_frozen(spec) for name, spec in evidence_config["evidence"].items()}
        canonical = evidence["raw"]
        for name, payload in evidence.items():
            _validate_payload(name, payload, evidence_config["evidence"][name], evidence_config)
            _require_equal(canonical, payload, name)
        means, stds, normalization = _load_normalization(evidence_config)
        means5 = means.float().reshape(1, 1, 6, 1, 1); stds5 = stds.float().reshape(1, 1, 6, 1, 1)
        truth_normalized = canonical["truth_normalized"].float()
        persistence_normalized = canonical["persistence_normalized"].float()
        truth_physical = decode_with_exact_sit_zero(truth_normalized, means, stds)
        persistence_physical = decode_with_exact_sit_zero(persistence_normalized, means, stds)
        mask = canonical["valid_mask"].float()
        visuals = output.parent / "matched_fixed_scale_members"
        visuals.mkdir()
        result: dict[str, Any] = {
            "status": "numerical_results_complete_pending_finalization",
            "scientific_role": "support_law_counterfactual_not_champion_selection",
            "code_identity": code_identity, "normalization_source": normalization,
            "sources": {key: config[key] for key in ("evidence_config", "source_rank_status", "source_rank_metrics")},
            "transformation": "all SIT horizons max(value,0); SIC unchanged; no fitting",
            "candidates": {},
        }
        for name in ("raw", "ordinary64", "threshold64"):
            payload = evidence[name]
            original_normalized = payload["forecast_normalized"].float()
            original_physical = decode_with_exact_sit_zero(original_normalized, means, stds)
            censored_physical = censor_sit(original_physical)
            censored_normalized = (censored_physical - means5) / stds5
            positive = original_physical[:, :, SIT_CHANNELS] > 0
            if not torch.equal(censored_physical[:, :, SIT_CHANNELS][positive], original_physical[:, :, SIT_CHANNELS][positive]):
                raise RuntimeError("left censoring changed positive SIT")
            for channel in SIT_CHANNELS:
                if not torch.equal(
                    original_physical[:, :, channel : channel + 1] <= 0.01,
                    censored_physical[:, :, channel : channel + 1] <= 0.01,
                ):
                    raise RuntimeError("left censoring changed raw SIT<=0.01 indicators")
            original_metrics = _evaluate(
                original_normalized, original_physical, truth_normalized, truth_physical,
                persistence_physical, mask, means, stds,
            )
            censored_metrics = _evaluate(
                censored_normalized, censored_physical, truth_normalized, truth_physical,
                persistence_physical, mask, means, stds,
            )
            result["candidates"][name] = {
                "original": original_metrics, "left_censored": censored_metrics,
                "paired_checks": _validate_expected_score_behavior(original_metrics, censored_metrics),
                "coarse_consistency_change": _coarse_consistency_change(
                    censored_normalized, payload["coarse_normalized"], mask,
                ),
            }
            for position in (0, 4, 7, 11):
                for variant, physical in (("original", original_physical), ("left_censored", censored_physical)):
                    path = visuals / f"{name}_{variant}_case{position:02d}_all8_fixed.png"
                    _save_contact_sheet(
                        path, f"{name}/{variant}", canonical["case_ids"][position],
                        physical[position], truth_physical[position], persistence_physical[position],
                        mask[position], {"sic": (0.0, 1.0), "sit": (0.0, 3.5)},
                    )
                    tracker.report_image("sit_left_censor/matched_members", f"{name}/{variant}/case{position:02d}", path, 0)
        _strict_atomic_json(metrics_path, result)
        tracker.upload_artifact("sit_left_censor_audit", metrics_path)
        _finish_success(
            tracker, output, metrics_path,
            {**reservation, "code_identity": code_identity, "clearml_task_id": str(tracker.task.id), "scientific_role": result["scientific_role"]},
        )
        tracker = None
        return result
    except BaseException as error:
        try: _record_failure(output, reservation, error)
        except Exception: pass
        raise
    finally:
        if tracker is not None:
            try: tracker.close()
            except Exception: pass


def main() -> None:
    signal.signal(signal.SIGTERM, _terminate); signal.signal(signal.SIGINT, _terminate)
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True, type=Path); parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(); print(json.dumps(run(args.config.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__": main()
