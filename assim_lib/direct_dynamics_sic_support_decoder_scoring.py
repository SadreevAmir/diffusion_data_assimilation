"""Frozen paired CPU counterfactual for one uniform bounded SIC decoder."""

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
from .direct_dynamics_cascade_checkpoint_evaluation import _save_rank_histograms
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
from .direct_dynamics_sic_support_decoder import (
    project_masked_blocks_to_unit_interval_mean,
)


SIC_CHANNELS = (0, 2, 4)
OUTPUTS = ("d3_sic", "d3_sit", "d6_sic", "d6_sit", "d9_sic", "d9_sit")


def apply_sic_support_decoder_physical(
    physical: torch.Tensor,
    coarse_physical: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Apply the same capped-simplex law to every valid-ocean SIC value."""
    if physical.ndim != 5 or physical.shape[2] != 6:
        raise ValueError("physical ensemble must be [case,member,6,y,x]")
    expected_coarse = (
        physical.shape[0], physical.shape[1], 6,
        physical.shape[-2] // 2, physical.shape[-1] // 2,
    )
    if coarse_physical.shape != expected_coarse:
        raise ValueError("coarse ensemble shape is incompatible with physical ensemble")
    if mask.shape != (physical.shape[0], 1, *physical.shape[-2:]):
        raise ValueError("mask shape is incompatible with physical ensemble")
    if not torch.isfinite(physical).all() or not torch.isfinite(coarse_physical).all():
        raise FloatingPointError("SIC decoder inputs contain NaN/Inf")

    decoded = physical.double().clone()
    cases, members = physical.shape[:2]
    expanded_mask = mask[:, None].expand(-1, members, -1, -1, -1).flatten(0, 1)
    for channel in SIC_CHANNELS:
        projected = project_masked_blocks_to_unit_interval_mean(
            physical[:, :, channel : channel + 1].flatten(0, 1),
            coarse_physical[:, :, channel : channel + 1].flatten(0, 1),
            expanded_mask,
        )
        decoded[:, :, channel : channel + 1] = projected.unflatten(0, (cases, members))
    if not torch.equal(decoded[:, :, 1::2], physical[:, :, 1::2].double()):
        raise RuntimeError("SIC support decoder changed SIT")
    return decoded


def _source_gate_is_complete(config: dict[str, Any]) -> dict[str, Any]:
    bound: dict[str, Any] = {}
    for name, spec in config["source_gate"].items():
        path = Path(spec["path"])
        if not path.is_file() or _sha256(path) != spec["sha256"]:
            raise ValueError(f"source gate SHA mismatch: {name}")
        bound[name] = {"path": str(path), "sha256": spec["sha256"]}
    contract = load_json(config["source_gate"]["contract"]["path"])
    if contract.get("code_identity") != config["source_code_identity"]:
        raise ValueError("source gate code identity differs")
    if contract.get("split") != "valid" or contract.get("optimizer_steps") != 0:
        raise ValueError("source must be zero-optimizer validation evidence")
    if contract.get("panel", {}).get("role") != "development_reuse":
        raise ValueError("source panel role differs")
    if contract.get("case_indices") != config["case_indices"] or contract.get("case_ids") != config["case_ids"]:
        raise ValueError("source panel differs")
    if contract.get("coarse") != config["coarse"] or contract.get("fine") != config["fine"]:
        raise ValueError("source model identity differs")
    if contract.get("refinement") != config["ordinary_refinement"]:
        raise ValueError("ordinary refinement identity differs")
    for label, filename in config["source_evidence_names"].items():
        if contract.get("evidence_sha256", {}).get(filename) != config["evidence"][label]["sha256"]:
            raise ValueError(f"source contract does not bind {label} evidence")
    return bound


def _physical(payload: dict[str, Any], means: torch.Tensor, stds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    forecast = canonical_physical_decode(payload["forecast_normalized"], means, stds)
    coarse = canonical_physical_decode(payload["coarse_normalized"], means, stds)
    return forecast, coarse


def canonical_physical_decode(
    normalized: torch.Tensor, means: torch.Tensor, stds: torch.Tensor
) -> torch.Tensor:
    """Decode without manufacturing SIC atoms through FP32 cancellation.

    SIT retains the historical FP32 affine path.  SIC uses FP64 arithmetic
    with the same promoted FP32 statistics, then restores physical zero/one
    only when the frozen normalized value is exactly its FP32 encoding.
    """
    if not normalized.is_floating_point() or normalized.shape[-3] != 6:
        raise ValueError("canonical decode requires six floating-point channels")
    source = normalized.float()
    means32 = means.float().reshape((1,) * (source.ndim - 3) + (6, 1, 1))
    stds32 = stds.float().reshape((1,) * (source.ndim - 3) + (6, 1, 1))
    if not torch.isfinite(means32).all() or not torch.isfinite(stds32).all() or torch.any(stds32 <= 0):
        raise ValueError("invalid normalization for canonical physical decode")
    physical = (source * stds32 + means32).double()
    for channel in SIC_CHANNELS:
        mean32 = means32.select(-3, channel)
        std32 = stds32.select(-3, channel)
        encoded_zero = -mean32 / std32
        encoded_one = (torch.ones_like(mean32) - mean32) / std32
        values = source.select(-3, channel)
        decoded = values.double() * std32.double() + mean32.double()
        decoded[values == encoded_zero] = 0.0
        decoded[values == encoded_one] = 1.0
        physical.select(-3, channel).copy_(decoded)
    return physical


def _stage_gate(
    candidate: dict[str, Any], control: dict[str, Any], comparison: dict[str, Any], spec: dict[str, Any]
) -> dict[str, Any]:
    failures: list[str] = []
    paired = comparison["paired_primary_delta"]
    if paired["estimate"] >= 0 or paired["paired_date_bootstrap_95_ci_high"] >= 0:
        failures.append("primary standardized fair CRPS paired-date CI does not exclude zero")
    tolerance = 1.0 + float(spec["per_output_score_tolerance"])
    if comparison["joint_energy_ratio"] > tolerance:
        failures.append("joint energy worsened beyond tolerance")
    for name, row in comparison["outputs"].items():
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
        if value > temporal_tolerance * control["temporal_increments"][key] + 1e-12:
            failures.append(f"temporal diagnostic worsened beyond tolerance: {key}")
    for name, events in candidate["boundary_events"].items():
        for event, row in events.items():
            if event == "open_water_haze":
                continue
            reference = control["boundary_events"][name][event]
            for metric in ("brier", "reliability_ece"):
                if row[metric] > tolerance * reference[metric] + 1e-8:
                    failures.append(f"{name} {event} {metric} worsened beyond tolerance")
    sic_support_passed = all(
        candidate["outputs"][name]["support"]["global_max_excess"] == 0
        for name in ("d3_sic", "d6_sic", "d9_sic")
    )
    if not sic_support_passed:
        failures.append("candidate violates exact SIC physical support")
    return {
        "passed_pending_visual_review": not failures,
        "failures": failures,
        "sic_exact_support_passed": sic_support_passed,
        "formal_comparator": "decoded_raw",
        "training_or_replacement_permitted": False,
    }


def _persist_before_reporting(path: Path, result: dict[str, Any], report: Callable[[], None]) -> None:
    _require_finite_tree(result)
    _strict_atomic_json(path, result)
    report()


def run(config_path: Path, output: Path) -> dict[str, Any]:
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    if torch.get_num_threads() != 6 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("failed to enforce reviewed 6/1 CPU thread envelope")
    if torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise RuntimeError("SIC support scoring requires CUDA_VISIBLE_DEVICES empty")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    config = load_json(config_path)
    if config.get("schema_version") != "sic_support_decoder_scoring_v1":
        raise ValueError("unreviewed SIC support scoring schema")
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = output.with_name(f"{output.stem}.metrics.json")
    if metrics_path.exists() or metrics_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {metrics_path}")
    reservation = {
        "status": "reserved", "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
    }
    with output.open("x") as handle:
        json.dump(reservation, handle, indent=2, allow_nan=False)
        handle.write("\n")
    tracker = None
    try:
        source_gate = _source_gate_is_complete(config)
        code_identity = _clean_code_identity(Path(__file__).resolve().parents[1])
        tracker = ClearMLTracker(
            config["project_name"], f'{config["task_name"]}-{output.parent.name}',
            tags=config["clearml"]["tags"], env_path=config["clearml"]["env_path"],
        )
        tracker.connect("audit_contract", config)
        _strict_atomic_json(output, {
            **reservation, "status": "running", "code_identity": code_identity,
            "clearml_task_id": str(tracker.task.id),
        })

        evidence = {name: _load_frozen(spec) for name, spec in config["evidence"].items()}
        raw = evidence["raw"]
        for name, payload in evidence.items():
            _validate_payload(name, payload, config["evidence"][name], config)
            _require_equal(raw, payload, name)
        means, stds, normalization = _load_normalization(config)
        mask = raw["valid_mask"].double()
        truth = canonical_physical_decode(raw["truth_normalized"], means, stds)
        persistence = canonical_physical_decode(
            raw["persistence_normalized"], means, stds
        )

        variants: dict[str, torch.Tensor] = {}
        for name, payload in evidence.items():
            physical, coarse = _physical(payload, means, stds)
            variants[f"original_{name}"] = physical
            variants[f"decoded_{name}"] = apply_sic_support_decoder_physical(
                physical, coarse, mask
            )
        for name in ("raw", "ordinary"):
            if not torch.equal(variants[f"decoded_{name}"][:, :, 1::2], variants[f"original_{name}"][:, :, 1::2]):
                raise RuntimeError(f"decoded {name} changed SIT")

        scores = {
            name: diagnostics(values, truth, mask, stds.double())
            for name, values in variants.items()
        }
        source_metrics = load_json(config["source_gate"]["metrics"]["path"])
        for name, source_label in config["source_candidate_labels"].items():
            expected = source_metrics["candidates"][source_label]["metrics"][
                "primary_standardized_fair_crps"
            ]["case_equal_mean"]
            actual = scores[f"original_{name}"]["primary_standardized_fair_crps"]
            # The source gate scores normalized FP32 directly; this audit
            # reconstructs physical FP32 before standardizing in float64.
            if abs(actual - expected) > 2e-7:
                raise RuntimeError(f"source-score replay mismatch: {name}")

        gate_spec = config["decision_gate"]
        comparisons = {
            "decoded_ordinary_versus_decoded_raw": compare(
                scores["decoded_ordinary"], scores["decoded_raw"], gate_spec, 0
            ),
            "decoded_raw_versus_original_raw": compare(
                scores["decoded_raw"], scores["original_raw"], gate_spec, 1
            ),
            "decoded_ordinary_versus_original_ordinary": compare(
                scores["decoded_ordinary"], scores["original_ordinary"], gate_spec, 2
            ),
        }
        gate = _stage_gate(
            scores["decoded_ordinary"], scores["decoded_raw"],
            comparisons["decoded_ordinary_versus_decoded_raw"], gate_spec,
        )
        result: dict[str, Any] = {
            "status": "numerical_results_complete_pending_visual_review",
            "scientific_role": "frozen_development_counterfactual_not_calibrator_acceptance",
            "training_performed": False,
            "sampling_performed": False,
            "code_identity": code_identity,
            "source_gate": source_gate,
            "normalization_source": normalization,
            "decoder_law": "uniform per-block Euclidean projection of SIC onto [0,1] with clipped source coarse mean",
            "scores": scores,
            "comparisons": comparisons,
            "gate": gate,
        }

        def report() -> None:
            visuals = output.parent / "fixed_scale_members"
            ranks = output.parent / "rank_histograms"
            visuals.mkdir()
            ranks.mkdir()
            for name in ("decoded_raw", "decoded_ordinary"):
                for position in (0, 4, 7, 11):
                    path = visuals / f"{name}_case{position:02d}_all8_fixed.png"
                    _save_contact_sheet(
                        path, name, raw["case_ids"][position], variants[name][position],
                        truth[position],
                        persistence[position],
                        mask[position], {"sic": (0.0, 1.0), "sit": (0.0, 3.5)},
                    )
                    tracker.report_image("sic_support_decoder/matched_members", f"{name}/case{position:02d}", path, 0)
                rank_path = ranks / f"{name}.png"
                _save_rank_histograms(rank_path, name, scores[name])
                tracker.report_image("sic_support_decoder/rank_histograms", name, rank_path, 0)

        _persist_before_reporting(metrics_path, result, report)
        tracker.upload_artifact("sic_support_decoder_scoring", metrics_path)
        _finish_success(
            tracker, output, metrics_path,
            {
                **reservation, "code_identity": code_identity,
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
