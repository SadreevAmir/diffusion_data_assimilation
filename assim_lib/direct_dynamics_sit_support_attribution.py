"""Frozen CPU attribution of threshold refinement after the same SIT decoder."""

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
from .direct_dynamics_cascade_coarse_fine_boundary_audit import (
    _finish_success,
    _load_frozen,
    _record_failure,
    _require_equal,
    _sha256,
    _strict_atomic_json,
)
from .direct_dynamics_cascade_fine_training import _clean_code_identity
from .direct_dynamics_cascade_memberwise_affine import compare, _require_finite_tree
from .direct_dynamics_cascade_proper_refinement_evaluation import (
    _apply_frozen_support_decoder,
)
from .direct_dynamics_sit_left_censor_audit import decode_with_exact_sit_zero
from .direct_dynamics_sit_support_decoder_scoring import (
    _score,
    _stage_gate,
)


def _verify(path: Path, sha256: str, label: str) -> None:
    if not path.is_file() or _sha256(path) != sha256:
        raise ValueError(f"frozen {label} SHA mismatch: {path}")


def _validate_payloads(
    raw: dict[str, Any], candidate: dict[str, Any], config: dict[str, Any]
) -> None:
    _require_equal(raw, candidate, "threshold64")
    shapes = {
        "forecast_normalized": (12, 8, 6, 320, 256),
        "coarse_normalized": (12, 8, 6, 160, 128),
        "truth_normalized": (12, 6, 320, 256),
        "persistence_normalized": (12, 6, 320, 256),
        "valid_mask": (12, 1, 320, 256),
    }
    for label, payload in (("raw", raw), ("threshold64", candidate)):
        for name, shape in shapes.items():
            value = payload.get(name)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"{label}/{name} violates frozen 12x8 shape")
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"{label}/{name} contains NaN/Inf")
        if list(payload.get("case_ids", ())) != config["case_ids"]:
            raise ValueError(f"{label} case ids differ from frozen attribution panel")
        if payload.get("case_indices") != config["case_indices"]:
            raise ValueError(f"{label} case indices differ from frozen attribution panel")
        if tuple(payload.get("member_indices", ())) != tuple(range(8)):
            raise ValueError(f"{label} member indices differ from frozen attribution panel")
    if raw.get("support_decoder") is not None:
        raise ValueError("raw evidence unexpectedly contains a support decoder")
    if candidate.get("support_decoder") != config["support_decoder"]:
        raise ValueError("candidate support decoder identity differs from frozen law")
    if not isinstance(candidate.get("pre_support_decoder_forecast_normalized"), torch.Tensor):
        raise ValueError("candidate lacks its frozen predecoder forecast")
    if raw.get("pre_support_decoder_forecast_normalized") is not None:
        raise ValueError("raw comparator unexpectedly has a predecoder variant")


def run(config_path: Path, output: Path) -> dict[str, Any]:
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("support attribution must run with CUDA_VISIBLE_DEVICES empty")
    if os.environ.get("CLEARML_REQUIRE_ONLINE") != "1":
        raise RuntimeError("support attribution requires online ClearML")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    config = load_json(config_path)
    if config.get("schema_version") != "sit_support_attribution_v1":
        raise ValueError("unreviewed support attribution schema")
    if (
        len(config.get("case_indices", ())) != 12
        or len(set(config["case_indices"])) != 12
        or len(config.get("case_ids", ())) != 12
        or len(set(config["case_ids"])) != 12
        or config.get("decision_gate", {}).get("bootstrap_draws") != 100000
    ):
        raise ValueError("attribution requires the frozen 12-case paired gate")
    repository_root = Path(__file__).resolve().parents[1]
    for spec, label in (
        (config["source"]["evaluator"], "source evaluator"),
        (config["support_decoder"]["implementation"], "support decoder"),
        (config["support_decoder"]["scoring"], "support decoder scorer"),
    ):
        relative = Path(spec["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe {label} path")
        _verify(repository_root / relative, spec["sha256"], label)

    source_root = Path(config["source"]["directory"])
    contract_path = source_root / "contract.json"
    numerical_path = source_root / "proper_refinement_e2e_evaluation.json"
    review_path = source_root / "visual_review.json"
    _verify(contract_path, config["source"]["contract_sha256"], "contract")
    _verify(numerical_path, config["source"]["numerical_sha256"], "numerical result")
    _verify(review_path, config["source"]["visual_review_sha256"], "visual review")
    contract = load_json(contract_path)
    numerical = load_json(numerical_path)
    review = load_json(review_path)
    if (
        numerical.get("frozen_confirmation", {}).get("numerical_decision")
        != "GO_PENDING_VISUAL_REVIEW"
        or review.get("status") != "visual_review_passed"
    ):
        raise ValueError("source confirmation did not complete numerical and visual review")
    if contract.get("optimizer_steps") != 0 or not contract.get("common_random_numbers"):
        raise ValueError("source is not the frozen zero-optimizer paired replay")
    if contract.get("support_decoder") != config["support_decoder"]:
        raise ValueError("source contract support decoder differs from attribution law")
    if contract.get("code_identity", {}).get("git_commit") != config["source"][
        "code_commit"
    ]:
        raise ValueError("source replay commit differs from frozen attribution source")
    if contract.get("case_indices") != config["case_indices"] or contract.get(
        "case_ids"
    ) != config["case_ids"]:
        raise ValueError("source contract panel differs from frozen attribution panel")
    if contract.get("evidence_sha256", {}).get(
        Path(config["evidence"]["raw"]["path"]).name
    ) != config["evidence"]["raw"]["sha256"]:
        raise ValueError("raw evidence is not registered in the source contract")
    if contract.get("evidence_sha256", {}).get(
        Path(config["evidence"]["threshold64"]["path"]).name
    ) != config["evidence"]["threshold64"]["sha256"]:
        raise ValueError("candidate evidence is not registered in the source contract")

    raw_spec = config["evidence"]["raw"]
    candidate_spec = config["evidence"]["threshold64"]
    raw = _load_frozen(raw_spec)
    candidate = _load_frozen(candidate_spec)
    _validate_payloads(raw, candidate, config)

    metadata_path = Path(contract["fine"]["run_dir"]) / "metadata.json"
    _verify(
        metadata_path,
        contract["fine"]["sha256"]["metadata.json"],
        "normalization metadata",
    )
    data_config = load_json(metadata_path)["data_config"]
    if data_config.get("fields") != ["siconc", "sithic"]:
        raise ValueError("normalization source is not the frozen SIC/SIT law")
    means = torch.tensor(data_config["means"] * 3, dtype=torch.float32)
    stds = torch.tensor(data_config["stds"] * 3, dtype=torch.float32)
    valid = raw["valid_mask"].float()
    truth = decode_with_exact_sit_zero(raw["truth_normalized"].float(), means, stds)

    raw_normalized, raw_decoded, raw_coarse_error = _apply_frozen_support_decoder(
        raw["forecast_normalized"], raw["coarse_normalized"], valid, means, stds
    )
    candidate_normalized, candidate_decoded, candidate_coarse_error = (
        _apply_frozen_support_decoder(
            candidate["pre_support_decoder_forecast_normalized"],
            candidate["coarse_normalized"],
            valid,
            means,
            stds,
        )
    )
    if not torch.equal(candidate_normalized, candidate["forecast_normalized"].float()):
        raise RuntimeError("replayed candidate decoder differs from frozen confirmation")

    scored = {
        "raw_plus_decoder": _score(raw_decoded, truth, valid, stds),
        "threshold64_plus_decoder": _score(candidate_decoded, truth, valid, stds),
    }
    comparison = compare(
        scored["threshold64_plus_decoder"],
        scored["raw_plus_decoder"],
        config["decision_gate"],
        0,
    )
    gate = _stage_gate(
        scored["threshold64_plus_decoder"],
        scored["raw_plus_decoder"],
        comparison,
        config["decision_gate"],
    )
    result = {
        "status": "complete",
        "scientific_role": "frozen_confirmation_attribution_no_fit_no_sampling",
        "training_performed": False,
        "sampling_performed": False,
        "code_identity": _clean_code_identity(Path(__file__).resolve().parents[1]),
        "source": config["source"],
        "evidence": config["evidence"],
        "support_decoder": config["support_decoder"],
        "coarse_consistency_error_max": {
            "raw_plus_decoder": raw_coarse_error,
            "threshold64_plus_decoder": candidate_coarse_error,
        },
        "scores": scored,
        "comparison_threshold64_minus_raw_after_common_decoder": comparison,
        "attribution_gate": gate,
        "claim": (
            "threshold refinement adds value beyond the common support decoder"
            if gate["passed_pending_visual_review"]
            else "additional value of threshold refinement is not established"
        ),
    }
    _require_finite_tree(result)

    output.parent.mkdir(parents=True, exist_ok=False)
    metrics_path = output.with_name("status.metrics.json")
    reservation = {
        "status": "reserved",
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
    }
    _strict_atomic_json(output, reservation)
    tracker = None
    try:
        tracker = ClearMLTracker(
            config["project_name"],
            f'{config["task_name"]}-{output.parent.name}',
            tags=config["clearml"]["tags"],
            env_path=config["clearml"]["env_path"],
        )
        tracker.connect("attribution_contract", config)
        _strict_atomic_json(metrics_path, result)
        tracker.upload_artifact("sit_support_attribution", metrics_path)
        terminal = {
            **reservation,
            "code_identity": result["code_identity"],
            "clearml_task_id": str(tracker.task.id),
            "attribution_passed": gate["passed_pending_visual_review"],
            "claim": result["claim"],
        }
        _finish_success(tracker, output, metrics_path, terminal)
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
    signal.signal(signal.SIGTERM, lambda signum, _frame: (_ for _ in ()).throw(
        TimeoutError(f"support attribution received signal {signum}")
    ))
    signal.signal(signal.SIGINT, lambda signum, _frame: (_ for _ in ()).throw(
        TimeoutError(f"support attribution received signal {signum}")
    ))
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.config.resolve(), arguments.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
