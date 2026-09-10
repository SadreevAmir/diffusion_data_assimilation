"""Immutable checkpoint-to-sample-to-image provenance and visual reviews."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_RATINGS = {"pass", "fail", "not_assessed"}
_VERDICTS = {"pass", "fail", "inconclusive"}
_CHECKPOINT_ROLES = {"raw", "ema", "calibrated"}
_DISPLAY_MODES = {"raw", "display_transformed"}
_REQUIRED_RATINGS = {
    "macro_coherence",
    "grain_or_pits",
    "coastline",
    "false_low_ice_sit",
    "support",
}


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evidence(path: str | Path) -> dict[str, str]:
    resolved = Path(path).expanduser().resolve()
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def _validate_panel(panel: dict[str, Any], image_count: int) -> None:
    required = {"image_index", "case_id", "member", "lead_days", "fields"}
    if set(panel) != required:
        raise ValueError(f"each panel binding must contain exactly {sorted(required)}")
    if not isinstance(panel["image_index"], int) or not 0 <= panel["image_index"] < image_count:
        raise ValueError("panel image_index is outside the evidence image list")
    if not isinstance(panel["case_id"], str) or not panel["case_id"].strip():
        raise ValueError("panel case_id must be explicit")
    if not isinstance(panel["member"], int) or panel["member"] < 0:
        raise ValueError("panel member must be a non-negative integer")
    if panel["lead_days"] != [3, 6, 9]:
        raise ValueError("visual panel must bind the declared d+3/d+6/d+9 trajectory")
    if panel["fields"] != ["sic", "sit"]:
        raise ValueError("visual panel must bind both SIC and SIT")


def _validate_replay_identity(
    source_commit: str,
    checkpoints: list[dict[str, str]],
    replay_identity: dict[str, str],
) -> None:
    if replay_identity.get("replay_code_commit") != source_commit:
        raise ValueError("replay identity commit differs from sample provenance")
    for checkpoint in checkpoints:
        stage = checkpoint["stage"]
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,31}", stage) is None:
            raise ValueError("checkpoint stage is unsafe for replay identity binding")
        if replay_identity.get(f"{stage}_checkpoint_sha256") != checkpoint["sha256"]:
            raise ValueError("replay identity checkpoint SHA differs from its file binding")


def write_sample_provenance_manifest(
    *,
    manifest_path: str | Path,
    source_commit: str,
    checkpoint_bindings: list[dict[str, str | Path]],
    sample_path: str | Path,
    case_ids: list[str],
    member_indices: list[int],
    solver: dict[str, Any],
    replay_identity: dict[str, str],
) -> Path:
    """Bind a saved sample tensor to the exact checkpoints and replay contract."""
    destination = Path(manifest_path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to replace sample provenance manifest: {destination}")
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("source_commit must be a full immutable git commit")
    if not checkpoint_bindings:
        raise ValueError("sample provenance requires at least one checkpoint")
    checkpoints = []
    stages = set()
    for binding in checkpoint_bindings:
        if set(binding) != {"stage", "role", "path"}:
            raise ValueError("checkpoint bindings require exactly stage, role, and path")
        stage = str(binding["stage"])
        role = str(binding["role"])
        if not stage or stage in stages or role not in _CHECKPOINT_ROLES:
            raise ValueError("checkpoint stage must be unique and role must be explicit")
        stages.add(stage)
        checkpoints.append({**_evidence(binding["path"]), "stage": stage, "role": role})
    if not case_ids or len(case_ids) != len(set(case_ids)) or any(not value for value in case_ids):
        raise ValueError("sample provenance requires unique non-empty case identities")
    if (
        not member_indices
        or len(member_indices) != len(set(member_indices))
        or any(not isinstance(value, int) or value < 0 for value in member_indices)
    ):
        raise ValueError("sample provenance requires unique non-negative member identities")
    if (
        not isinstance(solver, dict)
        or not solver
        or not isinstance(replay_identity, dict)
        or not replay_identity
    ):
        raise ValueError("sample provenance requires solver and replay identities")
    _validate_replay_identity(source_commit, checkpoints, replay_identity)
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "checkpoints": checkpoints,
        "sample_tensor": _evidence(sample_path),
        "case_ids": case_ids,
        "member_indices": member_indices,
        "solver": solver,
        "replay_identity": replay_identity,
    }
    _atomic_json(destination, payload)
    return destination


def _load_and_verify_sample_provenance(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("sample provenance manifest has an incompatible schema")
    if re.fullmatch(r"[0-9a-f]{40}", str(payload.get("source_commit", ""))) is None:
        raise ValueError("sample provenance lacks an immutable replay commit")
    checkpoints = payload.get("checkpoints")
    sample = payload.get("sample_tensor")
    if not isinstance(checkpoints, list) or not checkpoints or not isinstance(sample, dict):
        raise ValueError("sample provenance file bindings are incomplete")
    for item in [*checkpoints, sample]:
        if not isinstance(item, dict) or not {"path", "sha256"} <= set(item):
            raise ValueError("sample provenance contains an invalid file binding")
        if _sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError("sample provenance file hash no longer matches its manifest")
    stages = [item.get("stage") for item in checkpoints]
    if (
        len(stages) != len(set(stages))
        or any(not stage for stage in stages)
        or any(item.get("role") not in _CHECKPOINT_ROLES for item in checkpoints)
    ):
        raise ValueError("sample provenance checkpoint stages or roles are invalid")
    case_ids = payload.get("case_ids")
    members = payload.get("member_indices")
    if not isinstance(case_ids, list) or not isinstance(members, list):
        raise ValueError("sample provenance lacks case/member identities")
    replay_identity = payload.get("replay_identity")
    if not isinstance(replay_identity, dict):
        raise ValueError("sample provenance lacks its replay identity")
    _validate_replay_identity(payload["source_commit"], checkpoints, replay_identity)
    return resolved, payload


def write_visual_evidence_manifest(
    *,
    manifest_path: str | Path,
    sample_provenance_path: str | Path,
    image_paths: list[str | Path],
    panel_bindings: list[dict[str, Any]],
    renderer: str,
    origin: str,
    scales: str,
    display: str,
) -> Path:
    """Bind files produced by one render operation before anyone reviews them."""
    destination = Path(manifest_path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to replace visual evidence manifest: {destination}")
    provenance_path, provenance = _load_and_verify_sample_provenance(sample_provenance_path)
    if not image_paths or not panel_bindings:
        raise ValueError("visual evidence requires images and typed panel bindings")
    for panel in panel_bindings:
        if not isinstance(panel, dict):
            raise ValueError("panel bindings must be JSON objects")
        _validate_panel(panel, len(image_paths))
        if (
            panel["case_id"] not in provenance["case_ids"]
            or panel["member"] not in provenance["member_indices"]
        ):
            raise ValueError("panel binding is absent from the sample provenance")
    if not renderer.strip() or origin not in {"upper", "lower"} or not scales.strip():
        raise ValueError("renderer, origin, and scales must be explicit")
    if display not in _DISPLAY_MODES:
        raise ValueError(f"display must be one of {sorted(_DISPLAY_MODES)}")
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": provenance["source_commit"],
        "sample_provenance": {"path": str(provenance_path), "sha256": _sha256(provenance_path)},
        "checkpoints": provenance["checkpoints"],
        "sample_tensor": provenance["sample_tensor"],
        "images": [_evidence(path) for path in image_paths],
        "panel_bindings": panel_bindings,
        "renderer": renderer,
        "rendering": {"origin": origin, "scales": scales, "display": display},
    }
    _atomic_json(destination, payload)
    return destination


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _load_and_verify_evidence(manifest_path: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(manifest_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("visual evidence manifest has an incompatible schema")
    if re.fullmatch(r"[0-9a-f]{40}", str(payload.get("source_commit", ""))) is None:
        raise ValueError("visual evidence manifest lacks an immutable source commit")
    checkpoints = payload.get("checkpoints")
    sample = payload.get("sample_tensor")
    images = payload.get("images")
    panels = payload.get("panel_bindings")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError("visual evidence checkpoint binding is invalid")
    if not isinstance(sample, dict) or not isinstance(images, list) or not images:
        raise ValueError("visual evidence file bindings are incomplete")
    provenance_binding = payload.get("sample_provenance")
    if not isinstance(provenance_binding, dict):
        raise ValueError("visual evidence lacks sample provenance")
    provenance_path, provenance = _load_and_verify_sample_provenance(provenance_binding.get("path", ""))
    if _sha256(provenance_path) != provenance_binding.get("sha256"):
        raise ValueError("sample provenance manifest hash no longer matches visual evidence")
    if checkpoints != provenance["checkpoints"] or sample != provenance["sample_tensor"]:
        raise ValueError("visual evidence differs from its sample provenance")
    if payload.get("source_commit") != provenance["source_commit"]:
        raise ValueError("visual evidence source commit differs from sample provenance")
    for item in [*checkpoints, sample, *images]:
        if not isinstance(item, dict) or not {"path", "sha256"} <= set(item):
            raise ValueError("visual evidence contains an invalid file binding")
        if _sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError("visual evidence file hash no longer matches its manifest")
    if not isinstance(panels, list) or not panels:
        raise ValueError("visual evidence lacks typed panel bindings")
    for panel in panels:
        if not isinstance(panel, dict):
            raise ValueError("visual evidence panel binding is invalid")
        _validate_panel(panel, len(images))
        if (
            panel["case_id"] not in provenance["case_ids"]
            or panel["member"] not in provenance["member_indices"]
        ):
            raise ValueError("visual panel is absent from sample provenance")
    return path, payload


def write_visual_review(
    *,
    run_dir: str | Path,
    review_id: str,
    evidence_manifest_path: str | Path,
    inspected_panel_indices: list[int],
    reviewer: str,
    ratings: dict[str, str],
    observations: list[str],
    verdict: str,
) -> Path:
    """Create one non-overwritable JSON+Markdown review beside a run."""
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", review_id) is None:
        raise ValueError("review_id must be a safe non-empty identifier")
    evidence_path, evidence = _load_and_verify_evidence(evidence_manifest_path)
    if set(ratings) != _REQUIRED_RATINGS or any(value not in _RATINGS for value in ratings.values()):
        raise ValueError(
            f"ratings must contain exactly {sorted(_REQUIRED_RATINGS)} with values {sorted(_RATINGS)}"
        )
    if verdict not in _VERDICTS:
        raise ValueError(f"verdict must be one of {sorted(_VERDICTS)}")
    values = set(ratings.values())
    if ("fail" in values and verdict != "fail") or (verdict == "pass" and values != {"pass"}):
        raise ValueError("visual verdict contradicts the component ratings")
    panels = evidence["panel_bindings"]
    if (
        not inspected_panel_indices
        or len(inspected_panel_indices) != len(set(inspected_panel_indices))
        or any(
            not isinstance(index, int) or not 0 <= index < len(panels)
            for index in inspected_panel_indices
        )
    ):
        raise ValueError("inspected_panel_indices must uniquely select evidence-bound panels")
    if (
        not reviewer.strip()
        or not observations
        or any(not isinstance(value, str) or not value.strip() for value in observations)
    ):
        raise ValueError("a visual review requires a reviewer and concrete observations")

    root = Path(run_dir).expanduser().resolve()
    if evidence_path != root and root not in evidence_path.parents:
        raise ValueError("visual evidence manifest must be stored inside the reviewed run")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    review_dir = root / "visual_reviews" / f"{timestamp}_{review_id}"
    if any((root / "visual_reviews").glob(f"*_{review_id}")):
        raise FileExistsError(f"visual review id already exists in this run: {review_id}")
    review_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema_version": 1,
        "scope": "visual_review_only",
        "training_authorized": False,
        "review_id": review_id,
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "reviewer": reviewer,
        "run_dir": str(root),
        "evidence_manifest": {"path": str(evidence_path), "sha256": _sha256(evidence_path)},
        "source_commit": evidence["source_commit"],
        "sample_provenance": evidence["sample_provenance"],
        "checkpoints": evidence["checkpoints"],
        "sample_tensor": evidence["sample_tensor"],
        "images": evidence["images"],
        "inspected_panels": [panels[index] for index in inspected_panel_indices],
        "rendering": evidence["rendering"],
        "ratings": ratings,
        "observations": observations,
        "visual_verdict": verdict,
    }
    _atomic_json(review_dir / "visual_review.json", payload)
    lines = [
        f"# Visual review: {review_id}",
        "",
        f"- Verdict: **{verdict}**",
        f"- Reviewer: {reviewer}",
        "- Checkpoints: "
        + ", ".join(
            f"`{item['path']}` ({item['stage']}/{item['role']})" for item in payload["checkpoints"]
        ),
        f"- Sample tensor: `{payload['sample_tensor']['path']}`",
        f"- Evidence manifest: `{evidence_path}`",
        "- Training authorized: **false**",
        "",
        "## Ratings",
        "",
        *(f"- {key}: {value}" for key, value in ratings.items()),
        "",
        "## Observations",
        "",
        *(f"- {value}" for value in observations),
        "",
    ]
    markdown_path = review_dir / "visual_review.md"
    temporary = markdown_path.with_name(f".{markdown_path.name}.{os.getpid()}.incomplete")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, markdown_path)
    return review_dir


def write_visual_review_from_json(run_dir: str | Path, payload_path: str | Path) -> Path:
    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    if not isinstance(payload, dict):
        raise ValueError("visual review input must be a JSON object")
    return write_visual_review(run_dir=run_dir, **payload)
