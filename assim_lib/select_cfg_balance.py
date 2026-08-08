from __future__ import annotations

import argparse
import calendar
import csv
import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .compare_3dvar import (
    _aggregate_case_means,
    _json_safe,
    _write_csv,
)
from .compare_3dvar import run as run_comparison
from .config import load_json, resolve_path


def _prepare_output_dir(path: str | Path) -> Path:
    output_dir = Path(path).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"CFG-selection output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _metric_from_aggregate(path: str | Path, region: str, metric: str) -> float:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if row.get("region") == region]
    if len(matches) != 1:
        raise ValueError(f"Expected one aggregate row for region={region!r}, got {len(matches)}")
    if metric not in matches[0]:
        raise KeyError(f"Metric {metric!r} is absent from {path}")
    value = float(matches[0][metric])
    if not math.isfinite(value):
        raise ValueError(f"Selection metric is not finite: {region}/{metric}={value}")
    return value


def _comparison_args(
    *,
    experiment_config: Path,
    mode: str,
    run_dir: str,
    checkpoint_name: str,
    output_dir: Path,
    track_weight: float,
    device: str | None,
    ensemble_size: int | None,
    sample_batch_size: int | None,
    save_ensembles: bool | None,
    start_date: str | None = None,
    end_date: str | None = None,
    expected_num_cases: int | None = None,
    case_stride_days: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        config=str(experiment_config),
        mode=mode,
        run_dir=run_dir,
        checkpoint_name=checkpoint_name,
        output_dir=str(output_dir),
        start_date=start_date,
        end_date=end_date,
        target_hour=None,
        expected_num_cases=expected_num_cases,
        case_stride_days=case_stride_days,
        ensemble_size=ensemble_size,
        sample_batch_size=sample_batch_size,
        num_timesteps=None,
        method=None,
        rtol=None,
        atol=None,
        device=device,
        inference_precision=None,
        seed=None,
        field_protocol=None,
        conditioning_mode=None,
        track_weight=track_weight,
        save_ensembles=save_ensembles,
    )


def _select_result(results: list[dict[str, Any]], direction: str) -> dict[str, Any]:
    if direction == "minimize":
        sign = 1.0
    elif direction == "maximize":
        sign = -1.0
    else:
        raise ValueError(f"Unknown selection direction: {direction!r}")

    def key(row: dict[str, Any]) -> tuple[float, float, float]:
        return (
            sign * row["selection_value"],
            abs(row["track_weight"] - 0.5),
            row["track_weight"],
        )

    return min(results, key=key)


def _month_windows(start: date, end: date) -> list[tuple[int, date, date]]:
    if end < start:
        raise ValueError(f"Monthly test end {end} is before start {start}")
    if start.year != end.year:
        raise ValueError("Monthly CFG selection currently requires one test calendar year")
    windows = []
    for month in range(start.month, end.month + 1):
        month_start = max(start, date(start.year, month, 1))
        last_day = calendar.monthrange(start.year, month)[1]
        month_end = min(end, date(start.year, month, last_day))
        windows.append((month, month_start, month_end))
    return windows


def _read_case_rows(path: str | Path, track_weight: float) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["selected_track_weight"] = track_weight
        row["selected_background_weight"] = 1.0 - track_weight
        row["evaluated_points"] = float(row["evaluated_points"])
    return rows


def _run_monthly(
    *,
    args: argparse.Namespace,
    selection_path: Path,
    selection: dict[str, Any],
    experiment_config: Path,
    weights: list[float],
    validation_mode: str,
    test_mode: str,
    region: str,
    metric: str,
    direction: str,
    checkpoint_name: str,
) -> dict[str, Any]:
    output_dir = _prepare_output_dir(args.output_dir)
    validation_root = output_dir / "validation_monthly"
    test_root = output_dir / "test_monthly"
    validation_root.mkdir()

    test_start = date.fromisoformat(selection.get("test_start_date", "2023-01-01"))
    test_end = date.fromisoformat(selection.get("test_end_date", "2023-07-19"))
    validation_year = int(selection.get("validation_year", 2022))
    validation_stride = int(selection.get("validation_stride_days", 3))
    if validation_stride <= 0:
        raise ValueError("validation_stride_days must be positive")
    windows = _month_windows(test_start, test_end)

    manifest: dict[str, Any] = {
        "status": "running_monthly_validation",
        "started_at": datetime.now().isoformat(),
        "selection_scope": "monthly",
        "selection_config": str(selection_path),
        "experiment_config": str(experiment_config),
        "run_dir": str(Path(args.run_dir).expanduser().resolve()),
        "checkpoint_policy": checkpoint_name,
        "validation_mode": validation_mode,
        "test_mode": test_mode,
        "selection_region": region,
        "selection_metric": metric,
        "direction": direction,
        "track_weights": weights,
        "validation_stride_days": validation_stride,
        "months": [],
    }
    manifest_path = output_dir / "selection.json"
    _write_json(manifest_path, manifest)

    try:
        month_records = []
        for month, month_test_start, month_test_end in windows:
            validation_start = month_test_start.replace(year=validation_year)
            validation_end = month_test_end.replace(year=validation_year)
            validation_cases = len(range(0, (validation_end - validation_start).days + 1, validation_stride))
            month_record: dict[str, Any] = {
                "month": month,
                "validation_start_date": validation_start.isoformat(),
                "validation_end_date": validation_end.isoformat(),
                "validation_num_cases": validation_cases,
                "test_start_date": month_test_start.isoformat(),
                "test_end_date": month_test_end.isoformat(),
                "test_num_cases": (month_test_end - month_test_start).days + 1,
                "validation_results": [],
            }
            manifest["months"].append(month_record)
            month_validation_root = validation_root / f"month_{month:02d}"
            month_validation_root.mkdir()
            for weight in weights:
                tag = f"{weight:.6f}".rstrip("0").rstrip(".").replace(".", "p")
                weight_output = month_validation_root / f"track_weight_{tag}"
                comparison_result = run_comparison(
                    _comparison_args(
                        experiment_config=experiment_config,
                        mode=validation_mode,
                        run_dir=args.run_dir,
                        checkpoint_name=checkpoint_name,
                        output_dir=weight_output,
                        track_weight=weight,
                        device=args.device,
                        ensemble_size=args.validation_ensemble_size,
                        sample_batch_size=args.sample_batch_size,
                        save_ensembles=False,
                        start_date=validation_start.isoformat(),
                        end_date=validation_end.isoformat(),
                        expected_num_cases=validation_cases,
                        case_stride_days=validation_stride,
                    )
                )
                selection_value = _metric_from_aggregate(
                    comparison_result["aggregate_metrics"], region, metric
                )
                month_record["validation_results"].append(
                    {
                        "track_weight": weight,
                        "background_weight": 1.0 - weight,
                        "selection_value": selection_value,
                        "output_dir": comparison_result["output_dir"],
                        "aggregate_metrics": comparison_result["aggregate_metrics"],
                    }
                )
                _write_json(manifest_path, manifest)
            month_record["selected"] = _select_result(month_record["validation_results"], direction)
            month_records.append(month_record)
            _write_json(manifest_path, manifest)

        manifest["status"] = "running_monthly_test"
        selected_weights_path = output_dir / "monthly_selected_weights.csv"
        _write_csv(
            selected_weights_path,
            [
                {
                    "month": int(record["month"]),
                    "validation_start_date": record["validation_start_date"],
                    "validation_end_date": record["validation_end_date"],
                    "validation_num_cases": int(record["validation_num_cases"]),
                    "track_weight": float(record["selected"]["track_weight"]),
                    "background_weight": float(record["selected"]["background_weight"]),
                    "selection_value": float(record["selected"]["selection_value"]),
                }
                for record in month_records
            ],
        )
        manifest["monthly_selected_weights"] = str(selected_weights_path)
        _write_json(manifest_path, manifest)
        test_root.mkdir()
        combined_rows: list[dict[str, Any]] = []
        for month_record in month_records:
            month = int(month_record["month"])
            selected_weight = float(month_record["selected"]["track_weight"])
            month_output = test_root / f"month_{month:02d}"
            test_result = run_comparison(
                _comparison_args(
                    experiment_config=experiment_config,
                    mode=test_mode,
                    run_dir=args.run_dir,
                    checkpoint_name=checkpoint_name,
                    output_dir=month_output,
                    track_weight=selected_weight,
                    device=args.device,
                    ensemble_size=args.test_ensemble_size,
                    sample_batch_size=args.sample_batch_size,
                    save_ensembles=args.save_test_ensembles,
                    start_date=month_record["test_start_date"],
                    end_date=month_record["test_end_date"],
                    expected_num_cases=int(month_record["test_num_cases"]),
                    case_stride_days=1,
                )
            )
            month_record["test_result"] = test_result
            combined_rows.extend(_read_case_rows(test_result["per_case_metrics"], selected_weight))
            _write_json(manifest_path, manifest)

        expected_dates = {
            (test_start + timedelta(days=offset)).isoformat()
            for offset in range((test_end - test_start).days + 1)
        }
        actual_dates = {row["target_date"] for row in combined_rows}
        if actual_dates != expected_dates:
            raise AssertionError("Monthly test date union differs from the requested continuous test window")
        for expected_region in ("full", "track_imitation"):
            region_rows = [row for row in combined_rows if row["region"] == expected_region]
            region_dates = {row["target_date"] for row in region_rows}
            if region_dates != expected_dates or len(region_rows) != len(expected_dates):
                raise AssertionError(
                    f"Monthly test dates are missing or duplicated for region={expected_region!r}"
                )
        full_case_count = len(actual_dates)
        combined_root = output_dir / "test_combined"
        combined_root.mkdir()
        _write_csv(combined_root / "per_case_metrics.csv", combined_rows)
        aggregate_rows = _aggregate_case_means(combined_rows)
        _write_csv(combined_root / "aggregate_case_mean_metrics.csv", aggregate_rows)
        (combined_root / "aggregate_case_mean_metrics.json").write_text(
            json.dumps(_json_safe(aggregate_rows), indent=2) + "\n",
            encoding="utf-8",
        )

        manifest["status"] = "completed"
        manifest["completed_at"] = datetime.now().isoformat()
        manifest["test_combined"] = {
            "num_cases": full_case_count,
            "per_case_metrics": str(combined_root / "per_case_metrics.csv"),
            "aggregate_metrics": str(combined_root / "aggregate_case_mean_metrics.csv"),
        }
        _write_json(manifest_path, manifest)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failed_at"] = datetime.now().isoformat()
        manifest["error"] = str(error)
        _write_json(manifest_path, manifest)
        raise

    selected_weights = {
        f"{record['month']:02d}": float(record["selected"]["track_weight"]) for record in month_records
    }
    result = {
        "output_dir": str(output_dir),
        "selected_track_weights_by_month": selected_weights,
        "selection_metric": f"{region}/{metric}",
        "test_aggregate_metrics": manifest["test_combined"]["aggregate_metrics"],
        "manifest": str(manifest_path),
    }
    print(json.dumps(result, indent=2))
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    selection_path = Path(args.config).expanduser().resolve()
    config = load_json(selection_path)
    selection = config.get("selection", {})
    experiment_config = resolve_path(config["experiment_config"], selection_path.parent).resolve()

    weights = args.track_weights or selection.get("track_weights")
    if not weights:
        raise ValueError("At least one track weight is required")
    weights = [float(weight) for weight in weights]
    if any(not 0.0 <= weight <= 1.0 for weight in weights):
        raise ValueError(f"Track weights must be in [0, 1], got {weights}")
    if len(set(weights)) != len(weights):
        raise ValueError(f"Track weights must be unique, got {weights}")

    validation_mode = args.validation_mode or selection.get("validation_mode", "validation_balanced")
    test_mode = args.test_mode or selection.get("test_mode", "test_strict_cfg")
    region = args.selection_region or selection.get("selection_region", "full")
    metric = args.selection_metric or selection.get("selection_metric", "analysis_mean_rmse")
    direction = args.direction or selection.get("direction", "minimize")
    checkpoint_name = args.checkpoint_name or selection.get("checkpoint_name", "auto")
    selection_scope = args.selection_scope or selection.get("selection_scope", "global")
    if args.validation_ensemble_size is None:
        args.validation_ensemble_size = selection.get("validation_ensemble_size")
    if args.test_ensemble_size is None:
        args.test_ensemble_size = selection.get("test_ensemble_size")
    if args.sample_batch_size is None:
        args.sample_batch_size = selection.get("sample_batch_size")
    if args.save_test_ensembles is None:
        args.save_test_ensembles = selection.get("save_test_ensembles")

    if selection_scope == "monthly":
        return _run_monthly(
            args=args,
            selection_path=selection_path,
            selection=selection,
            experiment_config=experiment_config,
            weights=weights,
            validation_mode=validation_mode,
            test_mode=test_mode,
            region=region,
            metric=metric,
            direction=direction,
            checkpoint_name=checkpoint_name,
        )
    if selection_scope != "global":
        raise ValueError(f"Unknown selection_scope={selection_scope!r}; expected 'global' or 'monthly'")

    output_dir = _prepare_output_dir(args.output_dir)
    validation_root = output_dir / "validation"
    validation_root.mkdir()

    manifest: dict[str, Any] = {
        "status": "running_validation",
        "started_at": datetime.now().isoformat(),
        "selection_scope": "global",
        "selection_config": str(selection_path),
        "experiment_config": str(experiment_config),
        "run_dir": str(Path(args.run_dir).expanduser().resolve()),
        "checkpoint_policy": checkpoint_name,
        "validation_mode": validation_mode,
        "test_mode": test_mode,
        "selection_region": region,
        "selection_metric": metric,
        "direction": direction,
        "track_weights": weights,
        "validation_results": [],
    }
    manifest_path = output_dir / "selection.json"
    _write_json(manifest_path, manifest)

    try:
        results = []
        for weight in weights:
            tag = f"{weight:.6f}".rstrip("0").rstrip(".").replace(".", "p")
            weight_output = validation_root / f"track_weight_{tag}"
            comparison_result = run_comparison(
                _comparison_args(
                    experiment_config=experiment_config,
                    mode=validation_mode,
                    run_dir=args.run_dir,
                    checkpoint_name=checkpoint_name,
                    output_dir=weight_output,
                    track_weight=weight,
                    device=args.device,
                    ensemble_size=args.validation_ensemble_size,
                    sample_batch_size=args.sample_batch_size,
                    save_ensembles=False,
                )
            )
            selection_value = _metric_from_aggregate(comparison_result["aggregate_metrics"], region, metric)
            result = {
                "track_weight": weight,
                "background_weight": 1.0 - weight,
                "selection_value": selection_value,
                "output_dir": comparison_result["output_dir"],
                "aggregate_metrics": comparison_result["aggregate_metrics"],
            }
            results.append(result)
            manifest["validation_results"] = results
            _write_json(manifest_path, manifest)

        selected = _select_result(results, direction)
        selected_weight = float(selected["track_weight"])
        manifest["selected"] = selected
        manifest["status"] = "running_test"
        _write_json(manifest_path, manifest)

        test_output = output_dir / "test_selected"
        test_result = run_comparison(
            _comparison_args(
                experiment_config=experiment_config,
                mode=test_mode,
                run_dir=args.run_dir,
                checkpoint_name=checkpoint_name,
                output_dir=test_output,
                track_weight=selected_weight,
                device=args.device,
                ensemble_size=args.test_ensemble_size,
                sample_batch_size=args.sample_batch_size,
                save_ensembles=args.save_test_ensembles,
            )
        )
        manifest["status"] = "completed"
        manifest["completed_at"] = datetime.now().isoformat()
        manifest["test_result"] = test_result
        _write_json(manifest_path, manifest)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failed_at"] = datetime.now().isoformat()
        manifest["error"] = str(error)
        _write_json(manifest_path, manifest)
        raise

    result = {
        "output_dir": str(output_dir),
        "selected_track_weight": selected_weight,
        "selected_background_weight": 1.0 - selected_weight,
        "selection_metric": f"{region}/{metric}",
        "selection_value": selected["selection_value"],
        "test_aggregate_metrics": test_result["aggregate_metrics"],
        "manifest": str(manifest_path),
    }
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a background/track CFG balance on validation, then evaluate it on test."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--selection-scope", choices=("global", "monthly"), default=None)
    parser.add_argument("--validation-mode", default=None)
    parser.add_argument("--test-mode", default=None)
    parser.add_argument("--track-weights", type=float, nargs="+", default=None)
    parser.add_argument("--selection-region", default=None)
    parser.add_argument("--selection-metric", default=None)
    parser.add_argument("--direction", choices=("minimize", "maximize"), default=None)
    parser.add_argument("--validation-ensemble-size", type=int, default=None)
    parser.add_argument("--test-ensemble-size", type=int, default=None)
    parser.add_argument("--sample-batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--save-test-ensembles",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
