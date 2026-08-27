"""End-to-end fail-closed tests for server-only publication consumers."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from paper.make_calibration_summary_figure import (
    EXPECTED_EXPERIMENT as CALIBRATION_EXPERIMENT,
)
from paper.make_calibration_summary_figure import METRICS as CALIBRATION_METRICS
from paper.make_case_level_artifacts import EXPECTED_LONG_FORM_EXPERIMENT
from paper.make_joint_gate_figure import EXPECTED_EXPERIMENT as GATE_EXPERIMENT
from paper.make_joint_gate_figure import METHOD, METRICS as GATE_METRICS
from paper.validate_server_only_manifest import SCHEMA_VERSION


ROOT = Path(__file__).resolve().parents[1]


class ServerOnlyConsumerCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_json(self, name: str, payload: object) -> Path:
        path = self.directory / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def write_manifest(self, artifact: Path, experiment: str, *, valid: bool) -> Path:
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest() if valid else "0" * 64
        return self.write_json(
            f"{artifact.stem}_manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "experiment_id": experiment,
                "artifacts": {artifact.name: digest},
            },
        )

    def calibration_payload(self) -> list[dict[str, float]]:
        row = {"num_cases": 40.0}
        for index, (_, corrected, raw, _) in enumerate(CALIBRATION_METRICS, start=1):
            row[raw] = 0.1 * index
            row[corrected] = 0.09 * index
        return [row]

    def gate_payload(self) -> dict[str, object]:
        raw = {key: 0.1 * index for index, (_, key, _) in enumerate(GATE_METRICS, start=1)}
        candidate = {key: value * 0.98 for key, value in raw.items()}
        return {
            "gate": {"candidate_method": METHOD, "overall_eligible": False},
            "method_aggregates": [
                {"method": "raw", "region": "full", "num_cases": 40, **raw},
                {"method": METHOD, "region": "full", "num_cases": 40, **candidate},
            ],
        }

    def malformed_long_form(self) -> Path:
        path = self.directory / "per_case_metrics.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("target_date", "fold", "method"))
            writer.writeheader()
            writer.writerow({"target_date": "2022-01-01", "fold": 0, "method": "raw"})
        return path

    def assert_fails_without_outputs(self, command: list[str], outputs: tuple[Path, ...]) -> str:
        result = subprocess.run(
            [sys.executable, *command],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0, result.stdout)
        for output in outputs:
            self.assertFalse(output.exists(), f"partial output remains: {output}")
        return result.stderr

    def test_calibration_cli_rejects_manifest_then_payload_without_svg(self) -> None:
        artifact = self.write_json("aggregate_case_mean_metrics.json", self.calibration_payload())
        output = self.directory / "nested" / "calibration.svg"
        invalid_manifest = self.write_manifest(artifact, CALIBRATION_EXPERIMENT, valid=False)
        stderr = self.assert_fails_without_outputs(
            ["paper/make_calibration_summary_figure.py", str(artifact), str(output), "--manifest", str(invalid_manifest)],
            (output,),
        )
        self.assertIn("SHA-256 mismatch", stderr)
        artifact.write_text("{}", encoding="utf-8")
        valid_manifest = self.write_manifest(artifact, CALIBRATION_EXPERIMENT, valid=True)
        self.assert_fails_without_outputs(
            ["paper/make_calibration_summary_figure.py", str(artifact), str(output), "--manifest", str(valid_manifest)],
            (output,),
        )

    def test_joint_gate_cli_rejects_manifest_then_payload_without_svg(self) -> None:
        artifact = self.write_json("aggregate_case_mean_metrics.json", self.gate_payload())
        output = self.directory / "nested" / "gate.svg"
        invalid_manifest = self.write_manifest(artifact, GATE_EXPERIMENT, valid=False)
        stderr = self.assert_fails_without_outputs(
            ["paper/make_joint_gate_figure.py", str(artifact), str(output), "--manifest", str(invalid_manifest)],
            (output,),
        )
        self.assertIn("SHA-256 mismatch", stderr)
        artifact.write_text("{}", encoding="utf-8")
        valid_manifest = self.write_manifest(artifact, GATE_EXPERIMENT, valid=True)
        self.assert_fails_without_outputs(
            ["paper/make_joint_gate_figure.py", str(artifact), str(output), "--manifest", str(valid_manifest)],
            (output,),
        )

    def test_case_level_cli_rejects_manifest_then_payload_without_any_output(self) -> None:
        artifact = self.malformed_long_form()
        summary = self.directory / "nested" / "summary.json"
        figure = self.directory / "nested" / "cases.svg"
        base = [
            "paper/make_case_level_artifacts.py", str(artifact), "--long-form",
            "--date-column", "target_date", "--block-length", "4",
            "--summary", str(summary), "--figure", str(figure),
        ]
        invalid_manifest = self.write_manifest(artifact, EXPECTED_LONG_FORM_EXPERIMENT, valid=False)
        stderr = self.assert_fails_without_outputs(
            [*base, "--manifest", str(invalid_manifest)], (summary, figure)
        )
        self.assertIn("SHA-256 mismatch", stderr)
        valid_manifest = self.write_manifest(artifact, EXPECTED_LONG_FORM_EXPERIMENT, valid=True)
        self.assert_fails_without_outputs(
            [*base, "--manifest", str(valid_manifest)], (summary, figure)
        )


if __name__ == "__main__":
    unittest.main()
