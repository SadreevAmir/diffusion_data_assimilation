"""Regression tests for compact-input publication figure generators."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from paper.make_calibration_summary_figure import METRICS as CALIBRATION_METRICS
from paper.make_calibration_summary_figure import load_summary as load_calibration
from paper.make_calibration_summary_figure import make_svg as make_calibration_svg
from paper.make_joint_gate_figure import METHOD, METRICS as GATE_METRICS
from paper.make_joint_gate_figure import load_summary as load_gate
from paper.make_joint_gate_figure import make_svg as make_gate_svg


class PublicationFigureGeneratorTests(unittest.TestCase):
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

    def write_json(self, payload: object) -> Path:
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        with handle:
            json.dump(payload, handle)
        self.addCleanup(Path(handle.name).unlink, missing_ok=True)
        return Path(handle.name)

    def test_calibration_svg_is_deterministic_and_complete(self) -> None:
        row = load_calibration(self.write_json(self.calibration_payload()))
        first = make_calibration_svg(row)
        self.assertEqual(first, make_calibration_svg(row))
        self.assertEqual(first.count('class="raw"'), len(CALIBRATION_METRICS) + 1)
        self.assertEqual(first.count('class="corrected"'), len(CALIBRATION_METRICS) + 1)

    def test_calibration_rejects_negative_and_jointly_zero_metrics(self) -> None:
        for corrected, raw, values in (
            (CALIBRATION_METRICS[0][1], CALIBRATION_METRICS[0][2], (-0.1, 0.1)),
            (CALIBRATION_METRICS[0][1], CALIBRATION_METRICS[0][2], (0.0, 0.0)),
        ):
            payload = self.calibration_payload()
            payload[0][corrected], payload[0][raw] = values
            with self.assertRaises(ValueError):
                load_calibration(self.write_json(payload))

    def test_joint_gate_svg_is_deterministic_and_complete(self) -> None:
        raw, candidate = load_gate(self.write_json(self.gate_payload()))
        first = make_gate_svg(raw, candidate)
        self.assertEqual(first, make_gate_svg(raw, candidate))
        self.assertEqual(first.count('class="candidate"'), len(GATE_METRICS) + 1)
        self.assertEqual(first.count("0.980x"), len(GATE_METRICS))

    def test_joint_gate_rejects_zero_raw_denominator(self) -> None:
        payload = self.gate_payload()
        payload["method_aggregates"][0][GATE_METRICS[0][1]] = 0.0
        with self.assertRaises(ValueError):
            load_gate(self.write_json(payload))


if __name__ == "__main__":
    unittest.main()
