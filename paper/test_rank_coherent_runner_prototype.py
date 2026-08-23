#!/usr/bin/env python3
"""Synthetic fail-closed tests for the rank-coherent CPU prototype."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "rank_coherent_runner_prototype.py"
SOURCE = "joint_full_condition_validation_2022"


def synthetic_source():
    rows = columns = 5
    raw = []
    truth = []
    for case in range(40):
        members = []
        for member in range(10):
            field = []
            for row in range(rows):
                values = []
                for column in range(columns):
                    base = (.035 + .004 * case + .025 * (case % 5)
                            + .017 * row + .009 * column
                            + .00015 * case * row * row
                            + .00009 * (case % 7) * column * column)
                    anomaly = (member - 4.5) * (.003 + .0002 * ((row + column + case) % 4))
                    values.append(max(0.0, min(1.0, base + anomaly)))
                field.append(values)
            members.append(field)
        raw.append(members)
        truth.append([[max(0.0, min(1.0, .035 + .004 * case
                                    + .025 * (case % 5) + .017 * row
                                    + .009 * column + .00015 * case * row * row
                                    + .00009 * (case % 7) * column * column
                                    + .006 * ((case % 3) - 1)))
                       for column in range(columns)] for row in range(rows)])
    return {
        "schema_version": "rank-coherent-prototype-v1",
        "source_experiment": SOURCE,
        "dataset_split": "valid",
        "start_date": "2022-01-01",
        "end_date": "2022-07-15",
        "case_stride": 5,
        "artifact_policy": "summary_only",
        "raw_ensemble": raw,
        "truth": truth,
        "spatial_weights": [[1.0 + .01 * row for _ in range(columns)] for row in range(rows)],
        "truth_rank_tie_uniforms": [(case * .6180339887498949) % 1 for case in range(40)],
        "frozen_thresholds": {
            "boundary_mass_absolute_tolerance": 1.0,
            "member_semivariogram_absolute_tolerance": 1.0,
            "rank_l1_absolute_maximum": 1.0,
            "coverage_absolute_error_maximum": 1.0,
        },
    }


class PrototypeTests(unittest.TestCase):
    def invoke(self, source, extra_args=()):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        source_path = root / "source.json"
        source_path.write_text(json.dumps(source), encoding="utf-8")
        output = root / "output"
        environment = os.environ.copy()
        environment["RANK_COHERENT_SOURCE_JSON"] = str(source_path)
        environment["RANK_COHERENT_OUTPUT_DIR"] = str(output)
        process = subprocess.run(
            [sys.executable, str(RUNNER), "--source-experiment", SOURCE, *extra_args],
            env=environment, text=True, capture_output=True, check=False,
        )
        return temporary, output, process

    def test_exact_four_compact_outputs_and_no_arrays(self):
        temporary, output, process = self.invoke(synthetic_source())
        self.addCleanup(temporary.cleanup)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual({path.name for path in output.iterdir()}, {
            "run_status.json", "aggregate_case_mean_metrics.json",
            "per_case_metrics.csv", "metadata.json",
        })
        aggregate = json.loads((output / "aggregate_case_mean_metrics.json").read_text())
        self.assertEqual(len(aggregate["fold_selections"]), 5)
        self.assertLessEqual(aggregate["projection_diagnostics"]["maximum_mean_error"], 1e-10)
        self.assertFalse(aggregate["gate"]["overall_eligible"])
        self.assertFalse(aggregate["gate_inputs"]["operational"]["trusted_full_gate_integrated"])
        self.assertNotIn("raw_ensemble", json.dumps(aggregate))
        rows = (output / "per_case_metrics.csv").read_text().splitlines()
        self.assertEqual(len(rows), 81)

    def test_rejects_cli_knob(self):
        temporary, _, process = self.invoke(synthetic_source(), ("--alpha", "1.0"))
        self.addCleanup(temporary.cleanup)
        self.assertNotEqual(process.returncode, 0)

    def test_rejects_envelope_drift(self):
        source = synthetic_source(); source["raw_ensemble"].pop(); source["truth"].pop(); source["truth_rank_tie_uniforms"].pop()
        temporary, _, process = self.invoke(source)
        self.addCleanup(temporary.cleanup)
        self.assertEqual(process.returncode, 2)

    def test_rejects_nonfinite_source(self):
        source = synthetic_source(); source["raw_ensemble"][0][0][0][0] = float("nan")
        temporary, _, process = self.invoke(source)
        self.addCleanup(temporary.cleanup)
        self.assertEqual(process.returncode, 2)

    def test_rejects_unsealed_threshold_key(self):
        source = synthetic_source(); source["frozen_thresholds"]["posthoc"] = 1.0
        temporary, _, process = self.invoke(source)
        self.addCleanup(temporary.cleanup)
        self.assertEqual(process.returncode, 2)


if __name__ == "__main__":
    unittest.main()
