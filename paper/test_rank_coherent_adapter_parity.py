#!/usr/bin/env python3
"""Executable directory-level parity fixtures for the trusted adapter boundary."""
from __future__ import annotations
import json, sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rank_coherent_adapter_parity import validate_result_directory
from test_rank_coherent_runner_prototype import PrototypeTests, synthetic_source

class AdapterParityTests(PrototypeTests):
    def test_prototype_directory_has_nondecision_parity(self):
        temporary, output, process = self.invoke(synthetic_source()); self.addCleanup(temporary.cleanup)
        self.assertEqual(process.returncode, 0, process.stderr)
        validate_result_directory(output, decision_bearing=False)
    def test_rejects_status_gate_disagreement(self):
        temporary, output, process = self.invoke(synthetic_source()); self.addCleanup(temporary.cleanup)
        self.assertEqual(process.returncode, 0, process.stderr)
        path = output / "run_status.json"; status = json.loads(path.read_text()); status["overall_eligible"] = True; path.write_text(json.dumps(status))
        with self.assertRaisesRegex(ValueError, "status and compact gate"): validate_result_directory(output, decision_bearing=False)
    def test_rejects_csv_aggregate_disagreement(self):
        temporary, output, process = self.invoke(synthetic_source()); self.addCleanup(temporary.cleanup)
        self.assertEqual(process.returncode, 0, process.stderr)
        path = output / "per_case_metrics.csv"; text = path.read_text(); path.write_text(text.replace("0,raw,", "0,raw,9", 1))
        with self.assertRaisesRegex(ValueError, "per-case and aggregate"): validate_result_directory(output, decision_bearing=False)

if __name__ == "__main__": unittest.main()
