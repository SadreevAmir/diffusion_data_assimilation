from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from paper.probabilistic_da_adapter_parity import validate_result_directory
from paper.probabilistic_da_contract_oracle import CASE_COLUMNS
from paper.test_probabilistic_da_contract_oracle import fixture


class ProbabilisticDAAdapterParityTests(unittest.TestCase):
    def bundle(self) -> tuple[tempfile.TemporaryDirectory, Path]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        summary = {
            "case_count": 40, "letkf_fair_crps": 1.1,
            "raw_learned_joint_fair_crps": 1.0, "rank_uniformity_pass": True,
            "letkf_mean_rmse": 1.02, "var3d_mean_rmse": 1.0,
            "outcome": "PROBABILISTIC_DA_USEFUL",
        }
        summary_path = root / "probabilistic_da_summary.json"
        summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
        case_path = root / "probabilistic_da_per_case.csv"
        with case_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CASE_COLUMNS)
            writer.writeheader()
            for case_index in range(40):
                for method in ("LETKF", "raw_learned_joint", "3D-Var"):
                    row = {metric: 0.1 for metric in CASE_COLUMNS[4:]}
                    row.update({"case_index": case_index, "target_date": f"sealed-case-{case_index:02d}", "fold": case_index // 8, "method": method})
                    writer.writerow(row)
        manifest = fixture()
        manifest["artifact_hashes"] = {
            summary_path.name: hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            case_path.name: hashlib.sha256(case_path.read_bytes()).hexdigest(),
        }
        (root / "probabilistic_da_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return temporary, root

    def test_complete_bundle_is_admitted(self):
        _, root = self.bundle()
        self.assertEqual(validate_result_directory(root), "PROBABILISTIC_DA_USEFUL")

    def test_content_change_fails_artifact_hash(self):
        _, root = self.bundle()
        path = root / "probabilistic_da_per_case.csv"
        path.write_text(path.read_text().replace("0.1", "0.2", 1))
        with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
            validate_result_directory(root)

    def test_duplicate_row_fails_after_resealed_hash(self):
        _, root = self.bundle()
        case_path = root / "probabilistic_da_per_case.csv"
        lines = case_path.read_text().splitlines()
        lines[-1] = lines[1]
        case_path.write_text("\n".join(lines) + "\n")
        manifest_path = root / "probabilistic_da_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifact_hashes"][case_path.name] = hashlib.sha256(case_path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_result_directory(root)

    def test_claimed_outcome_mismatch_fails_after_resealed_hash(self):
        _, root = self.bundle()
        summary_path = root / "probabilistic_da_summary.json"
        summary = json.loads(summary_path.read_text())
        summary["outcome"] = "PROBABILISTIC_DA_NEGATIVE"
        summary_path.write_text(json.dumps(summary, sort_keys=True))
        manifest_path = root / "probabilistic_da_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifact_hashes"][summary_path.name] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "recomputed decision"):
            validate_result_directory(root)

    def test_extra_file_fails_atomic_boundary(self):
        _, root = self.bundle()
        (root / "debug.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "exactly three"):
            validate_result_directory(root)


if __name__ == "__main__":
    unittest.main()
