from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from paper.validate_server_only_manifest import SCHEMA_VERSION, validate_manifest


class ServerOnlyManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.artifact = self.directory / "aggregate_case_mean_metrics.json"
        self.artifact.write_text('[{"num_cases": 40}]\n', encoding="utf-8")
        self.manifest = self.directory / "compact_manifest.json"
        self.experiment = "joint_crossfit_spread_calibration_valid"
        self.payload = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": self.experiment,
            "artifacts": {
                self.artifact.name: hashlib.sha256(self.artifact.read_bytes()).hexdigest()
            },
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_manifest(self) -> None:
        self.manifest.write_text(json.dumps(self.payload), encoding="utf-8")

    def test_exact_sidecar_passes(self) -> None:
        self.write_manifest()
        validate_manifest(self.artifact, self.manifest, self.experiment)

    def test_missing_hash_entry_fails_closed(self) -> None:
        self.payload["artifacts"] = {}
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "hash entry is missing"):
            validate_manifest(self.artifact, self.manifest, self.experiment)

    def test_digest_mismatch_fails_closed(self) -> None:
        self.payload["artifacts"][self.artifact.name] = "0" * 64
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            validate_manifest(self.artifact, self.manifest, self.experiment)

    def test_producer_substitution_fails_closed(self) -> None:
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "producer experiment mismatch"):
            validate_manifest(self.artifact, self.manifest, "substituted_experiment")


if __name__ == "__main__":
    unittest.main()
