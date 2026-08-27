from __future__ import annotations
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from paper.reconcile_probabilistic_da_admission import reconcile_admission

SCRIPT = Path(__file__).with_name("reconcile_probabilistic_da_admission.py")
DIGEST = "a" * 64

class ProbabilisticDAAdmissionConsumerTests(unittest.TestCase):
    def test_exact_digest_releases_outcome(self) -> None:
        payload = {"outcome": "PROBABILISTIC_DA_USEFUL", "compact_directory_sha256": DIGEST}
        self.assertEqual(reconcile_admission(payload, DIGEST), "PROBABILISTIC_DA_USEFUL")

    def test_missing_digest_rejects_otherwise_valid_outcome(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly outcome"):
            reconcile_admission({"outcome": "PROBABILISTIC_DA_USEFUL"}, DIGEST)

    def test_mismatched_digest_rejects_otherwise_valid_outcome(self) -> None:
        payload = {"outcome": "PROBABILISTIC_DA_NEGATIVE", "compact_directory_sha256": "b" * 64}
        with self.assertRaisesRegex(ValueError, "does not match"):
            reconcile_admission(payload, DIGEST)

    def test_cli_requires_controller_supplied_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            admission = Path(temporary) / "admission.json"
            admission.write_text(json.dumps({"outcome": "PROBABILISTIC_DA_USEFUL"}), encoding="utf-8")
            process = subprocess.run([sys.executable, str(SCRIPT), str(admission), DIGEST], text=True, capture_output=True, check=False)
        self.assertNotEqual(process.returncode, 0)
        self.assertNotIn("PROBABILISTIC_DA_USEFUL\n", process.stdout)

if __name__ == "__main__":
    unittest.main()
