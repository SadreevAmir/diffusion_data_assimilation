from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from paper.test_probabilistic_da_adapter_parity import ProbabilisticDAAdapterParityTests
from paper.validate_probabilistic_da_admission import (
    admit_result_directory,
    directory_sha256,
)


SCRIPT = Path(__file__).with_name("validate_probabilistic_da_admission.py")


class ProbabilisticDAAdmissionTests(ProbabilisticDAAdapterParityTests):
    def test_combined_admission_returns_cli_identity(self) -> None:
        _, root = self.bundle()
        admitted = admit_result_directory(root)
        self.assertEqual(admitted.outcome, "PROBABILISTIC_DA_USEFUL")
        self.assertEqual(admitted.compact_directory_sha256, directory_sha256(root))
        process = subprocess.run(
            [sys.executable, str(SCRIPT), str(root)],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(json.loads(process.stdout), admitted.__dict__)
        self.assertEqual(process.stderr, "")

    def test_each_bundle_component_substitution_during_admission_fails(self) -> None:
        for name in (
            "probabilistic_da_summary.json",
            "probabilistic_da_per_case.csv",
            "probabilistic_da_manifest.json",
        ):
            with self.subTest(name=name):
                _, root = self.bundle()

                def mutate(_root: Path, *, target: str = name) -> str:
                    path = root / target
                    path.write_bytes(path.read_bytes() + b" ")
                    return "PROBABILISTIC_DA_USEFUL"

                with patch(
                    "paper.validate_probabilistic_da_admission.validate_result_directory",
                    side_effect=mutate,
                ):
                    with self.assertRaisesRegex(ValueError, "changed during combined admission"):
                        admit_result_directory(root)

    def test_directory_membership_substitution_during_admission_fails(self) -> None:
        _, root = self.bundle()

        def add_file(_root: Path) -> str:
            (root / "late_debug.json").write_text("{}", encoding="utf-8")
            return "PROBABILISTIC_DA_USEFUL"

        with patch(
            "paper.validate_probabilistic_da_admission.validate_result_directory",
            side_effect=add_file,
        ):
            with self.assertRaisesRegex(ValueError, "exactly three"):
                admit_result_directory(root)


if __name__ == "__main__":
    unittest.main()
