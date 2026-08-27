from __future__ import annotations

import unittest
from pathlib import Path

from paper.check_publication_artifacts import validate_reference_traceability


TRACEABILITY = Path(__file__).with_name("REFERENCE_TRACEABILITY.md")


class ReferenceTraceabilityTests(unittest.TestCase):
    def test_current_traceability_passes(self) -> None:
        validate_reference_traceability(TRACEABILITY.read_text(encoding="utf-8"))

    def test_missing_reference_fails(self) -> None:
        text = TRACEABILITY.read_text(encoding="utf-8")
        text = "\n".join(line for line in text.splitlines() if not line.startswith("| 8 |"))
        with self.assertRaises(ValueError):
            validate_reference_traceability(text)

    def test_changed_identity_fails(self) -> None:
        text = TRACEABILITY.read_text(encoding="utf-8").replace(
            "10.1002/qj.2270", "10.1002/qj.invalid", 1
        )
        with self.assertRaises(ValueError):
            validate_reference_traceability(text)

    def test_missing_empirical_boundary_fails(self) -> None:
        text = TRACEABILITY.read_text(encoding="utf-8").replace(
            "does not support project-specific empirical values",
            "is separate from empirical values",
        )
        with self.assertRaises(ValueError):
            validate_reference_traceability(text)


if __name__ == "__main__":
    unittest.main()
