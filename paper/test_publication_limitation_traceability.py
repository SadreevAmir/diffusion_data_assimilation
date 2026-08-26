from __future__ import annotations

import unittest
from pathlib import Path

from paper.check_publication_artifacts import validate_limitation_traceability


PAPER = Path(__file__).with_name("PAPER_DRAFT.md")
LEDGER = Path(__file__).with_name("CLAIM_LEDGER.md")
TRACEABILITY = Path(__file__).with_name("LIMITATION_TRACEABILITY.md")


class LimitationTraceabilityTests(unittest.TestCase):
    def validate(self, traceability: str) -> None:
        validate_limitation_traceability(
            PAPER.read_text(encoding="utf-8"),
            LEDGER.read_text(encoding="utf-8"),
            traceability,
        )

    def test_current_traceability_passes(self) -> None:
        self.validate(TRACEABILITY.read_text(encoding="utf-8"))

    def test_missing_limitation_fails(self) -> None:
        text = TRACEABILITY.read_text(encoding="utf-8")
        text = "\n".join(line for line in text.splitlines() if not line.startswith("| L8 |"))
        with self.assertRaises(ValueError):
            self.validate(text)

    def test_wrong_claim_mapping_fails(self) -> None:
        text = TRACEABILITY.read_text(encoding="utf-8").replace(
            "| `C2` | Direct observation-system limitation |",
            "| `C1` | Direct observation-system limitation |",
            1,
        )
        with self.assertRaises(ValueError):
            self.validate(text)

    def test_missing_unknown_evidence_record_fails(self) -> None:
        text = TRACEABILITY.read_text(encoding="utf-8").replace(
            "Explicit evidence absence (`C10` is `Unknown`)",
            "Explicit evidence absence",
            1,
        )
        with self.assertRaises(ValueError):
            self.validate(text)


if __name__ == "__main__":
    unittest.main()
