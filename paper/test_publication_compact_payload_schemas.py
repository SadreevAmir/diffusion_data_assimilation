"""Negative fixtures for fail-closed server-only compact payload consumers."""

from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from paper.make_case_level_artifacts import LONG_FORM_METRICS, read_long_form_cases


RAW_METHOD = "raw"
CORRECTED_METHOD = "crossfit_global_spread_fair_crps"
OTHER_METHODS = ("crossfit_crps", "crossfit_rank")


class CompactPayloadSchemaTests(unittest.TestCase):
    def make_rows(self) -> list[dict[str, object]]:
        rows = []
        first = date(2022, 1, 1)
        for case_index in range(40):
            target_date = (first + timedelta(days=case_index)).isoformat()
            for method in (RAW_METHOD, CORRECTED_METHOD, *OTHER_METHODS):
                row: dict[str, object] = {
                    "target_date": target_date,
                    "fold": case_index // 8,
                    "method": method,
                }
                row.update({metric: 0.1 for metric in LONG_FORM_METRICS})
                rows.append(row)
        return rows

    def write_csv(self, rows: list[dict[str, object]]) -> Path:
        handle = tempfile.NamedTemporaryFile(mode="w", newline="", suffix=".csv", delete=False)
        with handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        path = Path(handle.name)
        self.addCleanup(path.unlink, missing_ok=True)
        return path

    def load(self, rows: list[dict[str, object]]) -> None:
        read_long_form_cases(
            self.write_csv(rows),
            date_column="target_date",
            method_column="method",
            raw_method=RAW_METHOD,
            corrected_method=CORRECTED_METHOD,
        )

    def test_exact_long_form_schema_passes(self) -> None:
        self.assertEqual(len(self.load(self.make_rows()) or []), 0)

    def test_wrong_total_row_count_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 160 rows"):
            self.load(self.make_rows()[:-1])

    def test_duplicate_method_date_identity_fails_closed(self) -> None:
        rows = self.make_rows()
        rows[-1] = dict(rows[0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.load(rows)

    def test_missing_exact_candidate_fails_closed(self) -> None:
        rows = self.make_rows()
        for row in rows:
            if row["method"] == CORRECTED_METHOD:
                row["method"] = "substituted_candidate"
                break
        with self.assertRaisesRegex(ValueError, "two named methods"):
            self.load(rows)


if __name__ == "__main__":
    unittest.main()
