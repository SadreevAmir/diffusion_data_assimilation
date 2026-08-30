#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from paper.validate_raw_member_reweighting_review import (
    CONTRACT,
    REQUIRED_KEYS,
    build_admission_payload,
    load_and_validate,
    validate_record,
    validate_semantic_parity,
    validate_synthetic_result,
)
from paper.raw_member_reweighting_runner import synthetic_result as runner_synthetic_result


REFERENCE = Path(__file__).with_name("raw_member_reweighting_reference.py")
RUNNER = Path(__file__).with_name("raw_member_reweighting_runner.py")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_record(runner: Path, synthetic: Path) -> dict[str, object]:
    return {
        "reviewed_mode": "validation_reviewed_raw_member_reweighting",
        "publication_commit": "a" * 40,
        "runner_sha256": digest(runner),
        "contract_sha256": digest(CONTRACT),
        "synthetic_result_sha256": digest(synthetic),
        "test_command": "python -m unittest paper.test_runner",
        "test_sentinel": "raw_member_reweighting_runner=PASS",
        "decision_bearing_validation": "PASS",
        "deviations": [],
    }


def write_valid_synthetic(path: Path) -> None:
    path.write_text(
        json.dumps(runner_synthetic_result(), sort_keys=True) + "\n",
        encoding="utf-8",
    )


class ReviewAdmissionTests(unittest.TestCase):
    def test_reference_alone_is_not_an_admission_complete_runner(self):
        with self.assertRaisesRegex(ValueError, "lacks callable purged_folds"):
            validate_semantic_parity(REFERENCE)

    def test_separate_runner_satisfies_semantic_surface(self):
        validate_semantic_parity(RUNNER)

    def test_oracle_drift_fails_closed(self):
        source = RUNNER.read_text(encoding="utf-8").replace(
            "from .raw_member_reweighting_reference import (",
            "from paper.raw_member_reweighting_reference import (",
        ).replace(
            "return positions\n\n\ndef selection_diagnostics",
            "return list(reversed(positions))\n\n\ndef selection_diagnostics",
        )
        # Override the imported helper inside the reviewed runner copy so the
        # mutation exercises downstream oracle parity, not module discovery.
        source += "\nconstruct_selection = lambda ranks, means: {**__import__('paper.raw_member_reweighting_reference', fromlist=['construct_selection']).construct_selection(ranks, means), 'selected_rank_positions': list(reversed(__import__('paper.raw_member_reweighting_reference', fromlist=['construct_selection']).construct_selection(ranks, means)['selected_rank_positions']))}\n"
        with tempfile.TemporaryDirectory() as temporary:
            runner = Path(temporary) / "runner.py"
            runner.write_text(source, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "diverges"):
                validate_semantic_parity(runner)

    def test_fold_or_analog_semantic_drift_fails_closed(self):
        source = RUNNER.read_text(encoding="utf-8")
        mutations = (
            (
                "PURGE = 3",
                "PURGE = 2",
                "folds or non-circular purge diverge",
            ),
            (
                "sorted(distances)[:MEMBERS]",
                "sorted(distances, reverse=True)[:MEMBERS]",
                "analog selection, scaling or tie order diverges",
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            for index, (old, new, message) in enumerate(mutations):
                runner = Path(temporary) / f"runner_{index}.py"
                runner.write_text(source.replace(old, new), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    validate_semantic_parity(runner)

    def test_exact_record_and_artifact_bindings(self):
        with tempfile.TemporaryDirectory() as temporary:
            synthetic = Path(temporary) / "synthetic.json"
            write_valid_synthetic(synthetic)
            record = valid_record(RUNNER, synthetic)
            self.assertEqual(
                validate_record(record, RUNNER, synthetic), record["reviewed_mode"]
            )
            for key in ("runner_sha256", "contract_sha256", "synthetic_result_sha256"):
                bad = dict(record)
                bad[key] = "0" * 64
                with self.assertRaisesRegex(ValueError, key):
                    validate_record(bad, RUNNER, synthetic)

    def test_every_placeholder_and_nonpass_state_is_no_go(self):
        with tempfile.TemporaryDirectory() as temporary:
            synthetic = Path(temporary) / "synthetic.json"
            write_valid_synthetic(synthetic)
            good = valid_record(RUNNER, synthetic)
            mutations = (
                ("reviewed_mode", "<literal implemented trusted mode>"),
                ("publication_commit", None),
                ("test_command", "<exact command>"),
                ("test_sentinel", ""),
                ("decision_bearing_validation", "FAIL"),
                ("deviations", ["waiver"]),
            )
            for key, value in mutations:
                bad = dict(good)
                bad[key] = value
                with self.assertRaises((ValueError, TypeError), msg=key):
                    validate_record(bad, RUNNER, synthetic)
            bad = dict(good)
            bad["extra"] = True
            self.assertNotEqual(set(bad), REQUIRED_KEYS)
            with self.assertRaisesRegex(ValueError, "exact-schema"):
                validate_record(bad, RUNNER, synthetic)

    def test_combined_admission_returns_only_bound_literal_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            synthetic = root / "synthetic.json"
            write_valid_synthetic(synthetic)
            record_path = root / "review.json"
            record_path.write_text(
                json.dumps(valid_record(RUNNER, synthetic), sort_keys=True),
                encoding="utf-8",
            )
            self.assertEqual(
                load_and_validate(record_path, RUNNER, synthetic),
                "validation_reviewed_raw_member_reweighting",
            )

    def test_admission_payload_carries_every_verified_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            synthetic = root / "synthetic.json"
            write_valid_synthetic(synthetic)
            record_path = root / "review.json"
            record = valid_record(RUNNER, synthetic)
            record_path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
            payload = build_admission_payload(record_path, RUNNER, synthetic)
            self.assertEqual(payload["admission"], "GO")
            self.assertEqual(payload["reviewed_mode"], record["reviewed_mode"])
            self.assertEqual(payload["review_record_sha256"], digest(record_path))
            for key in (
                "publication_commit",
                "runner_sha256",
                "contract_sha256",
                "synthetic_result_sha256",
            ):
                self.assertEqual(payload[key], record[key])

    def test_synthetic_result_must_match_runner_and_remain_non_decision_bearing(self):
        with tempfile.TemporaryDirectory() as temporary:
            synthetic = Path(temporary) / "synthetic.json"
            write_valid_synthetic(synthetic)
            validate_synthetic_result(RUNNER, synthetic)
            for mutation in (
                {"decision_bearing": True},
                {"project_data_metrics_emitted": True},
                {"unexpected_metric": 0.123},
            ):
                bad = runner_synthetic_result()
                bad.update(mutation)
                synthetic.write_text(json.dumps(bad), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "synthetic result"):
                    validate_synthetic_result(RUNNER, synthetic)

    def test_synthetic_result_must_be_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            synthetic = Path(temporary) / "synthetic.json"
            synthetic.write_text("not-json", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "UTF-8 JSON"):
                validate_synthetic_result(RUNNER, synthetic)


if __name__ == "__main__":
    unittest.main()
