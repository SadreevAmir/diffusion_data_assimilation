#!/usr/bin/env python3

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from paper.validate_structured_gaussian_pilot_contract import CONTRACT, ROOT, validate


class StructuredGaussianPilotContractTests(unittest.TestCase):
    def test_current_contract_passes(self) -> None:
        contract = validate()
        self.assertFalse(contract["launch_authorized"])
        self.assertEqual(contract["training_semantics"]["minimum_optimizer_steps"], 2128)
        self.assertEqual(contract["evaluation"]["cases"], 8)
        self.assertEqual(contract["evaluation"]["ensemble_size"], 8)

    def test_identity_drift_fails_closed(self) -> None:
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        changed = next(iter(contract["required_identity_sha256"]))
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary)
            for relative in contract["required_identity_sha256"]:
                target = fixture / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, target)
            contract_target = fixture / CONTRACT.relative_to(ROOT)
            contract_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(CONTRACT, contract_target)
            changed_target = fixture / changed
            changed_target.write_bytes(changed_target.read_bytes() + b"\nidentity drift\n")
            with mock.patch("paper.validate_structured_gaussian_pilot_contract.subprocess.run") as run:
                run.return_value.stdout = contract["science_head"] + "\n"
                with self.assertRaisesRegex(ValueError, "immutable identity mismatch"):
                    validate(fixture)

    def test_launch_authorization_fails_closed(self) -> None:
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        contract["launch_authorized"] = True
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary)
            target = fixture / CONTRACT.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not launch-authorized"):
                validate(fixture)


if __name__ == "__main__":
    unittest.main()
