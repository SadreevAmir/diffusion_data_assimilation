#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from paper.validate_structured_gaussian_pilot_contract import (
    CONTRACT,
    ROOT,
    _validate_memory_admission,
    validate,
)


class StructuredGaussianPilotContractTests(unittest.TestCase):
    def test_current_contract_passes(self) -> None:
        contract = validate()
        self.assertFalse(contract["launch_authorized"])
        self.assertEqual(contract["training_semantics"]["minimum_optimizer_steps"], 2128)
        self.assertEqual(contract["evaluation"]["cases"], 8)
        self.assertEqual(contract["evaluation"]["ensemble_size"], 8)
        self.assertEqual(
            contract["base_science_head"],
            "62aafc4eaea483fc598d151c032dfb8eaaa00851",
        )
        self.assertTrue(contract["validated_current_head"])

    def test_git_validation_ignores_poisoned_repository_environment(self) -> None:
        poisoned = {
            "GIT_DIR": "/definitely/not/the/publication/repository/.git",
            "GIT_WORK_TREE": "/definitely/not/the/publication/repository",
            "GIT_INDEX_FILE": "/definitely/not/the/publication/repository/index",
            "GIT_COMMON_DIR": "/definitely/not/the/publication/repository/common",
        }
        with mock.patch.dict(os.environ, poisoned, clear=False):
            contract = validate()
        self.assertTrue(contract["validated_current_head"])

    def test_later_unrelated_changes_do_not_invalidate_frozen_identities(self) -> None:
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        required = set(contract["required_changes_from_base"])
        with mock.patch(
            "paper.validate_structured_gaussian_pilot_contract._git_output",
            side_effect=["f" * 40, "\n".join(sorted(required | {"paper/later_work.md"})), ""],
        ), mock.patch("subprocess.run") as run:
            run.return_value.returncode = 0
            head = __import__(
                "paper.validate_structured_gaussian_pilot_contract",
                fromlist=["_validate_git_contract"],
            )._validate_git_contract(ROOT, contract)
        self.assertEqual(head, "f" * 40)

    def test_identity_drift_fails_closed(self) -> None:
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        changed = next(iter(contract["required_identity_sha256"]))
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary)
            for relative in contract["required_identity_sha256"]:
                target = fixture / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / relative).read_bytes())
            contract_target = fixture / CONTRACT.relative_to(ROOT)
            contract_target.parent.mkdir(parents=True, exist_ok=True)
            contract_target.write_bytes(CONTRACT.read_bytes())
            changed_target = fixture / changed
            changed_target.write_bytes(changed_target.read_bytes() + b"\nidentity drift\n")
            with self.assertRaisesRegex(ValueError, "immutable identity mismatch"):
                validate(fixture, validate_git=False)

    def test_launch_authorization_fails_closed(self) -> None:
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        contract["launch_authorized"] = True
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary)
            target = fixture / CONTRACT.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not launch-authorized"):
                validate(fixture, validate_git=False)

    def test_boundary_stop_go_cannot_be_removed(self) -> None:
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        contract["quantitative_stop_go"].pop("boundary")
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary)
            for relative in contract["required_identity_sha256"]:
                target = fixture / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / relative).read_bytes())
            contract_target = fixture / CONTRACT.relative_to(ROOT)
            contract_target.parent.mkdir(parents=True, exist_ok=True)
            contract_target.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "boundary stop/go"):
                validate(fixture, validate_git=False)

    def test_memory_admission_requires_exact_hash_and_production_evidence(self) -> None:
        result = {
            "schema_version": "structured_gaussian_memory_admission_v1",
            "status": "passed",
            "purpose": "execution_memory_only_not_model_quality",
            "method_config": "method.json",
            "method_config_sha256": "1" * 64,
            "protocol": "protocol.json",
            "protocol_sha256": "2" * 64,
            "test_2023_used": False,
            "full_training_permitted": False,
            "cpu_parity": {
                "status": "passed", "forward_bitwise_equal": True,
                "rng_state_equal": True, "gradient_none_pattern_equal": True,
                "gradients_finite": True, "gradient_atol": 0.000001,
                "gradient_rtol": 0.00001, "gradient_max_abs_difference": 0.0,
                "gradient_max_relative_difference": 0.0,
                "post_adam_parameters_bitwise_equal": True,
                "post_ema_parameters_bitwise_equal": True,
                "optimizer_steps": 1, "scheduler_steps": 1, "ema_updates": 1,
            },
            "activation_memory_comparison": {
                "checkpointed_peak_strictly_lower": True,
                "uncheckpointed": {
                    "checkpointing": False, "loss_finite": True,
                    "memory": {"peak_reserved_bytes": 200},
                },
                "checkpointed": {
                    "checkpointing": True, "loss_finite": True,
                    "memory": {"peak_reserved_bytes": 100},
                },
            },
            "production_path": {
                "status": "passed", "train_batches": 2, "validation_batches": 1,
                "batch_size": 16, "validation_batch_size": 8,
                "image_size": [320, 256], "mixed_precision": "bf16",
                "gradient_accumulation_steps": 1, "validation_weight_source": "ema",
                "call_trace": ["backward", "optimizer", "scheduler", "ema"] * 2,
                "clearml_online": True, "clearml_task_id": "server-task",
                "checkpointed_modules": ["down_blocks.0"],
                "memory": {"peak_reserved_bytes": 80, "total_bytes": 100},
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "memory_admission.json"
            path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
            import hashlib
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            frozen = {
                "memory_admission": {
                    "result_sha256": digest,
                    "schema_version": result["schema_version"],
                    "status": result["status"],
                    "purpose": result["purpose"],
                    "method_config": result["method_config"],
                    "method_config_sha256": result["method_config_sha256"],
                    "protocol": result["protocol"],
                    "protocol_sha256": result["protocol_sha256"],
                }
            }
            self.assertEqual(_validate_memory_admission(path, digest, frozen), result)
            with self.assertRaisesRegex(ValueError, "path/hash mismatch"):
                _validate_memory_admission(path, "0" * 64, frozen)
            result["production_path"]["clearml_online"] = False
            path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
            bad_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            frozen["memory_admission"]["result_sha256"] = bad_digest
            with self.assertRaisesRegex(ValueError, "production evidence"):
                _validate_memory_admission(path, bad_digest, frozen)


if __name__ == "__main__":
    unittest.main()
