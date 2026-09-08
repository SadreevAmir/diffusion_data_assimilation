from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from assim_lib import structured_preconditioned_pilot as pilot
from assim_lib.config import load_json


def _metrics(value: float = 1.0) -> dict:
    return {
        "support": {
            "ensemble_support_violation_count": 0,
            "truth_support_violation_count": 0,
            "background_support_violation_count": 0,
            "lag0_exact_max_abs_error": 0.0,
            "decode_saturation_count": 0,
        },
        "leads": {
            ("d0" if lead == 0 else f"d+{lead}"): {
                "fields": {
                    field: {
                        "fair_crps": value,
                        "ensemble_mean_rmse": value,
                        "rank_tv_to_uniform": value,
                        "central_coverage_absolute_error": {
                            "central_50": value,
                            "central_80": value,
                            "central_90": value,
                        },
                    }
                    for field in ("sic", "sit")
                },
                "events": {
                    event: {
                        "brier_score": value,
                        "reliability_l1": value,
                    }
                    for event in ("ice_occurrence", "sic_archive_cap")
                },
            }
            for lead in range(4)
        },
        "temporal": {
            "transitions": {
                f"d+{lead}_to_d+{lead + 1}": {
                    "fields": {
                        field: {
                            "increment_fair_crps": value,
                            "ensemble_mean_increment_rmse": value,
                        }
                        for field in ("sic", "sit")
                    }
                }
                for lead in range(3)
            },
            "trajectory_variogram": {
                field: {"trajectory_variogram_score": value}
                for field in ("sic", "sit")
            },
        },
    }


def _spatial(value: float) -> dict:
    return {
        "aggregate": {
            f"lag{lag}_{field}": value
            for lag in pilot.SPATIAL_LAGS
            for field in ("sic", "sit")
        }
    }


class StructuredPreconditionedPilotTests(unittest.TestCase):
    def test_complete_quantitative_gate_passes_but_never_unlocks_full_training(self) -> None:
        raw = _metrics()
        candidate = _metrics(0.69)
        raw_spatial = _spatial(1.0)
        candidate_spatial = _spatial(0.70)
        solver = {"status": "converged", "pilot_permitted": True}
        result = pilot.paired_pilot_gate(
            raw, candidate, raw_spatial, candidate_spatial, solver
        )
        self.assertEqual(result["status"], "quantitative_pass")
        self.assertTrue(result["quantitative_passed"])
        self.assertEqual(result["visual_review_status"], "pending_independent_review")
        self.assertFalse(result["full_training_permitted"])
        self.assertFalse(result["test_2023_used"])
        self.assertEqual(len(result["checks"]), 79)

    def test_boundary_gate_fails_closed_on_worse_missing_and_nonfinite(self) -> None:
        raw = _metrics()
        candidate = _metrics(0.69)
        raw_spatial = _spatial(1.0)
        candidate_spatial = _spatial(0.70)
        solver = {"status": "converged", "pilot_permitted": True}

        mutations = []
        worse = copy.deepcopy(candidate)
        worse["leads"]["d+1"]["events"]["ice_occurrence"]["brier_score"] = 1.01
        mutations.append(worse)
        missing = copy.deepcopy(candidate)
        del missing["leads"]["d+2"]["events"]["sic_archive_cap"]["reliability_l1"]
        mutations.append(missing)
        nan = copy.deepcopy(candidate)
        nan["leads"]["d+3"]["events"]["ice_occurrence"]["reliability_l1"] = float("nan")
        mutations.append(nan)
        infinite = copy.deepcopy(candidate)
        infinite["leads"]["d0"]["events"]["sic_archive_cap"]["brier_score"] = float("inf")
        mutations.append(infinite)

        for adverse in mutations:
            with self.subTest(adverse=adverse):
                result = pilot.paired_pilot_gate(
                    raw, adverse, raw_spatial, candidate_spatial, solver
                )
                self.assertEqual(result["status"], "quantitative_fail")
                self.assertFalse(result["quantitative_passed"])

    def test_gate_fails_closed_on_missing_nonfinite_and_regression(self) -> None:
        raw = _metrics()
        candidate = _metrics(0.69)
        raw_spatial = _spatial(1.0)
        candidate_spatial = _spatial(0.70)
        solver = {"status": "converged", "pilot_permitted": True}
        adverse = copy.deepcopy(candidate)
        del adverse["leads"]["d+2"]["fields"]["sit"]["fair_crps"]
        self.assertEqual(
            pilot.paired_pilot_gate(
                raw, adverse, raw_spatial, candidate_spatial, solver
            )["status"],
            "quantitative_fail",
        )
        adverse = copy.deepcopy(candidate)
        adverse["temporal"]["transitions"]["d+1_to_d+2"]["fields"]["sic"][
            "increment_fair_crps"
        ] = float("nan")
        self.assertEqual(
            pilot.paired_pilot_gate(
                raw, adverse, raw_spatial, candidate_spatial, solver
            )["status"],
            "quantitative_fail",
        )
        adverse = copy.deepcopy(candidate)
        adverse["support"]["ensemble_support_violation_count"] = 1
        self.assertEqual(
            pilot.paired_pilot_gate(
                raw, adverse, raw_spatial, candidate_spatial, solver
            )["status"],
            "quantitative_fail",
        )
        self.assertEqual(
            pilot.paired_pilot_gate(
                raw,
                candidate,
                raw_spatial,
                candidate_spatial,
                {"status": "failed", "pilot_permitted": False},
            )["status"],
            "quantitative_fail",
        )

    def test_spatial_gate_never_pools_sic_and_sit_units(self) -> None:
        raw = _metrics()
        candidate = _metrics(0.69)
        raw_spatial = _spatial(1.0)
        candidate_spatial = _spatial(0.10)
        candidate_spatial["aggregate"]["lag2_sic"] = 0.71
        result = pilot.paired_pilot_gate(
            raw,
            candidate,
            raw_spatial,
            candidate_spatial,
            {"status": "converged", "pilot_permitted": True},
        )
        self.assertFalse(result["checks"]["spatial_variogram/lag2/sic"]["passed"])
        self.assertTrue(result["checks"]["spatial_variogram/lag2/sit"]["passed"])
        self.assertEqual(result["status"], "quantitative_fail")

    def test_frozen_panel_builds_validation_only_contract(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        data = load_json(repo / "config/data/m2m_2f_structured_joint_real_lagged.json")
        contract, indices, panel = pilot._load_panel_contract(
            repo / "paper/STRUCTURED_PAIRED_PILOT_PANEL.json",
            repo / "paper/STRUCTURED_GAUSSIAN_PRECONDITIONED_PILOT_PROTOCOL.json",
            data,
        )
        self.assertEqual(indices, [0, 45, 90, 135, 180, 225, 270, 315])
        self.assertEqual(len(contract.cases), 8)
        self.assertEqual(len(contract.member_seeds), 8)
        self.assertEqual(contract.blockers, ())
        self.assertFalse(panel["test_2023_permitted"])
        protocol = load_json(
            repo / "paper/STRUCTURED_GAUSSIAN_PRECONDITIONED_PILOT_PROTOCOL.json"
        )
        refinement = protocol["evaluation"]["candidate_solver_refinement"]
        self.assertEqual(refinement["case_positions"], [0, 2])
        self.assertEqual(refinement["member_positions"], [0, 1])
        self.assertEqual(pilot.CANDIDATE_SOLVER_CASE_POSITIONS, (0, 2))
        self.assertEqual(pilot.CANDIDATE_SOLVER_MEMBER_POSITIONS, (0, 1))

    def test_candidate_identity_is_bound_to_current_attempt_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            run = output / "training/seed1701"
            run.mkdir(parents=True)
            payload = {
                "schema_version": pilot.SCHEMA_VERSION,
                "status": "training_completed",
                "attempt_token": "attempt",
                "candidate_run_dir": str(run.resolve()),
                "optimizer_steps": 2128,
                "internal_sampling_disabled": True,
                "authoritative_evaluation": "separate_bare_model_strict_fp32",
                "metadata_sha256": "1" * 64,
                "ema_state_sha256": "2" * 64,
                "resume_sha256": "3" * 64,
            }
            (output / "training_result.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            hashes = pilot._load_candidate_training_identity(output, run, "attempt")
            self.assertEqual(hashes["ema_state_sha256"], "2" * 64)
            with self.assertRaisesRegex(ValueError, "pilot attempt"):
                pilot._load_candidate_training_identity(output, run, "other")

    def test_spatial_variogram_uses_both_spatial_axes(self) -> None:
        truth = torch.zeros((1, 8, 8, 9))
        truth[:, :, :, 1:] = 1.0
        ensemble = truth[:, None].repeat(1, 2, 1, 1, 1)
        valid = torch.ones((1, 1, 8, 9))
        lag0 = torch.zeros_like(valid)
        result = pilot._spatial_variogram_errors(ensemble, truth, valid, lag0)
        self.assertEqual(
            result["aggregate"],
            {
                f"lag{lag}_{field}": 0.0
                for lag in pilot.SPATIAL_LAGS
                for field in ("sic", "sit")
            },
        )

    def test_real_two_field_valid_mask_is_verified_before_collapse(self) -> None:
        mask = torch.ones((8, 2, 320, 256))
        collapsed = pilot._single_valid_mask(mask)
        self.assertEqual(tuple(collapsed.shape), (8, 1, 320, 256))
        adverse = mask.clone()
        adverse[0, 1, 0, 0] = 0
        with self.assertRaisesRegex(ValueError, "SIC and SIT"):
            pilot._single_valid_mask(adverse)

    def test_training_mode_disables_internal_sampling_and_keeps_exact_budget(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            output.mkdir()
            (output / ".structured_pilot_owner").write_text("attempt\n", encoding="utf-8")

            def fake_train(runtime, _config_dir):
                training = runtime["training"]
                self.assertEqual(training["sample_every_n_epochs"], 0)
                self.assertEqual(training["metric_every_n_epochs"], 0)
                self.assertFalse(training["metric_save_ensemble_samples"])
                self.assertEqual(training["num_workers_train"], 0)
                self.assertEqual(training["num_workers_val"], 0)
                self.assertTrue(training["preserve_persistent_worker_rng"])
                run = Path(training["base_output_dir"]) / training["run_name"]
                recovery = run / "structured_recovery/epoch_0016"
                recovery.mkdir(parents=True)
                (run / "metadata.json").write_text("{}", encoding="utf-8")
                (recovery / "ema_state.pth").write_bytes(b"ema")
                (recovery / "resume.json").write_text(
                    json.dumps({"next_epoch": 16, "global_step": 2128}),
                    encoding="utf-8",
                )
                return {"output_dir": str(run), "metrics_path": str(run / "metrics.json")}

            with mock.patch.dict(os.environ, {"STRUCTURED_PILOT_ATTEMPT": "attempt"}), mock.patch.object(
                pilot.torch.cuda, "device_count", return_value=1
            ), mock.patch.object(pilot, "train_main", side_effect=fake_train):
                checkpointed_experiment = repo / (
                    "config/experiments/"
                    "train_structured_joint_gaussian_preconditioned_checkpointed_pilot.json"
                )
                checkpointed_method = load_json(
                    repo
                    / "config/methods/"
                    "structured_joint_gaussian_preconditioned_checkpointed_pilot_2f.json"
                )
                self.assertTrue(checkpointed_method["activation_checkpointing"])
                result = pilot.run_training(checkpointed_experiment, output)
            self.assertEqual(result["optimizer_steps"], 2128)
            self.assertEqual(result["dataloader_workers_train"], 0)
            self.assertEqual(result["dataloader_workers_val"], 0)
            self.assertTrue(result["persistent_worker_rng_compatible"])
            self.assertTrue(result["internal_sampling_disabled"])

    def test_wrapper_refuses_existing_output_and_preserves_clearml_connectivity(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        script = repo / "scripts/run_structured_preconditioned_pilot.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw/structured_recovery/epoch_0016"
            raw.mkdir(parents=True)
            (raw.parents[1] / "metadata.json").write_text("{}", encoding="utf-8")
            (raw / "ema_state.pth").write_bytes(b"ema")
            (raw / "resume.json").write_text("{}", encoding="utf-8")
            refinement = root / "refinement"
            refinement.mkdir()
            result_path = refinement / "solver_control.json"
            gate_path = refinement / "solver_gate.json"
            result_path.write_text("{}", encoding="utf-8")
            gate_path.write_text("{}", encoding="utf-8")
            output = root / "existing"
            output.mkdir()
            sentinel = output / "sentinel"
            sentinel.write_text("keep", encoding="utf-8")
            env = {
                **os.environ,
                "REPO_DIR": str(repo), "RAW_RUN_DIR": str(raw.parents[1]),
                "REFINEMENT_RESULT_DIR": str(refinement), "OUTPUT_DIR": str(output),
                "RAW_METADATA_SHA256": "0" * 64, "RAW_EMA_SHA256": "1" * 64,
                "RAW_RESUME_SHA256": "2" * 64,
                "REFINEMENT_SOLVER_CONTROL_SHA256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
                "REFINEMENT_SOLVER_GATE_SHA256": hashlib.sha256(gate_path.read_bytes()).hexdigest(),
            }
            completed = subprocess.run(
                ["bash", str(script)], env=env, capture_output=True, text=True, check=False
            )
            self.assertEqual(completed.returncode, 73)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
        source = script.read_text(encoding="utf-8")
        self.assertIn("PILOT_TIMEOUT_SECONDS=86340", source)
        self.assertIn("--kill-after=\"${PILOT_KILL_GRACE_SECONDS}s\"", source)
        self.assertIn('[[ "${CLEARML_OFFLINE_MODE+x}" == x ]]', source)
        self.assertIn("export CLEARML_REQUIRE_ONLINE=1", source)
        self.assertIn("--memory-admission-result", source)
        self.assertIn("validate_structured_gaussian_pilot_contract.py", source)
        self.assertIn("--refinement-result-sha256", source)


if __name__ == "__main__":
    unittest.main()
