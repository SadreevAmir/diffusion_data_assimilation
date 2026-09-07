from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch

from assim_lib import structured_solver_control as control


class StructuredSolverControlTests(unittest.TestCase):
    def test_texture_energy_expands_valid_mask_over_members_and_leads(self) -> None:
        field = torch.tensor(
            [
                [
                    [[[0.0, 1.0], [0.0, 1.0]], [[0.0, 2.0], [0.0, 2.0]]],
                    [[[0.0, 3.0], [0.0, 3.0]], [[0.0, 4.0], [0.0, 4.0]]],
                ]
            ]
        )
        valid = torch.ones((1, 1, 1, 2, 2))
        expected = (1.0 + 4.0 + 9.0 + 16.0) / 8.0
        self.assertAlmostEqual(control._texture_energy(field, valid), expected)

    def test_texture_energy_ignores_invalid_edges(self) -> None:
        field = torch.tensor([[[[[0.0, 99.0], [2.0, 99.0]]]]])
        valid = torch.tensor([[[[[1.0, 0.0], [1.0, 0.0]]]]])
        self.assertAlmostEqual(control._texture_energy(field, valid), 4.0)

    def test_real_valid_mask_shape_expands_over_four_leads(self) -> None:
        field = torch.zeros((2, 2, 4, 3, 5))
        valid = torch.ones((2, 1, 2, 3, 5))
        self.assertEqual(control._texture_energy(field, valid[:, :, :1]), 0.0)

    def test_frozen_solver_design_is_small_paired_and_validation_only(self) -> None:
        self.assertEqual(control.CASE_INDICES, (0, 90))
        self.assertEqual(len(set(control.MEMBER_SEEDS)), 2)
        self.assertEqual(control.RK4_TIMEPOINTS, (33, 65, 129))
        self.assertEqual(control.FP32_TIMEPOINTS, 65)

    def test_paired_differences_exclude_exact_day0_track(self) -> None:
        left = torch.zeros((1, 2, 8, 2, 2))
        right = left.clone()
        left[:, :, 0, 0, 0] = 0.5
        valid = torch.ones((1, 2, 2, 2))
        lag0 = torch.zeros((1, 1, 2, 2))
        lag0[..., 0, 0] = 1
        result = control._paired_field_differences(
            left, right, valid, lag0, sic_cap=0.997
        )
        self.assertEqual(result["fields"]["lead0_sic"]["mean_abs_difference"], 0.0)
        self.assertEqual(
            result["events"]["lead0_occurrence"][
                "mean_absolute_probability_difference"
            ],
            0.0,
        )

    def test_solver_gate_is_executable_and_fail_closed(self) -> None:
        variant_names = (
            "rk4_32_intervals_bf16",
            "rk4_64_intervals_bf16",
            "rk4_128_intervals_bf16",
            "rk4_64_intervals_fp32",
        )
        variants = {
            name: {
                "status": "passed",
                "metrics": {
                    "lead0_sic_fair_crps": 0.2,
                    "lead0_sic_mean_rmse": 0.3,
                },
                "sic_neighbor_energy_by_lead": {f"lead{i}": 1.0 for i in range(4)},
            }
            for name in variant_names
        }
        comparison = {
            "events": {
                "lead0_occurrence": {
                    "mean_absolute_probability_difference": 0.0
                },
                "lead0_cap": {"mean_absolute_probability_difference": 0.0},
            }
        }
        comparisons = {
            "rk4_64_intervals_bf16_vs_rk4_128_intervals_bf16": {
                "variants": [
                    "rk4_64_intervals_bf16",
                    "rk4_128_intervals_bf16",
                ],
                **comparison,
            },
            "rk4_64_intervals_bf16_vs_fp32": {
                "variants": [
                    "rk4_64_intervals_bf16",
                    "rk4_64_intervals_fp32",
                ],
                **comparison,
            },
        }
        self.assertEqual(control._solver_gate(variants, comparisons)["status"], "converged")
        comparisons["rk4_64_intervals_bf16_vs_fp32"]["events"][
            "lead0_occurrence"
        ]["mean_absolute_probability_difference"] = 0.5
        failed = control._solver_gate(variants, comparisons)
        self.assertEqual(failed["status"], "failed")
        self.assertFalse(failed["pilot_permitted"])

    def test_wrapper_refuses_existing_output_without_modifying_it(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        script = repo / "scripts/run_structured_solver_control.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            recovery = run / "structured_recovery/epoch_0016"
            recovery.mkdir(parents=True)
            (run / "metadata.json").write_text("{}", encoding="utf-8")
            (recovery / "ema_state.pth").write_bytes(b"x")
            output = root / "existing"
            output.mkdir()
            sentinel = output / "sentinel"
            sentinel.write_bytes(b"do-not-touch")
            environment = {
                **os.environ,
                "REPO_DIR": str(repo),
                "RUN_DIR": str(run),
                "OUTPUT_DIR": str(output),
                "REFERENCE_METADATA_SHA256": "0" * 64,
                "REFERENCE_EMA_SHA256": "1" * 64,
                "REFERENCE_RESUME_SHA256": "2" * 64,
            }
            completed = subprocess.run(
                ["bash", str(script)],
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 73)
            self.assertEqual(sentinel.read_bytes(), b"do-not-touch")
            self.assertEqual({item.name for item in output.iterdir()}, {"sentinel"})


if __name__ == "__main__":
    unittest.main()
