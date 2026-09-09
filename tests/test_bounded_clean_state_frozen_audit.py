import unittest

import torch
from torch import nn

from assim_lib.bounded_clean_state_frozen_audit import (
    GENERATIVE_CONFIGS,
    SOURCE_CONTRACT_SHA256,
    SOURCE_CHECKPOINT_SHA256,
    SOURCE_RESULT_SHA256,
    SOURCE_SAMPLES_SHA256,
    TEACHER_TIMES,
    _background_average,
    _decision,
    _require_finite_scalars,
    _teacher_forced,
)


def _metrics(sic: float, sit: float, *, sit_zero: float | None = None) -> dict:
    if sit_zero is None:
        sit_zero = sit
    cases = []
    for case_index in range(4):
        cases.append(
            {
                "case": case_index,
                "leads": {
                    lead: {
                        "sic": {"open_water_mean": sic},
                        "sit": {
                            "open_water_mean": sit,
                            "truth_zero_count": 10,
                            "truth_zero_mean": sit_zero,
                        },
                    }
                    for lead in ("d3", "d6", "d9")
                },
            }
        )
    return {
        "cases": cases,
        "aggregate": {
            lead: {
                "sic": {"open_water_mean": sic},
                "sit": {"open_water_mean": sit},
            }
            for lead in ("d3", "d6", "d9")
        }
    }


class BoundedFrozenAuditTests(unittest.TestCase):
    def test_teacher_forced_probe_is_no_grad_and_masks_inactive_noise(self) -> None:
        class RecordingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.grad_enabled = None
                self.state = None

            def forward(self, value, timestep, return_dict=False):
                del timestep, return_dict
                self.grad_enabled = torch.is_grad_enabled()
                self.state = value[:, :6].detach().clone()
                return (torch.zeros_like(value[:, :6]),)

        model = RecordingModel()
        truth = torch.zeros((1, 6, 1, 2))
        noise = torch.ones((2, 6, 1, 2))
        noise[..., 1] = float("inf")
        valid = torch.tensor([[[[1.0, 0.0]]]])
        physical, normalized, state = _teacher_forced(
            model,
            truth_normalized=truth,
            initial_noise=noise,
            conditioning=torch.zeros((1, 15, 1, 2)),
            valid=valid,
            means=(0.2, 0.3) * 3,
            stds=(0.4, 0.5) * 3,
            time_value=0.01,
        )
        self.assertFalse(model.grad_enabled)
        self.assertTrue(torch.all(model.state[..., 1] == 0))
        self.assertTrue(torch.all(state[..., 1] == 0))
        self.assertTrue(torch.all(physical[..., 1] == 0))
        self.assertTrue(torch.all(normalized[..., 1] == 0))

    def test_protocol_is_exact_and_frozen(self) -> None:
        self.assertEqual(
            GENERATIVE_CONFIGS,
            (
                ("fp32_rk4_17_eps001", 17, 0.01),
                ("fp32_rk4_33_eps001", 33, 0.01),
                ("fp32_rk4_65_eps001", 65, 0.01),
                ("fp32_rk4_65_eps002", 65, 0.02),
            ),
        )
        self.assertEqual(TEACHER_TIMES, (0.01, 0.02))
        for digest in (
            SOURCE_CHECKPOINT_SHA256,
            SOURCE_SAMPLES_SHA256,
            SOURCE_RESULT_SHA256,
            SOURCE_CONTRACT_SHA256,
        ):
            self.assertEqual(len(digest), 64)

    def test_background_average_does_not_mix_fields(self) -> None:
        metrics = _metrics(0.02, 0.03)
        self.assertAlmostEqual(_background_average(metrics, "sic"), 0.02)
        self.assertAlmostEqual(_background_average(metrics, "sit"), 0.03)

    def test_decision_attributes_stable_solver_teacher_floor_to_denoiser(self) -> None:
        generative = {
            "fp32_rk4_17_eps001": _metrics(0.0204, 0.0254),
            "fp32_rk4_33_eps001": _metrics(0.0202, 0.0252),
            "fp32_rk4_65_eps001": _metrics(0.0200, 0.0250),
            "fp32_rk4_65_eps002": _metrics(0.0220, 0.0270),
        }
        teacher = {
            "teacher_forced_t0p01": _metrics(0.02, 0.02),
            "teacher_forced_t0p02": _metrics(0.02, 0.02),
        }
        decision = _decision(generative, teacher)
        self.assertTrue(
            decision[
                "solver_rk4_33_vs_65_stable_below_0p001_each_case_lead_field"
            ]
        )
        self.assertTrue(decision["teacher_forced_centimeter_sit_floor"])
        self.assertEqual(
            decision["diagnosis"], "denoiser_objective_or_finite_optimization_not_solver"
        )
        self.assertFalse(decision["permits_training"])

    def test_decision_rejects_nonfinite_metrics(self) -> None:
        generative = {
            "fp32_rk4_17_eps001": _metrics(float("nan"), 0.025),
            "fp32_rk4_33_eps001": _metrics(0.02, 0.025),
            "fp32_rk4_65_eps001": _metrics(0.02, 0.025),
            "fp32_rk4_65_eps002": _metrics(0.02, 0.025),
        }
        teacher = {
            "teacher_forced_t0p01": _metrics(0.02, 0.02),
            "teacher_forced_t0p02": _metrics(0.02, 0.02),
        }
        with self.assertRaises(FloatingPointError):
            _decision(generative, teacher)

    def test_solver_decision_cannot_hide_casewise_cancellation(self) -> None:
        reference = _metrics(0.02, 0.025)
        unstable = _metrics(0.02, 0.025)
        unstable["cases"][0]["leads"]["d3"]["sic"]["open_water_mean"] = 0.022
        unstable["cases"][1]["leads"]["d3"]["sic"]["open_water_mean"] = 0.018
        generative = {
            "fp32_rk4_17_eps001": reference,
            "fp32_rk4_33_eps001": unstable,
            "fp32_rk4_65_eps001": reference,
            "fp32_rk4_65_eps002": reference,
        }
        teacher = {
            "teacher_forced_t0p01": _metrics(0.0, 0.0, sit_zero=0.0),
            "teacher_forced_t0p02": _metrics(0.0, 0.0, sit_zero=0.0),
        }
        decision = _decision(generative, teacher)
        self.assertFalse(
            decision[
                "solver_rk4_33_vs_65_stable_below_0p001_each_case_lead_field"
            ]
        )
        self.assertEqual(decision["diagnosis"], "solver_sensitivity_requires_revision")

    def test_clean_teacher_endpoint_is_inconclusive_not_trajectory_claim(self) -> None:
        generative = {
            name: _metrics(0.02, 0.025) for name, _, _ in GENERATIVE_CONFIGS
        }
        teacher = {
            "teacher_forced_t0p01": _metrics(0.0, 0.0, sit_zero=0.001),
            "teacher_forced_t0p02": _metrics(0.0, 0.0, sit_zero=0.002),
        }
        decision = _decision(generative, teacher)
        self.assertFalse(decision["teacher_forced_centimeter_sit_floor"])
        self.assertEqual(
            decision["diagnosis"],
            "inconclusive_teacher_endpoint_can_represent_zero_sit",
        )

    def test_recursive_finite_guard_rejects_hidden_overflow_metric(self) -> None:
        with self.assertRaises(FloatingPointError):
            _require_finite_scalars(
                {"aggregate": {"rmse": 0.1}, "cases": [{"rmse": float("inf")}]}
            )


if __name__ == "__main__":
    unittest.main()
