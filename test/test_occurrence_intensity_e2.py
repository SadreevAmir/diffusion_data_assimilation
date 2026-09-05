import unittest

import torch

from assim_lib.occurrence_intensity_e2 import apply_operators_at_own_time


class E2OperatorTimeMappingTest(unittest.TestCase):
    def test_each_operator_uses_its_matching_generated_state(self):
        trajectory = torch.tensor(
            [[[[[1.0, 2.0]]], [[[10.0, 20.0]]], [[[100.0, 200.0]]]]]
        )
        observations = torch.tensor([[[[3.0, 4.0]], [[31.0, 61.0]], [[401.0, 801.0]]]])
        masks = torch.ones_like(observations)
        operators = (
            lambda state: state[:, 0],
            lambda state: 3.0 * state[:, 0],
            lambda state: 4.0 * state[:, 0],
        )

        predictions, innovations = apply_operators_at_own_time(
            trajectory, observations, masks, operators
        )

        self.assertTrue(torch.equal(predictions, torch.tensor(
            [[[[1.0, 2.0]], [[30.0, 60.0]], [[400.0, 800.0]]]]
        )))
        self.assertTrue(torch.equal(innovations, torch.ones_like(observations)))

    def test_missing_values_are_zeroed_without_nan_leakage(self):
        trajectory = torch.ones(1, 3, 1, 1, 2)
        observations = torch.tensor([[[[2.0, float("nan")]]] * 3])
        masks = torch.tensor([[[[1.0, 0.0]]] * 3])
        operators = (lambda state: state[:, 0],) * 3

        predictions, innovations = apply_operators_at_own_time(
            trajectory, observations, masks, operators
        )

        self.assertTrue(torch.all(torch.isfinite(predictions)))
        self.assertTrue(torch.all(torch.isfinite(innovations)))
        self.assertTrue(torch.all(predictions[..., 1] == 0))
        self.assertTrue(torch.all(innovations[..., 1] == 0))

    def test_current_state_broadcast_shape_and_operator_drift_fail_closed(self):
        observations = torch.zeros(1, 3, 1, 1)
        masks = torch.ones_like(observations)
        with self.assertRaisesRegex(ValueError, "state_trajectory"):
            apply_operators_at_own_time(
                torch.zeros(1, 1, 1, 1, 1), observations, masks,
                (lambda state: state[:, 0],) * 3,
            )
        with self.assertRaisesRegex(ValueError, "exactly three"):
            apply_operators_at_own_time(
                torch.zeros(1, 3, 1, 1, 1), observations, masks,
                (lambda state: state[:, 0],) * 2,
            )


if __name__ == "__main__":
    unittest.main()
