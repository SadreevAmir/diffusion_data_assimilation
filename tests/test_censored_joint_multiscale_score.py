import itertools
import unittest

import torch

from assim_lib.censored_joint_energy_overfit import joint_field_energy_score
from assim_lib.censored_joint_multiscale_score import (
    PATCHES_PER_CONDITION,
    _energy_score_with_mask,
    multiscale_joint_energy_score,
    patch_energy_score,
    sample_valid_centres,
)


class CensoredJointMultiscaleScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def setUp(self) -> None:
        torch.manual_seed(7)
        self.stds = (0.2, 0.8, 0.2, 0.8, 0.2, 0.8)

    def test_two_member_manual_formula(self) -> None:
        members = torch.tensor([[[[[0.0]]], [[[2.0]]]]], dtype=torch.float64)
        truth = torch.tensor([[[[1.0]]]], dtype=torch.float64)
        valid = torch.ones(1, 1, 1, 1, dtype=torch.float64)
        score = _energy_score_with_mask(members, truth, valid, (1.0,))
        self.assertAlmostEqual(float(score), 0.0)

    def test_global_component_matches_frozen_baseline_score(self) -> None:
        members = torch.rand(3, 2, 6, 5, 7, dtype=torch.float64)
        truth = torch.rand(3, 6, 5, 7, dtype=torch.float64)
        valid = torch.randint(0, 2, (3, 1, 5, 7), dtype=torch.int64).double()
        valid[:, :, 0, 0] = 1
        baseline = joint_field_energy_score(
            members, truth, valid, stds=self.stds
        )
        candidate = _energy_score_with_mask(members, truth, valid, self.stds)
        self.assertAlmostEqual(float(baseline), float(candidate), places=12)

    def test_zero_distance_backward_is_finite(self) -> None:
        members = torch.zeros(1, 2, 6, 4, 4, dtype=torch.float64, requires_grad=True)
        truth = torch.zeros(1, 6, 4, 4, dtype=torch.float64)
        valid = torch.ones(1, 1, 4, 4, dtype=torch.float64)
        score = _energy_score_with_mask(members, truth, valid, self.stds)
        score.backward()
        self.assertTrue(torch.all(torch.isfinite(members.grad)))

    def test_padding_outside_valid_mask_does_not_change_score(self) -> None:
        members = torch.randn(1, 3, 6, 3, 4, dtype=torch.float64)
        truth = torch.randn(1, 6, 3, 4, dtype=torch.float64)
        valid = torch.ones(1, 1, 3, 4, dtype=torch.float64)
        reference = _energy_score_with_mask(members, truth, valid, self.stds)
        padded_members = torch.full(
            (1, 3, 6, 7, 9), float("nan"), dtype=torch.float64
        )
        padded_truth = torch.full(
            (1, 6, 7, 9), float("inf"), dtype=torch.float64
        )
        padded_valid = torch.zeros(1, 1, 7, 9, dtype=torch.float64)
        padded_members[..., 2:5, 3:7] = members
        padded_truth[..., 2:5, 3:7] = truth
        padded_valid[..., 2:5, 3:7] = valid
        padded_members.requires_grad_()
        actual = _energy_score_with_mask(
            padded_members, padded_truth, padded_valid, self.stds
        )
        self.assertAlmostEqual(float(reference), float(actual), places=12)
        actual.backward()
        active = padded_valid[:, None].bool().expand_as(padded_members)
        self.assertTrue(torch.all(torch.isfinite(padded_members.grad[active])))
        self.assertTrue(torch.equal(padded_members.grad[~active], torch.zeros_like(padded_members.grad[~active])))

    def test_invalid_active_values_stds_masks_and_centres_fail(self) -> None:
        members = torch.zeros(1, 2, 6, 4, 4, dtype=torch.float64)
        truth = torch.zeros(1, 6, 4, 4, dtype=torch.float64)
        valid = torch.ones(1, 1, 4, 4, dtype=torch.float64)
        members[0, 0, 0, 1, 1] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "active members"):
            _energy_score_with_mask(members, truth, valid, self.stds)
        members[0, 0, 0, 1, 1] = 0
        with self.assertRaisesRegex(ValueError, "finite positive"):
            _energy_score_with_mask(
                members, truth, valid, (*self.stds[:-1], float("inf"))
            )
        fractional = valid.clone()
        fractional[0, 0, 0, 0] = 0.5
        with self.assertRaisesRegex(ValueError, "exactly binary"):
            _energy_score_with_mask(members, truth, fractional, self.stds)
        with self.assertRaisesRegex(ValueError, "exactly binary"):
            sample_valid_centres(
                fractional, 1, generator=torch.Generator().manual_seed(1)
            )
        with self.assertRaisesRegex(ValueError, "integer coordinates"):
            patch_energy_score(
                members,
                truth,
                valid,
                self.stds,
                torch.tensor([[[1.0, 1.0]]]),
                patch_size=4,
            )
        for centre in (torch.tensor([[[-1, 0]]]), torch.tensor([[[4, 0]]])):
            with self.assertRaisesRegex(ValueError, "outside the image"):
                patch_energy_score(
                    members, truth, valid, self.stds, centre, patch_size=4
                )

    def test_finite_input_overflow_fails_closed(self) -> None:
        members = torch.full((1, 2, 1, 1, 1), 1e308, dtype=torch.float64)
        members[:, 1] = -1e308
        truth = torch.zeros(1, 1, 1, 1, dtype=torch.float64)
        valid = torch.ones(1, 1, 1, 1, dtype=torch.float64)
        with self.assertRaisesRegex(FloatingPointError, "result"):
            _energy_score_with_mask(members, truth, valid, (1.0,))

    def test_population_score_and_iid_estimator_expectation(self) -> None:
        truth_law = torch.tensor([1.0, 3.0], dtype=torch.float64)
        correct_law = truth_law.clone()
        wrong_law = torch.tensor([0.0, 4.0], dtype=torch.float64)

        def population_score(forecast: torch.Tensor) -> float:
            observation = (forecast[:, None] - truth_law[None, :]).abs().mean()
            pair = (forecast[:, None] - forecast[None, :]).abs().mean()
            return float(observation - 0.5 * pair)

        def expected_two_member_estimator(forecast: torch.Tensor) -> float:
            estimates = []
            valid = torch.ones(1, 1, 1, 1, dtype=torch.float64)
            for first, second, target in itertools.product(
                range(2), range(2), range(2)
            ):
                ensemble = torch.tensor(
                    [[[[[forecast[first]]]], [[[forecast[second]]]]]],
                    dtype=torch.float64,
                )
                truth = truth_law[target].reshape(1, 1, 1, 1)
                estimates.append(
                    _energy_score_with_mask(ensemble, truth, valid, (1.0,))
                )
            return float(torch.stack(estimates).mean())

        self.assertAlmostEqual(population_score(correct_law), 0.5)
        self.assertAlmostEqual(population_score(wrong_law), 1.0)
        self.assertAlmostEqual(
            expected_two_member_estimator(correct_law),
            population_score(correct_law),
        )
        self.assertAlmostEqual(
            expected_two_member_estimator(wrong_law), population_score(wrong_law)
        )

    def test_sampled_centres_are_valid_and_rng_is_reproducible(self) -> None:
        valid = torch.zeros(2, 1, 5, 6)
        valid[0, 0, 1, 2] = 1
        valid[1, 0, 3, 4] = 1
        first = sample_valid_centres(
            valid,
            PATCHES_PER_CONDITION,
            generator=torch.Generator().manual_seed(91),
        )
        second = sample_valid_centres(
            valid,
            PATCHES_PER_CONDITION,
            generator=torch.Generator().manual_seed(91),
        )
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first[0], torch.tensor([[1, 2]] * 8)))
        self.assertTrue(torch.equal(first[1], torch.tensor([[3, 4]] * 8)))

    def test_finite_joint_law_prefers_spatially_correct_ensemble(self) -> None:
        count, channels, height, width = 16, 6, 8, 8
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, height, dtype=torch.float64),
            torch.linspace(-1, 1, width, dtype=torch.float64),
            indexing="ij",
        )
        fields = []
        for member in range(count):
            phase = 2 * torch.pi * member / count
            base = torch.sin(2.1 * x + phase) + torch.cos(1.7 * y - phase)
            joint = torch.stack(
                [torch.sigmoid(base + 0.1 * channel) for channel in range(channels)]
            )
            joint[0] = torch.where(joint[0] < 0.35, 0.0, joint[0])
            joint[2] = torch.where(joint[2] > 0.65, 1.0, joint[2])
            joint[1::2] = torch.relu(2.0 * joint[1::2] - 0.5)
            fields.append(joint)
        correct = torch.stack(fields)
        scrambled = torch.empty_like(correct)
        for row, column in itertools.product(range(height), range(width)):
            permutation = torch.randperm(count)
            scrambled[:, :, row, column] = correct[permutation, :, row, column]
        degenerate = correct.mean(dim=0, keepdim=True).expand_as(correct)
        valid = torch.ones(1, 1, height, width, dtype=torch.float64)
        centres = torch.tensor(
            [[[row, column] for row in range(height) for column in range(width)]]
        )

        def population_region_score(
            forecast: torch.Tensor,
            target: torch.Tensor,
            region_valid: torch.Tensor,
        ) -> torch.Tensor:
            mask = region_valid[0, 0].bool()
            scale = forecast.new_tensor(self.stds).view(1, channels, 1, 1)
            forecast_vectors = (forecast / scale)[:, :, mask].flatten(start_dim=1)
            target_vector = (target / scale[0])[:, mask].flatten()
            normalizer = (channels * int(mask.sum())) ** 0.5
            observation = torch.linalg.vector_norm(
                forecast_vectors - target_vector, dim=1
            ).mean() / normalizer
            pair = torch.linalg.vector_norm(
                forecast_vectors[:, None] - forecast_vectors[None, :], dim=2
            ).mean() / normalizer
            return observation - 0.5 * pair

        def population_patch_score(
            forecast: torch.Tensor, target: torch.Tensor, patch_size: int
        ) -> torch.Tensor:
            values = []
            half = patch_size // 2
            for centre_y, centre_x in centres[0].tolist():
                top, left = centre_y - half, centre_x - half
                bottom, right = top + patch_size, left + patch_size
                y0, y1 = max(0, top), min(height, bottom)
                x0, x1 = max(0, left), min(width, right)
                values.append(
                    population_region_score(
                        forecast[:, :, y0:y1, x0:x1],
                        target[:, y0:y1, x0:x1],
                        valid[:, :, y0:y1, x0:x1],
                    )
                )
            return torch.stack(values).mean()

        def expected_score(forecast: torch.Tensor) -> float:
            values = []
            for target in correct:
                global_score = population_region_score(
                    forecast, target, valid
                )
                patch_8 = population_patch_score(forecast, target, 8)
                patch_16 = population_patch_score(forecast, target, 16)
                values.append(0.5 * global_score + 0.25 * patch_8 + 0.25 * patch_16)
            return float(torch.stack(values).mean())

        def expected_pointwise_crps(forecast: torch.Tensor) -> torch.Tensor:
            observation = (
                forecast[:, None] - correct[None, :]
            ).abs().mean(dim=(0, 1))
            pair = (
                forecast[:, None] - forecast[None, :]
            ).abs().mean(dim=(0, 1))
            return (observation - 0.5 * pair).mean()

        correct_score = expected_score(correct)
        self.assertLess(correct_score, expected_score(scrambled))
        self.assertLess(correct_score, expected_score(degenerate))
        self.assertAlmostEqual(
            float(expected_pointwise_crps(correct)),
            float(expected_pointwise_crps(scrambled)),
            places=12,
        )
        for row, column in itertools.product(range(height), range(width)):
            self.assertTrue(
                torch.equal(
                    torch.sort(correct[:, :, row, column], dim=0).values,
                    torch.sort(scrambled[:, :, row, column], dim=0).values,
                )
            )

    def test_monte_carlo_patch_mean_matches_all_valid_centres(self) -> None:
        members = torch.randn(1, 3, 6, 6, 7, dtype=torch.float64)
        truth = torch.randn(1, 6, 6, 7, dtype=torch.float64)
        valid = torch.ones(1, 1, 6, 7, dtype=torch.float64)
        exhaustive = torch.tensor(
            [[[row, column] for row in range(6) for column in range(7)]]
        )
        exact_values = torch.stack(
            [
                patch_energy_score(
                    members,
                    truth,
                    valid,
                    self.stds,
                    exhaustive[:, index : index + 1],
                    patch_size=4,
                )
                for index in range(exhaustive.shape[1])
            ]
        )
        exact = exact_values.mean()
        sampled = sample_valid_centres(
            valid, 10_000, generator=torch.Generator().manual_seed(123)
        )
        flat_index = sampled[0, :, 0] * 7 + sampled[0, :, 1]
        counts = torch.bincount(flat_index, minlength=exhaustive.shape[1])
        estimate = (exact_values * counts).sum() / counts.sum()
        self.assertAlmostEqual(float(exact), float(estimate), delta=0.01)

    def test_frozen_multiscale_weights_are_applied(self) -> None:
        members = torch.randn(1, 2, 6, 20, 20, dtype=torch.float64)
        truth = torch.randn(1, 6, 20, 20, dtype=torch.float64)
        valid = torch.ones(1, 1, 20, 20, dtype=torch.float64)
        centres = sample_valid_centres(
            valid,
            PATCHES_PER_CONDITION,
            generator=torch.Generator().manual_seed(5),
        )
        total, parts = multiscale_joint_energy_score(
            members, truth, valid, stds=self.stds, centres=centres
        )
        expected = (
            0.5 * parts["global"]
            + 0.25 * parts["patch_8"]
            + 0.25 * parts["patch_16"]
        )
        self.assertAlmostEqual(float(total), float(expected), places=12)


if __name__ == "__main__":
    unittest.main()
