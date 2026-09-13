import unittest

import torch

from assim_lib.direct_dynamics_mixed_support_admission import (
    binary_dequantized_logit,
    coarse_occurrence,
    decode_binary_dequantized_logit,
    dequantized_cfm_gradient_check,
    occurrence_change_counts,
    synthetic_representation_birth_death_check,
)


class MixedSupportAdmissionTests(unittest.TestCase):
    def test_binary_dequantization_roundtrip_and_rejects_endpoints(self):
        binary = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
        uniform = torch.tensor([[[[0.25, 0.25], [0.75, 0.75]]]])
        latent = binary_dequantized_logit(binary, uniform)
        self.assertTrue(torch.equal(decode_binary_dequantized_logit(latent), binary))
        with self.assertRaises(ValueError):
            binary_dequantized_logit(binary, torch.zeros_like(binary))

    def test_coarse_target_is_any_positive_valid_sic(self):
        sic = torch.zeros((1, 1, 4, 4))
        valid = torch.ones_like(sic)
        sic[..., 1, 1] = 0.2
        target, support = coarse_occurrence(sic, valid, 2)
        self.assertEqual(tuple(target.shape), (1, 1, 2, 2))
        self.assertEqual(float(target.sum()), 1.0)
        self.assertTrue(torch.all(support))

    def test_change_retention_distinguishes_within_block_motion(self):
        initial = torch.zeros((1, 1, 4, 4))
        future = torch.zeros_like(initial)
        valid = torch.ones_like(initial)
        initial[..., 0, 0] = 1
        future[..., 0, 1] = 1
        counts = occurrence_change_counts(initial, future, valid, 2)
        self.assertEqual(counts["native_changed_cells"], 2)
        self.assertEqual(counts["coarse_changed_cells"], 0)
        self.assertEqual(counts["native_changed_cells_visible_at_coarse"], 0)

    def test_dequantized_cfm_has_real_gradients_without_ste(self):
        binary = torch.zeros((2, 3, 8, 8))
        binary[0, :, 2:5, 3:6] = 1
        binary[1, :, 1:7, 1:7] = 1
        result = dequantized_cfm_gradient_check(binary, torch.ones((2, 1, 8, 8)), seed=4)
        self.assertTrue(result["hard_decode_roundtrip_exact"])
        self.assertTrue(result["all_parameter_gradients_finite_positive"])
        self.assertEqual(result["objective"], "continuous_dequantized_binary_cfm_no_ste")

    def test_representation_codec_roundtrips_birth_without_initial_edge(self):
        result = synthetic_representation_birth_death_check(seed=9)
        self.assertFalse(result["initial_empty_has_edge"])
        self.assertGreater(result["future_birth_count"], 0)
        self.assertTrue(result["empty_edge_birth_roundtrip_exact"])
        self.assertTrue(result["death_roundtrip_exact"])
        self.assertEqual(result["scope"], "representation_codec_only_not_a_model_source_branch_test")


if __name__ == "__main__":
    unittest.main()
