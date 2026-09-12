import torch

from assim_lib.direct_dynamics_sit_support_decoder_audit import (
    _edge_means,
    _gradient_probe,
    _seam_audit,
)


def test_edge_means_separate_two_by_two_boundaries_from_interiors():
    # Constant within 2x2 blocks, one jump only at the vertical block boundary.
    value = torch.tensor(
        [[[[[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0]]]]]
    )
    mask = torch.ones((1, 1, 2, 4))
    metrics = _edge_means(value, mask)
    assert metrics["boundary_abs_gradient_case_equal"] > 0
    assert metrics["interior_abs_gradient_case_equal"] == 0


def test_seam_audit_is_zero_for_identity_decoder():
    value = torch.arange(16, dtype=torch.float32).reshape(1, 1, 1, 4, 4)
    mask = torch.ones((1, 1, 4, 4))
    metrics = _seam_audit(value, value.clone(), mask)
    assert metrics["decoded_minus_original_boundary_gradient"] == 0
    assert metrics["decoded_minus_original_interior_gradient"] == 0
    assert metrics["new_block_boundary_excess"] == 0
    assert metrics["boundary_to_interior_ratio_change_paired_case_equal"] == 0


def test_zero_interior_gradient_has_null_ratio_not_artificial_large_value():
    value = torch.tensor(
        [[[[[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0]]]]]
    )
    mask = torch.ones((1, 1, 2, 4))
    metrics = _edge_means(value, mask)
    assert metrics["per_case_zero_interior_gradient"] == [True]
    assert metrics["per_case_boundary_to_interior_ratio"] == [None]
    assert metrics["boundary_to_interior_ratio_case_equal_defined_cases"] is None


def test_seam_ratio_change_uses_only_paired_defined_cases():
    original = torch.tensor(
        [
            [[[[0.0, 1.0, 3.0, 4.0], [0.0, 1.0, 3.0, 4.0]]]],
            [[[[0.0, 1.0, 3.0, 4.0], [0.0, 1.0, 3.0, 4.0]]]],
        ]
    ).reshape(2, 1, 1, 2, 4)
    decoded = original.clone()
    decoded[0] = 0
    mask = torch.ones((2, 1, 2, 4))
    metrics = _seam_audit(original, decoded, mask)
    assert metrics["paired_ratio_defined_case_indices"] == [1]
    assert metrics["paired_ratio_defined_case_count"] == 1
    assert metrics["per_case_boundary_to_interior_ratio_change_paired"] == [0.0]
    assert metrics["boundary_to_interior_ratio_change_paired_case_equal"] == 0.0


def test_reviewed_gradient_probe_is_live_and_records_zero_negative_gradient():
    metrics = _gradient_probe()
    assert metrics["straight_through_estimator"] is False
    assert metrics["positive_example_gradient_l1"] > 0
    assert metrics["fully_negative_gradient_l1"] == 0
