import torch

from assim_lib.direct_dynamics_d9_sic_rank_atom_audit import (
    _coarse_atom_selectors,
    _fine_atom_selectors,
    conditional_rank_atom_metrics,
)


def test_atom_groups_reconstruct_rank_and_retain_bias():
    mean, std = 0.2, 0.4
    physical_truth = torch.tensor([[[[0.0, 0.5, 1.0]]]])
    truth = (physical_truth - mean) / std
    members_physical = torch.tensor(
        [[[[[0.0, 0.4, 0.9]]], [[[0.1, 0.6, 1.0]]]]]
    )
    members = (members_physical - mean) / std
    fraction = torch.ones_like(truth)
    result = conditional_rank_atom_metrics(
        members, truth, fraction, physical_mean=mean, physical_std=std
    )
    assert result["rank_mixture_reconstruction_max_abs"] < 1e-15
    assert result["groups"]["exact_zero"]["case_count"] == 1
    assert result["groups"]["interior"]["case_count"] == 1
    assert result["groups"]["exact_one"]["case_count"] == 1
    assert abs(result["groups"]["interior"]["case_equal_signed_bias_physical_over_nonempty_cases"]) < 1e-7


def test_coarse_atoms_come_from_fine_membership_with_real_normalization():
    mean, std = 0.19301218262209952, 0.36890927421384667
    mean_tensor = torch.tensor(mean, dtype=torch.float32)
    std_tensor = torch.tensor(std, dtype=torch.float32)
    zero = -mean_tensor / std_tensor
    one = (torch.ones((), dtype=torch.float32) - mean_tensor) / std_tensor
    near_zero = torch.nextafter(zero, torch.tensor(float("inf")))
    truth = torch.full((1, 1, 4, 12), zero)
    valid = torch.zeros_like(truth)
    blocks = []
    for row in range(2):
        for column in range(6):
            blocks.append((slice(2 * row, 2 * row + 2), slice(2 * column, 2 * column + 2)))

    # Four zero and four one blocks exercise every coastal occupancy n=1..4.
    for index, (ys, xs) in enumerate(blocks[:4], start=1):
        y0, x0 = ys.start, xs.start
        for offset in range(index):
            valid[0, 0, y0 + offset // 2, x0 + offset % 2] = 1
    for index, (ys, xs) in enumerate(blocks[4:8], start=1):
        y0, x0 = ys.start, xs.start
        for offset in range(index):
            valid[0, 0, y0 + offset // 2, x0 + offset % 2] = 1
            truth[0, 0, y0 + offset // 2, x0 + offset % 2] = one

    # Remaining blocks are interior: mixed atoms and a value one ULP above zero.
    ys, xs = blocks[8]
    valid[0, 0, ys, xs] = 1
    truth[0, 0, ys, xs] = torch.tensor([[zero, one], [zero, one]])
    ys, xs = blocks[9]
    valid[0, 0, ys.start, xs.start] = 1
    truth[0, 0, ys.start, xs.start] = near_zero

    fine = _fine_atom_selectors(truth, valid, physical_mean=mean, physical_std=std)
    coarse = _coarse_atom_selectors(fine, valid)
    assert int(coarse["exact_zero"].sum()) == 4
    assert int(coarse["exact_one"].sum()) == 4
    assert int(coarse["interior"].sum()) == 2
    assert torch.equal(
        coarse["exact_zero"] | coarse["interior"] | coarse["exact_one"],
        torch.nn.functional.avg_pool2d(valid, 2, 2) > 0,
    )
