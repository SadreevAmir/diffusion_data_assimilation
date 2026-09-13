from __future__ import annotations

import itertools

import torch

from assim_lib.direct_dynamics_geometry_score import (
    GROUP_WEIGHTS,
    _edge_band_masks,
    geometry_energy_score,
    geometry_observable_vectors,
    unbiased_energy_score_vectors,
)


def _inputs(batch: int = 2, members: int = 4, height: int = 16, width: int = 16):
    generator = torch.Generator().manual_seed(17)
    truth = torch.rand(batch, 6, height, width, generator=generator)
    truth[:, 1::2] *= 2.0
    ensemble = truth[:, None] + 0.08 * torch.randn(
        batch, members, 6, height, width, generator=generator
    )
    valid = torch.ones(batch, 1, height, width)
    valid[..., :2, :3] = 0
    initial_sic = torch.rand(batch, 1, height, width, generator=generator)
    area = torch.linspace(1.0, 1.4, height).view(1, 1, height, 1).expand(1, 1, height, width)
    return ensemble, truth, valid, initial_sic, area


def test_unbiased_energy_uses_all_six_unordered_pairs_and_is_permutation_invariant():
    members = torch.tensor([[[0.0], [1.0], [3.0], [7.0]]], dtype=torch.float64)
    truth = torch.tensor([[2.0]], dtype=torch.float64)
    observation = torch.tensor([2.0, 1.0, 1.0, 5.0], dtype=torch.float64).mean()
    pair_sum = sum(abs(left - right) for left, right in itertools.combinations((0, 1, 3, 7), 2))
    expected = observation - pair_sum / (4 * 3)
    actual = unbiased_energy_score_vectors(members, truth)
    assert actual == expected
    assert unbiased_energy_score_vectors(members[:, [2, 0, 3, 1]], truth) == actual


def test_geometry_score_is_zero_for_four_exact_members():
    _, truth, valid, initial_sic, area = _inputs()
    members = truth[:, None].repeat(1, 4, 1, 1, 1)
    score, slices = geometry_energy_score(
        members,
        truth,
        valid,
        initial_sic,
        area,
        sic_scale=0.3,
        sit_scale=0.5,
        product_scale=0.25,
    )
    assert score == 0
    assert set(slices) == set(GROUP_WEIGHTS)


def test_geometry_map_is_land_invariant_and_has_finite_nonzero_gradient():
    members, truth, valid, initial_sic, area = _inputs()
    members.requires_grad_(True)
    score, _ = geometry_energy_score(
        members,
        truth,
        valid,
        initial_sic,
        area,
        sic_scale=0.3,
        sit_scale=0.5,
        product_scale=0.25,
    )
    score.backward()
    assert torch.isfinite(score)
    assert members.grad is not None and torch.isfinite(members.grad).all()
    assert members.grad.abs().sum() > 0
    assert torch.all(members.grad[..., :2, :3] == 0)

    changed = members.detach().clone()
    changed[..., :2, :3] = 1e6
    changed_score, _ = geometry_energy_score(
        changed,
        truth,
        valid,
        initial_sic,
        area,
        sic_scale=0.3,
        sit_scale=0.5,
        product_scale=0.25,
    )
    assert torch.allclose(changed_score, score.detach())


def test_zero_ice_condition_produces_finite_edge_observables():
    members, truth, valid, _, area = _inputs(batch=1)
    initial_sic = torch.zeros(1, 1, 16, 16)
    member_vector, truth_vector, slices = geometry_observable_vectors(
        members,
        truth,
        valid,
        initial_sic,
        area,
        sic_scale=0.3,
        sit_scale=0.5,
        product_scale=0.25,
    )
    left, right = slices["initial_edge_profiles"]
    assert torch.isfinite(member_vector).all() and torch.isfinite(truth_vector).all()
    assert torch.all(member_vector[..., left:right] == 0)
    assert torch.all(truth_vector[..., left:right] == 0)


def test_coastline_is_not_mistaken_for_an_ice_open_water_edge():
    valid = torch.ones(1, 1, 24, 24)
    valid[..., :5, :] = 0
    valid[..., :, :3] = 0
    all_water_ice = valid.clone()
    bands = _edge_band_masks(all_water_ice, valid)
    assert torch.all(bands == 0)


def test_temporal_shuffle_changes_d80_increment_observables():
    members, truth, valid, initial_sic, area = _inputs(batch=1)
    original, _, slices = geometry_observable_vectors(
        members,
        truth,
        valid,
        initial_sic,
        area,
        sic_scale=0.3,
        sit_scale=0.5,
        product_scale=0.25,
    )
    shuffled = members[:, :, [4, 5, 2, 3, 0, 1]]
    changed, _, _ = geometry_observable_vectors(
        shuffled,
        truth,
        valid,
        initial_sic,
        area,
        sic_scale=0.3,
        sit_scale=0.5,
        product_scale=0.25,
    )
    left, right = slices["d80_increments"]
    assert not torch.allclose(original[..., left:right], changed[..., left:right])
