import torch

from assim_lib.direct_dynamics_mixed_support_mask_runner import (
    CfmBatch,
    build_matched_models,
    d0_edge_neighbourhood,
    masked_cfm_loss,
    sample_masks,
)


def _batch(d0: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> CfmBatch:
    condition = torch.zeros((d0.shape[0], 15, *d0.shape[-2:]))
    condition[:, :1] = d0
    return CfmBatch(target=target, condition=condition, d0_occurrence=d0, valid=valid)


def test_seeded_candidate_control_are_parameter_matched():
    candidate, control = build_matched_models(seed=17, hidden_channels=4)
    assert sum(p.numel() for p in candidate.parameters()) == sum(p.numel() for p in control.parameters())
    assert all(torch.equal(a, b) for a, b in zip(candidate.parameters(), control.parameters(), strict=True))


def test_empty_and_full_d0_have_no_artificial_front():
    valid = torch.ones((1, 1, 8, 8))
    assert torch.count_nonzero(d0_edge_neighbourhood(torch.zeros_like(valid), valid)) == 0
    assert torch.count_nonzero(d0_edge_neighbourhood(torch.ones_like(valid), valid)) == 0


def test_actual_source_head_gets_birth_and_death_gradients_without_d0_edge():
    valid = torch.ones((2, 1, 8, 8))
    d0 = torch.cat((torch.zeros((1, 1, 8, 8)), torch.ones((1, 1, 8, 8))))
    target = torch.cat((torch.ones((1, 3, 8, 8)), torch.zeros((1, 3, 8, 8))))
    batch = _batch(d0, target, valid)
    candidate, _ = build_matched_models(seed=3, hidden_channels=4)
    generator = torch.Generator().manual_seed(9)
    uniform = torch.rand(target.shape, generator=generator).clamp(1e-5, 1 - 1e-5)
    noise = torch.randn(target.shape, generator=generator)
    loss, per_case = masked_cfm_loss(
        candidate, batch, time=torch.tensor([0.25, 0.75]), noise=noise, uniform=uniform
    )
    loss.backward()
    assert torch.isfinite(per_case).all() and torch.all(per_case > 0)
    source_norms = [p.grad.norm() for p in candidate.source_head.parameters()]
    assert all(torch.isfinite(value) and value > 0 for value in source_norms)
    front_norms = [p.grad for p in candidate.front_head.parameters()]
    assert all(value is not None and torch.count_nonzero(value) == 0 for value in front_norms)


def test_land_perturbations_do_not_change_forward_or_sampling():
    valid = torch.ones((1, 1, 8, 8))
    valid[..., :2, :3] = 0
    d0 = torch.zeros_like(valid)
    target = torch.zeros((1, 3, 8, 8))
    batch = _batch(d0, target, valid)
    candidate, _ = build_matched_models(seed=5, hidden_channels=4)
    state = torch.randn_like(target)
    time = torch.tensor([0.4])
    reference = candidate(state, time, batch.condition, d0, valid)
    altered_state = state.clone()
    altered_condition = batch.condition.clone()
    altered_d0 = d0.clone()
    altered_state[..., :2, :3] = 1e6
    altered_condition[..., :2, :3] = -1e6
    altered_d0[..., :2, :3] = 1
    changed = candidate(altered_state, time, altered_condition, altered_d0, valid)
    assert torch.equal(reference, changed)
    sample_a = sample_masks(candidate, condition=batch.condition, d0_occurrence=d0, valid=valid, noise=state)
    sample_b = sample_masks(
        candidate, condition=altered_condition, d0_occurrence=altered_d0, valid=valid,
        noise=altered_state,
    )
    assert torch.equal(sample_a, sample_b)
    assert torch.count_nonzero(sample_a[..., :2, :3]) == 0


def test_empty_ocean_case_is_rejected_without_denominator_clamp():
    valid = torch.zeros((1, 1, 8, 8))
    target = torch.zeros((1, 3, 8, 8))
    batch = _batch(valid, target, valid)
    candidate, _ = build_matched_models(seed=1, hidden_channels=4)
    try:
        masked_cfm_loss(
            candidate, batch, time=torch.tensor([0.5]), noise=target, uniform=torch.full_like(target, 0.5)
        )
    except ValueError as error:
        assert "at least one active ocean" in str(error)
    else:
        raise AssertionError("empty-ocean case was silently normalized")
