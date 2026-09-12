import torch
from torchdiffeq import odeint

from assim_lib.direct_dynamics_coarse_budget_production_replay import (
    compact_first_divergence_diagnostic,
    production_frozen_prefix,
    production_terminal_sample,
    production_velocity,
    sequential_terminal_candidate_control,
)
from assim_lib.runtime import make_normalized_xy_grid


class NonlinearCoarse(torch.nn.Module):
    def forward(self, value, timestamp, return_dict=True):
        del return_dict
        time = timestamp.reshape(-1, 1, 1, 1) / 1000.0
        return (torch.tanh(value[:, :6]) + 0.03 * time,)


def _inputs(batch=1):
    torch.manual_seed(73)
    model = NonlinearCoarse().eval()
    state = torch.randn((batch, 6, 4, 4), dtype=torch.float32)
    condition = torch.randn((batch, 6, 4, 4), dtype=torch.float32)
    active = torch.ones((batch, 1, 4, 4), dtype=torch.float32)
    active[:, :, :1, :1] = 0
    state = torch.where(active.expand_as(state) > 0, state, torch.zeros_like(state))
    grid = make_normalized_xy_grid(
        4, 4, device=torch.device("cpu"), dtype=torch.float32
    )
    return model, state, condition, active, grid


def test_production_order_matches_every_torchdiffeq_state_batch_one():
    model, state, condition, active, grid = _inputs(batch=1)
    times = torch.linspace(1.0, 0.0, 17, dtype=torch.float32)

    def function(time, value):
        return production_velocity(model, value, condition, active, grid, time)

    reference = odeint(
        function,
        state,
        times,
        method="rk4",
        options={"step_size": 1.0 / 16.0},
    )
    prefix, states = production_frozen_prefix(
        model, state, condition, active, grid, return_states=True
    )
    candidate = production_terminal_sample(
        model, prefix, condition, active, grid
    )
    replay = torch.stack((*states, candidate))
    assert torch.equal(replay, reference)


def test_sequential_member_path_is_separate_from_batching_effect():
    model, state, condition, active, grid = _inputs(batch=4)
    candidate, control, prefix = sequential_terminal_candidate_control(
        model, model, state, condition, active, grid
    )
    assert torch.equal(candidate, control)
    expected = []
    for index in range(4):
        item_prefix = production_frozen_prefix(
            model,
            state[index : index + 1],
            condition[index : index + 1],
            active[index : index + 1],
            grid,
        )
        expected.append(
            production_terminal_sample(
                model,
                item_prefix,
                condition[index : index + 1],
                active[index : index + 1],
                grid,
            )
        )
    assert torch.equal(candidate, torch.cat(expected))
    assert prefix.shape == state.shape


def test_first_divergence_and_terminal_gradient_are_explicit():
    result = compact_first_divergence_diagnostic()
    assert result["status"] == "pass"
    assert result["first_legacy_divergence"] is not None
    assert result["production_call_count"] == 64
    assert result["production_order_all_states_bitwise_equal"]
    assert result["terminal_parameter_gradient_norm"] > 0
    assert result["prefix_has_no_graph"]
    assert result["frozen_parameter_gradients_absent"]
