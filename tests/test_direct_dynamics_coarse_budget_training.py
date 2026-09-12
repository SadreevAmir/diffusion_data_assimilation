from pathlib import Path

import torch

from assim_lib.config import load_json
from assim_lib.direct_dynamics_coarse_budget_training import (
    _indices_sha256,
    _scored_candidate,
    _validate_training_config,
)


def test_reviewed_training_config_is_exact_and_bounded():
    path = (
        Path(__file__).resolve().parents[1]
        / "config/experiments/train_direct_dynamics_coarse_budget_64_v1.json"
    )
    config = load_json(path)
    _validate_training_config(config)
    assert config["protocol"]["updates"] == 64
    assert config["protocol"]["extension"] == "forbidden"
    assert config["protocol"]["test_2023"] == "closed"
    assert _indices_sha256(config["train_indices"]) == config["train_indices_sha256"]


def test_step_zero_physical_objective_replays_and_has_candidate_gradient():
    height, width = 8, 6
    valid = torch.ones((1, 1, height, width), dtype=torch.float64)
    means = torch.zeros(6, dtype=torch.float64)
    stds = torch.ones(6, dtype=torch.float64)
    forecast = torch.full((1, 4, 6, height, width), 0.4, dtype=torch.float64)
    forecast[:, :, 1::2] = 0.2
    truth = forecast[:, 0].clone()
    coarse = torch.nn.functional.avg_pool2d(
        forecast.flatten(0, 1), 2, 2
    ).unflatten(0, (1, 4))
    control = coarse.clone()
    candidate = coarse.clone().requires_grad_(True)
    physical, objective, crps, energy = _scored_candidate(
        forecast, coarse, control, candidate, truth, valid, means, stds
    )
    assert torch.equal(physical, forecast)
    assert all(torch.isfinite(value) for value in (objective, crps, energy))
    objective.backward()
    assert candidate.grad is not None
    assert torch.isfinite(candidate.grad).all()
