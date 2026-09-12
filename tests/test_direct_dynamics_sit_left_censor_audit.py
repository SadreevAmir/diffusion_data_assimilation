import itertools

import torch

from assim_lib.direct_dynamics_cascade_paired_evaluation import _weighted_case_fair_crps
from assim_lib.direct_dynamics_sit_left_censor_audit import (
    censor_sit,
    decode_with_exact_sit_zero,
)


def test_left_censor_preserves_sic_and_positive_sit():
    value = torch.tensor([-1.0, -0.2, 0.0, 0.3, 1.2, 2.0]).reshape(1, 1, 6, 1, 1)
    actual = censor_sit(value)
    assert torch.equal(actual[:, :, 0::2], value[:, :, 0::2])
    assert torch.equal(actual[:, :, 1::2], torch.tensor([0.0, 0.3, 2.0]).reshape(1, 1, 3, 1, 1))


def test_left_censor_creates_fractional_zero_ties_and_keeps_le_0p01_event():
    values = torch.tensor([-0.1, 0.0, 0.1]).reshape(1, 3, 1, 1, 1)
    six = values.expand(-1, -1, 6, -1, -1).clone()
    censored = censor_sit(six)[:, :, 1:2]
    assert torch.equal(values <= 0.01, censored <= 0.01)
    truth = torch.zeros((1, 1, 1, 1))
    less = (censored < truth[:, None]).sum(dim=1)
    equal = (censored == truth[:, None]).sum(dim=1)
    assert int(less) == 0 and int(equal) == 2


def test_population_expected_fair_crps_does_not_increase_under_projection():
    atoms = torch.tensor([-2.0, -0.25, 0.5, 2.0], dtype=torch.float64)
    truth = torch.tensor([[[[0.75]]]], dtype=torch.float64)
    weight = torch.ones_like(truth)
    before, after = [], []
    for left, right in itertools.product(atoms, repeat=2):
        members = torch.tensor([left, right], dtype=torch.float64).reshape(1, 2, 1, 1, 1)
        before.append(_weighted_case_fair_crps(members, truth, weight)[0])
        after.append(_weighted_case_fair_crps(members.clamp_min(0), truth, weight)[0])
    assert sum(after) / len(after) <= sum(before) / len(before) + 1e-14


def test_actual_fp32_encoded_zero_decodes_to_exact_physical_tie():
    means = torch.tensor([0.19301218262209952, 0.18969311571248842] * 3)
    stds = torch.tensor([0.36890927421384667, 0.4359687842884708] * 3)
    normalized = torch.zeros((1, 6, 1, 1), dtype=torch.float32)
    for channel in (1, 3, 5):
        normalized[:, channel] = -means[channel] / stds[channel]
    physical = decode_with_exact_sit_zero(normalized, means, stds)
    assert torch.equal(physical[:, 1::2], torch.zeros_like(physical[:, 1::2]))
