import json

import pytest
import torch

from assim_lib.direct_dynamics_cascade import masked_block_average
from assim_lib import direct_dynamics_sic_support_decoder_scoring as scoring
from assim_lib.direct_dynamics_sic_support_decoder_scoring import (
    apply_sic_support_decoder_physical,
    canonical_physical_decode,
)


def test_joint_decoder_changes_only_sic_and_preserves_clipped_sic_coarse_means():
    physical = torch.tensor(
        [[[[[-0.2, 0.3], [0.8, 1.4]], [[0.1, 0.2], [0.3, 0.4]]] * 3]],
        dtype=torch.float32,
    )
    physical = physical.expand(1, 2, 6, 2, 2).clone()
    mask = torch.ones(1, 1, 2, 2)
    coarse, _ = masked_block_average(physical.flatten(0, 1), mask.expand(2, -1, -1, -1))
    coarse = coarse.unflatten(0, (1, 2))
    coarse[:, :, 0::2] = torch.tensor([[[[[1.2]], [[-0.1]], [[0.5]]]]])
    decoded = apply_sic_support_decoder_physical(physical, coarse, mask)
    assert torch.equal(decoded[:, :, 1::2], physical[:, :, 1::2].double())
    recovered, fraction = masked_block_average(
        decoded[:, :, 0::2].flatten(0, 1), mask[:, None].expand(-1, 2, -1, -1, -1).flatten(0, 1)
    )
    target = coarse[:, :, 0::2].double().clamp(0, 1).flatten(0, 1)
    assert torch.allclose(recovered[fraction.expand_as(recovered) > 0], target[fraction.expand_as(target) > 0], atol=2e-14, rtol=0)


def test_canonical_decode_restores_only_exact_fp32_sic_atoms():
    sic_mean = torch.tensor(0.19301218262209952, dtype=torch.float32)
    sic_std = torch.tensor(0.36890927421384667, dtype=torch.float32)
    means = torch.tensor(
        [sic_mean.item(), 1.5, sic_mean.item(), 1.5, sic_mean.item(), 1.5],
        dtype=torch.float32,
    )
    stds = torch.tensor(
        [sic_std.item(), 0.4, sic_std.item(), 0.4, sic_std.item(), 0.4],
        dtype=torch.float32,
    )
    encoded_zero = -sic_mean / sic_std
    encoded_one = (torch.tensor(1.0, dtype=torch.float32) - sic_mean) / sic_std
    below_zero_atom = torch.nextafter(encoded_zero, torch.tensor(-torch.inf))
    above_zero_atom = torch.nextafter(encoded_zero, torch.tensor(torch.inf))
    below_one_atom = torch.nextafter(encoded_one, torch.tensor(-torch.inf))
    above_one_atom = torch.nextafter(encoded_one, torch.tensor(torch.inf))
    normalized = torch.zeros(1, 6, 1, 6, dtype=torch.float32)
    normalized[0, 0, 0] = torch.stack(
        (
            encoded_zero,
            encoded_one,
            below_zero_atom,
            above_zero_atom,
            below_one_atom,
            above_one_atom,
        )
    )
    normalized[0, 1, 0] = torch.tensor((-1.0, 0.0, 1.0, 2.0, 3.0, 4.0))

    decoded = canonical_physical_decode(normalized, means, stds)

    assert decoded[0, 0, 0, 0].item() == 0.0
    assert decoded[0, 0, 0, 1].item() == 1.0
    neighbors = normalized[0, 0, 0, 2:6]
    assert torch.all(neighbors != encoded_zero)
    assert torch.all(neighbors != encoded_one)
    expected_neighbors = neighbors.double() * sic_std.double() + sic_mean.double()
    assert torch.all(expected_neighbors[:2] != 0.0)
    assert torch.all(expected_neighbors[2:] != 1.0)
    assert torch.equal(decoded[0, 0, 0, 2:6], expected_neighbors)
    expected_sit = (normalized[0, 1] * stds[1] + means[1]).double()
    assert torch.equal(decoded[0, 1], expected_sit)


def test_source_gate_failure_is_recorded_after_reservation(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"schema_version": "sic_support_decoder_scoring_v1"}),
        encoding="utf-8",
    )
    output = tmp_path / "status.json"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(scoring.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(scoring.torch, "set_num_threads", lambda _value: None)
    monkeypatch.setattr(scoring.torch, "set_num_interop_threads", lambda _value: None)
    monkeypatch.setattr(scoring.torch, "get_num_threads", lambda: 6)
    monkeypatch.setattr(scoring.torch, "get_num_interop_threads", lambda: 1)

    def fail_source_gate(_config):
        raise ValueError("injected source gate failure")

    monkeypatch.setattr(scoring, "_source_gate_is_complete", fail_source_gate)
    with pytest.raises(ValueError, match="injected source gate failure"):
        scoring.run(config_path, output)
    status = json.loads(output.read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["error_type"] == "ValueError"
    assert status["error"] == "injected source gate failure"
