import pytest
import torch

from assim_lib.direct_dynamics_sit_support_decoder_scoring import (
    _persist_scores_before_reporting,
    _score,
    apply_support_decoder_physical,
)


def test_support_scoring_decoder_changes_only_sit_and_preserves_positive_coarse_law():
    fine = torch.zeros((1, 2, 6, 4, 4), dtype=torch.float64)
    fine[:, :, 0::2] = torch.arange(16, dtype=torch.float64).reshape(1, 1, 1, 4, 4)
    fine[:, :, 1::2] = torch.tensor(
        [[[-1.0, 0.0, 1.0, 2.0], [0.0, 1.0, 2.0, 3.0],
          [-2.0, -1.0, 3.0, 4.0], [-1.0, 0.0, 4.0, 5.0]]],
        dtype=torch.float64,
    )
    coarse = torch.zeros((1, 2, 6, 2, 2), dtype=torch.float64)
    coarse[:, :, 1::2] = torch.tensor(
        [[[[0.25, 1.0], [-1.0, 2.0]]]], dtype=torch.float64
    )
    mask = torch.ones((1, 1, 4, 4), dtype=torch.float64)
    decoded = apply_support_decoder_physical(fine, coarse, mask)
    assert torch.equal(decoded[:, :, 0::2], fine[:, :, 0::2])
    assert float(decoded[:, :, 1::2].min()) >= 0
    blocks = (
        decoded[:, :, 1::2]
        .reshape(1, 2, 3, 2, 2, 2, 2)
        .permute(0, 1, 2, 3, 5, 4, 6)
        .reshape(1, 2, 3, 2, 2, 4)
    )
    assert torch.allclose(blocks.mean(-1), coarse[:, :, 1::2].clamp_min(0))


def test_support_scoring_decoder_rejects_bad_shapes():
    with pytest.raises(ValueError, match="physical ensemble"):
        apply_support_decoder_physical(
            torch.zeros((1, 1, 5, 4, 4)),
            torch.zeros((1, 1, 6, 2, 2)),
            torch.ones((1, 1, 4, 4)),
        )


def test_score_exposes_actual_fractional_rank_key():
    truth = torch.zeros((2, 6, 4, 4), dtype=torch.float64)
    offsets = torch.linspace(-0.2, 0.2, 8, dtype=torch.float64)
    ensemble = truth[:, None].expand(-1, 8, -1, -1, -1).clone()
    ensemble += offsets.reshape(1, 8, 1, 1, 1)
    mask = torch.ones((2, 1, 4, 4), dtype=torch.float64)
    result = _score(ensemble, truth, mask, torch.ones(6, dtype=torch.float64))
    assert len(result["d6_sit_fractional_rank"]["case_equal_fractional_rank_frequencies"]) == 9


def test_numerical_result_survives_reporting_failure(tmp_path):
    metrics = tmp_path / "metrics.json"

    def fail():
        raise RuntimeError("injected image reporting failure")

    with pytest.raises(RuntimeError, match="injected image"):
        _persist_scores_before_reporting(metrics, {"score": 1.25}, fail)
    assert metrics.read_text().strip() == '{\n  "score": 1.25\n}'
