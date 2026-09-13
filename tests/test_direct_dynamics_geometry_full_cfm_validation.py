import json
from pathlib import Path
from unittest.mock import patch

import torch

from assim_lib.direct_dynamics_geometry_full_cfm_validation import (
    _binary_calibration,
    _check_contract,
    _fractional_rank,
    _sample_frozen_laws,
    _stratified_diagnostics,
    _verify_optimizer_state,
)


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "config/experiments/evaluate_direct_dynamics_geometry_full_cfm_ab_v1.json"


def test_frozen_contract_accepts_only_exact_checked_in_protocol():
    experiment = json.loads(CONFIG.read_text())
    protocol = _check_contract(experiment)
    assert protocol["case_indices"] == [
        0,
        777,
        1553,
        2330,
        3107,
        3883,
        4660,
        5436,
        6213,
        6990,
        7766,
        8543,
    ]
    changed = json.loads(CONFIG.read_text())
    changed["protocol"]["members"] = 16
    try:
        _check_contract(changed)
    except ValueError as error:
        assert "protocol" in str(error)
    else:
        raise AssertionError("changed validation member count was accepted")


def test_fractional_rank_splits_exact_ties_and_nulls_empty_groups():
    members = torch.tensor([[[[[0.0]]], [[[0.0]]], [[[1.0]]]]])
    truth = torch.tensor([[[[0.0]]]])
    selected = torch.ones_like(truth, dtype=torch.bool)
    result = _fractional_rank(members, truth, selected)
    assert result["frequencies"] == [1 / 3, 1 / 3, 1 / 3, 0.0]
    empty = _fractional_rank(members, truth, torch.zeros_like(selected))
    assert empty == {
        "case_count": 0,
        "point_count": 0,
        "frequencies": None,
        "rank_tv_to_uniform": None,
    }


def test_binary_calibration_uses_case_equal_brier_and_reports_empty_bins():
    member_event = torch.tensor(
        [
            [[[[False, True]]], [[[False, True]]]],
            [[[[True, True]]], [[[False, True]]]],
        ]
    )
    truth = torch.tensor([[[[False, True]]], [[[True, False]]]])
    valid = torch.ones_like(truth)
    result = _binary_calibration(member_event, truth, valid)
    assert result["case_equal_brier"] == 0.3125
    assert sum(result["bin_counts"]) == 4
    assert result["ece_10_bins"] == 0.375


def test_real_stats_thresholds_stay_in_encoded_coordinates_with_nextafter():
    means = torch.tensor([0.19301218262209952, 0.18969311571248842] * 3)
    stds = torch.tensor([0.36890927421384667, 0.4359687842884708] * 3)
    raw_truth = torch.full((1, 6, 1, 1), 0.01, dtype=torch.float32)
    truth = (raw_truth - means.view(1, 6, 1, 1)) / stds.view(1, 6, 1, 1)
    exact_members = truth[:, None].expand(-1, 8, -1, -1, -1).clone()
    valid = torch.ones((1, 1, 1, 1))
    identities = [{"case_id": "2022-01-01_slice00"}]
    exact = _stratified_diagnostics(
        exact_members, truth, raw_truth, valid, identities, means, stds
    )
    assert exact["binary"]["d3_sic"]["open_water_le_0p01"]["case_equal_brier"] == 0.0
    assert exact["overall"]["d3_sic"]["frequencies"] == [1 / 9] * 9

    upward = torch.nextafter(exact_members, torch.full_like(exact_members, float("inf")))
    shifted = _stratified_diagnostics(
        upward, truth, raw_truth, valid, identities, means, stds
    )
    assert shifted["binary"]["d3_sic"]["open_water_le_0p01"]["case_equal_brier"] == 1.0


def test_sampling_runs_all_three_full_public_laws_and_commits_each_case(tmp_path):
    class Model:
        def __init__(self):
            self.loaded = []

        def load_state_dict(self, state, strict=True):
            self.loaded.append((state, strict))

        def eval(self):
            return self

    class Sampler:
        def __init__(self):
            self.model = Model()

    sampler = Sampler()
    cases = [(0, {"token": "case"})]
    identities = [{"case_id": "2022-01-01_slice00"}]
    fixed_noise = torch.zeros((1, 2, 6, 2, 2))
    evidence_root = tmp_path / "sampling"
    evidence_root.mkdir()
    status = tmp_path / "status.json"
    bindings = tmp_path / "bindings.json"
    calls = []

    def fake_public(current_sampler, _item, noise, _size, _device):
        calls.append((len(current_sampler.model.loaded), noise.clone()))
        return torch.full((1, 6, 2, 2), float(len(current_sampler.model.loaded)))

    with patch(
        "assim_lib.direct_dynamics_geometry_full_cfm_validation._public_sample",
        side_effect=fake_public,
    ):
        ensembles = _sample_frozen_laws(
            sampler,
            {"control512": {"x": torch.tensor(1)}, "treatment512": {"x": torch.tensor(2)}},
            cases,
            identities,
            fixed_noise,
            (2, 2),
            2,
            torch.device("cpu"),
            evidence_root,
            status,
            bindings,
            "task",
        )
    assert list(ensembles) == ["ema6", "control512", "treatment512"]
    assert [entry[0] for entry in sampler.model.loaded] == [
        {"x": torch.tensor(1)},
        {"x": torch.tensor(2)},
    ]
    assert len(calls) == 6
    assert all((evidence_root / label / "case_00.pt").is_file() for label in ensembles)
    assert json.loads(status.read_text())["completed_laws"] == 2


def test_nonfinite_sample_is_committed_before_failure(tmp_path):
    class Model:
        def load_state_dict(self, _state, strict=True):
            return strict

        def eval(self):
            return self

    class Sampler:
        model = Model()

    evidence_root = tmp_path / "sampling"
    evidence_root.mkdir()
    status = tmp_path / "status.json"
    bindings = tmp_path / "bindings.json"
    with patch(
        "assim_lib.direct_dynamics_geometry_full_cfm_validation._public_sample",
        return_value=torch.full((1, 6, 2, 2), float("nan")),
    ):
        try:
            _sample_frozen_laws(
                Sampler(),
                {"control512": {}, "treatment512": {}},
                [(0, {})],
                [{"case_id": "2022-01-01_slice00"}],
                torch.zeros((1, 1, 6, 2, 2)),
                (2, 2),
                1,
                torch.device("cpu"),
                evidence_root,
                status,
                bindings,
                "task",
            )
        except FloatingPointError:
            pass
        else:
            raise AssertionError("non-finite sample was accepted")
    path = evidence_root / "ema6/case_00.pt"
    assert path.is_file()
    assert torch.isnan(torch.load(path, weights_only=True)["normalized"]).all()
    manifest = json.loads((evidence_root / "ema6/manifest.json").read_text())
    assert manifest["completed_cases"][0]["sha256"]


def test_optimizer_parameter_identifiers_must_be_unique():
    model = torch.nn.Linear(1, 1)
    state = {
        "step": torch.tensor(512.0),
        "exp_avg": torch.zeros_like(model.weight),
        "exp_avg_sq": torch.zeros_like(model.weight),
    }
    payload = {
        "arm": "control",
        "completed_update": 512,
        "optimizer": {
            "param_groups": [{"lr": 1e-5, "weight_decay": 0.0, "params": [0, 0]}],
            "state": {0: state},
        },
    }
    try:
        _verify_optimizer_state(payload, model, "control", 512)
    except ValueError as error:
        assert "coverage" in str(error)
    else:
        raise AssertionError("duplicate AdamW identifiers were accepted")


def test_validation_persists_complete_samples_before_any_score():
    source = Path(
        "assim_lib/direct_dynamics_geometry_full_cfm_validation.py"
    ).read_text()
    save_position = source.index(
        'complete_path = output / "complete_three_law_ensembles_before_scoring.pt"'
    )
    score_position = source.index("scores: dict[str, Any] = {}", save_position)
    assert save_position < score_position
    assert '"frozen_numerical_result_before_plots"' not in source[source.index("_atomic_json(numerical_path, result)") :]
    assert 'checkpoint_manifest[arm]["512"]["optimizer_recovery"]' in source


def test_validation_wrapper_is_exact_commit_one_gpu_and_non_reusing():
    source = Path("scripts/run_direct_dynamics_geometry_full_cfm_validation.sh").read_text()
    assert "GEOMETRY_FULL_CFM_VALIDATION_EXPECTED_COMMIT" in source
    assert "git status --porcelain --untracked-files=all" in source
    assert "EXPECTED_GPU_UUID" in source
    assert "flock -n 9" in source
    assert "refusing to reuse" in source
    assert "assim_lib.direct_dynamics_geometry_full_cfm_validation" in source
