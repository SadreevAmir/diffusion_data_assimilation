from assim_lib.config import load_json
from assim_lib.direct_dynamics_mixed_support_train_stats import MODE


def test_stats_contract_excludes_heldout_and_test_years():
    config = load_json("config/admission/direct_dynamics_mixed_support_train2016_2020_stats_v1.json")
    assert config["mode"] == MODE
    assert config["train_years"] == [2016, 2017, 2018, 2019, 2020]
    assert config["heldout_year"] == 2021
    assert config["test_2023_access_allowed"] is False
    assert config["resource_kind"] == "cpu"


def test_training_config_binds_exact_stats_artifact():
    config = load_json("config/admission/direct_dynamics_mixed_support_train2016_2020_v2.json")
    assert config["conditioning_stats_artifact"]["sha256"] == (
        "8bf75a878ab4316a5fc9a83ce40ed239cd5292f397696e651353b720982e90c2"
    )
    assert config["test_2023_access_allowed"] is False
