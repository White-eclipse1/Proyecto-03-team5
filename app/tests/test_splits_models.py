import pytest
from pydantic import ValidationError

from splits.models import SplitsConfig, load_splits_config


def test_valid_splits_round_trip():
    config = SplitsConfig.model_validate({"train": 0.70, "val": 0.15, "test": 0.15, "seed": 7})
    assert config.seed == 7


def test_default_seed_is_42():
    config = SplitsConfig.model_validate({"train": 0.70, "val": 0.15, "test": 0.15})
    assert config.seed == 42


def test_ratios_not_summing_to_one_are_rejected():
    with pytest.raises(ValidationError, match=r"sumar 1\.0"):
        SplitsConfig.model_validate({"train": 0.5, "val": 0.3, "test": 0.3})


@pytest.mark.parametrize("overrides", [{"train": 0.0}, {"train": 1.0}, {"val": -0.1}])
def test_ratio_out_of_range_is_rejected(overrides):
    document = {"train": 0.70, "val": 0.15, "test": 0.15, **overrides}
    with pytest.raises(ValidationError):
        SplitsConfig.model_validate(document)


def test_real_splits_yaml_loads_and_validates():
    config = load_splits_config()
    assert config.seed == 42
    assert abs(config.train + config.val + config.test - 1.0) < 1e-9


def test_broken_splits_yaml_is_rejected_not_silently_accepted(tmp_path):
    splits_path = tmp_path / "splits.yaml"
    splits_path.write_text("train: 0.9\nval: 0.3\ntest: 0.3\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_splits_config(splits_path)
