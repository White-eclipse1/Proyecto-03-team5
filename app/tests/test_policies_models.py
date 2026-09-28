import pytest
from pydantic import ValidationError

from policies.models import QualityPolicy, load_quality_policy

VALID = {
    "min_images_per_class": {"threshold": 300, "action": "fail"},
    "max_imbalance_ratio": {"threshold": 20.0, "action": "warn"},
    "max_small_object_ratio": {
        "threshold": 0.40,
        "width_px": 32,
        "height_px": 32,
        "action": "warn",
    },
    "degenerate_boxes": {"threshold": 0, "action": "fail"},
    "cross_split_leakage": {"threshold": 0, "action": "fail"},
    "duplicate_similarity_threshold": {"threshold": 0.94, "action": "warn"},
    "min_spatial_dispersion": {"threshold": 0.15, "action": "warn"},
}


def test_valid_policy_round_trips():
    policy = QualityPolicy.model_validate(VALID)
    assert policy.min_images_per_class.threshold == 300
    assert policy.max_small_object_ratio.width_px == 32


def test_unknown_action_is_rejected_naming_action():
    document = {**VALID, "degenerate_boxes": {"threshold": 0, "action": "block"}}
    with pytest.raises(ValidationError, match="action"):
        QualityPolicy.model_validate(document)


def test_missing_rule_is_rejected_naming_the_rule():
    document = {key: value for key, value in VALID.items() if key != "cross_split_leakage"}
    with pytest.raises(ValidationError, match="cross_split_leakage"):
        QualityPolicy.model_validate(document)


def test_typo_ed_extra_key_is_rejected():
    document = {**VALID, "degenerate_boxes": {"threshold": 0, "action": "fail", "acton": "fail"}}
    with pytest.raises(ValidationError, match="acton"):
        QualityPolicy.model_validate(document)


def test_non_numeric_threshold_is_rejected():
    document = {**VALID, "max_imbalance_ratio": {"threshold": "veinte", "action": "warn"}}
    with pytest.raises(ValidationError, match="threshold"):
        QualityPolicy.model_validate(document)


def test_real_quality_yaml_loads_and_validates():
    policy = load_quality_policy()
    assert policy.min_images_per_class.action == "fail"
    assert policy.max_small_object_ratio.threshold == 0.40


def test_broken_quality_yaml_is_rejected_not_a_key_error(tmp_path):
    policy_path = tmp_path / "quality.yaml"
    policy_path.write_text(
        "min_images_per_class:\n  threshold: 300\n  action: fail\n", encoding="utf-8"
    )
    with pytest.raises(ValidationError):
        load_quality_policy(policy_path)
