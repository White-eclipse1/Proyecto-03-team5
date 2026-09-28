import json
from copy import deepcopy

import pytest
import yaml
from pydantic import ValidationError

from analyzers.base import AnalyzerResult
from analyzers.imbalance import ImbalanceConfig, analyze_imbalance
from policies.imbalance import load_imbalance_config
from policies.models import load_quality_policy
from presentation.contracts import QualityCheck


def sample_coco():
    pairs = [(1, 1), (1, 1), (2, 1), (3, 1), (4, 1), (1, 2), (2, 2)]
    return {
        "images": [
            {"id": i, "file_name": f"sample-{i}.jpg", "width": 100, "height": 100}
            for i in range(1, 6)
        ],
        "categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}],
        "annotations": [
            {
                "id": i,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [0, 0, 20, 20],
                "area": 400,
                "iscrowd": 0,
                "segmentation": [],
            }
            for i, (image_id, category_id) in enumerate(pairs, start=1)
        ],
    }


def config(minimum=3, threshold=2):
    return ImbalanceConfig(min_images_per_class=minimum, threshold=threshold)


def test_distinct_images_multiclass_and_manual_ratio():
    result = analyze_imbalance(sample_coco(), config())
    assert isinstance(result, AnalyzerResult)
    assert result.check_name == "max_imbalance_ratio"
    assert result.metric_value == 4 / 2
    assert result.passed
    counts = result.details["images_per_category"]
    assert counts == [
        {"category_id": 1, "category_name": "cat", "image_count": 4, "image_ids": [1, 2, 3, 4]},
        {"category_id": 2, "category_name": "dog", "image_count": 2, "image_ids": [1, 2]},
    ]
    assert result.details["majority_class"] == counts[0]
    assert result.details["minority_class"] == counts[1]
    assert result.details["classes_below_minimum"] == [counts[1]]
    assert result.details["ratio_defined"] is True
    assert result.details["empty_category_ids"] == []


def test_equality_to_minimum_is_not_below():
    result = analyze_imbalance(sample_coco(), config(minimum=2))
    assert result.details["classes_below_minimum"] == []


@pytest.mark.parametrize("threshold, passed", [(2, True), (1.99, False), (3, True)])
def test_ratio_threshold_is_inclusive(threshold, passed):
    result = analyze_imbalance(sample_coco(), config(threshold=threshold))
    assert result.passed is passed
    assert result.metric_value == 2


def test_minimum_is_reported_without_combining_two_policy_actions():
    result = analyze_imbalance(sample_coco(), config(minimum=5))
    assert len(result.details["classes_below_minimum"]) == 2
    assert result.passed  # This result evaluates max_imbalance_ratio only.


def test_input_is_not_modified_or_aliased():
    coco = sample_coco()
    original = deepcopy(coco)
    result = analyze_imbalance(coco, config())
    assert coco == original
    result.details["images_per_category"][0]["image_ids"].append(99)
    assert coco == original


@pytest.mark.parametrize("all_empty", [False, True])
def test_empty_category_is_explicit_finite_and_fails(all_empty):
    coco = sample_coco()
    coco["categories"].append({"id": 3, "name": "bird"})
    if all_empty:
        coco["annotations"] = []
    result = analyze_imbalance(coco, config())
    assert result.metric_value == 0.0
    assert not result.passed
    assert result.details["ratio_defined"] is False
    assert result.details["empty_category_ids"] == ([1, 2, 3] if all_empty else [3])
    assert 3 in [entry["category_id"] for entry in result.details["classes_below_minimum"]]
    assert result.details["minority_class"]["image_count"] == 0
    json.dumps(result.model_dump(), allow_nan=False)
    QualityCheck.model_validate({**result.model_dump(), "action": "warn"})


@pytest.mark.parametrize("field", ["category_id", "image_id"])
def test_unknown_references_are_explicit_errors(field):
    coco = sample_coco()
    coco["annotations"][0][field] = 99
    with pytest.raises(ValueError, match=f"Unknown {field}: 99"):
        analyze_imbalance(coco, config())


@pytest.mark.parametrize("section, field", [("images", "image_id"), ("categories", "category_id")])
def test_duplicate_declared_ids_are_rejected(section, field):
    coco = sample_coco()
    coco[section].append(deepcopy(coco[section][0]))
    with pytest.raises(ValueError, match=f"Duplicate {field}"):
        analyze_imbalance(coco, config())


def test_no_categories_is_an_explicit_error():
    coco = sample_coco()
    coco["categories"] = []
    with pytest.raises(ValueError, match="declared category"):
        analyze_imbalance(coco, config())


def test_ties_are_deterministic_by_category_id():
    coco = sample_coco()
    coco["annotations"] = [ann for ann in coco["annotations"] if ann["image_id"] <= 2]
    coco["categories"].reverse()
    result = analyze_imbalance(coco, config())
    assert result.metric_value == 1
    assert result.details["majority_class"]["category_id"] == 1
    assert result.details["minority_class"]["category_id"] == 1


def test_configuration_comes_from_real_quality_yaml():
    policy = load_quality_policy()
    loaded = load_imbalance_config()
    assert loaded.min_images_per_class == policy.min_images_per_class.threshold == 300
    assert loaded.threshold == policy.max_imbalance_ratio.threshold == 20
    result = analyze_imbalance(sample_coco(), loaded)
    assert result.passed
    assert len(result.details["classes_below_minimum"]) == 2


def test_changing_yaml_changes_both_parameters(tmp_path):
    document = load_quality_policy().model_dump()
    document["min_images_per_class"]["threshold"] = 2
    document["max_imbalance_ratio"]["threshold"] = 1.5
    path = tmp_path / "quality.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    result = analyze_imbalance(sample_coco(), load_imbalance_config(path))
    assert not result.passed
    assert result.details["classes_below_minimum"] == []
    assert result.details["max_imbalance_ratio"] == 1.5


@pytest.mark.parametrize(
    "minimum, threshold", [(-1, 2), (3, -1), (3, float("inf")), (float("nan"), 2)]
)
def test_invalid_parameters_are_rejected(minimum, threshold):
    with pytest.raises(ValidationError):
        config(minimum, threshold)
