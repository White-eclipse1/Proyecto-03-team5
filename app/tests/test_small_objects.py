from copy import deepcopy

import pytest
from pydantic import ValidationError

from analyzers.base import AnalyzerResult
from analyzers.small_objects import SmallObjectConfig, analyze_small_objects
from policies.small_objects import load_small_object_config


def config(**overrides):
    return SmallObjectConfig.model_validate(
        {"width_px": 32, "height_px": 32, "threshold": 0.40, **overrides}
    )


def coco_with_boxes(*boxes):
    return {
        "images": [{"id": 10, "width": 640, "height": 480, "file_name": "sample.jpg"}],
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "car"}],
        "annotations": [
            {"id": i, "image_id": 10, "category_id": 1, "bbox": list(box)}
            for i, box in enumerate(boxes, start=1)
        ],
    }


def test_20_and_50_pixel_boxes_report_only_first_and_do_not_mutate_input():
    coco = coco_with_boxes((0, 0, 20, 20), (30, 30, 50, 50))
    before = deepcopy(coco)
    result = analyze_small_objects(coco, config())
    assert isinstance(result, AnalyzerResult)
    assert result.model_dump() == {
        "check_name": "max_small_object_ratio",
        "passed": False,
        "metric_value": 0.5,
        "details": {
            "small_objects": 1,
            "total_objects": 2,
            "most_affected_class": {
                "category_id": 1,
                "category_name": "person",
                "small_objects": 1,
                "total_objects": 2,
                "ratio": 0.5,
            },
            "offending_samples": [
                {"annotation_id": 1, "image_id": 10, "category_id": 1, "bbox": [0, 0, 20, 20]}
            ],
        },
    }
    assert coco == before
    result.details["offending_samples"][0]["bbox"][2] = 1
    assert coco == before


def test_strict_dimension_boundaries_and_not_area():
    coco = coco_with_boxes((0, 0, 32, 32), (0, 0, 20, 50), (0, 0, 31, 32), (0, 0, 31, 31))
    for annotation in coco["annotations"]:
        annotation["area"] = 1  # Segmentation area must not determine bbox size.
    result = analyze_small_objects(coco, config())
    assert result.metric_value == 0.25
    assert result.passed
    assert [item["annotation_id"] for item in result.details["offending_samples"]] == [4]


def test_size_and_allowed_ratio_are_independently_configurable():
    coco = coco_with_boxes((0, 0, 20, 20), (0, 0, 50, 50))
    assert analyze_small_objects(coco, config(width_px=20)).metric_value == 0
    assert analyze_small_objects(coco, config(height_px=20)).metric_value == 0
    assert analyze_small_objects(coco, config(width_px=60, height_px=60)).metric_value == 1
    result = analyze_small_objects(coco, config(threshold=0.5))
    assert result.metric_value == 0.5
    assert result.passed  # Equality to the allowed ratio passes.


def test_most_affected_is_by_count_with_category_id_tiebreak():
    coco = coco_with_boxes(*([(0, 0, 20, 20)] * 3 + [(0, 0, 50, 50)] * 3))
    coco["annotations"][0]["category_id"] = 2
    result = analyze_small_objects(coco, config())
    affected = result.details["most_affected_class"]
    assert affected["category_id"] == 1  # 2 small boxes vs 1, despite a smaller class ratio.
    assert affected["ratio"] == 0.4
    coco["annotations"] = coco["annotations"][:2]
    result = analyze_small_objects(coco, config())
    assert result.details["most_affected_class"]["category_id"] == 1


@pytest.mark.parametrize("boxes", [(), ((0, 0, 50, 50),)])
def test_no_offenders(boxes):
    result = analyze_small_objects(coco_with_boxes(*boxes), config())
    assert result.metric_value == 0
    assert result.passed
    assert result.details == {
        "small_objects": 0,
        "total_objects": len(boxes),
        "most_affected_class": None,
        "offending_samples": [],
    }


def test_crowd_annotations_are_counted():
    coco = coco_with_boxes((0, 0, 20, 20))
    coco["annotations"][0]["iscrowd"] = 1
    assert analyze_small_objects(coco, config()).metric_value == 1


@pytest.mark.parametrize("bbox", [(0, 0, 0, 20), (0, 0, -1, 20), (0, 0, float("nan"), 20)])
def test_invalid_boxes_do_not_silently_distort_ratio(bbox):
    with pytest.raises(ValueError, match="bbox"):
        analyze_small_objects(coco_with_boxes(bbox), config())


@pytest.mark.parametrize(
    "overrides", [{"width_px": 0}, {"height_px": -1}, {"threshold": 1.1}, {"width_px": "32"}]
)
def test_invalid_configuration_is_rejected(overrides):
    with pytest.raises(ValidationError):
        config(**overrides)


def test_real_yaml_defaults_and_required_box_example():
    pytest.importorskip("yaml", reason="PyYAML declared in pyproject.toml is not installed")
    loaded = load_small_object_config()
    assert loaded.model_dump() == {"width_px": 32.0, "height_px": 32.0, "threshold": 0.40}
    result = analyze_small_objects(coco_with_boxes((0, 0, 20, 20), (0, 0, 50, 50)), loaded)
    assert result.metric_value == 0.5
    assert [item["annotation_id"] for item in result.details["offending_samples"]] == [1]


def test_yaml_configuration_is_used_without_python_defaults(tmp_path):
    pytest.importorskip("yaml", reason="PyYAML declared in pyproject.toml is not installed")
    policy = tmp_path / "quality.yaml"
    policy.write_text(
        "max_small_object_ratio:\n  width_px: 16\n  height_px: 24\n"
        "  threshold: 0.75\n  action: warn\n",
        encoding="utf-8",
    )
    loaded = load_small_object_config(policy)
    assert loaded.model_dump() == {"width_px": 16.0, "height_px": 24.0, "threshold": 0.75}
    assert analyze_small_objects(coco_with_boxes((0, 0, 20, 20)), loaded).metric_value == 0
    policy.write_text("max_small_object_ratio:\n  threshold: 0.40\n", encoding="utf-8")
    with pytest.raises(KeyError):
        load_small_object_config(policy)
