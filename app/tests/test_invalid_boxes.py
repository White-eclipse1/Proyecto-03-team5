import json
from copy import deepcopy

import pytest
import yaml
from pydantic import ValidationError

from analyzers.invalid_boxes import InvalidBoxConfig, analyze_invalid_boxes
from analyzers.small_objects import SmallObjectConfig, analyze_small_objects
from ingestion.models import CocoDataset
from policies.invalid_boxes import load_invalid_box_config
from policies.models import load_quality_policy
from presentation.contracts import QualityCheck


def document(bbox=None, area=100):
    return {
        "images": [{"id": 1, "file_name": "example.jpg", "width": 100, "height": 80}],
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 10, 10] if bbox is None else bbox,
                "area": area,
                "iscrowd": 0,
                "segmentation": [],
            }
        ],
    }


@pytest.mark.parametrize(
    "bbox, reason",
    [
        ([0, 0, -10, 10], "non_positive_width"),
        ([0, 0, 10, -10], "non_positive_height"),
        ([0, 0, 0, 10], "non_positive_width"),
        ([0, 0, 10, 0], "non_positive_height"),
        ([-1, 0, 10, 10], "negative_x"),
        ([0, -1, 10, 10], "negative_y"),
        ([91, 0, 10, 10], "exceeds_image_width"),
        ([0, 71, 10, 10], "exceeds_image_height"),
    ],
)
def test_each_geometric_error(bbox, reason):
    result = analyze_invalid_boxes(document(bbox, bbox[2] * bbox[3]), load_invalid_box_config())
    assert result.metric_value == 1
    assert not result.passed
    assert result.details["offending_samples"][0]["reasons"] == [reason]


def test_exact_image_border_and_correct_area_are_valid():
    result = analyze_invalid_boxes(document([90, 70, 10, 10]), load_invalid_box_config())
    assert result.check_name == "degenerate_boxes"
    assert result.metric_value == 0
    assert result.passed
    assert result.details == {
        "total_annotations": 1,
        "invalid_annotations": 0,
        "offending_samples": [],
    }


def test_area_mismatch_has_verifiable_evidence():
    result = analyze_invalid_boxes(document(area=120), load_invalid_box_config())
    assert result.details["offending_samples"] == [
        {
            "annotation_id": 1,
            "image_id": 1,
            "bbox": [0, 0, 10, 10],
            "reasons": ["area_mismatch"],
            "reported_area": 120,
            "calculated_area": 100,
            "image_width": 100,
            "image_height": 80,
        }
    ]
    QualityCheck.model_validate({**result.model_dump(), "action": "fail"})


@pytest.mark.parametrize("delta, invalid", [(5e-7, 0), (-5e-7, 0), (1e-3, 1)])
def test_absolute_area_tolerance(delta, invalid):
    result = analyze_invalid_boxes(document(area=100 + delta), load_invalid_box_config())
    assert result.metric_value == invalid


def test_normal_floating_point_roundoff_is_valid():
    result = analyze_invalid_boxes(document([0, 0, 0.1, 0.2], 0.02), load_invalid_box_config())
    assert result.passed


@pytest.mark.parametrize("delta, invalid", [(0.0005, 0), (0.01, 1)])
def test_relative_area_tolerance(delta, invalid):
    coco = document([0, 0, 1000, 1000], 1_000_000 + delta)
    coco["images"][0].update(width=1000, height=1000)
    result = analyze_invalid_boxes(coco, load_invalid_box_config())
    assert result.metric_value == invalid


def test_multiple_reasons_count_annotation_once():
    result = analyze_invalid_boxes(document([-1, -2, -3, -4], 99), load_invalid_box_config())
    assert result.metric_value == 1
    assert result.details["invalid_annotations"] == 1
    assert result.details["offending_samples"][0]["reasons"] == [
        "non_positive_width",
        "non_positive_height",
        "negative_x",
        "negative_y",
        "area_mismatch",
    ]


def test_unknown_image_is_reported_with_other_evaluable_errors():
    coco = document([-1, 0, 10, 10], 99)
    coco["annotations"][0]["image_id"] = 99
    result = analyze_invalid_boxes(coco, load_invalid_box_config())
    sample = result.details["offending_samples"][0]
    assert sample["reasons"] == ["negative_x", "unknown_image_id", "area_mismatch"]
    assert sample["calculated_area"] == 100
    assert sample["image_width"] is None
    assert sample["image_height"] is None


def test_both_injected_boxes_reported_before_strict_ingestion_and_small_objects():
    coco = document([0, 0, -10, 10], 100)
    second = deepcopy(coco["annotations"][0])
    second.update(id=2, bbox=[95, 0, 10, 10], area=100)
    coco["annotations"].append(second)
    result = analyze_invalid_boxes(coco, load_invalid_box_config())
    assert result.metric_value == 2
    assert not result.passed
    assert result.details["total_annotations"] == 2
    assert [sample["annotation_id"] for sample in result.details["offending_samples"]] == [1, 2]
    assert result.details["offending_samples"][0]["reasons"] == [
        "non_positive_width",
        "area_mismatch",
    ]
    assert result.details["offending_samples"][1]["reasons"] == ["exceeds_image_width"]
    with pytest.raises(ValidationError, match="bbox"):
        CocoDataset.model_validate(coco)
    with pytest.raises(ValueError, match="bbox"):
        analyze_small_objects(coco, SmallObjectConfig(width_px=32, height_px=32, threshold=0.4))


def test_input_not_mutated_or_aliased_and_result_deterministic():
    coco = document([-1, 0, 10, 10])
    before = deepcopy(coco)
    config = load_invalid_box_config()
    result = analyze_invalid_boxes(coco, config)
    assert result == analyze_invalid_boxes(coco, config)
    assert coco == before
    result.details["offending_samples"][0]["bbox"][0] = 999
    assert coco == before


def test_threshold_from_real_yaml_and_modified_configuration(tmp_path):
    policy = load_quality_policy()
    loaded = load_invalid_box_config()
    assert loaded.threshold == policy.degenerate_boxes.threshold == 0
    coco = document([-1, 0, 10, 10])
    assert not analyze_invalid_boxes(coco, loaded).passed
    raw = policy.model_dump()
    raw["degenerate_boxes"]["threshold"] = 1
    path = tmp_path / "quality.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert analyze_invalid_boxes(coco, load_invalid_box_config(path)).passed


def test_empty_annotation_list():
    coco = document()
    coco["annotations"] = []
    result = analyze_invalid_boxes(coco, load_invalid_box_config())
    assert result.passed
    assert result.metric_value == 0
    assert result.details["total_annotations"] == 0


@pytest.mark.parametrize("threshold", [-1, float("inf"), float("nan"), "0"])
def test_invalid_configuration(threshold):
    with pytest.raises(ValidationError):
        InvalidBoxConfig(threshold=threshold)


@pytest.mark.parametrize("bbox", [[0, 0, 10], [0, 0, float("nan"), 10], [0, 0, True, 10]])
def test_malformed_bbox_is_rejected_explicitly(bbox):
    with pytest.raises(ValueError, match="finite bbox"):
        analyze_invalid_boxes(document(bbox), load_invalid_box_config())


def test_non_finite_calculated_area_is_reported_without_invalid_json():
    coco = document([0, 0, 1e200, 1e200])
    result = analyze_invalid_boxes(coco, load_invalid_box_config())
    sample = result.details["offending_samples"][0]
    assert sample["calculated_area"] is None
    assert "non_finite_calculated_area" in sample["reasons"]
    json.dumps(result.model_dump(), allow_nan=False)
