import pytest
from pydantic import ValidationError

from analyzers.spatial_bias import SpatialBiasConfig, analyze_spatial_bias


def config(**overrides):
    return SpatialBiasConfig.model_validate({"min_std_dev": 0.15, **overrides})


def coco_with_boxes(*boxes, width=100, height=100):
    return {
        "images": [{"id": 1, "width": width, "height": height, "file_name": "sample.jpg"}],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [
            {"id": i, "image_id": 1, "category_id": 1, "bbox": list(box)}
            for i, box in enumerate(boxes, start=1)
        ],
    }


def test_boxes_spread_across_the_image_pass():
    # Centros en las 4 esquinas: máxima dispersión posible.
    coco = coco_with_boxes((0, 0, 20, 20), (80, 0, 20, 20), (0, 80, 20, 20), (80, 80, 20, 20))
    result = analyze_spatial_bias(coco, config())
    assert result.passed
    assert result.metric_value > 0.15


def test_boxes_always_centered_fail():
    # Todas las cajas centradas en el mismo punto: dispersión 0.
    coco = coco_with_boxes((40, 40, 20, 20), (41, 41, 20, 20), (39, 39, 20, 20), (40, 40, 20, 20))
    result = analyze_spatial_bias(coco, config())
    assert not result.passed
    assert result.metric_value < 0.15


def test_metric_value_is_the_smaller_of_the_two_axis_std_devs():
    # Dispersión amplia en x, nula en y: el sesgo lo marca el eje y (menor).
    coco = coco_with_boxes((0, 40, 10, 10), (30, 40, 10, 10), (60, 40, 10, 10), (90, 40, 10, 10))
    result = analyze_spatial_bias(coco, config())
    assert result.details["std_center_y"] == 0.0
    assert result.details["std_center_x"] > 0.0
    assert result.metric_value == result.details["std_center_y"]


def test_fewer_than_two_boxes_has_zero_dispersion_and_fails_a_positive_threshold():
    coco = coco_with_boxes((10, 10, 20, 20))
    result = analyze_spatial_bias(coco, config(min_std_dev=0.01))
    assert result.metric_value == 0.0
    assert not result.passed


def test_no_boxes_does_not_crash():
    coco = coco_with_boxes()
    result = analyze_spatial_bias(coco, config())
    assert result.metric_value == 0.0
    assert result.details["total_boxes"] == 0


def test_details_report_mean_center_and_total_boxes():
    coco = coco_with_boxes((0, 0, 20, 20), (80, 80, 20, 20), width=100, height=100)
    result = analyze_spatial_bias(coco, config())
    assert result.details["total_boxes"] == 2
    assert result.details["mean_center_x"] == pytest.approx(0.5)
    assert result.details["mean_center_y"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "overrides", [{"min_std_dev": -0.1}, {"min_std_dev": 0.6}, {"min_std_dev": "0.1"}]
)
def test_invalid_configuration_is_rejected(overrides):
    with pytest.raises(ValidationError):
        config(**overrides)


def test_exact_threshold_and_scale_invariance():
    boxes = ((0, 0, 20, 20), (50, 50, 20, 20))
    result = analyze_spatial_bias(coco_with_boxes(*boxes), config(min_std_dev=0.25))
    assert result.metric_value == pytest.approx(0.25)
    assert result.passed
    scaled = tuple(tuple(value * 2 for value in box) for box in boxes)
    second = analyze_spatial_bias(
        coco_with_boxes(*scaled, width=200, height=200), config(min_std_dev=0.25)
    )
    assert second == result


@pytest.mark.parametrize("boxes", [(), ((10, 10, 20, 20),)])
def test_degenerate_sample_count_passes_zero_threshold(boxes):
    result = analyze_spatial_bias(coco_with_boxes(*boxes), config(min_std_dev=0.0))
    assert result.metric_value == 0
    assert result.passed
