import pytest
from pydantic import ValidationError

from ingestion.models import Annotation, Category, CocoDataset, Image


def coco(images=None, categories=None, annotations=None):
    return {
        "images": images
        if images is not None
        else [{"id": 1, "file_name": "a.jpg", "width": 10, "height": 10}],
        "categories": categories if categories is not None else [{"id": 1, "name": "cat"}],
        "annotations": annotations if annotations is not None else [],
    }


def annotation(**overrides):
    return {
        "id": 1,
        "image_id": 1,
        "category_id": 1,
        "bbox": [0, 0, 5, 5],
        "area": 25.0,
        "iscrowd": 0,
        **overrides,
    }


def test_valid_dataset_round_trips():
    document = coco(annotations=[annotation()])
    dataset = CocoDataset.model_validate(document)
    assert dataset.images[0] == Image(id=1, file_name="a.jpg", width=10, height=10)
    assert dataset.categories[0] == Category(id=1, name="cat")
    assert isinstance(dataset.annotations[0], Annotation)


def test_bbox_with_three_elements_is_rejected_naming_bbox():
    document = coco(annotations=[annotation(bbox=[0, 0, 5])])
    with pytest.raises(ValidationError, match="bbox"):
        CocoDataset.model_validate(document)


@pytest.mark.parametrize("bbox", [[0, 0, 0, 5], [0, 0, 5, 0], [0, 0, -1, 5]])
def test_bbox_with_non_positive_size_is_rejected(bbox):
    document = coco(annotations=[annotation(bbox=bbox)])
    with pytest.raises(ValidationError, match="bbox"):
        CocoDataset.model_validate(document)


def test_unknown_category_id_is_rejected_naming_category_id():
    document = coco(annotations=[annotation(category_id=99)])
    with pytest.raises(ValidationError, match="category_id"):
        CocoDataset.model_validate(document)


def test_orphan_image_id_is_rejected_naming_image_id():
    document = coco(annotations=[annotation(image_id=99)])
    with pytest.raises(ValidationError, match="image_id"):
        CocoDataset.model_validate(document)


def test_no_images_is_rejected():
    document = coco(images=[])
    with pytest.raises(ValidationError):
        CocoDataset.model_validate(document)


def test_no_categories_is_rejected():
    document = coco(categories=[])
    with pytest.raises(ValidationError):
        CocoDataset.model_validate(document)


def test_extra_field_on_dataset_is_rejected():
    document = {**coco(), "extra": "nope"}
    with pytest.raises(ValidationError):
        CocoDataset.model_validate(document)


def test_extra_field_on_annotation_is_rejected():
    document = coco(annotations=[annotation(extra="nope")])
    with pytest.raises(ValidationError):
        CocoDataset.model_validate(document)
