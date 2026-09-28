"""ML-01: clases dog/cat congeladas y extracción validada de crops COCO."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image, ImageDraw

from crops.classes import CLASS_NAMES, CLASS_TO_INDEX, resolve_category_classes
from crops.extract import extract_crops, write_crop_report
from crops.models import CropReport
from ingestion.loader import merge_raw_batches

ROOT = Path(__file__).resolve().parents[2]
DOG, CAT, PERSON = 3, 4, 1
CATEGORIES = [
    {"id": PERSON, "name": "person"},
    {"id": DOG, "name": "dog"},
    {"id": CAT, "name": "cat"},
]


def _write_image(images_dir, file_name, *, size=(80, 60), boxes=()):
    """Fondo gris con cada `(bbox, color)` pintado, para verificar píxel a píxel."""
    image = Image.new("RGB", size, color=(128, 128, 128))
    draw = ImageDraw.Draw(image)
    for (x, y, width, height), color in boxes:
        draw.rectangle([x, y, x + width - 1, y + height - 1], fill=color)
    images_dir.mkdir(parents=True, exist_ok=True)
    image.save(images_dir / file_name, format="PNG")
    return image


def _annotation(ann_id, image_id, category_id, bbox):
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "area": float(bbox[2] * bbox[3]) if len(bbox) == 4 else 0.0,
        "iscrowd": 0,
    }


def _coco(images, annotations, categories=CATEGORIES):
    return {"images": images, "annotations": annotations, "categories": categories}


def _extract(tmp_path, coco):
    return extract_crops(
        coco,
        images_dir=tmp_path / "images",
        output_dir=tmp_path / "crops",
        dataset_version="v0.1.1",
    )


def _rejected_ids(report):
    return {rejection.annotation_id for rejection in report.rejections}


def _accepted_ids(report):
    return {crop.annotation_id for crop in report.crops}


# --- Clases oficiales -------------------------------------------------------


def test_official_classes_are_frozen_as_dog_and_cat():
    assert CLASS_NAMES == ("dog", "cat")
    assert isinstance(CLASS_NAMES, tuple)
    assert dict(CLASS_TO_INDEX) == {"dog": 0, "cat": 1}
    with pytest.raises(TypeError):
        CLASS_TO_INDEX["bird"] = 2  # type: ignore[index]


def test_resolve_category_classes_maps_release_ids_by_name():
    assert resolve_category_classes(CATEGORIES) == {DOG: "dog", CAT: "cat"}


def test_resolve_category_classes_rejects_release_without_both_classes():
    with pytest.raises(ValueError, match="cat"):
        resolve_category_classes([{"id": DOG, "name": "dog"}])


def test_resolve_category_classes_rejects_duplicated_class_names():
    with pytest.raises(ValueError, match="dog"):
        resolve_category_classes([*CATEGORIES, {"id": 9, "name": "dog"}])


# --- Crops válidos ----------------------------------------------------------


def test_valid_crop_preserves_provenance(tmp_path):
    _write_image(tmp_path / "images", "a.png", boxes=[((10, 5, 30, 20), (200, 0, 0))])
    coco = _coco(
        [{"id": 7, "file_name": "a.png", "width": 80, "height": 60}],
        [_annotation(70, 7, DOG, [10, 5, 30, 20])],
    )

    report = _extract(tmp_path, coco)

    assert report.rejections == []
    [crop] = report.crops
    assert crop.image_id == 7
    assert crop.annotation_id == 70
    assert crop.category_id == DOG
    assert crop.class_name == "dog"
    assert crop.bbox == [10, 5, 30, 20]
    assert crop.source_file_name == "a.png"
    assert (tmp_path / "crops" / crop.crop_path).is_file()


def test_crop_pixels_match_the_original_bbox_region(tmp_path):
    original = _write_image(
        tmp_path / "images",
        "a.png",
        boxes=[((12, 8, 25, 17), (0, 180, 40)), ((40, 30, 20, 20), (10, 20, 250))],
    )
    coco = _coco(
        [{"id": 1, "file_name": "a.png", "width": 80, "height": 60}],
        [_annotation(1, 1, CAT, [12, 8, 25, 17])],
    )

    [crop] = _extract(tmp_path, coco).crops

    with Image.open(tmp_path / "crops" / crop.crop_path) as stored:
        stored_pixels = np.asarray(stored.convert("RGB"))
    expected = np.asarray(original.crop((12, 8, 37, 25)))
    assert stored_pixels.shape == (17, 25, 3)
    assert np.array_equal(stored_pixels, expected)
    assert np.all(stored_pixels == (0, 180, 40))


def test_fractional_bbox_is_expanded_to_cover_the_whole_box(tmp_path):
    _write_image(tmp_path / "images", "a.png")
    coco = _coco(
        [{"id": 1, "file_name": "a.png", "width": 80, "height": 60}],
        [_annotation(1, 1, DOG, [10.4, 5.6, 20.2, 10.1])],
    )

    [crop] = _extract(tmp_path, coco).crops

    assert crop.bbox == [10.4, 5.6, 20.2, 10.1]
    assert crop.crop_box == [10, 5, 31, 16]


def test_multi_object_image_yields_one_crop_per_annotation(tmp_path):
    _write_image(
        tmp_path / "images",
        "both.png",
        boxes=[((0, 0, 30, 30), (255, 0, 0)), ((40, 20, 30, 30), (0, 0, 255))],
    )
    coco = _coco(
        [{"id": 5, "file_name": "both.png", "width": 80, "height": 60}],
        [_annotation(50, 5, DOG, [0, 0, 30, 30]), _annotation(51, 5, CAT, [40, 20, 30, 30])],
    )

    report = _extract(tmp_path, coco)

    by_annotation = {crop.annotation_id: crop for crop in report.crops}
    assert {crop.image_id for crop in report.crops} == {5}
    assert by_annotation[50].class_name == "dog"
    assert by_annotation[51].class_name == "cat"
    assert by_annotation[50].crop_id != by_annotation[51].crop_id
    for annotation_id, color in ((50, (255, 0, 0)), (51, (0, 0, 255))):
        with Image.open(tmp_path / "crops" / by_annotation[annotation_id].crop_path) as stored:
            assert np.all(np.asarray(stored.convert("RGB")) == color)


def test_crop_ids_are_deterministic_and_unique(tmp_path):
    _write_image(tmp_path / "images", "a.png")
    coco = _coco(
        [{"id": 1, "file_name": "a.png", "width": 80, "height": 60}],
        [_annotation(1, 1, DOG, [0, 0, 10, 10]), _annotation(2, 1, CAT, [20, 20, 10, 10])],
    )

    report_a = extract_crops(
        coco, images_dir=tmp_path / "images", output_dir=tmp_path / "a", dataset_version="v1"
    )
    report_b = extract_crops(
        coco, images_dir=tmp_path / "images", output_dir=tmp_path / "b", dataset_version="v1"
    )

    ids_a = [crop.crop_id for crop in report_a.crops]
    assert ids_a == [crop.crop_id for crop in report_b.crops]
    assert len(ids_a) == len(set(ids_a))
    assert [crop.sha256 for crop in report_a.crops] == [crop.sha256 for crop in report_b.crops]


# --- Rechazos ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("bbox", "reason"),
    [
        ([10, 10, 0, 10], "non_positive_width"),
        ([10, 10, -5, 10], "non_positive_width"),
        ([10, 10, 10, 0], "non_positive_height"),
        ([10, 10, 10, -1], "non_positive_height"),
    ],
)
def test_degenerate_bbox_is_rejected_without_crop_or_manifest_entry(tmp_path, bbox, reason):
    """Agent Test de ML-01: una bbox degenerada no produce crop ni entra al manifiesto."""
    _write_image(tmp_path / "images", "a.png")
    coco = _coco(
        [{"id": 1, "file_name": "a.png", "width": 80, "height": 60}],
        [_annotation(1, 1, DOG, [0, 0, 10, 10]), _annotation(2, 1, DOG, bbox)],
    )

    report = _extract(tmp_path, coco)

    assert _accepted_ids(report) == {1}
    [rejection] = report.rejections
    assert rejection.annotation_id == 2
    assert reason in rejection.reasons
    assert rejection.bbox == bbox
    assert len(list((tmp_path / "crops").rglob("*.png"))) == 1


@pytest.mark.parametrize(
    ("bbox", "reason"),
    [
        ([-1, 0, 10, 10], "negative_x"),
        ([0, -1, 10, 10], "negative_y"),
        ([75, 0, 10, 10], "exceeds_image_width"),
        ([0, 55, 10, 10], "exceeds_image_height"),
    ],
)
def test_out_of_bounds_bbox_is_rejected(tmp_path, bbox, reason):
    _write_image(tmp_path / "images", "a.png")
    coco = _coco(
        [{"id": 1, "file_name": "a.png", "width": 80, "height": 60}],
        [_annotation(1, 1, CAT, bbox)],
    )

    report = _extract(tmp_path, coco)

    assert report.crops == []
    assert reason in report.rejections[0].reasons
    assert "out_of_bounds" in report.rejections[0].reasons


def test_bbox_touching_image_border_is_accepted(tmp_path):
    _write_image(tmp_path / "images", "a.png")
    coco = _coco(
        [{"id": 1, "file_name": "a.png", "width": 80, "height": 60}],
        [_annotation(1, 1, CAT, [0, 0, 80, 60])],
    )

    assert _accepted_ids(_extract(tmp_path, coco)) == {1}


def test_missing_image_file_is_rejected(tmp_path):
    (tmp_path / "images").mkdir()
    coco = _coco(
        [{"id": 1, "file_name": "ghost.png", "width": 80, "height": 60}],
        [_annotation(1, 1, DOG, [0, 0, 10, 10])],
    )

    report = _extract(tmp_path, coco)

    assert report.crops == []
    assert report.rejections[0].reasons == ["missing_image"]


def test_unreadable_image_file_is_rejected(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "broken.png").write_bytes(b"not an image")
    coco = _coco(
        [{"id": 1, "file_name": "broken.png", "width": 80, "height": 60}],
        [_annotation(1, 1, DOG, [0, 0, 10, 10])],
    )

    assert _extract(tmp_path, coco).rejections[0].reasons == ["unreadable_image"]


def test_declared_size_different_from_pixels_is_rejected(tmp_path):
    _write_image(tmp_path / "images", "a.png", size=(40, 30))
    coco = _coco(
        [{"id": 1, "file_name": "a.png", "width": 80, "height": 60}],
        [_annotation(1, 1, DOG, [0, 0, 10, 10])],
    )

    assert _extract(tmp_path, coco).rejections[0].reasons == ["image_size_mismatch"]


def test_unknown_image_id_is_rejected(tmp_path):
    _write_image(tmp_path / "images", "a.png")
    coco = _coco(
        [{"id": 1, "file_name": "a.png", "width": 80, "height": 60}],
        [_annotation(1, 99, DOG, [0, 0, 10, 10])],
    )

    assert _extract(tmp_path, coco).rejections[0].reasons == ["unknown_image_id"]


def test_annotation_outside_official_classes_is_rejected(tmp_path):
    _write_image(tmp_path / "images", "a.png")
    coco = _coco(
        [{"id": 1, "file_name": "a.png", "width": 80, "height": 60}],
        [_annotation(1, 1, PERSON, [0, 0, 10, 10]), _annotation(2, 1, 42, [0, 0, 10, 10])],
    )

    report = _extract(tmp_path, coco)

    reasons = {rejection.annotation_id: rejection.reasons for rejection in report.rejections}
    assert reasons == {1: ["unsupported_category"], 2: ["unknown_category_id"]}
    assert report.rejections[0].class_name is None


@pytest.mark.parametrize("bbox", [[0, 0, 10], [0, 0, 10, 10, 1], [0, 0, float("nan"), 10]])
def test_malformed_bbox_is_rejected_instead_of_crashing(tmp_path, bbox):
    _write_image(tmp_path / "images", "a.png")
    annotation = _annotation(1, 1, DOG, [0, 0, 10, 10]) | {"bbox": bbox}
    coco = _coco([{"id": 1, "file_name": "a.png", "width": 80, "height": 60}], [annotation])

    report = _extract(tmp_path, coco)

    assert report.crops == []
    assert report.rejections[0].reasons == ["malformed_bbox"]


def test_extraction_refuses_non_empty_output_dir(tmp_path):
    _write_image(tmp_path / "images", "a.png")
    (tmp_path / "crops").mkdir()
    (tmp_path / "crops" / "stale.png").write_bytes(b"old")
    coco = _coco([{"id": 1, "file_name": "a.png", "width": 80, "height": 60}], [])

    with pytest.raises(FileExistsError):
        _extract(tmp_path, coco)


# --- Reporte ----------------------------------------------------------------


def test_report_summarizes_accepted_and_rejected(tmp_path):
    _write_image(tmp_path / "images", "a.png")
    coco = _coco(
        [
            {"id": 1, "file_name": "a.png", "width": 80, "height": 60},
            {"id": 2, "file_name": "ghost.png", "width": 80, "height": 60},
        ],
        [
            _annotation(1, 1, DOG, [0, 0, 10, 10]),
            _annotation(2, 1, CAT, [20, 20, 10, 10]),
            _annotation(3, 1, CAT, [20, 20, 0, 10]),
            _annotation(4, 2, DOG, [0, 0, 10, 10]),
        ],
    )

    report = _extract(tmp_path, coco)

    assert report.dataset_version == "v0.1.1"
    assert report.classes == ["dog", "cat"]
    assert report.summary.total_annotations == 4
    assert report.summary.accepted == 2
    assert report.summary.rejected == 2
    assert report.summary.accepted_per_class == {"dog": 1, "cat": 1}
    assert report.summary.rejected_per_reason == {"non_positive_width": 1, "missing_image": 1}


def test_report_is_written_and_round_trips(tmp_path):
    _write_image(tmp_path / "images", "a.png")
    coco = _coco(
        [{"id": 1, "file_name": "a.png", "width": 80, "height": 60}],
        [_annotation(1, 1, DOG, [0, 0, 10, 10]), _annotation(2, 1, DOG, [0, 0, 0, 0])],
    )
    report = _extract(tmp_path, coco)

    path = write_crop_report(report, tmp_path / "reports" / "crops.json")

    loaded = CropReport.model_validate_json(path.read_text(encoding="utf-8"))
    assert loaded == report
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert [crop["annotation_id"] for crop in raw["crops"]] == [1]
    assert [rejection["annotation_id"] for rejection in raw["rejections"]] == [2]


def test_report_contract_rejects_crop_that_is_also_rejected(tmp_path):
    _write_image(tmp_path / "images", "a.png")
    coco = _coco(
        [{"id": 1, "file_name": "a.png", "width": 80, "height": 60}],
        [_annotation(1, 1, DOG, [0, 0, 10, 10]), _annotation(2, 1, DOG, [0, 0, 0, 0])],
    )
    raw = _extract(tmp_path, coco).model_dump(mode="json")
    raw["rejections"][0]["annotation_id"] = 1

    with pytest.raises(ValueError, match="annotation_id"):
        CropReport.model_validate(raw)


# --- Integración con la ingesta y DVC ----------------------------------------


def test_merge_raw_batches_keeps_degenerate_boxes_for_per_annotation_rejection(tmp_path):
    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir()
    coco = _coco(
        [{"id": 1, "file_name": "a.png", "width": 80, "height": 60}],
        [_annotation(1, 1, DOG, [0, 0, 0, 10])],
    )
    (annotations_dir / "lote-1.json").write_text(json.dumps(coco), encoding="utf-8")

    merged = merge_raw_batches(annotations_dir)

    assert merged["annotations"][0]["bbox"] == [0, 0, 0, 10]
    assert {category["name"] for category in merged["categories"]} == {"person", "dog", "cat"}


def test_dvc_pipeline_has_crops_stage_after_quality_gate():
    stages = yaml.safe_load((ROOT / "dvc.yaml").read_text(encoding="utf-8"))["stages"]
    crops = stages["crops"]

    assert crops["cmd"].endswith("dvc_crops_stage.py")
    assert "../reports/.quality_gate.passed" in crops["deps"]
    assert "crops" in crops["deps"]
    assert "../data/crops" in crops["outs"]
    assert any(
        isinstance(output, dict) and "../reports/crops.json" in output for output in crops["outs"]
    )
