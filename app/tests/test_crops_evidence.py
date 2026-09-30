"""ML-01: evidencia reproducible sobre el release real v0.1.1 (ver tests/evidence/ml-01-crops.md).

Requiere `dvc pull` de `data/raw` y `dvc repro crops`; se activa con RUN_CROPS_EVIDENCE=1.
"""

import os
import random
from pathlib import Path

import pytest
from PIL import Image, ImageChops

from crops.classes import CLASS_NAMES, MIN_IMAGES_PER_CLASS
from crops.extract import assert_min_images_per_class, extract_crops
from crops.models import CropReport
from crops.preview import render_verification_sheet
from ingestion.loader import merge_raw_batches

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
CROPS = ROOT / "data" / "crops"
REPORT = ROOT / "reports" / "crops.json"
SAMPLE_SIZE = 40

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_CROPS_EVIDENCE") != "1",
    reason="Set RUN_CROPS_EVIDENCE=1 with data/raw and data/crops materialized",
)


def _report() -> CropReport:
    return CropReport.model_validate_json(REPORT.read_text(encoding="utf-8"))


def _sample(report: CropReport) -> list:
    """Muestra fija: todas las imágenes con dog y cat, más crops al azar (semilla 42)."""
    by_image: dict[int, list] = {}
    for crop in report.crops:
        by_image.setdefault(crop.image_id, []).append(crop)
    mixed = [
        crop
        for crops in by_image.values()
        if {crop.class_name for crop in crops} == set(CLASS_NAMES)
        for crop in crops
    ]
    rest = [crop for crop in report.crops if crop not in mixed]
    return mixed + random.Random(42).sample(rest, SAMPLE_SIZE - len(mixed))


def test_sampled_crops_match_the_original_coco():
    """Etiqueta, bbox y píxeles de cada crop muestreado contra el COCO y la imagen original."""
    report = _report()
    coco = merge_raw_batches(RAW / "annotations")
    annotations = {annotation["id"]: annotation for annotation in coco["annotations"]}
    images = {image["id"]: image for image in coco["images"]}
    categories = {category["id"]: category["name"] for category in coco["categories"]}

    sample = _sample(report)
    for crop in sample:
        annotation = annotations[crop.annotation_id]
        assert annotation["image_id"] == crop.image_id
        assert annotation["category_id"] == crop.category_id
        assert categories[annotation["category_id"]] == crop.class_name
        assert [float(value) for value in annotation["bbox"]] == crop.bbox
        assert images[crop.image_id]["file_name"] == crop.source_file_name
        with Image.open(RAW / "images" / crop.source_file_name) as source:
            expected = source.convert("RGB").crop(tuple(crop.crop_box))
        with Image.open(CROPS / crop.crop_path) as stored:
            assert ImageChops.difference(expected, stored.convert("RGB")).getbbox() is None

    per_image: dict[int, set[str]] = {}
    for crop in sample:
        per_image.setdefault(crop.image_id, set()).add(crop.class_name)
    mixed = sorted(image_id for image_id, names in per_image.items() if len(names) > 1)
    print(f"\nCrops verificados contra COCO e imagen original: {len(sample)}")
    print(f"Imágenes con dog y cat (un crop por objeto): {mixed}")


def test_real_release_meets_the_class_minimum():
    report = _report()

    assert_min_images_per_class(report, MIN_IMAGES_PER_CLASS)
    print(f"\n{report.summary.model_dump_json()}")


def test_injected_invalid_boxes_on_the_real_release_are_excluded(tmp_path):
    """Agent Test del issue #3 sobre el COCO real, en un directorio temporal."""
    coco = merge_raw_batches(RAW / "annotations")
    next_id = max(annotation["id"] for annotation in coco["annotations"]) + 1
    image = next(image for image in coco["images"] if image["id"] == 40)
    injected = {
        next_id: (40, 3, [10, 10, 0, 50], "non_positive_width"),
        next_id + 1: (40, 4, [10, 10, 40, -5], "non_positive_height"),
        next_id + 2: (40, 3, [10, 10, image["width"], 20], "exceeds_image_width"),
        next_id + 3: (99999, 4, [0, 0, 10, 10], "missing_image"),
    }
    coco["images"].append({"id": 99999, "file_name": "no-existe.jpg", "width": 100, "height": 100})
    for annotation_id, (image_id, category_id, bbox, _) in injected.items():
        coco["annotations"].append(
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox,
                "area": 0.0,
                "iscrowd": 0,
            }
        )

    report = extract_crops(
        coco,
        images_dir=RAW / "images",
        output_dir=tmp_path / "crops",
        dataset_version="v0.1.1",
        provenance=_report().provenance.model_dump(),
    )

    manifest = {crop.annotation_id for crop in report.crops}
    reasons = {rejection.annotation_id: rejection.reasons for rejection in report.rejections}
    for annotation_id, (_, _, _, reason) in injected.items():
        assert annotation_id not in manifest
        assert not list((tmp_path / "crops").rglob(f"*-ann{annotation_id}.png"))
        assert reason in reasons[annotation_id]
        print(f"\nann {annotation_id}: sin crop ni manifiesto; motivos={reasons[annotation_id]}")
    assert report.summary.accepted == _report().summary.accepted
    assert report.summary.rejected == len(injected)


def test_render_manual_verification_sheet(tmp_path):
    """Hoja imagen+bbox vs crop de la misma muestra, para la revisión visual del DoD."""
    report = _report()
    out = Path(os.environ.get("CROPS_EVIDENCE_SHEET", tmp_path / "ml-01-crops-verification.png"))
    sample = _sample(report)[:12]

    render_verification_sheet(
        report.model_copy(update={"crops": sample}),
        images_dir=RAW / "images",
        crops_dir=CROPS,
        out_path=out,
        limit=len(sample),
    )

    assert out.is_file()
    print(f"\nHoja de verificación: {out.name} ({len(sample)} crops)")
