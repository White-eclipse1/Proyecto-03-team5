"""Dataset dog/cat pequeño y controlado: crops de ML-01 + manifiesto con el contrato de OPS-02.

Las imágenes dog son claras con un cuadro rojo y las cat oscuras con un cuadro azul,
así un modelo pequeño puede separarlas en pocas épocas.
"""

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from manifests.models import P3Manifest
from PIL import Image, ImageDraw

from classification.manifest import manifest_hash
from crops.extract import extract_crops
from crops.models import CropReport

DOG, CAT = 3, 4


@dataclass(frozen=True)
class ControlledRelease:
    manifest_path: Path
    crop_report_path: Path
    crops_dir: Path
    dataset_version: str
    manifest_hash: str
    report: CropReport


def _image(image_id: int, is_dog: bool) -> Image.Image:
    background = (210, 200, 190) if is_dog else (40, 45, 60)
    image = Image.new("RGB", (64, 48), color=background)
    fill = (220, 30, 30) if is_dog else (30, 60, 220)
    offset = image_id % 5
    ImageDraw.Draw(image).rectangle([8 + offset, 8, 40 + offset, 30], fill=fill)
    return image


def write_controlled_release(
    root: Path, *, splits: dict[str, int] | None = None, dataset_version: str = "v0.1.1"
) -> ControlledRelease:
    """Crea `root/{images,crops}`, `crops.json` y `manifest.json` con hash y procedencia válidos.

    `splits` da cuántos crops por partición (alternando dog/cat); por defecto 8/4/4.
    """
    splits = splits or {"train": 8, "validation": 4, "test": 4}
    images_dir = root / "images"
    images_dir.mkdir(parents=True)
    assignment = [name for name, count in splits.items() for _ in range(count)]
    images, annotations = [], []
    for image_id, _ in enumerate(assignment, start=1):
        is_dog = image_id % 2 == 1
        _image(image_id, is_dog).save(images_dir / f"{image_id}.png")
        images.append({"id": image_id, "file_name": f"{image_id}.png", "width": 64, "height": 48})
        annotations.append(
            {
                "id": 100 + image_id,
                "image_id": image_id,
                "category_id": DOG if is_dog else CAT,
                "bbox": [4, 4, 48, 36],
                "area": 1728.0,
                "iscrowd": 0,
            }
        )
    report = extract_crops(
        {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": DOG, "name": "dog"}, {"id": CAT, "name": "cat"}],
        },
        images_dir=images_dir,
        output_dir=root / "crops",
        dataset_version=dataset_version,
        provenance={
            "release_version": dataset_version,
            "images_dvc_hash": "0" * 32 + ".dir",
            "annotations_dvc_hash": "1" * 32 + ".dir",
            "quality_report": f"reports/releases/{dataset_version}/quality.json",
        },
    )
    report_path = root / "crops.json"
    report_path.write_text(report.model_dump_json(), encoding="utf-8")

    by_image = {crop.image_id: crop for crop in report.crops}
    total = len(assignment)
    payload = {
        "schema_version": "1.0",
        "manifest_version": "p3-test",
        "dataset_version": dataset_version,
        "source_release": dataset_version,
        "provenance": {
            "release_version": dataset_version,
            "images_dvc_hash": report.provenance.images_dvc_hash,
            "annotations_dvc_hash": report.provenance.annotations_dvc_hash,
            "quality_report": report.provenance.quality_report,
            "crops_sha256": "sha256:" + sha256(report_path.read_bytes()).hexdigest(),
        },
        "seed": 42,
        "manifest_hash": "sha256:" + "0" * 64,
        "total_images": total,
        "splits": {
            name: {"image_count": count, "ratio": count / total} for name, count in splits.items()
        },
        "records": [
            {
                "crop_id": by_image[image_id].crop_id,
                "source_image_id": image_id,
                "duplicate_group": f"group-{image_id}",
                "class": by_image[image_id].class_name,
                "split": split,
            }
            for image_id, split in enumerate(assignment, start=1)
        ],
        "counts": {},
    }
    payload["manifest_hash"] = manifest_hash(P3Manifest.model_validate(payload))
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return ControlledRelease(
        manifest_path=manifest_path,
        crop_report_path=report_path,
        crops_dir=root / "crops",
        dataset_version=dataset_version,
        manifest_hash=payload["manifest_hash"],
        report=report,
    )
