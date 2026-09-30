"""ML-01 — extrae un crop por bounding box COCO válida del release de P2.

Cada anotación dog/cat se convierte en una muestra de clasificación
independiente: una imagen con varios objetos produce varios crops, nunca
una sola etiqueta para la imagen completa. Las anotaciones que no pueden
producir un crop fiel a su bbox se excluyen una por una y quedan
registradas en `CropReport.rejections` con sus motivos; no se reparan ni se
recortan contra el borde.

`crop_box` usa el formato de Pillow `[left, top, right, bottom]` en píxeles
enteros: `floor` del origen y `ceil` del extremo, de modo que el crop cubre
la bbox completa aunque venga con decimales. `bbox` conserva el valor COCO
original `[x, y, width, height]` sin redondear.
"""

import hashlib
import logging
from collections import Counter
from collections.abc import Mapping
from io import BytesIO
from math import ceil, floor, isfinite
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from crops.classes import CLASS_NAMES, resolve_category_classes
from crops.models import (
    CropProvenance,
    CropRecord,
    CropRejection,
    CropReport,
    CropSummary,
)

logger = logging.getLogger("crop-extraction")

BOUNDS_REASONS = {"negative_x", "negative_y", "exceeds_image_width", "exceeds_image_height"}


def _finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def _bbox_reasons(bbox, image: dict | None) -> list[str]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return ["malformed_bbox"]
    if not all(_finite_number(value) for value in bbox):
        return ["malformed_bbox"]

    x, y, width, height = bbox
    reasons = []
    if width <= 0:
        reasons.append("non_positive_width")
    if height <= 0:
        reasons.append("non_positive_height")
    if x < 0:
        reasons.append("negative_x")
    if y < 0:
        reasons.append("negative_y")
    if image is not None:
        if x + width > image["width"]:
            reasons.append("exceeds_image_width")
        if y + height > image["height"]:
            reasons.append("exceeds_image_height")
    if BOUNDS_REASONS.intersection(reasons):
        reasons.append("out_of_bounds")
    return reasons


def _crop_box(bbox) -> tuple[int, int, int, int]:
    x, y, width, height = bbox
    return floor(x), floor(y), ceil(x + width), ceil(y + height)


def _open_image(path: Path, declared: dict) -> tuple[Image.Image | None, str | None]:
    if not path.is_file():
        return None, "missing_image"
    try:
        with Image.open(path) as source:
            source.load()
            pixels = source.copy()
    except (UnidentifiedImageError, OSError):
        return None, "unreadable_image"
    if pixels.size != (declared["width"], declared["height"]):
        return None, "image_size_mismatch"
    return pixels, None


def _safe_bbox(bbox) -> list:
    """Copia la bbox cruda para el reporte, sin valores que rompan el JSON."""
    if not isinstance(bbox, (list, tuple)):
        return []
    return [value if _finite_number(value) else None for value in bbox]


def _index_by_id(items: list[dict], label: str) -> dict[int, dict]:
    indexed: dict[int, dict] = {}
    for item in items:
        if item["id"] in indexed:
            raise ValueError(f"{label} tiene ids repetidos: {item['id']}")
        indexed[item["id"]] = item
    return indexed


def extract_crops(
    coco: dict,
    *,
    images_dir: Path,
    output_dir: Path,
    dataset_version: str,
    provenance: Mapping[str, str],
) -> CropReport:
    """Escribe `output_dir/<clase>/<crop_id>.png` por cada bbox válida y devuelve el reporte.

    `coco` es el dict crudo de `ingestion.loader.merge_raw_batches`. Se exige
    un `output_dir` vacío para que el manifiesto nunca conviva con crops
    viejos de otra corrida.
    """
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"El directorio de crops no está vacío: {output_dir}")

    release_provenance = CropProvenance.model_validate(dict(provenance))

    class_by_category = resolve_category_classes(coco["categories"])
    known_categories = {category["id"] for category in coco["categories"]}
    images = _index_by_id(coco["images"], "images")
    annotations = sorted(
        _index_by_id(coco["annotations"], "annotations").values(),
        key=lambda annotation: (annotation["image_id"], annotation["id"]),
    )

    crops: list[CropRecord] = []
    rejections: list[CropRejection] = []
    # Las anotaciones van ordenadas por imagen: basta con mantener abierta la actual.
    current_image_id, current_pixels, current_error = None, None, None

    for annotation in annotations:
        image = images.get(annotation["image_id"])
        category_id = annotation["category_id"]
        class_name = class_by_category.get(category_id)
        bbox = annotation["bbox"]

        reasons = []
        if category_id not in known_categories:
            reasons.append("unknown_category_id")
        elif class_name is None:
            reasons.append("unsupported_category")
        if image is None:
            reasons.append("unknown_image_id")
        reasons.extend(_bbox_reasons(bbox, image))

        if not reasons:
            if current_image_id != image["id"]:
                current_image_id = image["id"]
                current_pixels, current_error = _open_image(images_dir / image["file_name"], image)
            if current_error is not None:
                reasons.append(current_error)

        if reasons:
            rejection = CropRejection(
                annotation_id=annotation["id"],
                image_id=annotation["image_id"],
                category_id=category_id,
                class_name=class_name,
                bbox=_safe_bbox(bbox),
                reasons=reasons,
            )
            rejections.append(rejection)
            logger.warning(
                "Crop excluido: annotation_id=%s image_id=%s motivos=%s",
                rejection.annotation_id,
                rejection.image_id,
                ",".join(reasons),
            )
            continue

        box = _crop_box(bbox)
        crop_id = f"img{image['id']}-ann{annotation['id']}"
        crop_path = f"{class_name}/{crop_id}.png"
        buffer = BytesIO()
        current_pixels.crop(box).save(buffer, format="PNG")
        data = buffer.getvalue()
        target = output_dir / crop_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

        crops.append(
            CropRecord(
                crop_id=crop_id,
                image_id=image["id"],
                annotation_id=annotation["id"],
                category_id=category_id,
                class_name=class_name,
                bbox=[float(value) for value in bbox],
                crop_box=list(box),
                source_file_name=image["file_name"],
                crop_path=crop_path,
                width=box[2] - box[0],
                height=box[3] - box[1],
                sha256=hashlib.sha256(data).hexdigest(),
            )
        )

    accepted_per_class = Counter(crop.class_name for crop in crops)
    accepted_images_per_class = {
        class_name: len({crop.image_id for crop in crops if crop.class_name == class_name})
        for class_name in CLASS_NAMES
    }
    rejected_per_reason = Counter(
        reason for rejection in rejections for reason in rejection.reasons
    )
    logger.info("Crops aceptados=%d rechazados=%d", len(crops), len(rejections))
    return CropReport(
        schema_version="1.0",
        dataset_version=dataset_version,
        provenance=release_provenance,
        classes=list(CLASS_NAMES),
        summary=CropSummary(
            total_annotations=len(annotations),
            accepted=len(crops),
            rejected=len(rejections),
            accepted_per_class={name: accepted_per_class[name] for name in CLASS_NAMES},
            accepted_images_per_class=accepted_images_per_class,
            rejected_per_reason=dict(sorted(rejected_per_reason.items())),
        ),
        crops=crops,
        rejections=rejections,
    )


def write_crop_report(report: CropReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path
