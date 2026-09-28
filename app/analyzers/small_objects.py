"""P2-13: análisis puro de cajas COCO [x, y, width, height], en píxeles."""

from collections import Counter
from math import isfinite

from pydantic import BaseModel, ConfigDict, Field

from analyzers.base import AnalyzerResult


class SmallObjectConfig(BaseModel):
    """Sin defaults: los límites se suministran desde la configuración de políticas."""

    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    width_px: float = Field(gt=0)
    height_px: float = Field(gt=0)
    threshold: float = Field(ge=0, le=1)


def analyze_small_objects(coco: dict, config: SmallObjectConfig) -> AnalyzerResult:
    """Cuenta todas las anotaciones (incluidas crowd), sin leer ni modificar el COCO.

    Pequeña: width < width_px AND height < height_px. La clase más afectada
    tiene más cajas pequeñas; los empates se resuelven por menor category_id.
    Se espera COCO válido con annotations y categories; cajas no finitas o
    degeneradas se rechazan para no producir una proporción engañosa.
    """
    categories = {category["id"]: category["name"] for category in coco["categories"]}
    totals = Counter()
    small_counts = Counter()
    offenders = []

    for annotation in coco["annotations"]:
        bbox = annotation["bbox"]
        if (
            len(bbox) != 4
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                for value in bbox
            )
            or bbox[2] <= 0
            or bbox[3] <= 0
        ):
            raise ValueError("Expected a finite COCO bbox [x, y, positive width, positive height]")
        category_id = annotation["category_id"]
        if category_id not in categories:
            raise ValueError("Annotation references an unknown category_id")
        totals[category_id] += 1
        if bbox[2] < config.width_px and bbox[3] < config.height_px:
            small_counts[category_id] += 1
            offenders.append(
                {
                    "annotation_id": annotation["id"],
                    "image_id": annotation["image_id"],
                    "category_id": category_id,
                    "bbox": list(bbox),
                }
            )

    total = sum(totals.values())
    small = len(offenders)
    ratio = small / total if total else 0.0
    most_affected = None
    if small_counts:
        category_id = min(small_counts, key=lambda key: (-small_counts[key], key))
        most_affected = {
            "category_id": category_id,
            "category_name": categories[category_id],
            "small_objects": small_counts[category_id],
            "total_objects": totals[category_id],
            "ratio": small_counts[category_id] / totals[category_id],
        }

    return AnalyzerResult(
        check_name="max_small_object_ratio",
        passed=ratio <= config.threshold,
        metric_value=ratio,
        details={
            "small_objects": small,
            "total_objects": total,
            "most_affected_class": most_affected,
            "offending_samples": offenders,
        },
    )
