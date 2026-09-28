"""P2-21: diagnóstico geométrico del COCO crudo, previo a ingesta estricta."""

from math import isclose, isfinite

from pydantic import BaseModel, ConfigDict, Field

from analyzers.base import AnalyzerResult

AREA_REL_TOL = 1e-9
AREA_ABS_TOL = 1e-6  # píxeles cuadrados


class InvalidBoxConfig(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    threshold: float = Field(ge=0)


def _finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def analyze_invalid_boxes(coco: dict, config: InvalidBoxConfig) -> AnalyzerResult:
    """Inspecciona bbox numéricas de cuatro componentes, sin I/O ni reparación.

    Las dimensiones proceden de coco['images'], no de los píxeles. Una referencia
    desconocida se reporta y no impide diagnosticar otras anotaciones. La forma
    estructural malformada se rechaza explícitamente, no se valida segmentación.
    """
    images = {}
    for image in coco["images"]:
        if image["id"] in images:
            raise ValueError(f"Duplicate image_id: {image['id']}")
        if any(
            not _finite_number(image[dimension]) or image[dimension] <= 0
            for dimension in ("width", "height")
        ):
            raise ValueError(f"Image {image['id']} must have positive finite dimensions")
        images[image["id"]] = image

    offenders = []
    for annotation in coco["annotations"]:
        bbox = annotation["bbox"]
        if (
            not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
            or not all(_finite_number(value) for value in bbox)
        ):
            raise ValueError(f"Annotation {annotation['id']} requires four finite bbox values")
        reported_area = annotation["area"]
        if not _finite_number(reported_area):
            raise ValueError(f"Annotation {annotation['id']} requires a finite area")

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

        image = images.get(annotation["image_id"])
        if image is None:
            reasons.append("unknown_image_id")
        else:
            if x + width > image["width"]:
                reasons.append("exceeds_image_width")
            if y + height > image["height"]:
                reasons.append("exceeds_image_height")

        calculated_area = width * height
        if not isfinite(calculated_area):
            calculated_area = None
            reasons.append("non_finite_calculated_area")
        elif not isclose(
            reported_area, calculated_area, rel_tol=AREA_REL_TOL, abs_tol=AREA_ABS_TOL
        ):
            reasons.append("area_mismatch")

        if reasons:
            offenders.append(
                {
                    "annotation_id": annotation["id"],
                    "image_id": annotation["image_id"],
                    "bbox": list(bbox),
                    "reasons": reasons,
                    "reported_area": reported_area,
                    "calculated_area": calculated_area,
                    "image_width": image["width"] if image is not None else None,
                    "image_height": image["height"] if image is not None else None,
                }
            )

    invalid_count = len(offenders)
    return AnalyzerResult(
        check_name="degenerate_boxes",
        passed=invalid_count <= config.threshold,
        metric_value=float(invalid_count),
        details={
            "total_annotations": len(coco["annotations"]),
            "invalid_annotations": invalid_count,
            "offending_samples": offenders,
        },
    )
