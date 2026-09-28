"""P2-19: desbalance por imágenes distintas, sin I/O ni evaluación de la compuerta."""

from pydantic import BaseModel, ConfigDict, Field

from analyzers.base import AnalyzerResult


class ImbalanceConfig(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    min_images_per_class: float = Field(ge=0)
    threshold: float = Field(ge=0)


def analyze_imbalance(coco: dict, config: ImbalanceConfig) -> AnalyzerResult:
    """Recibe COCO ya cargado; cuenta cada image_id una vez por categoría.

    Una categoría sin imágenes hace indefinido el ratio: metric_value=0.0
    es entonces un marcador, ratio_defined=False y passed=False. No se
    descartan categorías vacías ni se inventa un denominador distinto de cero.
    """
    categories = coco["categories"]
    if not categories:
        raise ValueError("At least one declared category is required")
    category_ids = [category["id"] for category in categories]
    image_ids = [image["id"] for image in coco["images"]]
    if len(set(category_ids)) != len(category_ids):
        raise ValueError("Duplicate category_id in categories")
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("Duplicate image_id in images")

    declared_images = set(image_ids)
    images_by_category = {category_id: set() for category_id in category_ids}
    for annotation in coco["annotations"]:
        category_id = annotation["category_id"]
        image_id = annotation["image_id"]
        if category_id not in images_by_category:
            raise ValueError(f"Unknown category_id: {category_id}")
        if image_id not in declared_images:
            raise ValueError(f"Unknown image_id: {image_id}")
        images_by_category[category_id].add(image_id)

    counts = [
        {
            "category_id": category["id"],
            "category_name": category["name"],
            "image_count": len(images_by_category[category["id"]]),
            "image_ids": sorted(images_by_category[category["id"]]),
        }
        for category in sorted(categories, key=lambda category: category["id"])
    ]
    # counts is sorted by category_id, so ties always select the smallest ID.
    majority = max(counts, key=lambda category: category["image_count"])
    minority = min(counts, key=lambda category: category["image_count"])
    ratio_defined = minority["image_count"] > 0
    ratio = majority["image_count"] / minority["image_count"] if ratio_defined else 0.0

    return AnalyzerResult(
        check_name="max_imbalance_ratio",
        passed=ratio_defined and ratio <= config.threshold,
        metric_value=ratio,
        details={
            "images_per_category": counts,
            "majority_class": majority,
            "minority_class": minority,
            "classes_below_minimum": [
                category
                for category in counts
                if category["image_count"] < config.min_images_per_class
            ],
            "empty_category_ids": [
                category["category_id"] for category in counts if category["image_count"] == 0
            ],
            "min_images_per_class": config.min_images_per_class,
            "max_imbalance_ratio": config.threshold,
            "ratio_defined": ratio_defined,
        },
    )
