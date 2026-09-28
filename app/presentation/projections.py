"""Load inputs, orchestrate pure projections, validate and atomically publish JSON."""

import logging
import os
import tempfile
from pathlib import Path

from projections.compute import compute_projections
from projections.models import ProjectionsConfig, load_projections_config, relative_image_name

from ingestion.loader import load_dataset, load_image_bytes
from presentation.projections_contracts import ProjectionsReport
from storage.settings import Settings


def build_projections_report(
    *, dataset_dir: Path, dataset_version: str, config: ProjectionsConfig
) -> ProjectionsReport:
    coco = load_dataset(dataset_dir / "annotations")
    images_dir = (dataset_dir / "images").resolve()
    for image in coco.images:
        relative_image_name(image.file_name)
        path = (images_dir / image.file_name).resolve()
        if not path.is_relative_to(images_dir):
            raise ValueError(f"Image path escapes dataset: COCO image_id={image.id}")
        if not path.is_file():
            raise ValueError(f"Missing COCO image_id={image.id}, file_name={image.file_name}")
    contents = load_image_bytes(coco, images_dir)
    result = compute_projections(coco, contents, config)
    by_id = {image.id: image for image in coco.images}
    labels = {image.id: set() for image in coco.images}
    for annotation in coco.annotations:
        labels[annotation.image_id].add(annotation.category_id)

    def points(coordinates):
        return [
            {
                "image_id": image_id,
                "file_name": by_id[image_id].file_name,
                "category_ids": sorted(labels[image_id]),
                "x": float(x),
                "y": float(y),
            }
            for image_id, (x, y) in zip(result.image_ids, coordinates, strict=True)
        ]

    return ProjectionsReport.model_validate(
        {
            "schema_version": "1.0",
            "dataset_version": dataset_version,
            "dataset_fingerprint": result.fingerprint,
            "total_images": len(result.image_ids),
            "categories": [c.model_dump() for c in sorted(coco.categories, key=lambda c: c.id)],
            "features": config.features.model_dump(),
            "pca": {
                "parameters": config.pca.model_dump(),
                "explained_variance_ratio": list(result.explained_variance_ratio),
                "points": points(result.pca),
            },
            "tsne": {"parameters": config.tsne.model_dump(), "points": points(result.tsne)},
        }
    )


def write_projections_report(report: ProjectionsReport, destination: Path) -> None:
    # Validate before creating the temporary file; never publish NaN/partial JSON.
    report = ProjectionsReport.model_validate(report.model_dump())
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, prefix=".projections-", delete=False
        ) as file:
            temporary = Path(file.name)
            file.write(report.model_dump_json(indent=2) + "\n")
            file.flush()
            os.fsync(file.fileno())
            # Public report must be readable by the separate nginx container.
        os.chmod(temporary, 0o644)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run(
    settings: Settings | None = None, config: ProjectionsConfig | None = None
) -> ProjectionsReport:
    settings = settings if settings is not None else Settings()
    config = config if config is not None else load_projections_config()
    report = build_projections_report(
        dataset_dir=settings.dataset_dir, dataset_version=settings.dataset_version, config=config
    )
    write_projections_report(report, settings.reports_dir / "projections.json")
    return report


def main() -> None:
    report = run()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).info(
        "Projected %s images (%s), fingerprint=%s",
        report.total_images,
        report.dataset_version,
        report.dataset_fingerprint,
    )


if __name__ == "__main__":
    main()
