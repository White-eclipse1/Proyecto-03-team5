"""DVC stage (ML-01) that extracts crops from a verified P2 release."""

import logging
import shutil
from pathlib import Path

import yaml
from dvc_gate_stage import assert_quality_gate_passed
from releases.service import P2ReleaseService

from crops.classes import MIN_IMAGES_PER_CLASS
from crops.extract import assert_min_images_per_class, extract_crops, write_crop_report
from ingestion.loader import merge_raw_batches

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
REPORTS_DIR = REPO_ROOT / "reports"
CROPS_DIR = REPO_ROOT / "data" / "crops"

RELEASE_SELECTION_PATH = APP_DIR / "releases" / "selection.yaml"


def _load_release_selection() -> tuple[str, Path]:
    """Load the P2 release selection from versioned configuration."""
    try:
        payload = yaml.safe_load(RELEASE_SELECTION_PATH.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"Invalid P2 release selection: {RELEASE_SELECTION_PATH}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("P2 release selection must be a mapping")

    release_version = payload.get("release_version")
    catalog = payload.get("catalog")

    if not isinstance(release_version, str) or not release_version:
        raise RuntimeError("P2 release selection is missing release_version")

    if not isinstance(catalog, str) or not catalog:
        raise RuntimeError("P2 release selection is missing catalog")

    catalog_path = Path(catalog)
    if not catalog_path.is_absolute():
        catalog_path = APP_DIR / catalog_path

    return release_version, catalog_path


P2_RELEASE_VERSION, P2_RELEASE_CATALOG = _load_release_selection()


def write_crops() -> None:
    """Generate crops only from a verified and approved P2 release."""
    assert_quality_gate_passed()

    release_service = P2ReleaseService.from_catalog(P2_RELEASE_CATALOG)
    release = release_service.verify_release(
        P2_RELEASE_VERSION,
        REPO_ROOT,
    )

    annotations_dir = REPO_ROOT / release["coco_path"]
    images_dir = REPO_ROOT / release["images_path"]

    # Verify provenance before deleting or creating ML outputs.
    shutil.rmtree(CROPS_DIR, ignore_errors=True)

    provenance = {
        "release_version": release["release_version"],
        "images_dvc_hash": release["images_dvc_hash"],
        "annotations_dvc_hash": release["annotations_dvc_hash"],
        "quality_report": release["quality_report"],
    }

    report = extract_crops(
        merge_raw_batches(annotations_dir),
        images_dir=images_dir,
        output_dir=CROPS_DIR,
        dataset_version=release["release_version"],
        provenance=provenance,
    )
    # Sin el mínimo por clase no se publica el reporte: DVC no registra la corrida.
    assert_min_images_per_class(report, MIN_IMAGES_PER_CLASS)
    write_crop_report(report, REPORTS_DIR / "crops.json")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    write_crops()
