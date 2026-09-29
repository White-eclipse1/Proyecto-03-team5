"""DVC stage (ML-01) that extracts crops from a verified P2 release."""

import logging
import os
import shutil
from pathlib import Path

from dvc_gate_stage import assert_quality_gate_passed
from releases.service import P2ReleaseService

from crops.extract import extract_crops, write_crop_report
from ingestion.loader import merge_raw_batches

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", REPO_ROOT / "reports"))
CROPS_DIR = Path(os.environ.get("CROPS_DIR", REPO_ROOT / "data" / "crops"))

P2_RELEASE_VERSION = os.environ.get("P2_RELEASE_VERSION", "v0.1.1")
P2_RELEASE_CATALOG = Path(
    os.environ.get(
        "P2_RELEASE_CATALOG",
        REPO_ROOT / "app" / "releases" / "p2_releases.json",
    )
)


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

    report = extract_crops(
        merge_raw_batches(annotations_dir),
        images_dir=images_dir,
        output_dir=CROPS_DIR,
        dataset_version=release["release_version"],
    )
    write_crop_report(report, REPORTS_DIR / "crops.json")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    write_crops()
