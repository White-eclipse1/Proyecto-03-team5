"""DVC stage (ML-01) that extracts dog/cat crops from the approved P2 release.

The release version and the per-class image minimum are DVC parameters in
`crops/crops.yaml`; there is no `local-dev` default. The stage validates the
frozen release quality report and the DVC fingerprint of the source data
before writing any crop, and records both in `reports/crops.json`.
"""

import logging
import shutil
from pathlib import Path

from crops.extract import extract_crops, write_crop_report
from crops.models import CropSource
from crops.release import (
    assert_dataset_matches_release,
    assert_min_images_per_class,
    dvc_fingerprint,
    load_approved_release,
    load_crops_config,
)
from ingestion.loader import merge_raw_batches

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "data" / "raw"
REPORTS_DIR = REPO_ROOT / "reports"
CROPS_DIR = REPO_ROOT / "data" / "crops"
logger = logging.getLogger("dvc-crops")


def write_crops() -> None:
    """Regenerate `data/crops` from scratch and persist `reports/crops.json`."""
    config = load_crops_config()
    release, quality = load_approved_release(REPORTS_DIR, config.dataset_version)
    source = CropSource(
        dataset_version=release.dataset_version,
        quality_file=release.quality_file,
        quality_status=quality.status,
        images=dvc_fingerprint(DATASET_DIR / "images.dvc", relative_to=REPO_ROOT),
        annotations=dvc_fingerprint(DATASET_DIR / "annotations.dvc", relative_to=REPO_ROOT),
    )
    coco = merge_raw_batches(DATASET_DIR / "annotations")
    assert_dataset_matches_release(coco, quality)
    logger.info("Release %s validado (quality=%s)", release.dataset_version, quality.status)

    # `data/crops` is this stage's own DVC output: rebuilding it avoids stale crops.
    shutil.rmtree(CROPS_DIR, ignore_errors=True)
    report = extract_crops(
        coco, images_dir=DATASET_DIR / "images", output_dir=CROPS_DIR, source=source
    )
    assert_min_images_per_class(report, config.min_images_per_class)
    write_crop_report(report, REPORTS_DIR / "crops.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    write_crops()
