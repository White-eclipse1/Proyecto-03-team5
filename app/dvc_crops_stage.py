"""DVC stage (ML-01) that extracts dog/cat crops once the quality gate passed."""

import logging
import os
import shutil
from pathlib import Path

from dvc_gate_stage import assert_quality_gate_passed

from crops.extract import extract_crops, write_crop_report
from ingestion.loader import merge_raw_batches

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = Path(os.environ.get("DATASET_DIR", REPO_ROOT / "data" / "raw"))
REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", REPO_ROOT / "reports"))
CROPS_DIR = Path(os.environ.get("CROPS_DIR", REPO_ROOT / "data" / "crops"))
DATASET_VERSION = os.environ.get("DATASET_VERSION", "local-dev")


def write_crops() -> None:
    """Regenerate `data/crops` from scratch and persist `reports/crops.json`."""
    assert_quality_gate_passed()
    # `data/crops` is this stage's own DVC output: rebuilding it avoids stale crops.
    shutil.rmtree(CROPS_DIR, ignore_errors=True)
    report = extract_crops(
        merge_raw_batches(DATASET_DIR / "annotations"),
        images_dir=DATASET_DIR / "images",
        output_dir=CROPS_DIR,
        dataset_version=DATASET_VERSION,
    )
    write_crop_report(report, REPORTS_DIR / "crops.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    write_crops()
