"""DVC stage that computes the quality report without deciding the gate result."""

import logging
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

os.environ.setdefault("DATABASE_URL", "unused")
os.environ.setdefault("MINIO_ENDPOINT", "unused")
os.environ.setdefault("MINIO_PORT", "9000")
os.environ.setdefault("MINIO_ACCESS_KEY", "unused")
os.environ.setdefault("MINIO_SECRET_KEY", "unused")
os.environ.setdefault("MINIO_BUCKET", "unused")
os.environ.setdefault("DATASET_DIR", str(REPO_ROOT / "data" / "raw"))
os.environ.setdefault("REPORTS_DIR", str(REPO_ROOT / "reports"))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from presentation.gate import run

    run()
