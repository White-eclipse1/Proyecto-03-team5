"""DVC-only entry point: fixed stage paths and a version tracked as a DVC parameter.

The general presentation entry point retains its environment-based Settings.
This wrapper never mutates environment variables or consults the release catalog.
"""

from pathlib import Path

from projections.models import load_projections_config

from presentation.projections import run
from storage.settings import Settings

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    config = load_projections_config(REPO_ROOT / "app/projections/projections.yaml")
    settings = Settings(
        dataset_dir=REPO_ROOT / "data/raw",
        reports_dir=REPO_ROOT / "reports",
        dataset_version=config.dataset_version,
        # Required by shared Settings, unused by this local-only stage.
        database_url="unused",
        minio_endpoint="unused",
        minio_port=9000,
        minio_access_key="unused",
        minio_secret_key="unused",
        minio_bucket="unused",
    )
    run(settings=settings, config=config)


if __name__ == "__main__":
    main()
