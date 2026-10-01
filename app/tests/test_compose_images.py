"""Issue #36: las imágenes de docker-compose.yml deben poder descargarse sin login.

`quay.io/minio/minio` y `minio/minio` dejaron de ser públicas; con ellas un clon
nuevo no puede hacer `docker compose up` (requisito M1).
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PRIVATE_IMAGES = ("quay.io/minio/minio", "minio/minio", "docker.io/minio/minio")


def _services() -> dict:
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))["services"]


def test_compose_does_not_use_images_that_require_login():
    offenders = {
        name: service["image"]
        for name, service in _services().items()
        if service.get("image", "").split(":")[0] in PRIVATE_IMAGES
    }
    assert not offenders, f"Imágenes que ya no son públicas: {offenders}"


def test_minio_runs_as_root_so_it_can_write_its_volume():
    """La imagen de Chainguard corre como 65532 y no puede escribir un volumen nuevo."""
    minio = _services()["minio"]
    assert minio["image"].startswith("cgr.dev/chainguard/minio")
    assert minio["user"] == "0:0"
    assert "minio_data:/data" in minio["volumes"]


def test_readme_snippets_use_the_same_minio_image():
    for readme in (ROOT / "README.md", ROOT / "backend" / "README.md"):
        text = readme.read_text(encoding="utf-8")
        assert "-d quay.io/minio/minio" not in text, readme
