"""Fixtures compartidos por los tests del servidor MCP y del Copilot (P2-34/P2-52).

Prefijo `_` para que pytest no lo recolecte como módulo de tests; mismo criterio
que `_dataset_fixtures.py`.
"""

import json
from pathlib import Path

from storage.settings import Settings

CATEGORY_IDS = {"dog": 3, "cat": 4}


def mcp_settings(monkeypatch, dataset_dir: Path, reports_dir: Path) -> Settings:
    monkeypatch.setenv("DATABASE_URL", "mysql+pymysql://u:p@localhost/db")
    monkeypatch.setenv("MINIO_ENDPOINT", "localhost")
    monkeypatch.setenv("MINIO_PORT", "9000")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "ak")
    monkeypatch.setenv("MINIO_SECRET_KEY", "sk")
    monkeypatch.setenv("MINIO_BUCKET", "bucket")
    monkeypatch.setenv("DATASET_DIR", str(dataset_dir))
    monkeypatch.setenv("REPORTS_DIR", str(reports_dir))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return Settings(_env_file=None)


def write_annotations(annotations_dir: Path, dog_count: int = 2) -> None:
    """Un COCO mínimo: 1 imagen de gato y `dog_count` de perro, una anotación por imagen."""
    annotations_dir.mkdir(parents=True, exist_ok=True)
    file_names = ["cat.0.jpg", *(f"dog.{index}.jpg" for index in range(dog_count))]
    images = []
    annotations = []
    for image_id, file_name in enumerate(file_names, start=1):
        images.append({"id": image_id, "file_name": file_name, "width": 10, "height": 10})
        annotations.append(
            {
                "id": image_id,
                "image_id": image_id,
                "category_id": CATEGORY_IDS[file_name.split(".")[0]],
                "bbox": [0, 0, 5, 5],
                "area": 25.0,
                "iscrowd": 0,
            }
        )
    doc = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 3, "name": "dog"}, {"id": 4, "name": "cat"}],
    }
    (annotations_dir / "lote.json").write_text(json.dumps(doc), encoding="utf-8")


def write_releases(reports_dir: Path, statuses: dict[str, str]) -> None:
    """Escribe versions.json + releases/<v>/{quality,splits}.json, como `cut_release`."""
    catalog: dict = {"schema_version": "1.0", "releases": []}
    for version, status in statuses.items():
        release_dir = reports_dir / "releases" / version
        release_dir.mkdir(parents=True)
        quality = {
            "schema_version": "1.0",
            "dataset_version": version,
            "status": status,
            "checks": [],
        }
        splits = {"schema_version": "1.0", "dataset_version": version, "total_images": 10}
        (release_dir / "quality.json").write_text(json.dumps(quality), encoding="utf-8")
        (release_dir / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
        catalog["releases"].append(
            {
                "dataset_version": version,
                "quality_file": f"releases/{version}/quality.json",
                "splits_file": f"releases/{version}/splits.json",
            }
        )
    (reports_dir / "versions.json").write_text(json.dumps(catalog), encoding="utf-8")
