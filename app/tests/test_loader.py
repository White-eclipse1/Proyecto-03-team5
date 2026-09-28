import json

import pytest
from pydantic import ValidationError

from ingestion.loader import load_dataset

CATEGORIES = [{"id": 1, "name": "person"}, {"id": 3, "name": "dog"}, {"id": 4, "name": "cat"}]


def _write_lote(directory, name, *, image_id, ann_id, file_name, category_id):
    doc = {
        "images": [{"id": image_id, "file_name": file_name, "width": 10, "height": 10}],
        "annotations": [
            {
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [0, 0, 5, 5],
                "area": 25.0,
                "iscrowd": 0,
            }
        ],
        "categories": CATEGORIES,
    }
    (directory / name).write_text(json.dumps(doc), encoding="utf-8")


def test_merges_multiple_lote_files(tmp_path):
    _write_lote(tmp_path, "lote-a.json", image_id=1, ann_id=1, file_name="cat.0.jpg", category_id=4)
    _write_lote(tmp_path, "lote-b.json", image_id=2, ann_id=2, file_name="dog.0.jpg", category_id=3)

    dataset = load_dataset(tmp_path)

    assert len(dataset.images) == 2
    assert len(dataset.annotations) == 2
    assert {category.id for category in dataset.categories} == {1, 3, 4}


def test_dedupes_repeated_categories_by_id(tmp_path):
    _write_lote(tmp_path, "lote-a.json", image_id=1, ann_id=1, file_name="cat.0.jpg", category_id=4)
    _write_lote(tmp_path, "lote-b.json", image_id=2, ann_id=2, file_name="dog.0.jpg", category_id=3)

    dataset = load_dataset(tmp_path)

    assert len(dataset.categories) == 3


def test_category_id_collision_with_different_name_is_rejected(tmp_path):
    # id=4 significa "cat" en un lote y "wolf" en el otro: una colisión real,
    # no la repetición esperada del seeder (mismo id, mismo nombre).
    doc_a = {
        "images": [{"id": 1, "file_name": "cat.0.jpg", "width": 10, "height": 10}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 4,
                "bbox": [0, 0, 5, 5],
                "area": 25.0,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 4, "name": "cat"}],
    }
    doc_b = {
        "images": [{"id": 2, "file_name": "wolf.0.jpg", "width": 10, "height": 10}],
        "annotations": [
            {
                "id": 2,
                "image_id": 2,
                "category_id": 4,
                "bbox": [0, 0, 5, 5],
                "area": 25.0,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 4, "name": "wolf"}],
    }
    (tmp_path / "lote-a.json").write_text(json.dumps(doc_a), encoding="utf-8")
    (tmp_path / "lote-b.json").write_text(json.dumps(doc_b), encoding="utf-8")

    with pytest.raises(ValueError, match="category id=4"):
        load_dataset(tmp_path)


def test_cross_file_id_collision_is_rejected_not_silently_merged(tmp_path):
    # Mismo image_id/annotation_id en dos lotes distintos: el bug real de
    # project_dataset_id_collision_bug, reproducido a propósito.
    _write_lote(tmp_path, "lote-a.json", image_id=1, ann_id=1, file_name="cat.0.jpg", category_id=4)
    _write_lote(tmp_path, "lote-b.json", image_id=1, ann_id=1, file_name="dog.0.jpg", category_id=3)

    with pytest.raises(ValidationError):
        load_dataset(tmp_path)


def test_empty_directory_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path)


def test_loads_real_shared_dataset():
    """No es un fixture sintético: valida que el dataset real (data/raw/annotations)
    todavía pasa como un solo CocoDataset consistente."""
    from pathlib import Path

    real_dir = Path(__file__).resolve().parents[2] / "data" / "raw" / "annotations"
    pytest.importorskip("yaml")
    if not real_dir.exists() or not any(real_dir.glob("*.json")):
        pytest.skip("data/raw/annotations no está disponible en este entorno (falta dvc pull)")

    dataset = load_dataset(real_dir)
    assert len(dataset.images) > 0
    assert len(dataset.annotations) > 0
