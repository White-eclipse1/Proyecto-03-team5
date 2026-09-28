"""Synthetic images only: real Pillow/sklearn pipeline and independent JSON contract."""

import copy
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageOps
from projections.compute import compute_projections, dataset_fingerprint, extract_features
from projections.models import ProjectionsConfig, load_projections_config
from pydantic import ValidationError

from ingestion.loader import load_dataset, load_image_bytes
from ingestion.models import CocoDataset
from presentation.projections import build_projections_report, run, write_projections_report
from presentation.projections_contracts import ProjectionsReport
from tests._dataset_fixtures import jpeg_bytes, write_coco_dataset


@pytest.fixture
def config():
    doc = load_projections_config().model_dump()
    doc["tsne"].update(perplexity=2.0, max_iter=300)
    return ProjectionsConfig.model_validate(doc)


@pytest.fixture
def dataset(tmp_path):
    return write_coco_dataset(tmp_path / "raw", cats=3, dogs=3)


@pytest.fixture
def inputs(dataset):
    coco = load_dataset(dataset / "annotations")
    return coco, load_image_bytes(coco, dataset / "images")


@pytest.fixture
def report(dataset, config):
    return build_projections_report(dataset_dir=dataset, dataset_version="test-v1", config=config)


def test_features_shape_range_order_and_exif(config):
    contents = {2: jpeg_bytes(2), 1: jpeg_bytes(1)}
    features = extract_features(contents, config.features)
    assert features.shape == (2, 768)
    assert 0 <= features.min() <= features.max() <= 1
    np.testing.assert_array_equal(
        features, extract_features(dict(reversed(contents.items())), config.features)
    )
    source = Image.open(BytesIO(jpeg_bytes(3)))
    exif = source.getexif()
    exif[274] = 6
    encoded = BytesIO()
    source.save(encoded, format="JPEG", exif=exif)
    with Image.open(BytesIO(encoded.getvalue())) as image:
        expected = (
            np.asarray(
                ImageOps.exif_transpose(image)
                .convert("RGB")
                .resize((16, 16), Image.Resampling.LANCZOS),
                dtype=float,
            ).reshape(-1)
            / 255
        )
    np.testing.assert_array_equal(
        extract_features({9: encoded.getvalue()}, config.features)[0], expected
    )


def test_real_algorithms_reproducible_and_input_order_independent(inputs, config):
    coco, contents = inputs
    before = coco.model_dump()
    first = compute_projections(coco, contents, config)
    reordered = CocoDataset.model_validate(
        {key: list(reversed(value)) for key, value in before.items()}
    )
    second = compute_projections(reordered, dict(reversed(contents.items())), config)
    assert first.image_ids == tuple(sorted(contents))
    for actual, expected in [(first.pca, second.pca), (first.tsne, second.tsne)]:
        assert actual.shape == (6, 2)
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert first.fingerprint == second.fingerprint
    assert coco.model_dump() == before


def test_multilabel_repeated_annotations_and_unlabeled(dataset, config):
    import json

    path = dataset / "annotations/lote.json"
    doc = json.loads(path.read_text())
    doc["annotations"] = [ann for ann in doc["annotations"] if ann["image_id"] != 6]
    for ann_id, category_id in [(20, 3), (21, 4)]:
        doc["annotations"].append(
            {**doc["annotations"][0], "id": ann_id, "category_id": category_id}
        )
    path.write_text(json.dumps(doc))
    result = build_projections_report(dataset_dir=dataset, dataset_version="test-v1", config=config)
    assert len(result.pca.points) == len(result.tsne.points) == 6
    assert result.pca.points[0].category_ids == [3, 4]
    assert result.pca.points[-1].category_ids == []
    assert result.pca.points[0].file_name == "cat.0.jpg"


def test_fingerprint_tracks_coco_and_encoded_contents(inputs):
    coco, contents = inputs
    original = dataset_fingerprint(coco, contents)
    assert original == dataset_fingerprint(coco.model_copy(deep=True), dict(contents))
    altered = dict(contents)
    altered[1] = jpeg_bytes(99)
    assert dataset_fingerprint(coco, altered) != original
    changed = coco.model_copy(deep=True)
    changed.annotations[0].bbox[0] = 1.0
    assert dataset_fingerprint(changed, contents) != original


@pytest.mark.parametrize("perplexity", [0, -1, float("nan"), float("inf")])
def test_invalid_config(perplexity, config):
    doc = config.model_dump()
    doc["tsne"]["perplexity"] = perplexity
    with pytest.raises(ValidationError):
        ProjectionsConfig.model_validate(doc)


def test_too_large_perplexity(inputs):
    coco, contents = inputs
    with pytest.raises(ValueError, match="perplexity"):
        compute_projections(coco, contents, load_projections_config())


def test_small_dataset_and_missing_bytes(inputs, config):
    coco, contents = inputs
    small = CocoDataset.model_validate(
        {
            "images": [coco.images[0].model_dump()],
            "annotations": [],
            "categories": [c.model_dump() for c in coco.categories],
        }
    )
    with pytest.raises(ValueError, match="three"):
        compute_projections(small, {1: contents[1]}, config)
    with pytest.raises(ValueError, match="exactly"):
        compute_projections(coco, {1: contents[1]}, config)


def test_no_variation(inputs, config):
    coco, contents = inputs
    same = dict.fromkeys(contents, contents[1])
    with pytest.raises(ValueError, match="variation"):
        compute_projections(coco, same, config)


@pytest.mark.parametrize("mode", ["missing", "corrupt"])
def test_missing_or_corrupt_image_never_publishes_partial(dataset, config, tmp_path, mode):
    destination = tmp_path / "projections.json"
    destination.write_text("previous valid report")
    image = dataset / "images/cat.0.jpg"
    if mode == "missing":
        image.unlink()
    else:
        image.write_bytes(b"not an image")
    settings = SimpleNamespace(dataset_dir=dataset, dataset_version="test-v1", reports_dir=tmp_path)
    with pytest.raises(ValueError, match="image_id=1"):
        run(settings=settings, config=config)
    assert destination.read_text() == "previous valid report"


def test_round_trip_and_atomic_failure(report, tmp_path, monkeypatch):
    destination = tmp_path / "projections.json"
    write_projections_report(report, destination)
    assert ProjectionsReport.model_validate_json(destination.read_text()) == report
    before = destination.read_bytes()

    def fail_replace(*args):
        raise OSError("simulated replace failure")

    monkeypatch.setattr("presentation.projections.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        write_projections_report(report, destination)
    assert destination.read_bytes() == before
    assert list(tmp_path.glob(".projections-*")) == []


@pytest.mark.parametrize(
    "mutation",
    [
        "nan",
        "inf",
        "count",
        "duplicate",
        "coverage",
        "filename",
        "absolute",
        "category",
        "labels",
        "variance",
        "empty",
        "metadata",
    ],
)
def test_contract_rejects_invalid_report(report, mutation):
    doc = copy.deepcopy(report.model_dump())
    point = doc["tsne"]["points"][0]
    if mutation in ("nan", "inf"):
        point["x"] = float(mutation)
    elif mutation == "count":
        doc["total_images"] += 1
    elif mutation == "duplicate":
        doc["tsne"]["points"][1] = copy.deepcopy(point)
    elif mutation == "coverage":
        point["image_id"] = 999
    elif mutation == "filename":
        point["file_name"] = "other.jpg"
    elif mutation == "absolute":
        point["file_name"] = "/private/image.jpg"
    elif mutation == "category":
        point["category_ids"] = [999]
    elif mutation == "labels":
        point["category_ids"] = []
    elif mutation == "variance":
        doc["pca"]["explained_variance_ratio"] = [0.8, 0.8]
    elif mutation == "empty":
        doc["total_images"] = 0
    else:
        del doc["features"]["width"]
    with pytest.raises(ValidationError):
        ProjectionsReport.model_validate(doc)


@pytest.mark.parametrize("value", [0, 1.0, 9007199254740991])
def test_interoperable_ids_accept_mathematical_integers(value):
    from presentation.projections_contracts import ProjectionCategory, ProjectionPoint

    category = ProjectionCategory.model_validate({"id": value, "name": "cat"})
    point = ProjectionPoint.model_validate(
        {
            "image_id": value,
            "file_name": "cat.jpg",
            "category_ids": [value],
            "x": 0,
            "y": 0,
        }
    )
    assert category.id == int(value)
    assert point.image_id == int(value)
    assert point.category_ids == [int(value)]


@pytest.mark.parametrize("value", [9007199254740992, -1, 1.5, True, False, "1"])
@pytest.mark.parametrize("field", ["image_id", "category_ids", "category.id"])
def test_interoperable_ids_reject_invalid_numbers(value, field):
    from presentation.projections_contracts import ProjectionCategory, ProjectionPoint

    doc = {"image_id": 1, "file_name": "cat.jpg", "category_ids": [1], "x": 0, "y": 0}
    model = ProjectionPoint
    if field == "category.id":
        model = ProjectionCategory
        doc = {"id": value, "name": "cat"}
    else:
        doc[field] = [value] if field == "category_ids" else value
    with pytest.raises(ValidationError):
        model.model_validate(doc)


@pytest.mark.parametrize("value", [False, True])
def test_whiten_accepts_only_real_booleans(value, config):
    from projections.models import PCAConfig

    doc = config.pca.model_dump()
    doc["whiten"] = value
    assert PCAConfig.model_validate(doc).whiten is value


@pytest.mark.parametrize("value", [0, 1, 0.0, 1.0, "false", "true"])
def test_whiten_rejects_boolean_coercion(value, config):
    from projections.models import PCAConfig

    doc = config.pca.model_dump()
    doc["whiten"] = value
    with pytest.raises(ValidationError):
        PCAConfig.model_validate(doc)


def test_dvc_wrapper_ignores_environment_and_tracks_version(tmp_path, monkeypatch):
    import os
    from pathlib import Path

    import dvc_projections_stage as wrapper
    import yaml

    repo = Path(__file__).resolve().parents[2]
    stage = yaml.safe_load((repo / "dvc.yaml").read_text())["stages"]["projections"]
    tracked_parameters = stage["params"][0]["projections/projections.yaml"]
    assert "dataset_version" in tracked_parameters
    config_doc = load_projections_config().model_dump()
    config_path = tmp_path / "app/projections/projections.yaml"
    config_path.parent.mkdir(parents=True)
    monkeypatch.setattr(wrapper, "REPO_ROOT", tmp_path)
    for key, value in {
        "DATASET_DIR": "/untrusted/raw",
        "REPORTS_DIR": "/untrusted/reports",
        "DATASET_VERSION": "untracked-version",
        "DATABASE_URL": "unchanged-infrastructure",
        "MINIO_ENDPOINT": "unchanged-infrastructure",
    }.items():
        monkeypatch.setenv(key, value)
    before = dict(os.environ)
    calls = []
    monkeypatch.setattr(wrapper, "run", lambda **kwargs: calls.append(kwargs))
    for version in ("local-dev", "tracked-v2"):
        config_doc["dataset_version"] = version
        config_path.write_text(yaml.safe_dump(config_doc))
        wrapper.main()
        effective = calls[-1]
        settings = effective["settings"]
        assert settings.dataset_version == version
        assert effective["config"].dataset_version == version
        wdir = tmp_path / stage["wdir"]
        for relative in ("annotations", "images"):
            declared = next(path for path in stage["deps"] if path.endswith(f"raw/{relative}"))
            assert settings.dataset_dir / relative == (wdir / declared).resolve()
        output = next(iter(stage["outs"][0]))
        assert settings.reports_dir / "projections.json" == (wdir / output).resolve()
    assert len(calls) == 2
    assert dict(os.environ) == before
    assert not (tmp_path / "reports").exists()
    assert not (tmp_path / "data").exists()
