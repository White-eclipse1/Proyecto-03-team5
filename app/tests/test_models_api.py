"""APP-06: API de la pantalla Models (`training/models_view.py`, `ml-api`).

Lee el registro de OPS-06 (`reports/models/registry.json`, paquetes en `data/models`)
y la publicación en S3 de OPS-07 (`reports/models/s3_publications.json`).

TDD Requirement del issue #23: model version → artifact resolution. Agent Test: dos
versiones resuelven checkpoints (y sha256) distintos.
"""

import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from starlette.testclient import TestClient

from presentation.ml_contracts import ErrorResponse, ModelsResponse, RegisteredModelVersion
from tests._model_registry import BUCKET, MODEL, ROOT, Registry, make_checkpoint, sha256
from training.queue import TrainingJobQueue
from training.server import create_app

RUN_A = "a" * 32
RUN_B = "b" * 32


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory) -> dict[str, Path]:
    folder = tmp_path_factory.mktemp("checkpoints")
    return {
        "a": make_checkpoint(folder / "a.pt", bias=(6.0, -6.0)),
        "b": make_checkpoint(folder / "b.pt", bias=(-6.0, 6.0)),
    }


@pytest.fixture
def registry(tmp_path, checkpoints) -> Registry:
    built = Registry(tmp_path / "repo")
    built.register("1.0.0", RUN_A, checkpoints["a"])
    built.register("1.1.0", RUN_B, checkpoints["b"])
    return built


def client_for(tmp_path: Path, registry: Registry) -> TestClient:
    queue = TrainingJobQueue(create_engine(f"sqlite:///{tmp_path / 'jobs.db'}"))
    queue.create_tables()
    reports = registry.root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    return TestClient(create_app(queue=queue, reports_dir=reports, repo_root=registry.root))


def models(client: TestClient) -> dict[str, RegisteredModelVersion]:
    response = client.get("/models")
    assert response.status_code == 200, response.text
    parsed = ModelsResponse.model_validate(response.json())
    return {model.model_version: model for model in parsed.models}


# --- Versiones del registro ----------------------------------------------------------


def test_every_registered_version_is_listed_with_its_traceability(tmp_path, registry, checkpoints):
    listed = models(client_for(tmp_path, registry))

    assert set(listed) == {"1.0.0", "1.1.0"}, "la versión anterior sigue disponible"
    model = listed["1.0.0"]
    entry = registry.packages[0]
    assert (model.model_name, model.run_id) == (MODEL, RUN_A)
    assert model.checkpoint == f"runs:/{RUN_A}/checkpoints/best.pt"
    assert model.checkpoint_sha256 == sha256(checkpoints["a"])
    assert (model.dataset_version, model.manifest_hash) == (
        entry["dataset_version"],
        entry["manifest_hash"],
    )
    assert (model.architecture, model.image_size) == ("resnet18", 64)
    assert model.test_metrics.accuracy_top1 == entry["metrics"]["accuracy_top1"]
    assert model.model_card.purpose == entry["model_card"]["purpose"]
    assert model.model_card.limitations == entry["model_card"]["limitations"]
    assert [f.name for f in model.files] == list(entry["files"])
    assert model.servable and all(f.available for f in model.files)


def test_agent_test_each_version_resolves_its_own_checkpoint(tmp_path, registry, checkpoints):
    client = client_for(tmp_path, registry)

    first = RegisteredModelVersion.model_validate(client.get("/models/1.0.0").json())
    second = RegisteredModelVersion.model_validate(client.get("/models/1.1.0").json())

    assert (first.run_id, first.checkpoint_sha256) == (RUN_A, sha256(checkpoints["a"]))
    assert (second.run_id, second.checkpoint_sha256) == (RUN_B, sha256(checkpoints["b"]))


def test_unknown_version(tmp_path, registry):
    response = client_for(tmp_path, registry).get("/models/9.9.9")

    assert response.status_code == 404
    assert ErrorResponse.model_validate(response.json()).error.code == "model_not_found"


def test_without_registry_there_are_no_models(tmp_path):
    empty = Registry(tmp_path / "repo")

    assert models(client_for(tmp_path, empty)) == {}


def test_unreadable_registry(tmp_path, registry):
    registry.path.write_text("{no es json", encoding="utf-8")

    response = client_for(tmp_path, registry).get("/models")

    assert response.status_code == 503
    assert ErrorResponse.model_validate(response.json()).error.code == "registry_unavailable"


# --- Paquete en este servidor (servable) ---------------------------------------------


def test_package_not_pulled_is_not_servable(tmp_path, registry):
    registry.checkpoint("1.0.0").unlink()

    model = models(client_for(tmp_path, registry))["1.0.0"]

    assert not model.servable
    assert {f.name: f.available for f in model.files}["checkpoint/best.pt"] is False


def test_altered_package_file_is_not_available(tmp_path, registry):
    card = registry.root / registry.packages[0]["package_path"] / "model-card.md"
    card.write_text("otra tarjeta\n", encoding="utf-8")

    model = models(client_for(tmp_path, registry))["1.0.0"]

    assert not model.servable
    assert {f.name: f.available for f in model.files}["model-card.md"] is False


def test_version_with_metadata_only_is_listed_but_not_servable(tmp_path, registry):
    registry.register("0.9.0", "c" * 32, None, checkpoint_sha256="5" * 64)

    model = models(client_for(tmp_path, registry))["0.9.0"]

    assert not model.servable and model.files[0].name == "checkpoint/best.pt"


# --- Publicación en S3 ---------------------------------------------------------------


def test_published_version_shows_bucket_and_keys(tmp_path, registry):
    registry.publish("1.0.0")

    listed = models(client_for(tmp_path, registry))

    publication = listed["1.0.0"].publication
    assert (publication.status, publication.bucket, publication.region) == (
        "published",
        BUCKET,
        "us-east-1",
    )
    keys = {obj.name: obj.key for obj in publication.objects}
    assert keys["checkpoint/best.pt"] == f"models/{MODEL}/1.0.0/checkpoint/best.pt"
    assert "package.json" in keys
    assert listed["1.1.0"].publication.status == "not_published"


def _break_file(field: str, value: str):
    def mutate(record: dict) -> None:
        record["files"]["checkpoint/best.pt"][field] = value

    return mutate


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda r: r.update(checkpoint_sha256="0" * 64), "checkpoint"),
        (lambda r: r.update(run_id="f" * 32), "run"),
        (_break_file("s3_checksum_sha256", "0" * 64), "S3"),
        (_break_file("download_sha256", "0" * 64), "descarga"),
        (_break_file("sha256", "0" * 64), "registr"),
        (lambda r: r["files"].pop("model-card.md"), "model-card.md"),
        (lambda r: r.update(bucket=""), "bucket"),
    ],
    ids=[
        "other-checkpoint",
        "other-run",
        "s3-checksum",
        "download",
        "other-file-sha",
        "missing-file",
        "no-bucket",
    ],
)
def test_publication_that_does_not_match_is_never_shown_as_published(
    tmp_path, registry, mutate, fragment
):
    record = registry.publish("1.0.0")
    mutate(record)
    registry.save_publications()

    publication = models(client_for(tmp_path, registry))["1.0.0"].publication

    assert publication.status == "inconsistent"
    assert publication.objects == [] and publication.bucket is None
    assert fragment in publication.problem


def test_unreadable_publications_do_not_hide_the_models(tmp_path, registry):
    registry.publications_path.parent.mkdir(parents=True, exist_ok=True)
    registry.publications_path.write_text("{no es json", encoding="utf-8")

    publication = models(client_for(tmp_path, registry))["1.0.0"].publication

    assert publication.status == "inconsistent"
    assert "s3_publications.json" in publication.problem


# --- Descargas autorizadas -----------------------------------------------------------


@pytest.mark.parametrize("name", ["model-card.md", "dependencies.json", "checkpoint/best.pt"])
def test_package_files_can_be_downloaded(tmp_path, registry, name):
    path = registry.root / registry.packages[0]["package_path"] / name

    response = client_for(tmp_path, registry).get(f"/models/1.0.0/files/{name}")

    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    assert response.content == path.read_bytes()


@pytest.mark.parametrize(
    "name",
    ["package.json", "../../registry.json", "..%2F..%2F..%2Freports%2Fmodels%2Fregistry.json"],
    ids=["not-in-the-package", "traversal", "encoded-traversal"],
)
def test_only_registered_package_files_are_served(tmp_path, registry, name):
    response = client_for(tmp_path, registry).get(f"/models/1.0.0/files/{name}")

    assert response.status_code == 404
    # `../` sin codificar lo normaliza el cliente HTTP (llega a /models/registry.json).
    assert ErrorResponse.model_validate(response.json()).error.code in {
        "file_not_found",
        "model_not_found",
    }
    assert registry.path.read_text(encoding="utf-8") not in response.text


def test_encoded_traversal_reaches_the_whitelist(tmp_path, registry):
    response = client_for(tmp_path, registry).get(
        "/models/1.0.0/files/..%2F..%2F..%2Freports%2Fmodels%2Fregistry.json"
    )

    assert ErrorResponse.model_validate(response.json()).error.code == "file_not_found"


def test_altered_or_missing_file_is_not_served(tmp_path, registry, checkpoints):
    registry.checkpoint("1.0.0").write_bytes(checkpoints["b"].read_bytes())
    registry.checkpoint("1.1.0").unlink()
    client = client_for(tmp_path, registry)

    altered = client.get("/models/1.0.0/files/checkpoint/best.pt")
    missing = client.get("/models/1.1.0/files/checkpoint/best.pt")

    assert altered.status_code == 409
    assert ErrorResponse.model_validate(altered.json()).error.code == "file_not_servable"
    assert missing.status_code == 404
    assert ErrorResponse.model_validate(missing.json()).error.code == "file_not_available"


# --- Datos reales del repo (OPS-06 + OPS-07) -----------------------------------------


def test_real_registry_and_s3_publication(tmp_path):
    queue = TrainingJobQueue(create_engine(f"sqlite:///{tmp_path / 'jobs.db'}"))
    queue.create_tables()
    client = TestClient(create_app(queue=queue, reports_dir=ROOT / "reports", repo_root=ROOT))
    registry = json.loads((ROOT / "reports" / "models" / "registry.json").read_text("utf-8"))

    listed = models(client)

    entry = registry["models"][0]
    model = listed[entry["model_version"]]
    assert (model.run_id, model.checkpoint_sha256) == (entry["run_id"], entry["checkpoint_sha256"])
    assert model.publication.status == "published"
    assert model.publication.bucket == BUCKET
    assert {obj.name for obj in model.publication.objects} >= set(entry["files"])
