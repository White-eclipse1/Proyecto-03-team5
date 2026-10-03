"""APP-07: inferencia dog/cat real en `ml-api` (`training/inference.py`).

TDD Requirement del issue #24: validación de la entrada (tipo, tamaño, archivo
ilegible) y esquema de la respuesta (`InferenceResponse`).

Agent Test: con la misma imagen, cambiar de model version cambia el checkpoint (y
su sha256) que se carga y la predicción; el nombre del archivo no influye.

Las versiones salen del registro de OPS-06 (`reports/models/registry.json`, paquetes
en `data/models/<modelo>/<versión>/`, `classification.registry`). Aquí es un registro
temporal con el mismo formato y checkpoints reales (`save_checkpoint` de ML-03): la
inferencia carga pesos de verdad y aplica el preprocesamiento de evaluación de ML-02.
"""

import hashlib
import io
import json
from pathlib import Path

import pytest
from PIL import Image
from sqlalchemy import create_engine
from starlette.testclient import TestClient

from classification.model import predict_proba
from classification.transforms import eval_transform
from presentation.ml_contracts import ErrorResponse, InferenceResponse
from tests._model_registry import IMAGE_SIZE, MODEL, RELEASE, Registry, make_checkpoint, sha256
from training.inference import PackageRegistryResolver
from training.queue import TrainingJobQueue
from training.server import create_app

RUN_DOG = "a" * 32
RUN_CAT = "b" * 32
RUN_REAL = "c" * 32


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory) -> dict[str, Path]:
    folder = tmp_path_factory.mktemp("checkpoints")
    return {
        "dog": make_checkpoint(folder / "dog.pt", bias=(6.0, -6.0)),
        "cat": make_checkpoint(folder / "cat.pt", bias=(-6.0, 6.0)),
        "real": make_checkpoint(folder / "real.pt", bias=None, seed=3),
    }


@pytest.fixture
def registry(tmp_path, checkpoints) -> Registry:
    built = Registry(tmp_path / "repo")
    built.register("1.0.0", RUN_DOG, checkpoints["dog"])
    built.register("1.1.0", RUN_CAT, checkpoints["cat"])
    built.register("2.0.0", RUN_REAL, checkpoints["real"])
    return built


def client_for(tmp_path, registry=None, *, crops_dir=None, reports_dir=None, **limits):
    queue = TrainingJobQueue(create_engine(f"sqlite:///{tmp_path / 'jobs.db'}"))
    queue.create_tables()
    reports = reports_dir or tmp_path / "reports"
    reports.mkdir(exist_ok=True)
    models = (
        PackageRegistryResolver(registry.path, repo_root=registry.root)
        if registry is not None
        else None
    )
    app = create_app(queue=queue, reports_dir=reports, crops_dir=crops_dir, models=models, **limits)
    return TestClient(app)


def image_bytes(color=(200, 120, 40), size=(80, 60), fmt="PNG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format=fmt)
    return buffer.getvalue()


def striped(size=(80, 60)) -> bytes:
    image = Image.new("RGB", size, (0, 0, 0))
    for x in range(0, size[0], 4):
        for y in range(size[1]):
            image.putpixel((x, y), (255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def upload(client, data: bytes, *, filename="foto.png", version="1.0.0", content_type="image/png"):
    return client.post(
        "/inference/upload",
        data={"model_name": MODEL, "model_version": version},
        files={"file": (filename, data, content_type)},
    )


def error_code(response) -> str:
    return ErrorResponse.model_validate(response.json()).error.code


# --- Imagen subida: esquema de la respuesta ------------------------------------------


def test_upload_returns_a_valid_inference_response(tmp_path, registry, checkpoints):
    data = image_bytes()
    response = upload(client_for(tmp_path, registry), data)

    assert response.status_code == 200, response.text
    result = InferenceResponse.model_validate(response.json())
    assert (result.source, result.crop) == ("upload", None)
    assert result.upload.sha256 == hashlib.sha256(data).hexdigest()
    assert (result.upload.width, result.upload.height) == (80, 60)
    assert (result.upload.content_type, result.upload.size_bytes) == ("image/png", len(data))
    assert (result.model_name, result.model_version) == (MODEL, "1.0.0")
    assert result.run_id == RUN_DOG
    assert result.checkpoint == f"runs:/{RUN_DOG}/checkpoints/best.pt"
    assert result.checkpoint_sha256 == sha256(checkpoints["dog"])
    assert (result.dataset_version, result.image_size) == (RELEASE, IMAGE_SIZE)
    assert set(result.probabilities) == {"dog", "cat"}
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    assert result.predicted_class == "dog"


def test_jpeg_is_detected_by_content_not_by_name(tmp_path, registry):
    response = upload(
        client_for(tmp_path, registry),
        image_bytes(fmt="JPEG"),
        filename="foto.png",
        content_type="image/png",
    )

    assert response.status_code == 200, response.text
    assert response.json()["upload"]["content_type"] == "image/jpeg"


# --- Agent Test: la versión elegida decide el artefacto ------------------------------


def test_agent_test_changing_model_version_changes_the_loaded_checkpoint(
    tmp_path, registry, checkpoints
):
    client = client_for(tmp_path, registry)
    data = image_bytes()

    first = InferenceResponse.model_validate(upload(client, data, version="1.0.0").json())
    second = InferenceResponse.model_validate(upload(client, data, version="1.1.0").json())

    assert (first.run_id, first.checkpoint_sha256) == (RUN_DOG, sha256(checkpoints["dog"]))
    assert (second.run_id, second.checkpoint_sha256) == (RUN_CAT, sha256(checkpoints["cat"]))
    assert (first.predicted_class, second.predicted_class) == ("dog", "cat")


def test_the_file_name_does_not_decide_the_class(tmp_path, registry):
    client = client_for(tmp_path, registry)
    data = image_bytes()

    as_cat = upload(client, data, filename="cat.png").json()
    as_dog = upload(client, data, filename="dog.png").json()

    assert as_cat["predicted_class"] == "dog"  # la versión 1.0.0 siempre dice dog
    assert as_cat["probabilities"] == as_dog["probabilities"]


def test_the_prediction_comes_from_the_model_and_the_eval_preprocessing(
    tmp_path, registry, checkpoints
):
    from classification.model import load_checkpoint

    client = client_for(tmp_path, registry)
    plain, stripes = image_bytes(), striped()

    results = [
        upload(client, data, version="2.0.0").json()["probabilities"] for data in (plain, stripes)
    ]

    assert results[0] != results[1], "con pesos reales, imágenes distintas dan otra salida"
    model = load_checkpoint(checkpoints["real"])
    tensor = eval_transform(IMAGE_SIZE)(Image.open(io.BytesIO(stripes)).convert("RGB"))
    expected = predict_proba(model, tensor.unsqueeze(0))[0].tolist()
    assert [results[1]["dog"], results[1]["cat"]] == pytest.approx(expected, abs=1e-6)


def test_each_model_version_is_loaded_once(tmp_path, registry, monkeypatch):
    import training.inference as inference

    loads = []
    original = inference.load_checkpoint
    monkeypatch.setattr(
        inference, "load_checkpoint", lambda path: loads.append(path) or original(path)
    )
    client = client_for(tmp_path, registry)
    for version in ("1.0.0", "1.0.0", "1.1.0", "1.0.0"):
        assert upload(client, image_bytes(), version=version).status_code == 200
    assert len(loads) == 2


# --- Validación de la entrada --------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "filename"),
    [
        (b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj\n", "foto.pdf"),
        (image_bytes(fmt="GIF"), "foto.gif"),
        (b"no soy una imagen", "foto.png"),
    ],
    ids=["pdf", "gif", "text-named-png"],
)
def test_only_png_and_jpeg_are_accepted(tmp_path, registry, data, filename):
    response = upload(client_for(tmp_path, registry), data, filename=filename)

    assert response.status_code == 415
    assert error_code(response) == "unsupported_image_type"
    assert "PNG o JPEG" in response.json()["error"]["message"]


@pytest.mark.parametrize("data", [b"", image_bytes()[:120]], ids=["empty", "truncated-png"])
def test_unreadable_image_gets_a_useful_message(tmp_path, registry, data):
    response = upload(client_for(tmp_path, registry), data)

    assert response.status_code == 422
    assert error_code(response) == "invalid_image"


def test_file_over_the_size_limit_is_rejected(tmp_path, registry):
    data = striped(size=(400, 400))
    client = client_for(tmp_path, registry, max_upload_bytes=len(data) - 1)

    response = upload(client, data)

    assert response.status_code == 413
    assert error_code(response) == "image_too_large"


def test_image_with_too_many_pixels_is_rejected(tmp_path, registry):
    client = client_for(tmp_path, registry, max_image_pixels=100 * 100)

    response = upload(client, image_bytes(size=(101, 100)))

    assert response.status_code == 413
    assert error_code(response) == "image_too_large"


@pytest.mark.parametrize(
    "fields",
    [
        {"model_version": "1.0.0"},
        {"model_name": MODEL},
        {"model_name": MODEL, "model_version": "1"},
        {"model_name": MODEL, "model_version": "v1.0.0"},
    ],
    ids=["no-name", "no-version", "mlflow-integer", "v-prefix"],
)
def test_missing_or_invalid_model_fields(tmp_path, registry, fields):
    response = client_for(tmp_path, registry).post(
        "/inference/upload", data=fields, files={"file": ("foto.png", image_bytes(), "image/png")}
    )

    assert response.status_code == 422
    assert error_code(response) == "invalid_request"


def test_missing_file(tmp_path, registry):
    response = client_for(tmp_path, registry).post(
        "/inference/upload", data={"model_name": MODEL, "model_version": "1.0.0"}
    )

    assert response.status_code == 422
    assert error_code(response) == "invalid_request"


# --- Modelo inexistente o no servible: error controlado -----------------------------


@pytest.mark.parametrize(
    ("model_name", "version"), [(MODEL, "9.9.9"), ("otro-modelo", "1.0.0")], ids=["version", "name"]
)
def test_unknown_model_version(tmp_path, registry, model_name, version):
    response = client_for(tmp_path, registry).post(
        "/inference/upload",
        data={"model_name": model_name, "model_version": version},
        files={"file": ("foto.png", image_bytes(), "image/png")},
    )

    # 422 y no 404: el portal lee un 404 como "servicio no conectado".
    assert response.status_code == 422
    assert error_code(response) == "model_not_found"


def test_version_with_metadata_but_no_package(tmp_path, registry):
    registry.register("3.0.0", "d" * 32, None)

    response = upload(client_for(tmp_path, registry), image_bytes(), version="3.0.0")

    assert response.status_code == 409
    assert error_code(response) == "model_not_servable"


def test_package_not_pulled_says_how_to_get_it(tmp_path, registry):
    registry.checkpoint("1.0.0").unlink()

    response = upload(client_for(tmp_path, registry), image_bytes())

    assert response.status_code == 503
    assert error_code(response) == "model_package_missing"
    assert "dvc pull data/models.dvc" in response.json()["error"]["message"]


def test_altered_checkpoint_is_not_served(tmp_path, registry, checkpoints):
    registry.checkpoint("1.0.0").write_bytes(checkpoints["cat"].read_bytes())

    response = upload(client_for(tmp_path, registry), image_bytes())

    assert response.status_code == 409
    assert error_code(response) == "model_not_servable"


def test_checkpoint_that_does_not_match_the_registered_architecture(tmp_path, registry):
    entry = registry.packages[0]
    entry["architecture"]["image_size"] = 128
    entry["preprocessing"]["image_size"] = 128
    registry.save()

    response = upload(client_for(tmp_path, registry), image_bytes())

    assert response.status_code == 409
    assert error_code(response) == "model_not_servable"
    assert "image_size" in response.json()["error"]["message"]


def test_unreadable_registry(tmp_path, registry):
    registry.path.write_text("{no es json", encoding="utf-8")

    response = upload(client_for(tmp_path, registry), image_bytes())

    assert response.status_code == 503
    assert error_code(response) == "registry_unavailable"


def test_inference_not_configured(tmp_path):
    response = upload(client_for(tmp_path), image_bytes())

    assert response.status_code == 503
    assert error_code(response) == "inference_not_configured"


# --- Recorte del portal --------------------------------------------------------------


@pytest.fixture
def crops(tmp_path) -> tuple[Path, Path]:
    crops_dir = tmp_path / "crops"
    (crops_dir / "cat").mkdir(parents=True)
    png = striped()
    (crops_dir / "cat" / "img12-ann11.png").write_bytes(png)
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "crops.json").write_text(
        json.dumps(
            {
                "dataset_version": RELEASE,
                "crops": [
                    {
                        "crop_id": "img12-ann11",
                        "image_id": 12,
                        "annotation_id": 11,
                        "crop_path": "cat/img12-ann11.png",
                        "sha256": hashlib.sha256(png).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return crops_dir, reports


def classify_crop(
    client, *, version="2.0.0", dataset_version=RELEASE, image_id=12, annotation_id=11
):
    return client.post(
        "/inference",
        json={
            "schema_version": "1.0",
            "model_name": MODEL,
            "model_version": version,
            "crop": {
                "dataset_version": dataset_version,
                "image_id": image_id,
                "annotation_id": annotation_id,
            },
        },
    )


def test_crop_of_the_portal_is_classified(tmp_path, registry, crops, checkpoints):
    crops_dir, reports = crops
    client = client_for(tmp_path, registry, crops_dir=crops_dir, reports_dir=reports)

    response = classify_crop(client)

    assert response.status_code == 200, response.text
    result = InferenceResponse.model_validate(response.json())
    assert (result.source, result.upload) == ("crop", None)
    assert (result.crop.image_id, result.crop.annotation_id) == (12, 11)
    assert result.checkpoint_sha256 == sha256(checkpoints["real"])
    stripes = upload(client, striped(), version="2.0.0").json()["probabilities"]
    assert result.probabilities == pytest.approx(stripes, abs=1e-6)


@pytest.mark.parametrize(
    "selection",
    [{"image_id": 99, "annotation_id": 98}, {"dataset_version": "v0.1.0"}],
    ids=["unknown-crop", "crop-of-another-release"],
)
def test_crop_that_does_not_exist(tmp_path, registry, crops, selection):
    crops_dir, reports = crops
    client = client_for(tmp_path, registry, crops_dir=crops_dir, reports_dir=reports)

    response = classify_crop(client, **selection)

    assert response.status_code == 422
    assert error_code(response) == "crop_not_found"


def test_altered_crop_is_not_classified(tmp_path, registry, crops):
    crops_dir, reports = crops
    (crops_dir / "cat" / "img12-ann11.png").write_bytes(image_bytes())
    client = client_for(tmp_path, registry, crops_dir=crops_dir, reports_dir=reports)

    response = classify_crop(client)

    assert response.status_code == 422
    assert error_code(response) == "crop_not_available"


def test_crop_inference_without_crops_dir(tmp_path, registry, crops):
    _, reports = crops
    client = client_for(tmp_path, registry, reports_dir=reports)

    response = classify_crop(client)

    assert response.status_code == 503
    assert error_code(response) == "crops_not_configured"


@pytest.mark.parametrize(
    ("body", "status", "code"),
    [("{no es json", 400, "invalid_json"), ('{"schema_version": "1.0"}', 422, "invalid_request")],
    ids=["json", "contract"],
)
def test_bad_crop_request(tmp_path, registry, body, status, code):
    response = client_for(tmp_path, registry).post(
        "/inference", content=body, headers={"content-type": "application/json"}
    )

    assert response.status_code == status
    assert error_code(response) == code
