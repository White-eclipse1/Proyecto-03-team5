"""APP-07: inferencia dog/cat real en `ml-api` (`training/inference.py`).

TDD Requirement del issue #24: validación de la entrada (tipo, tamaño, archivo
ilegible) y esquema de la respuesta (`InferenceResponse`).

Agent Test: con la misma imagen, cambiar de model version cambia el checkpoint (y
su sha256) que se carga y la predicción; el nombre del archivo no influye.

El registry es un MLflow falso que sirve checkpoints reales (`save_checkpoint` de
ML-03): la inferencia carga pesos de verdad y aplica el preprocesamiento de
evaluación de ML-02.
"""

import hashlib
import io
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from PIL import Image
from sqlalchemy import create_engine
from starlette.testclient import TestClient

from classification.model import ModelConfig, build_model, predict_proba, save_checkpoint
from classification.transforms import eval_transform
from presentation.ml_contracts import ErrorResponse, InferenceResponse
from training.inference import MlflowRegistryResolver
from training.queue import TrainingJobQueue
from training.server import create_app

MODEL = "pet-classifier"
RELEASE = "v0.1.1"
MANIFEST = "sha256:178b28bddb1bef5c05975fc7150710658627e31af08a1a12d94b98f9e99cb2b2"
RUN_DOG = "a" * 32
RUN_CAT = "b" * 32
RUN_REAL = "c" * 32
IMAGE_SIZE = 64


def make_checkpoint(path: Path, *, bias: tuple[float, float] | None, seed: int = 0) -> Path:
    """ResNet18 sin pesos ImageNet. Con `bias`, la cabeza ignora los pixeles y siempre
    favorece la misma clase; sin `bias`, son pesos aleatorios (depende de la imagen)."""
    torch.manual_seed(seed)
    model = build_model(
        ModelConfig(image_size=IMAGE_SIZE, hidden_layers=[], dropout=0.0, pretrained=False),
        download_weights=False,
    )
    if bias is not None:
        with torch.no_grad():
            model.head[-1].weight.zero_()
            model.head[-1].bias.copy_(torch.tensor(bias))
    return save_checkpoint(model, path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeMlflow:
    """Lo que `MlflowRegistryResolver` usa de `MlflowClient`."""

    def __init__(self):
        self.versions: dict[tuple[str, str], SimpleNamespace] = {}
        self.files: dict[str, Path] = {}
        self.downloads = 0
        self.down = False

    def register(self, version: str, run_id: str, checkpoint: Path, **changes) -> None:
        self.files[run_id] = checkpoint
        entry = SimpleNamespace(
            name=MODEL,
            version=version,
            run_id=run_id,
            source=f"runs:/{run_id}/checkpoints/best.pt",
            status="READY",
            tags={},
        )
        for key, value in changes.items():
            setattr(entry, key, value)
        self.versions[(MODEL, version)] = entry

    def get_model_version(self, name: str, version: str):
        if self.down:
            raise ConnectionError("MLflow no responde")
        if (name, version) not in self.versions:
            raise MlflowException("no existe", error_code=RESOURCE_DOES_NOT_EXIST)
        return self.versions[(name, version)]

    def get_run(self, run_id: str):
        tags = {"dataset_version": RELEASE, "manifest_hash": MANIFEST}
        return SimpleNamespace(info=SimpleNamespace(run_id=run_id), data=SimpleNamespace(tags=tags))

    def download_artifacts(self, run_id: str, path: str, dst_path: str) -> str:
        assert path == "checkpoints/best.pt"
        self.downloads += 1
        target = Path(dst_path) / "best.pt"
        shutil.copyfile(self.files[run_id], target)
        return str(target)


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory) -> dict[str, Path]:
    folder = tmp_path_factory.mktemp("checkpoints")
    return {
        "dog": make_checkpoint(folder / "dog.pt", bias=(6.0, -6.0)),
        "cat": make_checkpoint(folder / "cat.pt", bias=(-6.0, 6.0)),
        "real": make_checkpoint(folder / "real.pt", bias=None, seed=3),
    }


@pytest.fixture
def mlflow(checkpoints) -> FakeMlflow:
    fake = FakeMlflow()
    fake.register("1", RUN_DOG, checkpoints["dog"])
    fake.register("2", RUN_CAT, checkpoints["cat"])
    fake.register("3", RUN_REAL, checkpoints["real"])
    return fake


def client_for(tmp_path, mlflow=None, *, crops_dir=None, reports_dir=None, **limits):
    queue = TrainingJobQueue(create_engine(f"sqlite:///{tmp_path / 'jobs.db'}"))
    queue.create_tables()
    reports = reports_dir or tmp_path / "reports"
    reports.mkdir(exist_ok=True)
    models = MlflowRegistryResolver(mlflow) if mlflow is not None else None
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


def upload(client, data: bytes, *, filename="foto.png", version="1", content_type="image/png"):
    return client.post(
        "/inference/upload",
        data={"model_name": MODEL, "model_version": version},
        files={"file": (filename, data, content_type)},
    )


def error_code(response) -> str:
    return ErrorResponse.model_validate(response.json()).error.code


# --- Imagen subida: esquema de la respuesta ------------------------------------------


def test_upload_returns_a_valid_inference_response(tmp_path, mlflow, checkpoints):
    data = image_bytes()
    response = upload(client_for(tmp_path, mlflow), data)

    assert response.status_code == 200, response.text
    result = InferenceResponse.model_validate(response.json())
    assert (result.source, result.crop) == ("upload", None)
    assert result.upload.sha256 == hashlib.sha256(data).hexdigest()
    assert (result.upload.width, result.upload.height) == (80, 60)
    assert (result.upload.content_type, result.upload.size_bytes) == ("image/png", len(data))
    assert (result.model_name, result.model_version) == (MODEL, "1")
    assert result.run_id == RUN_DOG
    assert result.checkpoint == f"runs:/{RUN_DOG}/checkpoints/best.pt"
    assert result.checkpoint_sha256 == sha256(checkpoints["dog"])
    assert (result.dataset_version, result.image_size) == (RELEASE, IMAGE_SIZE)
    assert set(result.probabilities) == {"dog", "cat"}
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    assert result.predicted_class == "dog"


def test_jpeg_is_detected_by_content_not_by_name(tmp_path, mlflow):
    response = upload(
        client_for(tmp_path, mlflow),
        image_bytes(fmt="JPEG"),
        filename="foto.png",
        content_type="image/png",
    )

    assert response.status_code == 200, response.text
    assert response.json()["upload"]["content_type"] == "image/jpeg"


# --- Agent Test: la versión elegida decide el artefacto ------------------------------


def test_agent_test_changing_model_version_changes_the_loaded_checkpoint(
    tmp_path, mlflow, checkpoints
):
    client = client_for(tmp_path, mlflow)
    data = image_bytes()

    first = InferenceResponse.model_validate(upload(client, data, version="1").json())
    second = InferenceResponse.model_validate(upload(client, data, version="2").json())

    assert (first.run_id, first.checkpoint_sha256) == (RUN_DOG, sha256(checkpoints["dog"]))
    assert (second.run_id, second.checkpoint_sha256) == (RUN_CAT, sha256(checkpoints["cat"]))
    assert (first.predicted_class, second.predicted_class) == ("dog", "cat")


def test_the_file_name_does_not_decide_the_class(tmp_path, mlflow):
    client = client_for(tmp_path, mlflow)
    data = image_bytes()

    as_cat = upload(client, data, filename="cat.png").json()
    as_dog = upload(client, data, filename="dog.png").json()

    assert as_cat["predicted_class"] == "dog"  # la versión 1 siempre dice dog
    assert as_cat["probabilities"] == as_dog["probabilities"]


def test_the_prediction_comes_from_the_model_and_the_eval_preprocessing(
    tmp_path, mlflow, checkpoints
):
    from classification.model import load_checkpoint

    client = client_for(tmp_path, mlflow)
    plain, stripes = image_bytes(), striped()

    results = [
        upload(client, data, version="3").json()["probabilities"] for data in (plain, stripes)
    ]

    assert results[0] != results[1], "con pesos reales, imágenes distintas dan otra salida"
    model = load_checkpoint(checkpoints["real"])
    tensor = eval_transform(IMAGE_SIZE)(Image.open(io.BytesIO(stripes)).convert("RGB"))
    expected = predict_proba(model, tensor.unsqueeze(0))[0].tolist()
    assert [results[1]["dog"], results[1]["cat"]] == pytest.approx(expected, abs=1e-6)


def test_each_model_version_is_downloaded_once(tmp_path, mlflow):
    client = client_for(tmp_path, mlflow)
    for _ in range(3):
        assert upload(client, image_bytes()).status_code == 200
    assert mlflow.downloads == 1


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
def test_only_png_and_jpeg_are_accepted(tmp_path, mlflow, data, filename):
    response = upload(client_for(tmp_path, mlflow), data, filename=filename)

    assert response.status_code == 415
    assert error_code(response) == "unsupported_image_type"
    assert "PNG o JPEG" in response.json()["error"]["message"]


@pytest.mark.parametrize("data", [b"", image_bytes()[:120]], ids=["empty", "truncated-png"])
def test_unreadable_image_gets_a_useful_message(tmp_path, mlflow, data):
    response = upload(client_for(tmp_path, mlflow), data)

    assert response.status_code == 422
    assert error_code(response) == "invalid_image"


def test_file_over_the_size_limit_is_rejected(tmp_path, mlflow):
    data = striped(size=(400, 400))
    client = client_for(tmp_path, mlflow, max_upload_bytes=len(data) - 1)

    response = upload(client, data)

    assert response.status_code == 413
    assert error_code(response) == "image_too_large"


def test_image_with_too_many_pixels_is_rejected(tmp_path, mlflow):
    client = client_for(tmp_path, mlflow, max_image_pixels=100 * 100)

    response = upload(client, image_bytes(size=(101, 100)))

    assert response.status_code == 413
    assert error_code(response) == "image_too_large"


@pytest.mark.parametrize(
    "fields",
    [{"model_version": "1"}, {"model_name": MODEL}, {"model_name": MODEL, "model_version": "v1"}],
    ids=["no-name", "no-version", "bad-version"],
)
def test_missing_or_invalid_model_fields(tmp_path, mlflow, fields):
    response = client_for(tmp_path, mlflow).post(
        "/inference/upload", data=fields, files={"file": ("foto.png", image_bytes(), "image/png")}
    )

    assert response.status_code == 422
    assert error_code(response) == "invalid_request"


def test_missing_file(tmp_path, mlflow):
    response = client_for(tmp_path, mlflow).post(
        "/inference/upload", data={"model_name": MODEL, "model_version": "1"}
    )

    assert response.status_code == 422
    assert error_code(response) == "invalid_request"


# --- Modelo inexistente o no servible: error controlado -----------------------------


def test_unknown_model_version(tmp_path, mlflow):
    response = upload(client_for(tmp_path, mlflow), image_bytes(), version="9")

    # 422 y no 404: el portal lee un 404 como "servicio no conectado".
    assert response.status_code == 422
    assert error_code(response) == "model_not_found"


def test_model_version_that_is_not_ready(tmp_path, mlflow, checkpoints):
    mlflow.register("4", "d" * 32, checkpoints["dog"], status="PENDING_REGISTRATION")

    response = upload(client_for(tmp_path, mlflow), image_bytes(), version="4")

    assert response.status_code == 409
    assert error_code(response) == "model_not_ready"


@pytest.mark.parametrize(
    "changes",
    [
        {"tags": {"checkpoint_sha256": "0" * 64}},
        {"source": f"runs:/{'e' * 32}/checkpoints/best.pt"},
        {"source": "models:/m-123"},
    ],
    ids=["other-sha256", "checkpoint-of-another-run", "not-a-run-checkpoint"],
)
def test_model_version_whose_artifact_cannot_be_trusted(tmp_path, mlflow, checkpoints, changes):
    mlflow.register("5", "d" * 32, checkpoints["dog"], **changes)

    response = upload(client_for(tmp_path, mlflow), image_bytes(), version="5")

    assert response.status_code == 409
    assert error_code(response) == "model_not_servable"


def test_registry_down_is_retryable(tmp_path, mlflow):
    mlflow.down = True

    response = upload(client_for(tmp_path, mlflow), image_bytes())

    assert response.status_code == 503
    body = ErrorResponse.model_validate(response.json())
    assert (body.error.code, body.error.retryable) == ("registry_unavailable", True)


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


def classify_crop(client, *, version="3", dataset_version=RELEASE, image_id=12, annotation_id=11):
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


def test_crop_of_the_portal_is_classified(tmp_path, mlflow, crops, checkpoints):
    crops_dir, reports = crops
    client = client_for(tmp_path, mlflow, crops_dir=crops_dir, reports_dir=reports)

    response = classify_crop(client)

    assert response.status_code == 200, response.text
    result = InferenceResponse.model_validate(response.json())
    assert (result.source, result.upload) == ("crop", None)
    assert (result.crop.image_id, result.crop.annotation_id) == (12, 11)
    assert result.checkpoint_sha256 == sha256(checkpoints["real"])
    stripes = upload(client, striped(), version="3").json()["probabilities"]
    assert result.probabilities == pytest.approx(stripes, abs=1e-6)


@pytest.mark.parametrize(
    "selection",
    [{"image_id": 99, "annotation_id": 98}, {"dataset_version": "v0.1.0"}],
    ids=["unknown-crop", "crop-of-another-release"],
)
def test_crop_that_does_not_exist(tmp_path, mlflow, crops, selection):
    crops_dir, reports = crops
    client = client_for(tmp_path, mlflow, crops_dir=crops_dir, reports_dir=reports)

    response = classify_crop(client, **selection)

    assert response.status_code == 422
    assert error_code(response) == "crop_not_found"


def test_altered_crop_is_not_classified(tmp_path, mlflow, crops):
    crops_dir, reports = crops
    (crops_dir / "cat" / "img12-ann11.png").write_bytes(image_bytes())
    client = client_for(tmp_path, mlflow, crops_dir=crops_dir, reports_dir=reports)

    response = classify_crop(client)

    assert response.status_code == 422
    assert error_code(response) == "crop_not_available"


def test_crop_inference_without_crops_dir(tmp_path, mlflow, crops):
    _, reports = crops
    client = client_for(tmp_path, mlflow, reports_dir=reports)

    response = classify_crop(client)

    assert response.status_code == 503
    assert error_code(response) == "crops_not_configured"


@pytest.mark.parametrize(
    ("body", "status", "code"),
    [("{no es json", 400, "invalid_json"), ('{"schema_version": "1.0"}', 422, "invalid_request")],
    ids=["json", "contract"],
)
def test_bad_crop_request(tmp_path, mlflow, body, status, code):
    response = client_for(tmp_path, mlflow).post(
        "/inference", content=body, headers={"content-type": "application/json"}
    )

    assert response.status_code == status
    assert error_code(response) == code
