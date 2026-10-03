"""APP-09: casos adversariales contra el portal levantado, como los probaría el evaluador.

Pasa por nginx (`/api/ml/...`), igual que el navegador, con el stack completo y datos
reales (release v0.1.1, recortes de ML-01, candidato de ML-08, evaluación de ML-09,
paquete 1.0.0 de OPS-06 y publicación de OPS-07). Se omite sin `APP09_PORTAL_URL`.

    APP09_PORTAL_URL=http://localhost:8080 MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \\
    APP09_AWS_PROFILE=mlops-p2 uv run pytest tests/test_app09_portal_adversarial.py -v

`APP09_AWS_PROFILE` es opcional: con él, cada objeto que Models marca como publicado
se busca en el S3 real (head-object y ChecksumSHA256). La evidencia está en
`tests/evidence/app-09-adversarial-verification.md`.
"""

import base64
import io
import json
import os
import time
from pathlib import Path

import pytest
import requests
from PIL import Image

from presentation.ml_contracts import (
    ErrorResponse,
    EvaluationOverview,
    InferenceResponse,
    ModelsResponse,
    RunCurvesResponse,
    RunsResponse,
    TrainingJob,
    TrainingJobsResponse,
)

ROOT = Path(__file__).resolve().parents[2]
PORTAL = os.environ.get("APP09_PORTAL_URL", "")
pytestmark = pytest.mark.skipif(not PORTAL, reason="Requiere el stack: APP09_PORTAL_URL")

RELEASE = "v0.1.1"
MANIFEST = json.loads(
    (ROOT / "reports" / "releases" / RELEASE / "manifest.json").read_text("utf-8")
)
SHORT_PARAMS = {
    "optimizer": "adam",
    "batch_size": 128,
    "max_epochs": 1,
    "learning_rate": 0.001,
    "image_size": 32,
    "hidden_layers": [],
    "dropout": 0.0,
    "seed": 42,
    "patience": 1,
    "min_delta": 0.0,
}


class Portal:
    """`/api/ml` del portal (nginx), como lo llama el navegador."""

    def __init__(self, base: str):
        self.base = f"{base.rstrip('/')}/api/ml"
        self.session = requests.Session()

    def get(self, path: str) -> requests.Response:
        return self.session.get(self.base + path, timeout=60)

    def post(self, path: str, **kwargs) -> requests.Response:
        return self.session.post(self.base + path, timeout=60, **kwargs)


@pytest.fixture(scope="module")
def api():
    portal = Portal(PORTAL)
    yield portal
    portal.session.close()


def job_request(**changes) -> dict:
    request = {
        "schema_version": "1.0",
        "dataset_version": RELEASE,
        "manifest_hash": MANIFEST["manifest_hash"],
        "params": dict(SHORT_PARAMS),
    }
    request.update(changes)
    return request


def error(response: requests.Response) -> str:
    return ErrorResponse.model_validate(response.json()).error.code


def job_ids(api) -> set[str]:
    response = TrainingJobsResponse.model_validate(api.get("/training/jobs").json())
    return {job.job_id for job in response.jobs}


# --- Training ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "status", "code"),
    [
        ({"params": {**SHORT_PARAMS, "batch_size": 0}}, 422, "invalid_request"),
        ({"params": {**SHORT_PARAMS, "image_size": 100}}, 422, "invalid_request"),
        ({"params": {**SHORT_PARAMS, "patience": 5}}, 422, "invalid_request"),
        ({"params": {**SHORT_PARAMS, "hidden_layers": [8] * 6}}, 422, "invalid_request"),
        ({"params": {**SHORT_PARAMS, "learning_rate": 0}}, 422, "invalid_request"),
        ({"dataset_version": "v9.9.9"}, 422, "release_not_found"),
        ({"dataset_version": "v0.1.0"}, 409, "training_blocked"),
        ({"manifest_hash": "sha256:" + "0" * 64}, 409, "training_blocked"),
    ],
    ids=[
        "batch-0",
        "image-size-100",
        "patience-above-epochs",
        "six-layers",
        "lr-0",
        "unknown-release",
        "gate-failed",
        "other-manifest",
    ],
)
def test_training_rejects_invalid_requests_without_creating_jobs(api, changes, status, code):
    before = job_ids(api)

    response = api.post("/training/jobs", json=job_request(**changes))

    assert (response.status_code, error(response)) == (status, code)
    assert job_ids(api) == before


@pytest.fixture(scope="module")
def trained_job(api) -> TrainingJob:
    """Un training corto real: la cola lo persiste y el worker lo corre fuera del request."""
    started = time.monotonic()
    response = api.post("/training/jobs", json=job_request())
    assert response.status_code == 202, response.text
    assert time.monotonic() - started < 5, "el POST no debe esperar al entrenamiento"
    job = TrainingJob.model_validate(response.json())
    seen = {job.status}
    deadline = time.monotonic() + 900
    while job.status not in {"succeeded", "failed"} and time.monotonic() < deadline:
        time.sleep(2)
        job = TrainingJob.model_validate(api.get(f"/training/jobs/{job.job_id}").json())
        seen.add(job.status)
    assert job.status == "succeeded", job.error
    job.__dict__["_seen"] = seen
    return job


def test_training_job_persists_and_is_found_again_like_after_a_refresh(api, trained_job):
    again = TrainingJob.model_validate(api.get(f"/training/jobs/{trained_job.job_id}").json())

    assert again == trained_job
    assert trained_job.job_id in job_ids(api)
    assert "running" in trained_job.__dict__["_seen"] or trained_job.progress is not None
    logs = api.get(f"/training/jobs/{trained_job.job_id}/logs").json()["entries"]
    assert logs, "el job deja logs"
    assert trained_job.checkpoint == f"runs:/{trained_job.run_id}/checkpoints/best.pt"


# --- Experiments ---------------------------------------------------------------------


def test_experiments_shows_the_real_run_and_reflects_changes_in_mlflow(api, trained_job):
    from tracking.client import tracking_client

    runs = {run.run_id: run for run in RunsResponse.model_validate(api.get("/runs").json()).runs}
    run = runs[trained_job.run_id]
    assert (run.dataset_version, run.manifest_hash) == (RELEASE, MANIFEST["manifest_hash"])
    curves = RunCurvesResponse.model_validate(api.get(f"/runs/{run.run_id}/curves").json())
    assert curves.curves["val_loss"], "la curva de val_loss viene de MLflow"

    tracking_client().log_metric(run.run_id, "val_accuracy_top1", 0.4242)

    changed = {r.run_id: r for r in RunsResponse.model_validate(api.get("/runs").json()).runs}
    assert changed[run.run_id].metrics["val_accuracy_top1"] == pytest.approx(0.4242)


# --- Evaluation ----------------------------------------------------------------------


def test_evaluation_shows_the_frozen_candidate_and_a_matrix_that_adds_up(api):
    overview = EvaluationOverview.model_validate(api.get("/evaluation").json())
    candidate = json.loads(
        (ROOT / "reports" / "candidates" / "ml08_candidate.json").read_text("utf-8")
    )

    assert overview.state == "evaluated" and overview.problem is None
    assert overview.candidate.run_id == candidate["run_id"]
    evaluation = overview.evaluation
    assert evaluation.split == "test"
    total = sum(map(sum, evaluation.confusion_matrix))
    test_crops = sum(1 for record in MANIFEST["records"] if record["split"] == "test")
    assert total == len(evaluation.predictions) == test_crops
    csv = api.get("/evaluation/predictions.csv")
    assert csv.status_code == 200 and csv.text.strip().count("\n") == total


# --- Models --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def models(api) -> ModelsResponse:
    return ModelsResponse.model_validate(api.get("/models").json())


def test_models_lists_the_real_registered_version(models):
    registry = json.loads((ROOT / "reports" / "models" / "registry.json").read_text("utf-8"))
    listed = {model.model_version: model for model in models.models}

    for entry in registry["models"]:
        model = listed[entry["model_version"]]
        assert (model.run_id, model.checkpoint_sha256) == (
            entry["run_id"],
            entry["checkpoint_sha256"],
        )
        assert model.servable, "el stack monta el paquete de data/models"


def test_published_objects_really_exist_in_s3(models):
    profile = os.environ.get("APP09_AWS_PROFILE")
    if not profile:
        pytest.skip("Sin APP09_AWS_PROFILE no se consulta el S3 real")
    import boto3

    s3 = boto3.Session(profile_name=profile).client("s3")
    published = [m for m in models.models if m.publication.status == "published"]
    assert published
    for model in published:
        for obj in model.publication.objects:
            head = s3.head_object(
                Bucket=model.publication.bucket, Key=obj.key, ChecksumMode="ENABLED"
            )
            checksum = base64.b64decode(head["ChecksumSHA256"]).hex()
            assert checksum == obj.sha256, obj.key


def test_downloads_are_only_the_verified_package_files(api, models):
    model = models.models[0]
    card = api.get(f"/models/{model.model_version}/files/model-card.md")
    assert card.status_code == 200 and model.model_version in card.text
    for name in ("package.json", "..%2F..%2F..%2Freports%2Fmodels%2Fregistry.json"):
        response = api.get(f"/models/{model.model_version}/files/{name}")
        assert response.status_code == 404


# --- Inference -----------------------------------------------------------------------


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (120, 90, 60)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("filename", "data", "status", "code"),
    [
        ("foto.pdf", b"%PDF-1.7\n1 0 obj\n", 415, "unsupported_image_type"),
        ("foto.png", b"no soy una imagen", 415, "unsupported_image_type"),
        ("foto.png", b"", 422, "invalid_image"),
        ("foto.png", _png()[:60], 422, "invalid_image"),
    ],
    ids=["pdf", "text-named-png", "empty", "truncated"],
)
def test_inference_rejects_invalid_files_with_a_useful_message(
    api, models, filename, data, status, code
):
    model = models.models[0]

    response = api.post(
        "/inference/upload",
        data={"model_name": model.model_name, "model_version": model.model_version},
        files={"file": (filename, data, "image/png")},
    )

    assert (response.status_code, error(response)) == (status, code)
    assert response.json()["error"]["message"]


def test_inference_accepts_a_valid_image_of_several_megabytes(api, models):
    """Bug de APP-09: nginx respondía 413 en HTML a imágenes válidas de más de 1 MB."""
    image = Image.frombytes("RGB", (1100, 1100), os.urandom(1100 * 1100 * 3))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    assert 3 * 1024 * 1024 < len(buffer.getvalue()) < 10 * 1024 * 1024
    model = models.models[0]

    response = api.post(
        "/inference/upload",
        data={"model_name": model.model_name, "model_version": model.model_version},
        files={"file": ("grande.png", buffer.getvalue(), "image/png")},
    )

    assert response.status_code == 200, response.text[:200]
    assert InferenceResponse.model_validate(response.json()).upload.size_bytes == len(
        buffer.getvalue()
    )


def test_inference_of_an_unknown_version_is_a_controlled_error(api, models):
    response = api.post(
        "/inference/upload",
        data={"model_name": models.models[0].model_name, "model_version": "9.9.9"},
        files={"file": ("foto.png", _png(), "image/png")},
    )

    assert (response.status_code, error(response)) == (422, "model_not_found")


def test_inference_uses_the_chosen_version_and_reproduces_ml09(api, models):
    model = models.models[0]
    overview = EvaluationOverview.model_validate(api.get("/evaluation").json())
    expected = overview.evaluation.predictions[0]

    response = api.post(
        "/inference",
        json={
            "schema_version": "1.0",
            "model_name": model.model_name,
            "model_version": model.model_version,
            "crop": {
                "dataset_version": overview.evaluation.dataset_version,
                "image_id": expected.image_id,
                "annotation_id": expected.annotation_id,
            },
        },
    )

    assert response.status_code == 200, response.text
    result = InferenceResponse.model_validate(response.json())
    assert result.checkpoint_sha256 == model.checkpoint_sha256
    assert result.predicted_class == expected.predicted_class
    assert result.probabilities == pytest.approx(expected.probabilities, abs=1e-5)
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
