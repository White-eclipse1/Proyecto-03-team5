"""APP-10: recorrido completo desde el portal con IDs y datos reales (issue #52).

Hace, en orden, lo que hace una persona en el portal, y lee lo mismo que cada pantalla:
los reportes estáticos (`/reports/...`) y las APIs (`/api/ml/...`, `/api/images/...`)
detrás de nginx. Guarda cada ID y comprueba que la cadena encaja:

    release DVC → manifest hash → training job → run de MLflow → candidato →
    model version → inferencia → registro en la cola de anotación

Necesita el stack completo con los datos reales (`dvc pull crops`, `data/models.dvc` y
`data/mlflow-snapshot.dvc` restaurado con `tracking.snapshot restore`). Sin
`APP10_PORTAL_URL` se omite. Con `APP10_TRACE_OUT` deja la cadena en un JSON.

    APP10_PORTAL_URL=http://localhost:8080 uv run pytest tests/test_app10_portal_smoke.py -v
"""

import json
import os
import time
from pathlib import Path

import pytest
import requests

from presentation.ml_contracts import (
    EvaluationOverview,
    InferenceResponse,
    ModelsResponse,
    ReleaseProvenance,
    RunsResponse,
    TrainingJob,
    TrainingJobsResponse,
    TrainingManifest,
)

PORTAL = os.environ.get("APP10_PORTAL_URL", "").rstrip("/")
pytestmark = pytest.mark.skipif(not PORTAL, reason="Requiere el stack: APP10_PORTAL_URL")

SHORT_PARAMS = {
    "optimizer": "adam",
    "batch_size": 128,
    "max_epochs": 1,
    "learning_rate": 0.001,
    "image_size": 32,
    "hidden_layers": [],
    "dropout": 0.0,
    "seed": 7,
    "patience": 1,
    "min_delta": 0.0,
}


def get(path: str) -> requests.Response:
    response = requests.get(PORTAL + path, timeout=60)
    assert response.status_code == 200, f"GET {path}: {response.status_code} {response.text[:200]}"
    return response


def post(path: str, **kwargs) -> requests.Response:
    return requests.post(PORTAL + path, timeout=120, **kwargs)


def test_full_flow_from_the_portal_with_traceable_ids():
    trace: dict[str, object] = {}

    # 1. Training: el release P2 aprobado, su procedencia DVC y su manifiesto.
    versions = get("/reports/versions.json").json()["releases"]
    approved = [
        release["dataset_version"]
        for release in versions
        if get(f"/reports/{release['quality_file']}").json()["status"] != "failed"
    ]
    assert approved, "no hay ningún release aprobado"
    release = approved[-1]
    provenance = ReleaseProvenance.model_validate(
        get(f"/reports/releases/{release}/provenance.json").json()
    )
    manifest = TrainingManifest.model_validate(
        get(f"/reports/releases/{release}/manifest.json").json()
    )
    dvc = {output.path: output.md5 for output in provenance.dvc_outputs}
    assert manifest.dataset_version == release
    assert manifest.provenance.annotations_dvc_hash == dvc["annotations"]
    assert manifest.provenance.images_dvc_hash == dvc["images"]
    trace["release"] = {"dataset_version": release, "dvc": dvc}
    trace["manifest_hash"] = manifest.manifest_hash

    # 2. Lanzar un training corto; el job queda persistido y el worker lo corre.
    response = post(
        "/api/ml/training/jobs",
        json={
            "schema_version": "1.0",
            "dataset_version": release,
            "manifest_hash": manifest.manifest_hash,
            "params": SHORT_PARAMS,
        },
    )
    assert response.status_code == 202, response.text
    job = TrainingJob.model_validate(response.json())
    deadline = time.monotonic() + 900
    while job.status not in {"succeeded", "failed"} and time.monotonic() < deadline:
        time.sleep(2)
        job = TrainingJob.model_validate(get(f"/api/ml/training/jobs/{job.job_id}").json())
    assert job.status == "succeeded", job.error
    listed = TrainingJobsResponse.model_validate(get("/api/ml/training/jobs").json()).jobs
    assert job.job_id in {entry.job_id for entry in listed}
    assert (job.dataset_version, job.manifest_hash) == (release, manifest.manifest_hash)
    trace["training_job"] = {"job_id": job.job_id, "run_id": job.run_id}

    # 3. Experiments: el run del job y el del candidato, con el mismo release y manifiesto.
    runs = {run.run_id: run for run in RunsResponse.model_validate(get("/api/ml/runs").json()).runs}
    job_run = runs[job.run_id]
    assert (job_run.dataset_version, job_run.manifest_hash) == (release, manifest.manifest_hash)

    # 4. Evaluation: el candidato congelado, visible, y su evaluación final de test.
    overview = EvaluationOverview.model_validate(get("/api/ml/evaluation").json())
    assert overview.state == "evaluated"
    candidate, evaluation = overview.candidate, overview.evaluation
    assert candidate.run_id in runs, "el run del candidato aparece en Experiments"
    assert runs[candidate.run_id].manifest_hash == manifest.manifest_hash
    assert (candidate.dataset_version, candidate.manifest_hash) == (
        release,
        manifest.manifest_hash,
    )
    assert sum(map(sum, evaluation.confusion_matrix)) == len(evaluation.predictions)
    trace["candidate"] = {
        "run_id": candidate.run_id,
        "checkpoint": candidate.checkpoint,
        "checkpoint_sha256": candidate.checkpoint_sha256,
        "test_accuracy": evaluation.metrics.accuracy_top1,
    }

    # 5. Models: la versión real del candidato, publicada en S3 y lista para usar.
    models = ModelsResponse.model_validate(get("/api/ml/models").json()).models
    model = next(m for m in models if m.run_id == candidate.run_id)
    assert model.checkpoint_sha256 == candidate.checkpoint_sha256
    assert (model.dataset_version, model.manifest_hash) == (release, manifest.manifest_hash)
    assert model.publication.status == "published" and model.servable
    trace["model_version"] = {
        "model_name": model.model_name,
        "model_version": model.model_version,
        "s3": [obj.key for obj in model.publication.objects],
    }

    # 6. Inference: la versión elegida en Models clasifica un recorte real del test.
    sample = evaluation.predictions[0]
    response = post(
        "/api/ml/inference",
        json={
            "schema_version": "1.0",
            "model_name": model.model_name,
            "model_version": model.model_version,
            "crop": {
                "dataset_version": release,
                "image_id": sample.image_id,
                "annotation_id": sample.annotation_id,
            },
        },
    )
    assert response.status_code == 200, response.text
    inference = InferenceResponse.model_validate(response.json())
    assert (inference.run_id, inference.checkpoint_sha256) == (
        candidate.run_id,
        candidate.checkpoint_sha256,
    )
    assert inference.predicted_class == sample.predicted_class
    assert inference.probabilities == pytest.approx(sample.probabilities, abs=1e-5)
    trace["inference"] = {
        "request_id": inference.request_id,
        "crop": f"img{sample.image_id}-ann{sample.annotation_id}",
        "predicted_class": inference.predicted_class,
        "probabilities": inference.probabilities,
    }

    # 7. Enviar a la cola de anotación: crea un registro real con la trazabilidad.
    image = get(f"/api/ml/crops/img{sample.image_id}-ann{sample.annotation_id}").content
    metadata = {
        "sourceKind": "crop",
        "sourceRef": f"{release}:img{sample.image_id}-ann{sample.annotation_id}",
        "modelName": inference.model_name,
        "modelVersion": inference.model_version,
        "runId": inference.run_id,
        "checkpointSha256": inference.checkpoint_sha256,
        "predictedClass": inference.predicted_class,
        "probabilities": inference.probabilities,
    }
    response = post(
        "/api/images/from-inference",
        files={"image": ("crop.png", image, "image/png")},
        data={"metadata": json.dumps(metadata)},
    )
    assert response.status_code in (200, 201), response.text
    image_id = response.json()["imageId"]
    assert get(f"/api/images/{image_id}/file").content == image
    trace["annotation_queue"] = {"image_id": image_id, **response.json()}

    out = os.environ.get("APP10_TRACE_OUT")
    if out:
        Path(out).write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
