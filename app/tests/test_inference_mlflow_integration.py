"""APP-07 (integración): inferencia con versiones reales del Model Registry de MLflow.

Necesita el stack (`docker compose up -d --wait mariadb minio mlflow`) y
`MLFLOW_INTEGRATION=1`; sin esa variable se omite. En CI la corre el job
"MLflow persistente (OPS-03)".

Agent Test del issue #24: dos versiones del mismo modelo con checkpoints distintos;
cambiar de versión cambia el checkpoint (y su sha256) que carga `ml-api`.
"""

import hashlib
import io
import os
from uuid import uuid4

import pytest
import torch
from PIL import Image
from sqlalchemy import create_engine
from starlette.testclient import TestClient

from classification.model import ModelConfig, build_model, save_checkpoint
from classification.training import CHECKPOINT_ARTIFACT
from presentation.ml_contracts import InferenceResponse
from tracking.client import tracking_client
from tracking.run_schema import TAG_DATASET_VERSION, TAG_MANIFEST_HASH
from training.inference import CHECKPOINT_SHA256_TAG, MlflowRegistryResolver
from training.queue import TrainingJobQueue
from training.server import create_app

pytestmark = pytest.mark.skipif(
    os.environ.get("MLFLOW_INTEGRATION") != "1",
    reason="Prueba de integración: requiere MLFLOW_INTEGRATION=1 y el stack de Docker",
)

EXPERIMENT = "app-07-inference-check"
RELEASE = "v0.1.1"
MANIFEST = "sha256:178b28bddb1bef5c05975fc7150710658627e31af08a1a12d94b98f9e99cb2b2"


def _checkpoint(path, bias):
    torch.manual_seed(0)
    model = build_model(
        ModelConfig(image_size=64, hidden_layers=[], dropout=0.0, pretrained=False),
        download_weights=False,
    )
    with torch.no_grad():
        model.head[-1].weight.zero_()
        model.head[-1].bias.copy_(torch.tensor(bias))
    return save_checkpoint(model, path)


@pytest.fixture
def mlflow_client():
    return tracking_client()


@pytest.fixture
def registered(tmp_path, mlflow_client):
    """Dos versiones de un modelo nuevo: v1 siempre dog, v2 siempre cat."""
    experiment = mlflow_client.get_experiment_by_name(EXPERIMENT)
    experiment_id = (
        experiment.experiment_id if experiment else mlflow_client.create_experiment(EXPERIMENT)
    )
    name = f"app07-{uuid4().hex[:8]}"
    mlflow_client.create_registered_model(name)
    digests = {}
    for bias, label in (((6.0, -6.0), "dog"), ((-6.0, 6.0), "cat")):
        path = _checkpoint(tmp_path / label / "best.pt", bias)
        run = mlflow_client.create_run(
            experiment_id, tags={TAG_DATASET_VERSION: RELEASE, TAG_MANIFEST_HASH: MANIFEST}
        )
        run_id = run.info.run_id
        mlflow_client.log_artifact(run_id, str(path), CHECKPOINT_ARTIFACT.rsplit("/", 1)[0])
        mlflow_client.set_terminated(run_id)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        version = mlflow_client.create_model_version(
            name,
            f"runs:/{run_id}/{CHECKPOINT_ARTIFACT}",
            run_id=run_id,
            tags={CHECKPOINT_SHA256_TAG: digest},
        )
        digests[label] = (version.version, run_id, digest)
    yield name, digests
    mlflow_client.delete_registered_model(name)


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (80, 60), (180, 90, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_agent_test_switching_registry_versions_switches_the_checkpoint(
    tmp_path, mlflow_client, registered
):
    name, digests = registered
    queue = TrainingJobQueue(create_engine(f"sqlite:///{tmp_path / 'jobs.db'}"))
    queue.create_tables()
    reports = tmp_path / "reports"
    reports.mkdir()
    api = TestClient(
        create_app(queue=queue, reports_dir=reports, models=MlflowRegistryResolver(mlflow_client))
    )

    results = {}
    for label, (version, run_id, digest) in digests.items():
        response = api.post(
            "/inference/upload",
            data={"model_name": name, "model_version": version},
            files={"file": ("foto.png", _png(), "image/png")},
        )
        assert response.status_code == 200, response.text
        result = InferenceResponse.model_validate(response.json())
        assert (result.run_id, result.checkpoint_sha256) == (run_id, digest)
        assert result.checkpoint == f"runs:/{run_id}/{CHECKPOINT_ARTIFACT}"
        assert result.dataset_version == RELEASE
        results[label] = result.predicted_class

    assert results == {"dog": "dog", "cat": "cat"}
