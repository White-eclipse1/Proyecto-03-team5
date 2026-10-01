"""APP-04 (integración): la API de Experiments lee un servidor MLflow real.

Necesita el stack (`docker compose up -d --wait mariadb minio mlflow`) y
`MLFLOW_INTEGRATION=1`; sin esa variable se omite. En CI la corre el job
"MLflow persistente (OPS-03)".

Agent Test del issue #16: modificar un tag y un valor de un run en MLflow y
comprobar que la API (y por lo tanto la UI) refleja el cambio.
"""

import os
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from starlette.testclient import TestClient

from presentation.ml_contracts import RunCurvesResponse, RunsResponse
from tracking.client import tracking_client
from tracking.run_schema import (
    CURVE_METRICS,
    TAG_DATASET_VERSION,
    TAG_GIT_COMMIT,
    TAG_MANIFEST_HASH,
    TAG_TRAINING_JOB,
)
from training.queue import TrainingJobQueue
from training.server import create_app

pytestmark = pytest.mark.skipif(
    os.environ.get("MLFLOW_INTEGRATION") != "1",
    reason="Prueba de integración: requiere MLFLOW_INTEGRATION=1 y el stack de Docker",
)

EXPERIMENT = "app-04-experiments-check"
MANIFEST = "sha256:178b28bddb1bef5c05975fc7150710658627e31af08a1a12d94b98f9e99cb2b2"
COMMIT = "9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c4b3a2f1e0d"
PARAMS = {
    "optimizer": "adamw",
    "batch_size": "64",
    "max_epochs": "3",
    "learning_rate": "0.0005",
    "image_size": "224",
    "hidden_layers": "512,256",
    "dropout": "0.2",
    "seed": "7",
    "patience": "2",
    "min_delta": "0.001",
}
HISTORY = {
    "train_loss": [0.64, 0.42, 0.31],
    "val_loss": [0.66, 0.45, 0.37],
    "train_accuracy": [0.63, 0.80, 0.87],
    "val_accuracy": [0.62, 0.78, 0.85],
}


@pytest.fixture
def mlflow_client():
    return tracking_client()


@pytest.fixture
def api(tmp_path, mlflow_client) -> TestClient:
    reports = tmp_path / "reports"
    reports.mkdir()
    queue = TrainingJobQueue(create_engine(f"sqlite:///{tmp_path / 'jobs.db'}"))
    queue.create_tables()
    return TestClient(create_app(queue=queue, reports_dir=reports, tracking=mlflow_client))


@pytest.fixture
def training_run(mlflow_client):
    experiment = mlflow_client.get_experiment_by_name(EXPERIMENT)
    experiment_id = (
        experiment.experiment_id if experiment else mlflow_client.create_experiment(EXPERIMENT)
    )
    run = mlflow_client.create_run(
        experiment_id,
        run_name=f"resnet18-adamw-{uuid4().hex[:6]}",
        tags={
            TAG_DATASET_VERSION: "v0.1.1",
            TAG_MANIFEST_HASH: MANIFEST,
            TAG_GIT_COMMIT: COMMIT,
            TAG_TRAINING_JOB: "job-integration",
        },
    )
    run_id = run.info.run_id
    for key, value in PARAMS.items():
        mlflow_client.log_param(run_id, key, value)
    for name in CURVE_METRICS:
        for epoch, value in enumerate(HISTORY[name], start=1):
            mlflow_client.log_metric(run_id, name, value, step=epoch)
    mlflow_client.set_terminated(run_id, status="FINISHED")
    yield run_id
    mlflow_client.delete_run(run_id)


def find(api: TestClient, run_id: str):
    response = api.get("/runs")
    assert response.status_code == 200, response.text
    runs = {run.run_id: run for run in RunsResponse.model_validate(response.json()).runs}
    return runs.get(run_id)


def test_real_run_is_listed_with_provenance_params_and_metrics(api, training_run):
    run = find(api, training_run)

    assert run is not None
    assert run.status == "FINISHED" and run.end_time is not None
    assert (run.dataset_version, run.manifest_hash, run.git_commit) == ("v0.1.1", MANIFEST, COMMIT)
    assert run.params == PARAMS
    assert run.metrics == {name: values[-1] for name, values in HISTORY.items()}


def test_real_curves_match_the_logged_history(api, training_run):
    response = api.get(f"/runs/{training_run}/curves")

    assert response.status_code == 200, response.text
    curves = RunCurvesResponse.model_validate(response.json())
    assert {
        name: [(p.step, p.value) for p in points] for name, points in curves.curves.items()
    } == {name: list(enumerate(values, start=1)) for name, values in HISTORY.items()}


def test_changing_the_run_in_mlflow_is_reflected_by_the_api(api, mlflow_client, training_run):
    """Agent Test del issue #16."""
    mlflow_client.set_tag(training_run, TAG_DATASET_VERSION, "v0.1.2")
    mlflow_client.log_metric(training_run, "val_accuracy", 0.88, step=4)

    run = find(api, training_run)
    curves = RunCurvesResponse.model_validate(api.get(f"/runs/{training_run}/curves").json())

    assert run.dataset_version == "v0.1.2"
    assert run.metrics["val_accuracy"] == 0.88
    assert curves.curves["val_accuracy"][-1].step == 4


def test_deleted_run_disappears(api, mlflow_client, training_run):
    mlflow_client.delete_run(training_run)
    mlflow_client.restore_run(training_run)
    assert find(api, training_run) is not None
    mlflow_client.delete_run(training_run)
    assert find(api, training_run) is None
    mlflow_client.restore_run(training_run)  # el fixture lo borra al final
