"""OPS-03 (prueba de integración): un run de MLflow sobrevive al reinicio del stack.

Necesita Docker y el stack levantado (`docker compose up -d --wait mariadb minio mlflow`).
Se activa con `MLFLOW_INTEGRATION=1`; sin esa variable se omite, para que `pytest -q`
siga corriendo sin Docker. En CI la ejecuta el job "MLflow persistente (OPS-03)".

Flujo (Agent Test del issue #8): crear un run con parámetros, métricas por época,
un checkpoint y una curva → recrear los contenedores de MariaDB, MinIO y MLflow
(los volúmenes se conservan) → recuperar el mismo run_id por API y comparar todo.
"""

import hashlib
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("MLFLOW_INTEGRATION") != "1",
    reason="Prueba de integración: requiere MLFLOW_INTEGRATION=1 y el stack de Docker",
)

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "ops-03-persistence-check"
PARAMS = {
    "optimizer": "adam",
    "batch_size": "32",
    "max_epochs": "3",
    "learning_rate": "0.001",
    "image_size": "224",
    "hidden_layers": "[256]",
    "dropout": "0.3",
}
EPOCH_METRICS = {
    "train_loss": [0.69, 0.52, 0.41],
    "val_loss": [0.71, 0.58, 0.55],
    "val_accuracy": [0.55, 0.71, 0.78],
}


def _tracking_uri() -> str:
    from storage.settings import TrackingSettings

    return TrackingSettings().mlflow_tracking_uri


def _wait_until_healthy(uri: str, timeout_s: float = 180) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            with urllib.request.urlopen(f"{uri}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        if time.monotonic() > deadline:
            pytest.fail(f"MLflow no respondió en {uri}/health tras {timeout_s:.0f} s")
        time.sleep(2)


def _recreate_stack() -> None:
    """Contenedores nuevos (sistema de archivos vacío); los volúmenes nombrados se conservan."""
    subprocess.run(
        [
            "docker",
            "compose",
            "up",
            "-d",
            "--wait",
            "--force-recreate",
            "mariadb",
            "minio",
            "mlflow",
        ],
        cwd=ROOT,
        check=True,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def client():
    from tracking.client import tracking_client

    uri = _tracking_uri()
    _wait_until_healthy(uri)
    return tracking_client()


def test_run_survives_restart_with_params_metrics_and_artifacts(client, tmp_path):
    experiment = client.get_experiment_by_name(EXPERIMENT)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(EXPERIMENT)
    run_id = client.create_run(experiment_id, tags={"ops-03": "persistence"}).info.run_id

    for key, value in PARAMS.items():
        client.log_param(run_id, key, value)
    for key, values in EPOCH_METRICS.items():
        for epoch, value in enumerate(values):
            client.log_metric(run_id, key, value, step=epoch)

    checkpoint = tmp_path / "upload" / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(os.urandom(256 * 1024))
    curve = tmp_path / "upload" / "curves" / "loss.json"
    curve.parent.mkdir(parents=True)
    curve.write_text(json.dumps(EPOCH_METRICS), encoding="utf-8")
    client.log_artifact(run_id, str(checkpoint), artifact_path="checkpoints")
    client.log_artifact(run_id, str(curve), artifact_path="curves")
    client.set_terminated(run_id, status="FINISHED")

    # Los artefactos pasan por el servidor (no por el disco del contenedor).
    assert client.get_run(run_id).info.artifact_uri.startswith("mlflow-artifacts:")

    _recreate_stack()
    _wait_until_healthy(_tracking_uri())

    run = client.get_run(run_id)
    assert run.info.run_id == run_id
    assert run.info.status == "FINISHED"
    assert run.data.params == PARAMS
    assert run.data.tags["ops-03"] == "persistence"

    for key, values in EPOCH_METRICS.items():
        history = sorted(client.get_metric_history(run_id, key), key=lambda m: m.step)
        assert [(m.step, m.value) for m in history] == list(enumerate(values))

    downloaded = Path(client.download_artifacts(run_id, "", str(tmp_path / "download")))
    assert _sha256(downloaded / "checkpoints" / "best.pt") == _sha256(checkpoint)
    assert json.loads((downloaded / "curves" / "loss.json").read_text()) == EPOCH_METRICS


def test_ui_is_reachable():
    uri = _tracking_uri()
    _wait_until_healthy(uri)
    with urllib.request.urlopen(uri, timeout=10) as response:
        assert response.status == 200
        assert "text/html" in response.headers["Content-Type"]
