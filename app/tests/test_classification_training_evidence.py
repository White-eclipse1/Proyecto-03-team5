"""ML-04: corrida real contra el servidor MLflow (ver tests/evidence/ml-04-training.md).

Requiere `docker compose up -d --wait mlflow`, `data/crops` de DVC y red para los
pesos ImageNet la primera vez. Se activa con RUN_TRAINING_EVIDENCE=1 y usa
`MLFLOW_TRACKING_URI` (por ejemplo http://127.0.0.1:5050).
"""

import json
import os
from pathlib import Path

import pytest

from classification.model import load_checkpoint
from classification.training import METRIC_NAMES, DataPaths, run_training
from presentation.ml_contracts import TrainingParams
from tracking.client import check_server, tracking_client

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_TRAINING_EVIDENCE") != "1",
    reason="Set RUN_TRAINING_EVIDENCE=1 con MLflow levantado y data/crops descargado",
)


def test_real_run_is_recoverable_through_the_mlflow_api(tmp_path):
    print(f"\nMLflow: {check_server()}")
    manifest = json.loads(
        (ROOT / "reports" / "releases" / "v0.1.1" / "manifest.json").read_text(encoding="utf-8")
    )
    params = TrainingParams(
        optimizer="adam",
        batch_size=32,
        max_epochs=2,
        learning_rate=0.001,
        image_size=128,
        hidden_layers=[128],
        dropout=0.2,
        seed=42,
        patience=2,
        min_delta=0.0,
    )

    result = run_training(
        params,
        dataset_version=manifest["dataset_version"],
        manifest_hash=manifest["manifest_hash"],
        data=DataPaths.for_release("v0.1.1"),
        client=tracking_client(),
        run_name="ml04-evidencia",
    )

    # Agent Test: todo se recupera con la API de MLflow, sin usar `result.history`.
    client = tracking_client()
    run = client.get_run(result.run_id)
    print(f"run {result.run_id}: {run.info.status}, {result.optimizer_steps} pasos")
    assert run.info.status == "FINISHED"
    assert run.data.params["batch_size"] == "32" and run.data.params["train_samples"] == "469"
    assert run.data.tags["dataset_version"] == "v0.1.1"
    assert run.data.tags["manifest_hash"] == manifest["manifest_hash"]
    assert run.data.tags["classes"] == "dog,cat"
    for name in METRIC_NAMES:
        history = client.get_metric_history(result.run_id, name)
        print(f"{name}: {[(m.step, round(m.value, 4)) for m in history]}")
        assert [m.step for m in history] == [1, 2]
    assert result.optimizer_steps == 2 * 15  # ceil(469 / 32) batches por época
    local = client.download_artifacts(result.run_id, "checkpoints/best.pt", str(tmp_path))
    model = load_checkpoint(Path(local))
    assert model.config.image_size == 128 and model.class_map == {"dog": 0, "cat": 1}
    print(f"checkpoint {result.checkpoint_uri} descargado y recargado")
