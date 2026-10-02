"""OPS-04 - integracion real cola -> worker -> MLflow -> checkpoint."""

import json
import os
import time
from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient
from sqlalchemy import create_engine

from presentation.ml_contracts import TrainingJobRequest
from training.queue import TrainingJobQueue

pytestmark = pytest.mark.skipif(
    os.environ.get("TRAINING_WORKER_E2E") != "1",
    reason="requiere el stack real de OPS-04",
)


def test_real_training_job_reaches_mlflow_checkpoint(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = repo_root / "reports" / "releases" / "v0.1.1" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    request = TrainingJobRequest.model_validate(
        {
            "schema_version": "1.0",
            "dataset_version": manifest["dataset_version"],
            "manifest_hash": manifest["manifest_hash"],
            "params": {
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
            },
        }
    )

    queue = TrainingJobQueue(create_engine(os.environ["TRAINING_QUEUE_DATABASE_URL"]))
    queue.create_tables()
    created = queue.enqueue(request)

    deadline = time.monotonic() + 900
    persisted = created

    while time.monotonic() < deadline:
        current = queue.get(created.job_id)
        assert current is not None
        persisted = current

        if current.status in {"succeeded", "failed"}:
            break

        time.sleep(2)

    if persisted.status == "failed":
        pytest.fail(
            f"worker fallo: {persisted.error}; "
            f"logs={[entry.message for entry in queue.logs(created.job_id)]}"
        )

    assert persisted.status == "succeeded"
    assert persisted.run_id is not None
    assert persisted.experiment_id is not None
    assert persisted.progress is not None
    assert persisted.progress.epoch == 1

    expected_checkpoint = f"runs:/{persisted.run_id}/checkpoints/best.pt"
    assert persisted.checkpoint == expected_checkpoint

    client = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
    run = client.get_run(persisted.run_id)

    assert run.info.status == "FINISHED"
    assert run.data.tags["dataset_version"] == "v0.1.1"
    assert run.data.tags["manifest_hash"] == manifest["manifest_hash"]
    assert run.data.tags["git_commit"] == os.environ["GIT_COMMIT"]

    checkpoint = Path(
        client.download_artifacts(
            persisted.run_id,
            "checkpoints/best.pt",
            str(tmp_path),
        )
    )

    assert checkpoint.is_file()
    assert checkpoint.stat().st_size > 0
