"""ML-05: Agent Test con datos reales (ver tests/evidence/ml-05-reproducibility.md).

Dos corridas cortas con la misma semilla contra el servidor MLflow, comparando el
orden de muestras recuperado por la API. Requiere `docker compose up -d --wait
mlflow` y `data/crops`; se activa con RUN_REPRODUCIBILITY_EVIDENCE=1 y usa
`MLFLOW_TRACKING_URI`.
"""

import json
import os
from pathlib import Path

import pytest

from classification.training import SAMPLE_ORDER_ARTIFACT, DataPaths, run_training
from presentation.ml_contracts import TrainingParams
from tracking.client import tracking_client

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_REPRODUCIBILITY_EVIDENCE") != "1",
    reason="Set RUN_REPRODUCIBILITY_EVIDENCE=1 con MLflow levantado y data/crops descargado",
)


def _run(seed: int, name: str):
    manifest = json.loads(
        (ROOT / "reports" / "releases" / "v0.1.1" / "manifest.json").read_text(encoding="utf-8")
    )
    params = TrainingParams(
        optimizer="adam",
        batch_size=32,
        max_epochs=2,
        learning_rate=0.001,
        image_size=96,
        hidden_layers=[64],
        dropout=0.2,
        seed=seed,
        patience=1,
        min_delta=0.0,
    )
    return run_training(
        params,
        dataset_version=manifest["dataset_version"],
        manifest_hash=manifest["manifest_hash"],
        data=DataPaths.for_release("v0.1.1"),
        client=tracking_client(),
        run_name=name,
    )


def _order_from_mlflow(client, run_id, tmp_path):
    path = client.download_artifacts(run_id, SAMPLE_ORDER_ARTIFACT, str(tmp_path / run_id))
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_two_real_runs_with_the_same_seed_see_the_same_sample_order(tmp_path):
    first, again, other = (
        _run(42, "ml05-seed42-a"),
        _run(42, "ml05-seed42-b"),
        _run(7, "ml05-seed7"),
    )
    client = tracking_client()
    orders = {
        r.run_id: _order_from_mlflow(client, r.run_id, tmp_path) for r in (first, again, other)
    }
    tags = {r.run_id: client.get_run(r.run_id).data.tags for r in (first, again, other)}
    params = client.get_run(first.run_id).data.params

    for result in (first, again, other):
        order = orders[result.run_id]
        print(
            f"\n{result.run_id} {client.get_run(result.run_id).info.run_name}: "
            f"train_order_sha256={tags[result.run_id]['train_order_sha256'][:16]}… "
            f"época 1 empieza {order['1'][:3]}"
        )
        print(f"  métricas: {[e.as_dict() for e in result.history]}")
    print("semillas:", {k: v for k, v in params.items() if k.startswith("seed")})
    print(
        "entorno:",
        {
            k: tags[first.run_id][k]
            for k in (
                "python_version",
                "torch_version",
                "torchvision_version",
                "numpy_version",
                "platform",
                "torch_num_threads",
            )
        },
    )

    assert orders[first.run_id] == orders[again.run_id]
    assert len(orders[first.run_id]["1"]) == 469
    assert orders[first.run_id]["1"] != orders[first.run_id]["2"]
    assert first.history == again.history
    assert orders[other.run_id] != orders[first.run_id]
    assert params["seed_split"] == "42"
