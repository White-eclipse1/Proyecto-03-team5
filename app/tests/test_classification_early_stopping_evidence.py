"""ML-06: early stopping real y restauración verificada (tests/evidence/ml-06-early-stopping.md).

Entrena con datos reales contra el servidor MLflow hasta que el early stopping corta,
descarga `checkpoints/best.pt` por la API y lo vuelve a evaluar en validation: su
`val_loss` debe ser la de `best_epoch`, no la de la última época. Requiere
`docker compose up -d --wait mlflow` y `data/crops`; se activa con
RUN_EARLY_STOPPING_EVIDENCE=1 y usa `MLFLOW_TRACKING_URI`.
"""

import json
import os
from pathlib import Path

import pytest
from torch import nn

from classification.dataset import build_dataloader, load_split
from classification.model import load_checkpoint
from classification.training import (
    CHECKPOINT_ARTIFACT,
    CURVES_ARTIFACT,
    HISTORY_ARTIFACT,
    DataPaths,
    _evaluate,
    run_training,
)
from presentation.ml_contracts import TrainingParams
from tracking.client import tracking_client

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_EARLY_STOPPING_EVIDENCE") != "1",
    reason="Set RUN_EARLY_STOPPING_EVIDENCE=1 con MLflow levantado y data/crops descargado",
)


def test_real_run_restores_the_best_epoch_checkpoint(tmp_path):
    manifest = json.loads(
        (ROOT / "reports" / "releases" / "v0.1.1" / "manifest.json").read_text(encoding="utf-8")
    )
    params = TrainingParams(
        optimizer="adam",
        batch_size=32,
        max_epochs=15,
        learning_rate=0.001,
        image_size=128,
        hidden_layers=[128],
        dropout=0.2,
        seed=42,
        patience=2,
        min_delta=0.0,
    )
    data = DataPaths.for_release("v0.1.1")
    result = run_training(
        params,
        dataset_version=manifest["dataset_version"],
        manifest_hash=manifest["manifest_hash"],
        data=data,
        client=tracking_client(),
        run_name="ml06-evidencia",
    )

    client = tracking_client()
    run = client.get_run(result.run_id)
    val_loss = {m.step: m.value for m in client.get_metric_history(result.run_id, "val_loss")}
    best = int(run.data.metrics["best_epoch"])
    last = int(run.data.metrics["epochs_completed"])
    print(
        f"\nrun {result.run_id}: best_epoch={best}, épocas={last}, "
        f"early_stopped={run.data.tags['early_stopped']}"
    )
    print(f"val_loss: {[(e, round(v, 4)) for e, v in sorted(val_loss.items())]}")
    assert run.info.status == "FINISHED"
    assert best == min(val_loss, key=val_loss.get)
    assert {a.path for a in client.list_artifacts(result.run_id, "curves")} == {
        CURVES_ARTIFACT,
        HISTORY_ARTIFACT,
    }

    local = client.download_artifacts(result.run_id, CHECKPOINT_ARTIFACT, str(tmp_path))
    model = load_checkpoint(Path(local))
    validation = load_split(
        data.manifest,
        data.crop_report,
        crops_dir=data.crops_dir,
        split="validation",
        params=params,
    )
    loss, _ = _evaluate(
        model, build_dataloader(validation, batch_size=32, seed=42), nn.CrossEntropyLoss()
    )
    print(f"best.pt re-evaluado: val_loss={loss:.6f} (época {best}: {val_loss[best]:.6f})")
    assert loss == pytest.approx(val_loss[best], abs=1e-5)
    if last != best:
        assert loss != pytest.approx(val_loss[last], abs=1e-5)
