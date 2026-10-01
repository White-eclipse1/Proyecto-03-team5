"""ML-06: early stopping sobre val_loss y restauración del mejor checkpoint."""

import json
import math
from pathlib import Path

import pytest
import torch
from mlflow.tracking import MlflowClient
from PIL import Image

import classification.training as training_module
from classification.early_stopping import MONITOR, MONITOR_MODE, EarlyStopping
from classification.training import (
    CHECKPOINT_ARTIFACT,
    CURVES_ARTIFACT,
    HISTORY_ARTIFACT,
    DataPaths,
    run_training,
)
from presentation.ml_contracts import TrainingParams
from tests._classification_fixtures import write_controlled_release

COMMIT = "0123456789abcdef0123456789abcdef01234567"


# --- Lógica de early stopping con secuencias artificiales ---------------------------------


def _feed(stopper, values):
    """Alimenta `values` época por época hasta que pide parar; devuelve la época de parada."""
    for epoch, value in enumerate(values, start=1):
        stopper.update(epoch, value)
        if stopper.should_stop:
            return epoch
    return None


def test_monitored_metric_is_validation_loss_minimized():
    assert MONITOR == "val_loss"
    assert MONITOR_MODE == "min"


def test_stops_after_patience_epochs_without_improvement():
    stopper = EarlyStopping(patience=2, min_delta=0.0)

    stopped = _feed(stopper, [1.0, 0.8, 0.85, 0.9, 0.1])

    assert stopped == 4
    assert stopper.best_epoch == 2
    assert stopper.best_value == 0.8
    assert stopper.stopped_epoch == 4


def test_improvement_resets_the_patience_counter():
    stopper = EarlyStopping(patience=2, min_delta=0.0)

    stopped = _feed(stopper, [1.0, 1.1, 0.9, 0.95, 0.85, 0.86, 0.87])

    assert stopped == 7
    assert stopper.best_epoch == 5


def test_min_delta_ignores_tiny_improvements():
    stopper = EarlyStopping(patience=2, min_delta=0.1)

    stopped = _feed(stopper, [1.0, 0.95, 0.91, 0.5])

    assert stopped == 3
    assert stopper.best_epoch == 1


def test_an_equal_value_is_not_an_improvement():
    stopper = EarlyStopping(patience=1, min_delta=0.0)

    assert _feed(stopper, [0.5, 0.5]) == 2
    assert stopper.best_epoch == 1


def test_nan_never_counts_as_improvement():
    stopper = EarlyStopping(patience=1, min_delta=0.0)

    assert _feed(stopper, [0.7, math.nan]) == 2
    assert stopper.best_epoch == 1


def test_nan_in_the_first_epoch_is_not_the_best():
    stopper = EarlyStopping(patience=3, min_delta=0.0)

    assert _feed(stopper, [math.nan, 0.9, 0.8]) is None
    assert stopper.best_epoch == 3
    assert stopper.best_value == 0.8


def test_never_stops_while_improving():
    stopper = EarlyStopping(patience=1, min_delta=0.0)

    assert _feed(stopper, [0.9, 0.8, 0.7, 0.6]) is None
    assert stopper.best_epoch == 4
    assert stopper.stopped_epoch is None


def test_update_reports_whether_the_epoch_improved():
    stopper = EarlyStopping(patience=3, min_delta=0.0)

    assert [stopper.update(e, v) for e, v in enumerate([1.0, 0.9, 0.95, 0.8], 1)] == [
        True,
        True,
        False,
        True,
    ]


@pytest.mark.parametrize(("patience", "min_delta"), [(0, 0.0), (1, -0.1)])
def test_invalid_configuration_is_rejected(patience, min_delta):
    with pytest.raises(ValueError):
        EarlyStopping(patience=patience, min_delta=min_delta)


# --- Integración con run_training (Agent Test) --------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    return MlflowClient(tracking_uri=(tmp_path / "mlruns").as_uri())


@pytest.fixture
def release(tmp_path):
    return write_controlled_release(tmp_path / "release")


def _params(**overrides):
    values = {
        "optimizer": "adam",
        "batch_size": 4,
        "max_epochs": 6,
        "learning_rate": 0.001,
        "image_size": 32,
        "hidden_layers": [16],
        "dropout": 0.1,
        "seed": 7,
        "patience": 2,
        "min_delta": 0.0,
    }
    values.update(overrides)
    return TrainingParams(**values)


@pytest.fixture
def injected(monkeypatch):
    """Sustituye la evaluación en validation por una secuencia controlada de val_loss.

    Guarda una copia de los pesos del modelo en cada época para compararlos con el
    checkpoint final.
    """
    state = {"losses": [], "weights": []}

    def fake_evaluate(model, loader, criterion):
        state["weights"].append({k: v.detach().clone() for k, v in model.state_dict().items()})
        loss = state["losses"][len(state["weights"]) - 1]
        return loss, 1.0 - loss / 2

    monkeypatch.setattr(training_module, "_evaluate", fake_evaluate)
    return state


def _train(client, release, params, tmp_path):
    result = run_training(
        params,
        dataset_version=release.dataset_version,
        manifest_hash=release.manifest_hash,
        data=DataPaths(release.manifest_path, release.crop_report_path, release.crops_dir),
        client=client,
        pretrained=False,
        git_commit=COMMIT,
    )
    local = client.download_artifacts(result.run_id, CHECKPOINT_ARTIFACT, str(tmp_path / "dl"))
    return result, torch.load(local, weights_only=True)


def _same_weights(left, right):
    return all(torch.equal(left[key], right[key]) for key in left)


def test_injected_metrics_force_early_stopping_and_restore_the_best_epoch(
    client, release, injected, tmp_path
):
    injected["losses"] = [0.9, 0.5, 0.6, 0.7, 0.3, 0.2]

    result, checkpoint = _train(client, release, _params(max_epochs=6, patience=2), tmp_path)

    assert [epoch.epoch for epoch in result.history] == [1, 2, 3, 4]
    assert result.best_epoch == 2
    assert result.stopped_epoch == 4
    assert result.checkpoint_uri == f"runs:/{result.run_id}/{CHECKPOINT_ARTIFACT}"
    # El checkpoint final son los pesos de la época 2, no los de la última (4).
    assert _same_weights(checkpoint["state_dict"], injected["weights"][1])
    assert not _same_weights(checkpoint["state_dict"], injected["weights"][3])
    assert checkpoint["metadata"]["best_epoch"] == 2
    assert checkpoint["metadata"]["stopped_epoch"] == 4
    assert checkpoint["metadata"]["best_val_loss"] == 0.5


def test_early_stopping_is_recorded_in_mlflow(client, release, injected, tmp_path):
    injected["losses"] = [0.9, 0.5, 0.6, 0.7, 0.3, 0.2]

    result, _ = _train(client, release, _params(max_epochs=6, patience=2), tmp_path)

    run = client.get_run(result.run_id)
    assert run.data.params["early_stopping_monitor"] == "val_loss"
    assert run.data.params["early_stopping_mode"] == "min"
    assert run.data.params["patience"] == "2"
    assert run.data.params["min_delta"] == "0.0"
    assert run.data.metrics["best_epoch"] == 2
    assert run.data.metrics["best_val_loss"] == 0.5
    assert run.data.metrics["stopped_epoch"] == 4
    assert run.data.metrics["epochs_completed"] == 4
    assert run.data.tags["early_stopped"] == "True"
    assert [m.step for m in client.get_metric_history(result.run_id, "val_loss")] == [1, 2, 3, 4]
    assert [a.path for a in client.list_artifacts(result.run_id, "checkpoints")] == [
        CHECKPOINT_ARTIFACT
    ]


def test_without_early_stop_the_best_epoch_can_be_the_last(client, release, injected, tmp_path):
    injected["losses"] = [0.9, 0.8, 0.7]

    result, checkpoint = _train(client, release, _params(max_epochs=3, patience=1), tmp_path)

    run = client.get_run(result.run_id)
    assert result.stopped_epoch is None
    assert result.best_epoch == 3
    assert run.data.tags["early_stopped"] == "False"
    assert "stopped_epoch" not in run.data.metrics
    assert run.data.metrics["epochs_completed"] == 3
    assert _same_weights(checkpoint["state_dict"], injected["weights"][2])


def test_best_epoch_in_the_middle_of_a_full_run_is_restored(client, release, injected, tmp_path):
    injected["losses"] = [0.9, 0.4, 0.6, 0.5]

    result, checkpoint = _train(client, release, _params(max_epochs=4, patience=3), tmp_path)

    assert result.stopped_epoch is None
    assert len(result.history) == 4
    assert result.best_epoch == 2
    assert _same_weights(checkpoint["state_dict"], injected["weights"][1])
    assert not _same_weights(checkpoint["state_dict"], injected["weights"][3])


@pytest.mark.parametrize(
    ("patience", "min_delta", "stopped", "best"),
    [(1, 0.0, 3, 2), (3, 0.0, None, 5), (2, 0.15, 3, 1)],
)
def test_patience_and_min_delta_come_from_the_training_params(
    client, release, injected, tmp_path, patience, min_delta, stopped, best
):
    injected["losses"] = [0.9, 0.8, 0.85, 0.82, 0.7]

    result, _ = _train(
        client, release, _params(max_epochs=5, patience=patience, min_delta=min_delta), tmp_path
    )

    assert result.stopped_epoch == stopped
    assert result.best_epoch == best


def test_a_run_without_any_valid_val_loss_fails_instead_of_saving_a_checkpoint(
    client, release, injected, tmp_path
):
    injected["losses"] = [math.nan, math.nan]

    with pytest.raises(RuntimeError, match="val_loss"):
        _train(client, release, _params(max_epochs=2, patience=2), tmp_path)

    [run] = client.search_runs([client.get_experiment_by_name("dogcat-classifier").experiment_id])
    assert run.info.status == "FAILED"
    assert not client.list_artifacts(run.info.run_id, "checkpoints")


# --- Curvas con las métricas reales ----------------------------------------------------------


def test_curves_are_logged_from_the_real_epoch_metrics(client, release, tmp_path):
    result, _ = _train(client, release, _params(max_epochs=3, patience=3), tmp_path)

    history_path = client.download_artifacts(result.run_id, HISTORY_ARTIFACT, str(tmp_path / "h"))
    logged = json.loads(Path(history_path).read_text(encoding="utf-8"))
    assert logged["best_epoch"] == result.best_epoch
    assert logged["stopped_epoch"] is None
    assert logged["monitor"] == "val_loss"
    for name in ("train_loss", "train_accuracy", "val_loss", "val_accuracy"):
        mlflow_values = [m.value for m in client.get_metric_history(result.run_id, name)]
        assert logged["epochs"][name] == mlflow_values
    curves = client.download_artifacts(result.run_id, CURVES_ARTIFACT, str(tmp_path / "c"))
    with Image.open(curves) as image:
        assert image.format == "PNG"
        assert image.width >= 600 and image.height >= 300
