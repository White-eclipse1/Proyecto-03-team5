"""ML-04: loop de entrenamiento por minibatches con instrumentación MLflow.

Usa un dataset pequeño y controlado (`_classification_fixtures`) y un MLflow local en
`tmp_path` (backend de archivos), sin servidor ni red: `pretrained=False` construye la
misma ResNet18 sin descargar pesos ImageNet.
"""

import json
import math
from pathlib import Path

import pytest
import torch
from mlflow.tracking import MlflowClient
from torch import nn

import classification.training as training_module
from classification.model import load_checkpoint
from classification.training import (
    EXPERIMENT_NAME,
    METRIC_NAMES,
    DataPaths,
    JobQueueHooks,
    build_optimizer,
    resolve_git_commit,
    run_training,
)
from presentation.ml_contracts import TrainingParams
from tests._classification_fixtures import write_controlled_release

COMMIT = "0123456789abcdef0123456789abcdef01234567"


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
        "max_epochs": 2,
        "learning_rate": 0.001,
        "image_size": 32,
        "hidden_layers": [16],
        "dropout": 0.1,
        "seed": 7,
        "patience": 1,
        "min_delta": 0.0,
    }
    values.update(overrides)
    return TrainingParams(**values)


def _data(release):
    return DataPaths(
        manifest=release.manifest_path,
        crop_report=release.crop_report_path,
        crops_dir=release.crops_dir,
    )


def _train(client, release, params=None, **kwargs):
    kwargs.setdefault("pretrained", False)
    kwargs.setdefault("git_commit", COMMIT)
    return run_training(
        params or _params(),
        dataset_version=release.dataset_version,
        manifest_hash=release.manifest_hash,
        data=_data(release),
        client=client,
        **kwargs,
    )


class RecordingHooks:
    def __init__(self):
        self.events = []

    def on_run_started(self, experiment_id, run_id):
        self.events.append(("started", experiment_id, run_id))

    def on_epoch_end(self, epoch, metrics):
        self.events.append(("epoch", epoch, dict(metrics)))

    def log(self, message, level="info"):
        self.events.append(("log", level, message))


# --- Minibatches y optimizer.step() por batch ------------------------------------------


@pytest.mark.parametrize(("batch_size", "steps_per_epoch"), [(2, 4), (3, 3), (4, 2), (8, 1)])
def test_optimizer_steps_once_per_minibatch(client, release, batch_size, steps_per_epoch):
    result = _train(client, release, _params(batch_size=batch_size, max_epochs=2))

    assert result.optimizer_steps == 2 * steps_per_epoch == 2 * math.ceil(8 / batch_size)


def test_last_train_batch_of_one_sample_is_dropped(client, tmp_path):
    release = write_controlled_release(
        tmp_path / "nine", splits={"train": 9, "validation": 4, "test": 4}
    )

    result = _train(client, release, _params(batch_size=4, max_epochs=2))

    assert result.optimizer_steps == 2 * 2
    assert client.get_run(result.run_id).data.params["train_drop_last"] == "True"


def test_batch_size_one_is_rejected_before_creating_a_run(client, release):
    with pytest.raises(ValueError, match="batch_size=1"):
        _train(client, release, _params(batch_size=1))

    assert client.get_experiment_by_name(EXPERIMENT_NAME) is None


def test_each_step_uses_one_batch_of_batch_size(client, release, monkeypatch):
    seen = []
    real_build = training_module.build_optimizer

    def spy_build(name, parameters, learning_rate):
        optimizer = real_build(name, parameters, learning_rate)
        real_step = optimizer.step

        def step(*args, **kwargs):
            seen.append("step")
            return real_step(*args, **kwargs)

        optimizer.step = step
        return optimizer

    monkeypatch.setattr(training_module, "build_optimizer", spy_build)

    result = _train(client, release, _params(batch_size=3, max_epochs=1))

    assert len(seen) == result.optimizer_steps == 3


# --- Parámetros configurables que afectan el entrenamiento ------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [("adam", torch.optim.Adam), ("adamw", torch.optim.AdamW), ("sgd", torch.optim.SGD)],
)
def test_build_optimizer_uses_the_configured_optimizer_and_learning_rate(name, expected):
    parameter = nn.Parameter(torch.zeros(2))

    optimizer = build_optimizer(name, [parameter], 0.0123)

    assert type(optimizer) is expected
    assert optimizer.param_groups[0]["lr"] == 0.0123


def test_build_optimizer_rejects_unknown_names():
    with pytest.raises(ValueError, match="optimizer"):
        build_optimizer("rmsprop", [nn.Parameter(torch.zeros(1))], 0.1)


@pytest.mark.parametrize(
    "change",
    [{"optimizer": "sgd"}, {"learning_rate": 0.1}, {"batch_size": 2}],
)
def test_optimizer_learning_rate_and_batch_size_change_the_result(client, release, change):
    base = _train(client, release, _params(max_epochs=1))
    other = _train(client, release, _params(max_epochs=1, **change))

    assert base.history[0].train_loss != other.history[0].train_loss


def test_same_params_and_seed_reproduce_the_history(client, release):
    first = _train(client, release, _params(max_epochs=2))
    again = _train(client, release, _params(max_epochs=2))

    assert first.history == again.history


def test_max_epochs_sets_the_number_of_epochs(client, release):
    result = _train(client, release, _params(max_epochs=3))

    assert [epoch.epoch for epoch in result.history] == [1, 2, 3]


def test_checkpoint_model_uses_image_size_hidden_layers_and_dropout(client, release, tmp_path):
    params = _params(image_size=64, hidden_layers=[32, 8], dropout=0.3, max_epochs=1)
    result = _train(client, release, params)

    path = client.download_artifacts(result.run_id, "checkpoints/last.pt", str(tmp_path))
    model = load_checkpoint(Path(path))

    assert model.config.image_size == 64
    assert model.config.hidden_layers == [32, 8]
    assert model.config.dropout == 0.3
    assert model.config.pretrained is False


# --- Métricas por época ----------------------------------------------------------------


def test_epoch_metrics_are_logged_to_mlflow_per_epoch(client, release):
    result = _train(client, release, _params(max_epochs=3))

    assert set(METRIC_NAMES) == {"train_loss", "train_accuracy", "val_loss", "val_accuracy"}
    for name in METRIC_NAMES:
        history = client.get_metric_history(result.run_id, name)
        assert [metric.step for metric in history] == [1, 2, 3]
        assert [metric.value for metric in history] == [
            getattr(epoch, name) for epoch in result.history
        ]
    for epoch in result.history:
        assert 0 <= epoch.train_accuracy <= 1 and 0 <= epoch.val_accuracy <= 1
        assert epoch.train_loss > 0 and epoch.val_loss > 0


def test_validation_metrics_come_from_the_validation_split(client, release, monkeypatch):
    sizes = {}
    real_evaluate = training_module._evaluate

    def spy(model, loader, criterion):
        sizes.setdefault("validation", len(loader.dataset))
        return real_evaluate(model, loader, criterion)

    monkeypatch.setattr(training_module, "_evaluate", spy)

    _train(client, release, _params(max_epochs=1))

    assert sizes == {"validation": 4}


def test_controlled_dataset_is_learned(client, release):
    result = _train(
        client, release, _params(max_epochs=6, learning_rate=0.003, batch_size=4, dropout=0.0)
    )

    assert result.history[-1].train_loss < result.history[0].train_loss


# --- Parámetros, procedencia y artefactos en MLflow (Agent Test) ------------------------


def test_run_records_params_provenance_classes_and_checkpoint(client, release, tmp_path):
    params = _params(hidden_layers=[16, 8], max_epochs=2)
    result = _train(client, release, params, run_name="agent-test")

    run = client.get_run(result.run_id)
    assert run.info.status == "FINISHED"
    assert run.info.run_name == "agent-test"
    assert client.get_experiment(result.experiment_id).name == EXPERIMENT_NAME
    assert run.data.params == {
        "optimizer": "adam",
        "batch_size": "4",
        "max_epochs": "2",
        "learning_rate": "0.001",
        "image_size": "32",
        "hidden_layers": "16,8",
        "dropout": "0.1",
        "seed": "7",
        "patience": "1",
        "min_delta": "0.0",
        "architecture": "resnet18",
        "pretrained": "False",
        "trainable": "layer4",
        "train_samples": "8",
        "validation_samples": "4",
        "train_drop_last": "False",
    }
    tags = run.data.tags
    assert tags["mlflow.source.git.commit"] == COMMIT
    assert tags["git_commit"] == COMMIT
    assert tags["dataset_version"] == release.dataset_version
    assert tags["manifest_hash"] == release.manifest_hash
    assert tags["dvc_images_hash"] == release.report.provenance.images_dvc_hash
    assert tags["dvc_annotations_hash"] == release.report.provenance.annotations_dvc_hash
    assert tags["classes"] == "dog,cat"
    assert json.loads(tags["class_map"]) == {"dog": 0, "cat": 1}

    assert result.checkpoint_uri == f"runs:/{result.run_id}/checkpoints/last.pt"
    assert [artifact.path for artifact in client.list_artifacts(result.run_id, "checkpoints")] == [
        "checkpoints/last.pt"
    ]
    local = client.download_artifacts(result.run_id, "checkpoints/last.pt", str(tmp_path))
    payload = torch.load(local, weights_only=True)
    assert payload["metadata"] == {
        "run_id": result.run_id,
        "dataset_version": release.dataset_version,
        "manifest_hash": release.manifest_hash,
        "git_commit": COMMIT,
        "epochs": 2,
    }


def test_hooks_receive_the_run_and_every_epoch(client, release):
    hooks = RecordingHooks()

    result = _train(client, release, _params(max_epochs=2), hooks=hooks)

    started = [event for event in hooks.events if event[0] == "started"]
    epochs = [event for event in hooks.events if event[0] == "epoch"]
    assert started == [("started", result.experiment_id, result.run_id)]
    assert hooks.events.index(started[0]) < hooks.events.index(epochs[0])
    assert [event[1] for event in epochs] == [1, 2]
    assert set(epochs[0][2]) == set(METRIC_NAMES)
    assert all(isinstance(value, float) for value in epochs[0][2].values())
    assert any(event[0] == "log" for event in hooks.events)


# --- Fallos ---------------------------------------------------------------------------------


def test_failure_during_training_marks_the_run_failed(client, release, monkeypatch):
    calls = {"epochs": 0}
    real_train_epoch = training_module._train_epoch

    def failing(*args, **kwargs):
        calls["epochs"] += 1
        if calls["epochs"] == 2:
            raise RuntimeError("fallo inyectado en la época 2")
        return real_train_epoch(*args, **kwargs)

    monkeypatch.setattr(training_module, "_train_epoch", failing)
    hooks = RecordingHooks()

    with pytest.raises(RuntimeError, match="fallo inyectado"):
        _train(client, release, _params(max_epochs=3), hooks=hooks)

    [started] = [event for event in hooks.events if event[0] == "started"]
    run = client.get_run(started[2])
    assert run.info.status == "FAILED"
    assert "fallo inyectado" in run.data.tags["error"]
    assert [m.step for m in client.get_metric_history(run.info.run_id, "train_loss")] == [1]
    assert any(event[:2] == ("log", "error") for event in hooks.events)


@pytest.mark.parametrize(
    ("field", "value"),
    [("manifest_hash", "sha256:" + "a" * 64), ("dataset_version", "v0.1.0")],
)
def test_job_pinned_to_another_manifest_fails_before_creating_a_run(client, release, field, value):
    kwargs = {"dataset_version": release.dataset_version, "manifest_hash": release.manifest_hash}
    kwargs[field] = value

    with pytest.raises(ValueError, match=field):
        run_training(
            _params(),
            data=_data(release),
            client=client,
            pretrained=False,
            git_commit=COMMIT,
            **kwargs,
        )

    assert client.get_experiment_by_name(EXPERIMENT_NAME) is None or not client.search_runs(
        [client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id]
    )


# --- Git commit -----------------------------------------------------------------------------


def test_git_commit_comes_from_the_environment_first(monkeypatch):
    monkeypatch.setenv("GIT_COMMIT", COMMIT)

    assert resolve_git_commit() == COMMIT


def test_git_commit_falls_back_to_the_repository(monkeypatch):
    monkeypatch.delenv("GIT_COMMIT", raising=False)

    commit = resolve_git_commit()

    assert len(commit) == 40 and int(commit, 16) >= 0


@pytest.mark.parametrize("value", ["abc", "z" * 40, ""])
def test_invalid_git_commit_is_rejected(monkeypatch, value):
    monkeypatch.setenv("GIT_COMMIT", value)

    with pytest.raises(ValueError, match="GIT_COMMIT"):
        resolve_git_commit()


# --- Adaptador para la cola de APP-03 (worker de OPS-04) ------------------------------------


class FakeQueue:
    """Mismas firmas que `training.queue.TrainingJobQueue` (APP-03, PR #38)."""

    def __init__(self):
        self.calls = []

    def start(self, job_id, *, experiment_id, run_id):
        self.calls.append(("start", job_id, experiment_id, run_id))

    def report_progress(self, job_id, *, epoch, metrics):
        self.calls.append(("progress", job_id, epoch, dict(metrics)))

    def log(self, job_id, message, *, level="info"):
        # `level` es solo por nombre, como en la cola real.
        assert level in ("info", "warning", "error")
        self.calls.append(("log", job_id, level, message))


def test_job_queue_hooks_drive_the_queue_during_a_run(client, release):
    queue = FakeQueue()

    result = _train(client, release, _params(max_epochs=2), hooks=JobQueueHooks(queue, "job-1"))

    assert queue.calls[0] == ("start", "job-1", result.experiment_id, result.run_id)
    progress = [call for call in queue.calls if call[0] == "progress"]
    assert [call[2] for call in progress] == [1, 2]
    assert progress[-1][3] == result.history[-1].as_dict()
    assert all(call[1] == "job-1" for call in queue.calls)
    assert any(call[0] == "log" and "Época 2/2" in call[3] for call in queue.calls)


def test_job_queue_hooks_forward_error_logs(client, release, monkeypatch):
    queue = FakeQueue()
    monkeypatch.setattr(
        training_module, "_evaluate", lambda *args: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    with pytest.raises(RuntimeError, match="boom"):
        _train(client, release, _params(max_epochs=1), hooks=JobQueueHooks(queue, "job-2"))

    assert queue.calls[0][0] == "start"
    assert any(call[0] == "log" and call[2] == "error" for call in queue.calls)
    assert not [call for call in queue.calls if call[0] == "progress"]
