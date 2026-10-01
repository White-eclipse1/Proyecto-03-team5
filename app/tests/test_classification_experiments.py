"""ML-07: matriz de experimentos, ejecución reanudable y reporte por la API de MLflow."""

import json

import pytest
import yaml
from mlflow.tracking import MlflowClient

import classification.experiments as experiments_module
from classification.experiments import (
    MATRIX_TAG,
    VARIED_PARAMS,
    identical_result_groups,
    load_matrix,
    matrix_report,
    run_matrix,
)
from classification.training import EXPERIMENT_NAME, DataPaths
from tests._classification_fixtures import write_controlled_release
from tests._mlflow_paths import artifact_dir

COMMIT = "0123456789abcdef0123456789abcdef01234567"
BASE = {
    "optimizer": "adam",
    "batch_size": 4,
    "max_epochs": 2,
    "learning_rate": 0.001,
    "image_size": 32,
    "hidden_layers": [8],
    "dropout": 0.1,
    "seed": 7,
    "patience": 2,
    "min_delta": 0.0,
}
# Una corrida por cambio respecto de la base: cada parámetro termina con 2 valores.
CHANGES = [
    {},
    {"optimizer": "sgd", "learning_rate": 0.01},
    {"batch_size": 8},
    {"max_epochs": 3},
    {"image_size": 64},
    {"hidden_layers": [16, 8]},
    {"dropout": 0.3},
]


def _matrix_doc(changes=None, **extra):
    changes = CHANGES if changes is None else changes
    doc = {
        "matrix_id": "ml07-test",
        "dataset_version": "v0.1.1",
        "base": BASE,
        "runs": [{"name": f"r{i:02d}", **change} for i, change in enumerate(changes, start=1)],
    }
    doc.update(extra)
    return doc


def _write(tmp_path, doc):
    path = tmp_path / "matrix.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path


# --- Validación de la matriz ---------------------------------------------------------


def test_matrix_expands_base_and_changes_into_training_params(tmp_path):
    matrix = load_matrix(_write(tmp_path, _matrix_doc()))

    assert matrix.matrix_id == "ml07-test"
    assert [entry.name for entry in matrix.entries] == [f"r{i:02d}" for i in range(1, 8)]
    sgd = matrix.entries[1].params
    assert (sgd.optimizer, sgd.learning_rate, sgd.batch_size) == ("sgd", 0.01, 4)
    assert matrix.entries[5].params.hidden_layers == [16, 8]


def test_the_seven_requested_parameters_are_varied():
    assert set(VARIED_PARAMS) == {
        "optimizer",
        "batch_size",
        "max_epochs",
        "learning_rate",
        "image_size",
        "hidden_layers",
        "dropout",
    }


@pytest.mark.parametrize("missing", ["dropout", "image_size", "hidden_layers"])
def test_matrix_must_vary_every_requested_parameter(tmp_path, missing):
    changes = [change for change in CHANGES if missing not in change]

    with pytest.raises(ValueError, match=missing):
        load_matrix(_write(tmp_path, _matrix_doc(changes)))


def test_matrix_rejects_duplicate_configurations(tmp_path):
    changes = [*CHANGES, {"dropout": 0.3}]

    with pytest.raises(ValueError, match="duplicad"):
        load_matrix(_write(tmp_path, _matrix_doc(changes)))


def test_matrix_rejects_repeated_names(tmp_path):
    doc = _matrix_doc()
    doc["runs"][1]["name"] = "r01"

    with pytest.raises(ValueError, match="r01"):
        load_matrix(_write(tmp_path, doc))


def test_matrix_rejects_invalid_training_params(tmp_path):
    with pytest.raises(ValueError):
        load_matrix(_write(tmp_path, _matrix_doc([*CHANGES, {"image_size": 48}])))


def test_committed_ml07_matrix_is_valid():
    from pathlib import Path

    path = Path(experiments_module.__file__).with_name("ml07_matrix.yaml")
    matrix = load_matrix(path)

    assert len(matrix.entries) >= 10
    assert matrix.dataset_version == "v0.1.1"


# --- Ejecución -----------------------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    return MlflowClient(tracking_uri=(tmp_path / "mlruns").as_uri())


@pytest.fixture
def release(tmp_path):
    return write_controlled_release(tmp_path / "release")


def _data(release):
    return DataPaths(release.manifest_path, release.crop_report_path, release.crops_dir)


def _run(client, release, matrix, **kwargs):
    kwargs.setdefault("pretrained", False)
    kwargs.setdefault("git_commit", COMMIT)
    return run_matrix(
        matrix,
        manifest_hash=release.manifest_hash,
        data=_data(release),
        client=client,
        **kwargs,
    )


def _small(tmp_path, changes):
    return load_matrix(_write(tmp_path, _matrix_doc(changes)))


def test_run_matrix_tags_each_run_with_the_matrix_and_entry(client, release, tmp_path):
    matrix = load_matrix(_write(tmp_path, _matrix_doc()))

    results = _run(client, release, matrix, only=["r01", "r02"])

    assert [name for name, _ in results] == ["r01", "r02"]
    for name, run_id in results:
        run = client.get_run(run_id)
        assert run.info.status == "FINISHED"
        assert run.info.run_name == f"ml07-test-{name}"
        assert run.data.tags[MATRIX_TAG] == "ml07-test"
        assert run.data.tags["matrix_entry"] == name
    assert client.get_run(results[1][1]).data.params["optimizer"] == "sgd"


def test_run_matrix_skips_entries_that_already_finished(client, release, tmp_path, monkeypatch):
    matrix = load_matrix(_write(tmp_path, _matrix_doc()))
    first = dict(_run(client, release, matrix, only=["r01"]))
    calls = []
    real = experiments_module.run_training

    def spy(params, **kwargs):
        calls.append(kwargs["run_name"])
        return real(params, **kwargs)

    monkeypatch.setattr(experiments_module, "run_training", spy)

    again = dict(_run(client, release, matrix, only=["r01", "r02"]))

    assert calls == ["ml07-test-r02"]
    assert again["r01"] == first["r01"]


def test_run_matrix_refuses_a_dirty_working_tree(client, release, tmp_path, monkeypatch):
    matrix = load_matrix(_write(tmp_path, _matrix_doc()))
    monkeypatch.setattr(experiments_module, "_dirty_files", lambda: ["app/classification/x.py"])

    with pytest.raises(RuntimeError, match="commit"):
        run_matrix(
            matrix,
            manifest_hash=release.manifest_hash,
            data=_data(release),
            client=client,
            pretrained=False,
        )


# --- Reporte por la API de MLflow (Agent Test) ----------------------------------------


def test_report_lists_valid_runs_and_checks_every_criterion(client, release, tmp_path):
    matrix = load_matrix(_write(tmp_path, _matrix_doc()))
    _run(client, release, matrix)

    report = matrix_report(client, matrix, min_runs=7)

    assert report["matrix_id"] == "ml07-test"
    assert len(report["valid_runs"]) == 7
    assert all(report["checks"].values()), report["checks"]
    assert report["varied_params"]["optimizer"] == ["adam", "sgd"]
    assert report["varied_params"]["hidden_layers"] == ["16,8", "8"]
    run = report["valid_runs"][0]
    assert set(run) >= {
        "run_id",
        "entry",
        "params",
        "best_epoch",
        "best_val_loss",
        "best_val_accuracy",
        "epochs_completed",
        "checkpoint",
        "git_commit",
    }
    assert run["checkpoint"] == f"runs:/{run['run_id']}/checkpoints/best.pt"
    assert report["manifest_hash"] == release.manifest_hash
    json.dumps(report)  # serializable para guardarlo en el repo


def test_report_fails_when_there_are_fewer_runs_than_required(client, release, tmp_path):
    matrix = load_matrix(_write(tmp_path, _matrix_doc()))
    _run(client, release, matrix, only=["r01", "r02", "r03"])

    report = matrix_report(client, matrix, min_runs=10)

    assert report["checks"]["at_least_min_runs"] is False
    assert report["checks"]["seven_params_vary"] is False


def test_report_ignores_failed_and_foreign_runs(client, release, tmp_path, monkeypatch):
    matrix = load_matrix(_write(tmp_path, _matrix_doc()))
    _run(client, release, matrix, only=["r01"])
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
    failed = client.create_run(experiment, tags={MATRIX_TAG: "ml07-test", "matrix_entry": "r02"})
    client.set_terminated(failed.info.run_id, "FAILED")
    client.create_run(experiment, tags={MATRIX_TAG: "otra-matriz", "matrix_entry": "r03"})

    report = matrix_report(client, matrix, min_runs=1)

    assert [run["entry"] for run in report["valid_runs"]] == ["r01"]
    assert report["excluded_runs"] == [
        {"run_id": failed.info.run_id, "entry": "r02", "reason": "status FAILED"}
    ]


def test_runs_with_the_same_best_result_are_grouped():
    runs = [
        {"entry": "a", "best_epoch": 5, "best_val_loss": 0.08, "best_val_accuracy": 0.97},
        {"entry": "b", "best_epoch": 3, "best_val_loss": 0.04, "best_val_accuracy": 0.98},
        {"entry": "c", "best_epoch": 5, "best_val_loss": 0.08, "best_val_accuracy": 0.97},
    ]

    assert identical_result_groups(runs) == [["a", "c"]]


def test_report_counts_only_distinct_results(client, release, tmp_path):
    matrix = load_matrix(_write(tmp_path, _matrix_doc()))
    _run(client, release, matrix)

    report = matrix_report(client, matrix, min_runs=7)

    groups = report["identical_results"]
    assert report["distinct_results"] == 7 - sum(len(group) - 1 for group in groups)
    assert report["checks"]["at_least_min_distinct_results"] is (report["distinct_results"] >= 7)


def test_report_fails_when_identical_results_leave_fewer_distinct_runs(
    client, release, tmp_path, monkeypatch
):
    matrix = load_matrix(_write(tmp_path, _matrix_doc()))
    _run(client, release, matrix)
    real_summary = experiments_module._run_summary

    def same_best_result(client, run):
        summary = real_summary(client, run)
        if summary["entry"] in ("r01", "r04"):
            summary.update(best_epoch=2, best_val_loss=0.5, best_val_accuracy=0.75)
        return summary

    monkeypatch.setattr(experiments_module, "_run_summary", same_best_result)

    report = matrix_report(client, matrix, min_runs=7)

    assert ["r01", "r04"] in report["identical_results"]
    assert report["distinct_results"] < 7
    assert report["checks"]["at_least_min_runs"] is True
    assert report["checks"]["at_least_min_distinct_results"] is False


def test_report_detects_a_run_without_checkpoint(client, release, tmp_path):
    import shutil

    matrix = load_matrix(_write(tmp_path, _matrix_doc()))
    runs = dict(_run(client, release, matrix))
    artifacts = artifact_dir(client, runs["r02"])
    shutil.rmtree(artifacts / "checkpoints")

    report = matrix_report(client, matrix, min_runs=7)

    by_entry = {run["entry"]: run for run in report["valid_runs"]}
    assert by_entry["r02"]["checkpoint"] is None
    assert report["checks"]["all_have_checkpoint"] is False
