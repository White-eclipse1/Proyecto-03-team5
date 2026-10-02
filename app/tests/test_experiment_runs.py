"""APP-04: de la API de MLflow a los contratos de la pantalla Experiments.

TDD Requirement del issue #16: "mapping MLflow API → UI models". Usa entidades reales
de MLflow (`Run`, `RunInfo`, `Metric`, ...) servidas por un cliente falso; la prueba
contra un servidor MLflow real está en `test_experiments_mlflow_integration.py`.
"""

import json
import math
import sys

import pytest
from mlflow.entities import Experiment, Metric, Param, Run, RunData, RunInfo, RunTag
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from sqlalchemy import create_engine
from starlette.testclient import TestClient

from presentation.ml_contracts import ErrorResponse, RunCurvesResponse, RunsResponse
from tracking.run_schema import (
    CURVE_METRICS,
    TAG_DATASET_VERSION,
    TAG_GIT_COMMIT,
    TAG_MANIFEST_HASH,
)
from training.experiments import list_runs, run_curves, run_to_contract
from training.queue import TrainingJobQueue
from training.server import create_app

FINISHED = "0a1b2c3d4e5f60718293a4b5c6d7e8f9"
RUNNING = "f1e2d3c4b5a697887766554433221100"
NOT_TRAINING = "aaaabbbbccccddddeeeeffff00001111"
COMMIT = "9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c4b3a2f1e0d"
MANIFEST = "sha256:" + "a" * 64
START_MS = 1_790_000_000_000  # 2026-09-21T14:13:20Z


def make_run(
    run_id,
    *,
    status="FINISHED",
    tags=None,
    params=None,
    metrics=None,
    run_name="resnet18-adam",
    experiment_id="1",
):
    end = None if status in ("RUNNING", "SCHEDULED") else START_MS + 90_500
    default_tags = {
        TAG_DATASET_VERSION: "v0.1.1",
        TAG_MANIFEST_HASH: MANIFEST,
        TAG_GIT_COMMIT: COMMIT,
        "mlflow.runName": run_name,
    }
    return Run(
        RunInfo(
            run_id=run_id,
            experiment_id=experiment_id,
            user_id="worker",
            status=status,
            start_time=START_MS,
            end_time=end,
            lifecycle_stage="active",
            artifact_uri=f"mlflow-artifacts:/{experiment_id}/{run_id}/artifacts",
            run_name=run_name,
        ),
        RunData(
            metrics=[
                Metric(key, value, START_MS, step)
                for key, (value, step) in (metrics or {"val_accuracy": (0.869, 5)}).items()
            ],
            params=[Param(key, value) for key, value in (params or {"optimizer": "adam"}).items()],
            tags=[
                RunTag(key, value)
                for key, value in (default_tags if tags is None else tags).items()
            ],
        ),
    )


class FakeMlflow:
    """Implementa solo lo que `training/experiments.py` usa de `MlflowClient`."""

    def __init__(self, runs, histories=None, *, fail=False):
        self.runs = {run.info.run_id: run for run in runs}
        self.histories = histories or {}
        self.fail = fail

    def _check(self):
        if self.fail:
            raise MlflowException("connection refused")

    def search_experiments(self, **_):
        self._check()
        ids = sorted({run.info.experiment_id for run in self.runs.values()})
        return [Experiment(i, f"exp-{i}", "", "active") for i in ids]

    def search_runs(self, experiment_ids, **_):
        self._check()
        return [run for run in self.runs.values() if run.info.experiment_id in experiment_ids]

    def get_run(self, run_id):
        self._check()
        if run_id not in self.runs:
            raise MlflowException("no run", error_code=RESOURCE_DOES_NOT_EXIST)
        return self.runs[run_id]

    def get_metric_history(self, run_id, key):
        self._check()
        return [
            Metric(key, value, START_MS + i, step)
            for i, (step, value) in enumerate(self.histories.get((run_id, key), []))
        ]


# --- Mapeo de un run ---------------------------------------------------------------


def test_finished_run_maps_every_field_the_screen_needs():
    run = run_to_contract(
        make_run(
            FINISHED,
            params={"optimizer": "adam", "batch_size": "32"},
            metrics={"val_accuracy": (0.869, 5), "train_loss": (0.231, 5)},
        )
    )

    assert run.run_id == FINISHED
    assert run.experiment_id == "1"
    assert run.run_name == "resnet18-adam"
    assert run.status == "FINISHED"
    assert run.start_time == "2026-09-21T14:13:20Z"
    assert run.end_time == "2026-09-21T14:14:50.500000Z"
    assert (run.dataset_version, run.manifest_hash) == ("v0.1.1", MANIFEST)
    assert run.git_commit == COMMIT
    assert run.params == {"optimizer": "adam", "batch_size": "32"}
    assert run.metrics == {"val_accuracy": 0.869, "train_loss": 0.231}


def test_a_nan_latest_metric_keeps_the_run_visible_without_inventing_a_value():
    """Revisión de #41: con val_loss=NaN el run desaparecía de Experiments."""
    run = run_to_contract(
        make_run(FINISHED, metrics={"val_loss": (math.nan, 3), "val_accuracy": (0.81, 3)})
    )

    assert run is not None
    assert run.metrics == {"val_loss": None, "val_accuracy": 0.81}


@pytest.mark.parametrize(
    "value",
    [math.inf, -math.inf, sys.float_info.max, -sys.float_info.max],
    ids=["inf", "-inf", "sql-clamped-inf", "sql-clamped--inf"],
)
def test_infinite_metrics_are_null_too(value):
    """MLflow sobre MySQL guarda ±inf como ±1.797e308: tampoco es un valor real."""
    run = run_to_contract(make_run(FINISHED, metrics={"train_loss": (value, 1)}))
    assert run.metrics == {"train_loss": None}


def test_running_run_has_no_end_time():
    run = run_to_contract(make_run(RUNNING, status="RUNNING"))
    assert run.status == "RUNNING" and run.end_time is None


def test_missing_commit_is_shown_as_missing_not_invented():
    tags = {TAG_DATASET_VERSION: "v0.1.1", TAG_MANIFEST_HASH: MANIFEST}
    assert run_to_contract(make_run(FINISHED, tags=tags)).git_commit is None


def test_run_without_name_falls_back_to_its_id():
    run = make_run(FINISHED, run_name=None)
    assert run_to_contract(run).run_name == FINISHED


@pytest.mark.parametrize(
    "tags",
    [
        {},
        {TAG_DATASET_VERSION: "v0.1.1"},
        {TAG_MANIFEST_HASH: MANIFEST},
        {TAG_DATASET_VERSION: "v0.1.1", TAG_MANIFEST_HASH: "not-a-hash"},
    ],
    ids=["no-tags", "no-manifest", "no-dataset", "bad-manifest"],
)
def test_runs_without_valid_training_provenance_are_not_listed(tags):
    """Un run sin release ni manifest no es un entrenamiento trazable (rúbrica 3.2)."""
    assert run_to_contract(make_run(NOT_TRAINING, tags=tags)) is None


# --- Lista de runs ------------------------------------------------------------------


def test_list_runs_returns_training_runs_newest_first_across_experiments():
    older = make_run(FINISHED, experiment_id="1")
    newer = make_run(RUNNING, status="RUNNING", experiment_id="2")
    newer.info._start_time = START_MS + 60_000
    stray = make_run(NOT_TRAINING, tags={})

    runs = list_runs(FakeMlflow([older, newer, stray]))

    assert [run.run_id for run in runs] == [RUNNING, FINISHED]
    RunsResponse(schema_version="1.0", runs=runs)


def test_runs_started_in_the_same_millisecond_have_a_stable_order():
    """Empate en start_time: se desempata por run_id, igual que la pantalla."""
    first = make_run("1" * 32)
    second = make_run("2" * 32)

    for runs in ([first, second], [second, first]):
        assert [r.run_id for r in list_runs(FakeMlflow(runs))] == ["1" * 32, "2" * 32]


def test_sub_second_start_times_keep_newest_first():
    older = make_run("1" * 32)
    newer = make_run("2" * 32)
    newer.info._start_time = START_MS + 500

    runs = list_runs(FakeMlflow([older, newer]))

    assert [r.run_id for r in runs] == ["2" * 32, "1" * 32]
    assert runs[0].start_time == "2026-09-21T14:13:20.500000Z"
    assert runs[1].start_time == "2026-09-21T14:13:20Z"


# --- Curvas -------------------------------------------------------------------------


def test_curves_come_from_the_metric_history_ordered_by_step():
    history = {
        (FINISHED, "val_accuracy"): [(2, 0.78), (1, 0.62), (3, 0.85)],
        (FINISHED, "train_loss"): [(1, 0.64), (2, 0.42), (3, 0.31)],
    }
    run = make_run(FINISHED, metrics={"val_accuracy": (0.85, 3), "train_loss": (0.31, 3)})

    curves = run_curves(FakeMlflow([run], history), FINISHED)

    assert curves.run_id == FINISHED
    assert [(p.step, p.value) for p in curves.curves["val_accuracy"]] == [
        (1, 0.62),
        (2, 0.78),
        (3, 0.85),
    ]
    assert set(curves.curves) == {"val_accuracy", "train_loss"}


def test_a_nan_epoch_is_a_null_point_in_its_curve():
    """Revisión de #41: un NaN en una época intermedia hacía fallar la consulta de curvas."""
    history = {(FINISHED, "val_loss"): [(1, 0.70), (2, math.nan), (3, 0.52)]}
    run = make_run(FINISHED, metrics={"val_loss": (0.52, 3)})

    points = run_curves(FakeMlflow([run], history), FINISHED).curves["val_loss"]

    assert [(p.step, p.value) for p in points] == [(1, 0.70), (2, None), (3, 0.52)]


def test_a_step_logged_twice_keeps_the_last_value():
    history = {(FINISHED, "val_loss"): [(1, 0.70), (2, 0.50), (2, 0.48)]}
    run = make_run(FINISHED, metrics={"val_loss": (0.48, 2)})

    points = run_curves(FakeMlflow([run], history), FINISHED).curves["val_loss"]

    assert [(p.step, p.value) for p in points] == [(1, 0.70), (2, 0.48)]


def test_curves_of_unknown_or_non_training_runs_are_none():
    client = FakeMlflow([make_run(NOT_TRAINING, tags={})])
    assert run_curves(client, "ffffffffffffffffffffffffffffffff") is None
    assert run_curves(client, NOT_TRAINING) is None


def test_the_agreed_curve_metrics_are_train_and_validation_loss_and_accuracy():
    assert set(CURVE_METRICS) == {"train_loss", "val_loss", "train_accuracy", "val_accuracy"}


# --- Endpoints ----------------------------------------------------------------------


def client_with(tmp_path, tracking) -> TestClient:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "versions.json").write_text(json.dumps({"releases": []}), encoding="utf-8")
    queue = TrainingJobQueue(create_engine(f"sqlite:///{tmp_path / 'jobs.db'}"))
    queue.create_tables()
    return TestClient(create_app(queue=queue, reports_dir=reports, tracking=tracking))


def test_get_runs_serves_the_contract(tmp_path):
    client = client_with(tmp_path, FakeMlflow([make_run(FINISHED)]))

    response = client.get("/runs")

    assert response.status_code == 200
    assert [r.run_id for r in RunsResponse.model_validate(response.json()).runs] == [FINISHED]


def test_get_curves_serves_the_contract(tmp_path):
    history = {(FINISHED, "val_accuracy"): [(1, 0.6), (2, 0.8)]}
    run = make_run(FINISHED, metrics={"val_accuracy": (0.8, 2)})
    client = client_with(tmp_path, FakeMlflow([run], history))

    response = client.get(f"/runs/{FINISHED}/curves")

    assert response.status_code == 200
    curves = RunCurvesResponse.model_validate(response.json())
    assert [p.value for p in curves.curves["val_accuracy"]] == [0.6, 0.8]


def reject_non_json(value):
    raise ValueError(f"no es JSON estándar: {value}")


def test_api_serves_non_finite_values_as_json_null(tmp_path):
    history = {(FINISHED, "val_loss"): [(1, 0.7), (2, math.nan), (3, math.nan)]}
    run = make_run(FINISHED, metrics={"val_loss": (math.nan, 3), "val_accuracy": (0.8, 3)})
    client = client_with(tmp_path, FakeMlflow([run], history))

    runs = client.get("/runs")
    curves = client.get(f"/runs/{FINISHED}/curves")

    assert runs.status_code == 200 and curves.status_code == 200
    listed = json.loads(runs.text, parse_constant=reject_non_json)["runs"][0]
    assert listed["metrics"] == {"val_loss": None, "val_accuracy": 0.8}
    points = json.loads(curves.text, parse_constant=reject_non_json)["curves"]["val_loss"]
    assert [p["value"] for p in points] == [0.7, None, None]


def test_unknown_run_curves_are_404(tmp_path):
    client = client_with(tmp_path, FakeMlflow([]))
    response = client.get(f"/runs/{FINISHED}/curves")
    assert response.status_code == 404
    assert ErrorResponse.model_validate(response.json()).error.code == "run_not_found"


def test_malformed_run_id_is_400(tmp_path):
    response = client_with(tmp_path, FakeMlflow([])).get("/runs/not-a-run/curves")
    assert response.status_code == 400


@pytest.mark.parametrize("path", ["/runs", f"/runs/{FINISHED}/curves"])
def test_mlflow_down_is_a_retryable_503(tmp_path, path):
    response = client_with(tmp_path, FakeMlflow([], fail=True)).get(path)

    assert response.status_code == 503
    error = ErrorResponse.model_validate(response.json()).error
    assert (error.code, error.retryable) == ("mlflow_unavailable", True)


@pytest.mark.parametrize("path", ["/runs", f"/runs/{FINISHED}/curves"])
def test_service_without_mlflow_configuration_says_so(tmp_path, path):
    response = client_with(tmp_path, None).get(path)

    assert response.status_code == 503
    assert ErrorResponse.model_validate(response.json()).error.code == "mlflow_not_configured"
