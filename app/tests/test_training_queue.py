"""APP-03: cola persistente de jobs de entrenamiento (`training/queue.py`).

La usan la API del portal (crear y consultar jobs) y el worker de OPS-04 (tomar un
job, reportar progreso/logs y cerrarlo). Corre sobre SQLite; con
`TRAINING_QUEUE_DATABASE_URL` apuntando a MariaDB, las mismas pruebas validan el
motor real (así lo hace el job de CI de APP-03).
"""

import json
import math
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from presentation.ml_contracts import (
    ContractError,
    TrainingJob,
    TrainingJobRequest,
    TrainingJobsResponse,
)
from training.queue import InvalidTransitionError, JobNotFoundError, TrainingJobQueue

EXAMPLES = Path(__file__).resolve().parents[1] / "presentation" / "examples" / "ml"
RUN_ID = "0a1b2c3d4e5f60718293a4b5c6d7e8f9"


def request(**changes) -> TrainingJobRequest:
    document = json.loads((EXAMPLES / "training_request.json").read_text(encoding="utf-8"))
    document["params"].update(changes)
    return TrainingJobRequest.model_validate(document)


class Clock:
    def __init__(self):
        self.now = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


@pytest.fixture
def database_url(tmp_path):
    external = os.environ.get("TRAINING_QUEUE_DATABASE_URL")
    if not external:
        yield f"sqlite:///{tmp_path / 'queue.db'}"
        return
    engine = create_engine(external)
    with engine.begin() as connection:
        connection.execute(text("DROP TABLE IF EXISTS ml_training_job_logs"))
        connection.execute(text("DROP TABLE IF EXISTS ml_training_jobs"))
    engine.dispose()
    yield external


@pytest.fixture
def queue(database_url):
    queue = TrainingJobQueue(create_engine(database_url), clock=Clock())
    queue.create_tables()
    return queue


def started(queue: TrainingJobQueue) -> TrainingJob:
    job = queue.enqueue(request())
    claimed = queue.claim_next("worker-1")
    assert claimed is not None and claimed.job_id == job.job_id
    return queue.start(job.job_id, experiment_id="1", run_id=RUN_ID)


# --- Creación y consulta (lado de la API) -----------------------------------------


def test_enqueued_job_is_queued_and_valid_for_the_contract(queue):
    job = queue.enqueue(request())

    assert job.status == "queued"
    assert job.experiment_id is None and job.run_id is None and job.progress is None
    assert job.started_at is None and job.finished_at is None
    assert job.params == request().params
    assert job.created_at == "2026-10-01T12:00:01Z"
    TrainingJobsResponse(schema_version="1.0", jobs=queue.list_jobs())


def test_jobs_are_listed_newest_first_and_found_by_id(queue):
    first = queue.enqueue(request())
    second = queue.enqueue(request(batch_size=64))

    assert [job.job_id for job in queue.list_jobs()] == [second.job_id, first.job_id]
    assert queue.get(first.job_id) == first
    assert queue.get("job-does-not-exist") is None


def test_jobs_survive_a_new_connection(database_url):
    """Refrescar el portal = otra conexión a la misma base: el job sigue ahí."""
    original = TrainingJobQueue(create_engine(database_url), clock=Clock())
    original.create_tables()
    job = original.enqueue(request())

    reopened = TrainingJobQueue(create_engine(database_url))
    reopened.create_tables()
    assert reopened.get(job.job_id) == job


# --- Ciclo de vida (lado del worker, OPS-04) ---------------------------------------


def test_worker_claims_the_oldest_queued_job_once(queue):
    first = queue.enqueue(request())
    second = queue.enqueue(request())

    assert queue.claim_next("worker-1").job_id == first.job_id
    assert queue.claim_next("worker-2").job_id == second.job_id
    assert queue.claim_next("worker-3") is None


def test_full_successful_lifecycle(queue):
    job = started(queue)
    assert job.status == "running"
    assert (job.experiment_id, job.run_id) == ("1", RUN_ID)
    assert job.started_at is not None

    queue.report_progress(job.job_id, epoch=1, metrics={"train_loss": 0.6, "val_loss": 0.65})
    progress = queue.report_progress(job.job_id, epoch=2, metrics={"val_accuracy": 0.8}).progress
    assert (progress.epoch, progress.max_epochs) == (2, job.params.max_epochs)
    assert progress.metrics == {"val_accuracy": 0.8}

    done = queue.succeed(job.job_id, checkpoint=f"runs:/{RUN_ID}/checkpoints/best.pt")
    assert done.status == "succeeded"
    assert done.finished_at is not None
    assert done.progress.epoch == 2


def test_failure_is_persisted_with_its_error(queue):
    job = started(queue)
    error = ContractError(code="out_of_memory", message="Sin memoria.", retryable=True)

    failed = queue.fail(job.job_id, error)

    assert failed.status == "failed"
    assert failed.error == error
    assert failed.finished_at is not None
    assert queue.get(job.job_id).error == error


def test_a_job_can_fail_before_the_run_exists(queue):
    job = queue.enqueue(request())
    queue.claim_next("worker-1")
    error = ContractError(code="manifest_unavailable", message="No hay recortes.", retryable=False)

    failed = queue.fail(job.job_id, error)

    assert failed.status == "failed" and failed.run_id is None


@pytest.mark.parametrize(
    "action",
    [
        lambda q, job_id: q.report_progress(job_id, epoch=1, metrics={}),
        lambda q, job_id: q.succeed(job_id, checkpoint=f"runs:/{RUN_ID}/best.pt"),
        lambda q, job_id: q.start(job_id, experiment_id="1", run_id=RUN_ID),
    ],
    ids=["progress", "succeed", "start-unclaimed"],
)
def test_illegal_transitions_from_queued_are_rejected(queue, action):
    job = queue.enqueue(request())
    with pytest.raises(InvalidTransitionError):
        action(queue, job.job_id)
    assert queue.get(job.job_id).status == "queued"


def test_terminal_jobs_cannot_change(queue):
    job = started(queue)
    queue.succeed(job.job_id, checkpoint=f"runs:/{RUN_ID}/best.pt")
    error = ContractError(code="late", message="Tarde.", retryable=False)

    with pytest.raises(InvalidTransitionError):
        queue.fail(job.job_id, error)
    with pytest.raises(InvalidTransitionError):
        queue.report_progress(job.job_id, epoch=3, metrics={})


@pytest.mark.parametrize(
    "value",
    [math.nan, math.inf, -math.inf, sys.float_info.max],
    ids=["nan", "inf", "-inf", "sql-clamped-inf"],
)
def test_non_finite_progress_metrics_are_stored_as_null_and_the_job_keeps_running(queue, value):
    """Un val_loss=NaN en una época no debe hacer fallar el job por la validación de la cola.

    El loop de ML-04 llama a report_progress dentro de su try: si la cola lanzara, el run
    terminaría FAILED por nuestra validación y no por el entrenamiento.
    """
    job = started(queue)

    progress = queue.report_progress(
        job.job_id, epoch=2, metrics={"train_loss": 0.41, "val_loss": value}
    ).progress

    assert progress.metrics == {"train_loss": 0.41, "val_loss": None}
    stored = queue.get(job.job_id)
    assert stored.status == "running"
    assert stored.progress.metrics["val_loss"] is None


def test_progress_beyond_max_epochs_is_rejected(queue):
    job = started(queue)
    with pytest.raises(InvalidTransitionError, match="max_epochs"):
        queue.report_progress(job.job_id, epoch=job.params.max_epochs + 1, metrics={})


def test_checkpoint_must_belong_to_the_run(queue):
    job = started(queue)
    with pytest.raises(InvalidTransitionError):
        queue.succeed(job.job_id, checkpoint="runs:/ffffffffffffffffffffffffffffffff/best.pt")
    assert queue.get(job.job_id).status == "running"


def test_unknown_job_is_reported(queue):
    with pytest.raises(JobNotFoundError):
        queue.start("job-missing", experiment_id="1", run_id=RUN_ID)


# --- Logs ---------------------------------------------------------------------------


def test_logs_are_ordered_and_paginated_by_seq(queue):
    job = started(queue)
    queue.log(job.job_id, "Run creado.")
    queue.log(job.job_id, "Época 1/50.")
    queue.log(job.job_id, "val_loss no mejoró.", level="warning")

    entries = queue.logs(job.job_id)
    assert [(e.seq, e.level, e.message) for e in entries] == [
        (1, "info", "Run creado."),
        (2, "info", "Época 1/50."),
        (3, "warning", "val_loss no mejoró."),
    ]
    assert [e.seq for e in queue.logs(job.job_id, after_seq=1)] == [2, 3]


def test_logs_of_another_job_are_not_mixed(queue):
    job = started(queue)
    other = queue.enqueue(request())
    queue.log(job.job_id, "del primero")
    queue.log(other.job_id, "del segundo")

    assert [e.message for e in queue.logs(other.job_id)] == ["del segundo"]


def test_empty_log_lines_are_rejected(queue):
    job = started(queue)
    with pytest.raises(ValueError):
        queue.log(job.job_id, "")


# --- Concurrencia (solo MariaDB: SQLite no implementa SKIP LOCKED) -----------------


@pytest.mark.skipif(
    not os.environ.get("TRAINING_QUEUE_DATABASE_URL"),
    reason="Requiere MariaDB (TRAINING_QUEUE_DATABASE_URL)",
)
def test_claim_skips_a_job_another_worker_is_claiming(queue, database_url):
    """Mientras otro worker tiene la fila bloqueada, claim_next la salta sin esperar."""
    busy = queue.enqueue(request())
    free = queue.enqueue(request())
    other_worker = create_engine(database_url)

    with other_worker.begin() as connection:
        connection.execute(
            text("SELECT job_id FROM ml_training_jobs WHERE job_id = :job_id FOR UPDATE"),
            {"job_id": busy.job_id},
        )
        claimed = queue.claim_next("worker-2")

    assert claimed is not None and claimed.job_id == free.job_id
    other_worker.dispose()
