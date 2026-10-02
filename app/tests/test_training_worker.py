"""OPS-04 — worker asíncrono de entrenamiento."""

import contextlib
import json
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import create_engine

from presentation.ml_contracts import TrainingJobRequest
from training.queue import TrainingJobQueue
from training.worker import process_one

EXAMPLES = Path(__file__).resolve().parents[1] / "presentation" / "examples" / "ml"
RUN_ID = "0a1b2c3d4e5f60718293a4b5c6d7e8f9"


def request() -> TrainingJobRequest:
    document = json.loads((EXAMPLES / "training_request.json").read_text(encoding="utf-8"))
    return TrainingJobRequest.model_validate(document)


def test_worker_executes_one_job_and_persists_success(tmp_path):
    queue = TrainingJobQueue(create_engine(f"sqlite:///{tmp_path / 'worker.db'}"))
    queue.create_tables()
    job = queue.enqueue(request())

    def fake_trainer(
        params,
        *,
        dataset_version,
        manifest_hash,
        data,
        client,
        hooks,
        **kwargs,
    ):
        assert params == job.params
        assert dataset_version == job.dataset_version
        assert manifest_hash == job.manifest_hash

        hooks.on_run_started("1", RUN_ID)
        hooks.log("Entrenamiento iniciado.")
        hooks.on_epoch_end(1, {"val_accuracy": 0.8})

        return SimpleNamespace(checkpoint_uri=f"runs:/{RUN_ID}/checkpoints/best.pt")

    processed = process_one(
        queue,
        client=object(),
        trainer=fake_trainer,
        worker_id="worker-test",
    )

    assert processed is True

    persisted = queue.get(job.job_id)
    assert persisted is not None
    assert persisted.status == "succeeded"
    assert persisted.run_id == RUN_ID
    assert persisted.checkpoint == f"runs:/{RUN_ID}/checkpoints/best.pt"
    assert persisted.progress.epoch == 1
    assert [entry.message for entry in queue.logs(job.job_id)] == ["Entrenamiento iniciado."]


def test_worker_persists_training_failure(tmp_path):
    queue = TrainingJobQueue(create_engine(f"sqlite:///{tmp_path / 'worker-failure.db'}"))
    queue.create_tables()
    job = queue.enqueue(request())

    def failing_trainer(*args, **kwargs):
        raise RuntimeError("trainer exploded")

    processed = process_one(
        queue,
        client=object(),
        trainer=failing_trainer,
        worker_id="worker-test",
    )

    assert processed is True

    persisted = queue.get(job.job_id)
    assert persisted is not None
    assert persisted.status == "failed"
    assert persisted.finished_at is not None
    assert persisted.error is not None
    assert persisted.error.code == "training_error"
    assert persisted.error.message == "trainer exploded"
    assert persisted.error.retryable is False


def test_worker_recovers_abandoned_running_job(tmp_path):
    from training.worker import recover_abandoned_jobs

    engine = create_engine(f"sqlite:///{tmp_path / 'worker-recovery.db'}")
    queue = TrainingJobQueue(engine)
    queue.create_tables()

    job = queue.enqueue(request())
    queue.claim_next("dead-worker")
    queue.start(job.job_id, experiment_id="1", run_id=RUN_ID)

    terminated = []
    client = SimpleNamespace(
        get_run=lambda run_id: SimpleNamespace(info=SimpleNamespace(status="RUNNING")),
        set_terminated=lambda run_id, status: terminated.append((run_id, status)),
    )

    recovered = recover_abandoned_jobs(
        queue,
        client=client,
        worker_id="dead-worker",
    )

    assert recovered == 1
    assert terminated == [(RUN_ID, "KILLED")]

    persisted = queue.get(job.job_id)
    assert persisted is not None
    assert persisted.status == "failed"
    assert persisted.error is not None
    assert persisted.error.code == "worker_interrupted"
    assert persisted.error.retryable is True


def test_worker_recovers_abandoned_claimed_queued_job(tmp_path):
    from training.worker import recover_abandoned_jobs

    engine = create_engine(f"sqlite:///{tmp_path / 'worker-queued-recovery.db'}")
    queue = TrainingJobQueue(engine)
    queue.create_tables()

    job = queue.enqueue(request())
    claimed = queue.claim_next("dead-worker")
    assert claimed is not None

    def unexpected_termination(*args, **kwargs):
        raise AssertionError("Un job queued sin run_id no debe cerrar ningun run de MLflow")

    recovered = recover_abandoned_jobs(
        queue,
        client=SimpleNamespace(
            get_run=unexpected_termination,
            set_terminated=unexpected_termination,
        ),
        worker_id="dead-worker",
    )

    assert recovered == 1

    persisted = queue.get(job.job_id)
    assert persisted is not None
    assert persisted.status == "failed"
    assert persisted.run_id is None
    assert persisted.error is not None
    assert persisted.error.code == "worker_interrupted"
    assert persisted.error.retryable is True


def test_worker_loop_processes_jobs_and_sleeps_when_idle(tmp_path):
    from training.worker import run_worker

    queue = TrainingJobQueue(create_engine(f"sqlite:///{tmp_path / 'worker-loop.db'}"))
    queue.create_tables()
    job = queue.enqueue(request())

    calls = []

    def fake_trainer(
        params,
        *,
        dataset_version,
        manifest_hash,
        data,
        client,
        hooks,
        **kwargs,
    ):
        hooks.on_run_started("1", RUN_ID)
        return SimpleNamespace(checkpoint_uri=f"runs:/{RUN_ID}/checkpoints/best.pt")

    def fake_sleep(seconds):
        calls.append(seconds)
        raise KeyboardInterrupt

    with contextlib.suppress(KeyboardInterrupt):
        run_worker(
            queue,
            client=object(),
            trainer=fake_trainer,
            worker_id="worker-loop-test",
            poll_interval_s=0.25,
            sleep=fake_sleep,
        )

    persisted = queue.get(job.job_id)
    assert persisted is not None
    assert persisted.status == "succeeded"
    assert calls == [0.25]


def test_main_wires_queue_tracking_recovery_and_loop(monkeypatch):
    import training.worker as worker

    calls = []

    class FakeQueue:
        def create_tables(self):
            calls.append("create_tables")

        def ping(self):
            calls.append("ping")

    fake_queue = FakeQueue()

    monkeypatch.setenv("TRAINING_WORKER_ID", "worker-main-test")
    monkeypatch.setattr(worker, "TrainingJobQueue", lambda engine: fake_queue)

    monkeypatch.setattr(
        "storage.db.get_engine",
        lambda: object(),
    )
    monkeypatch.setattr(
        "tracking.client.check_server",
        lambda: calls.append("check_server"),
    )
    monkeypatch.setattr(
        "tracking.client.tracking_client",
        lambda: "mlflow-client",
    )
    monkeypatch.setattr(
        worker,
        "recover_abandoned_jobs",
        lambda queue, *, client, worker_id: calls.append(("recover", queue, client, worker_id)),
    )
    monkeypatch.setattr(
        worker,
        "run_worker",
        lambda queue, *, client, worker_id: calls.append(("run_worker", queue, client, worker_id)),
    )

    worker.main()

    assert calls == [
        "create_tables",
        "ping",
        "check_server",
        ("recover", fake_queue, "mlflow-client", "worker-main-test"),
        ("run_worker", fake_queue, "mlflow-client", "worker-main-test"),
    ]
