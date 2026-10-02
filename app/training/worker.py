"""OPS-04 — worker persistente de entrenamiento."""

import time
from collections.abc import Callable
from typing import Any

from classification.training import DataPaths, JobQueueHooks, run_training
from presentation.ml_contracts import ContractError
from training.queue import TrainingJobQueue


def process_one(
    queue: TrainingJobQueue,
    *,
    client: Any,
    trainer: Callable[..., Any] = run_training,
    worker_id: str,
) -> bool:
    """Procesa como máximo un job; devuelve False si no había trabajo."""
    job = queue.claim_next(worker_id)
    if job is None:
        return False

    try:
        result = trainer(
            job.params,
            dataset_version=job.dataset_version,
            manifest_hash=job.manifest_hash,
            data=DataPaths.for_release(job.dataset_version),
            client=client,
            hooks=JobQueueHooks(queue, job.job_id),
        )
        queue.succeed(job.job_id, checkpoint=result.checkpoint_uri)
    except Exception as exc:
        queue.fail(
            job.job_id,
            ContractError(
                code="training_error",
                message=str(exc) or exc.__class__.__name__,
                retryable=False,
            ),
        )

    return True


def recover_abandoned_jobs(
    queue: TrainingJobQueue,
    *,
    client: Any,
    worker_id: str,
) -> int:
    """Cierra runs de MLflow y falla jobs no terminales de un worker anterior."""
    job_ids = queue.claimed_by(worker_id)
    recovered = 0

    for job_id in job_ids:
        job = queue.get(job_id)
        if job is None:
            continue

        if job.run_id is not None:
            run = client.get_run(job.run_id)
            if run.info.status == "RUNNING":
                client.set_terminated(job.run_id, status="KILLED")

        queue.fail(
            job_id,
            ContractError(
                code="worker_interrupted",
                message="El worker anterior se interrumpio antes de terminar el entrenamiento.",
                retryable=True,
            ),
        )
        recovered += 1

    return recovered


def run_worker(
    queue: TrainingJobQueue,
    *,
    client: Any,
    trainer: Callable[..., Any] = run_training,
    worker_id: str,
    poll_interval_s: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Procesa jobs continuamente y espera cuando la cola está vacía."""
    while True:
        processed = process_one(
            queue,
            client=client,
            trainer=trainer,
            worker_id=worker_id,
        )

        if not processed:
            sleep(poll_interval_s)


def main() -> None:
    """Arranca el worker real de OPS-04."""
    import os
    import socket

    from storage.db import get_engine
    from tracking.client import check_server, tracking_client

    queue = TrainingJobQueue(get_engine())
    queue.create_tables()
    queue.ping()

    check_server()
    client = tracking_client()

    worker_id = os.environ.get("TRAINING_WORKER_ID", socket.gethostname())

    recover_abandoned_jobs(
        queue,
        client=client,
        worker_id=worker_id,
    )

    run_worker(
        queue,
        client=client,
        worker_id=worker_id,
    )


if __name__ == "__main__":
    main()
