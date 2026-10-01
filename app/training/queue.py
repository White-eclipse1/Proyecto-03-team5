"""APP-03 — cola persistente de jobs de entrenamiento en MariaDB.

La comparten dos procesos:

- La API del portal (`training/server.py`): `enqueue`, `list_jobs`, `get`, `logs`.
  Crear un job solo inserta una fila `queued`; el entrenamiento nunca corre dentro
  del request HTTP.
- El worker de OPS-04: `claim_next` → `start` → (`report_progress`/`log`)* →
  `succeed` o `fail`. `claim_next` usa `SELECT ... FOR UPDATE SKIP LOCKED`, así que
  dos workers nunca toman el mismo job.

Cada cambio se valida contra `TrainingJob` (ml_contracts.py) antes de guardarse:
una transición que deje un job fuera del contrato se rechaza con
`InvalidTransitionError` y la fila no cambia.
"""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import ValidationError
from sqlalchemy import (
    Boolean,
    Column,
    Engine,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    func,
    select,
)
from sqlalchemy.engine import Connection, RowMapping

from presentation.ml_contracts import (
    ContractError,
    TrainingJob,
    TrainingJobRequest,
    TrainingLogEntry,
)

metadata = MetaData()

jobs_table = Table(
    "ml_training_jobs",
    metadata,
    Column("job_id", String(64), primary_key=True),
    Column("status", String(16), nullable=False, index=True),
    Column("dataset_version", String(128), nullable=False),
    Column("manifest_hash", String(80), nullable=False),
    Column("experiment_id", String(32)),
    Column("run_id", String(32)),
    Column("checkpoint", String(512)),
    Column("params", Text, nullable=False),
    # Timestamps en el formato del contrato (ISO UTC con "Z"): se ordenan como texto.
    Column("created_at", String(32), nullable=False, index=True),
    Column("started_at", String(32)),
    Column("finished_at", String(32)),
    Column("error", Text),
    Column("progress", Text),
    # Uso interno del worker: quién tomó el job (no forma parte del contrato).
    Column("claimed", Boolean, nullable=False, default=False),
    Column("claimed_by", String(128)),
    Column("claimed_at", String(32)),
    mysql_charset="utf8mb4",
)

logs_table = Table(
    "ml_training_job_logs",
    metadata,
    Column("job_id", String(64), ForeignKey("ml_training_jobs.job_id"), primary_key=True),
    Column("seq", Integer, primary_key=True, autoincrement=False),
    Column("timestamp", String(32), nullable=False),
    Column("level", String(16), nullable=False),
    Column("message", Text, nullable=False),
    mysql_charset="utf8mb4",
)


class JobNotFoundError(LookupError):
    """No existe un job con ese `job_id`."""


class InvalidTransitionError(ValueError):
    """El cambio pedido no es válido para el estado actual del job."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _timestamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _dump(value: Any) -> str | None:
    return None if value is None else json.dumps(value, separators=(",", ":"))


def _load(value: str | None) -> Any:
    return None if value is None else json.loads(value)


class TrainingJobQueue:
    def __init__(self, engine: Engine, *, clock: Callable[[], datetime] = _utc_now):
        self._engine = engine
        self._clock = clock

    def create_tables(self) -> None:
        """Idempotente: crea las tablas si no existen (al arrancar API o worker)."""
        metadata.create_all(self._engine)

    def ping(self) -> None:
        with self._engine.connect() as connection:
            connection.execute(select(1))

    # --- API del portal ---------------------------------------------------------

    def enqueue(self, request: TrainingJobRequest) -> TrainingJob:
        job = TrainingJob(
            job_id=f"job-{uuid4().hex}",
            status="queued",
            dataset_version=request.dataset_version,
            manifest_hash=request.manifest_hash,
            experiment_id=None,
            run_id=None,
            checkpoint=None,
            params=request.params,
            created_at=self._now(),
            started_at=None,
            finished_at=None,
            error=None,
            progress=None,
        )
        with self._engine.begin() as connection:
            connection.execute(jobs_table.insert().values(**self._row(job), claimed=False))
        return job

    def list_jobs(self) -> list[TrainingJob]:
        query = select(jobs_table).order_by(
            jobs_table.c.created_at.desc(), jobs_table.c.job_id.desc()
        )
        with self._engine.connect() as connection:
            return [self._job(row) for row in connection.execute(query).mappings()]

    def get(self, job_id: str) -> TrainingJob | None:
        with self._engine.connect() as connection:
            row = self._fetch(connection, job_id)
        return None if row is None else self._job(row)

    def logs(self, job_id: str, *, after_seq: int = 0) -> list[TrainingLogEntry]:
        query = (
            select(logs_table)
            .where(logs_table.c.job_id == job_id, logs_table.c.seq > after_seq)
            .order_by(logs_table.c.seq)
        )
        with self._engine.connect() as connection:
            return [
                TrainingLogEntry(
                    seq=row["seq"],
                    timestamp=row["timestamp"],
                    level=row["level"],
                    message=row["message"],
                )
                for row in connection.execute(query).mappings()
            ]

    # --- Worker (OPS-04) --------------------------------------------------------

    def claim_next(self, worker_id: str) -> TrainingJob | None:
        """Toma el job `queued` más antiguo que nadie haya tomado; `None` si no hay."""
        query = (
            select(jobs_table)
            .where(jobs_table.c.status == "queued", jobs_table.c.claimed.is_(False))
            .order_by(jobs_table.c.created_at, jobs_table.c.job_id)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        with self._engine.begin() as connection:
            row = connection.execute(query).mappings().first()
            if row is None:
                return None
            connection.execute(
                jobs_table.update()
                .where(jobs_table.c.job_id == row["job_id"])
                .values(claimed=True, claimed_by=worker_id, claimed_at=self._now())
            )
            return self._job(row)

    def claimed_by(self, worker_id: str) -> list[str]:
        """IDs no terminales que quedaron reclamados por `worker_id`."""
        query = (
            select(jobs_table.c.job_id)
            .where(
                jobs_table.c.claimed.is_(True),
                jobs_table.c.claimed_by == worker_id,
                jobs_table.c.status.in_(("queued", "running")),
            )
            .order_by(jobs_table.c.claimed_at, jobs_table.c.job_id)
        )
        with self._engine.connect() as connection:
            return list(connection.execute(query).scalars())

    def start(self, job_id: str, *, experiment_id: str, run_id: str) -> TrainingJob:
        def change(job: TrainingJob, row: RowMapping) -> dict:
            self._require(job, "queued", "start")
            if not row["claimed"]:
                raise InvalidTransitionError(f"{job_id}: claim_next debe tomar el job primero")
            return {
                "status": "running",
                "experiment_id": experiment_id,
                "run_id": run_id,
                "started_at": self._now(),
            }

        return self._update(job_id, change)

    def report_progress(self, job_id: str, *, epoch: int, metrics: dict[str, float]) -> TrainingJob:
        def change(job: TrainingJob, _row: RowMapping) -> dict:
            self._require(job, "running", "report_progress")
            return {
                "progress": {
                    "epoch": epoch,
                    "max_epochs": job.params.max_epochs,
                    "metrics": {name: float(value) for name, value in metrics.items()},
                    "updated_at": self._now(),
                }
            }

        return self._update(job_id, change)

    def log(
        self,
        job_id: str,
        message: str,
        *,
        level: Literal["info", "warning", "error"] = "info",
    ) -> TrainingLogEntry:
        entry_time = self._now()
        with self._engine.begin() as connection:
            if self._fetch(connection, job_id, lock=True) is None:
                raise JobNotFoundError(job_id)
            last = connection.execute(
                select(func.max(logs_table.c.seq)).where(logs_table.c.job_id == job_id)
            ).scalar()
            entry = TrainingLogEntry(
                seq=(last or 0) + 1, timestamp=entry_time, level=level, message=message
            )
            connection.execute(logs_table.insert().values(job_id=job_id, **entry.model_dump()))
        return entry

    def succeed(self, job_id: str, *, checkpoint: str) -> TrainingJob:
        def change(job: TrainingJob, _row: RowMapping) -> dict:
            self._require(job, "running", "succeed")
            return {"status": "succeeded", "checkpoint": checkpoint, "finished_at": self._now()}

        return self._update(job_id, change)

    def fail(self, job_id: str, error: ContractError) -> TrainingJob:
        def change(job: TrainingJob, _row: RowMapping) -> dict:
            self._require(job, ("queued", "running"), "fail")
            return {"status": "failed", "error": error.model_dump(), "finished_at": self._now()}

        return self._update(job_id, change)

    # --- Internos ---------------------------------------------------------------

    def _now(self) -> str:
        return _timestamp(self._clock())

    @staticmethod
    def _require(job: TrainingJob, allowed: str | tuple[str, ...], action: str) -> None:
        allowed = (allowed,) if isinstance(allowed, str) else allowed
        if job.status not in allowed:
            raise InvalidTransitionError(f"{job.job_id}: no se puede {action} un job {job.status}")

    @staticmethod
    def _fetch(connection: Connection, job_id: str, *, lock: bool = False) -> RowMapping | None:
        query = select(jobs_table).where(jobs_table.c.job_id == job_id)
        if lock:
            query = query.with_for_update()
        return connection.execute(query).mappings().first()

    def _update(
        self, job_id: str, change: Callable[[TrainingJob, RowMapping], dict]
    ) -> TrainingJob:
        with self._engine.begin() as connection:
            row = self._fetch(connection, job_id, lock=True)
            if row is None:
                raise JobNotFoundError(job_id)
            current = self._job(row)
            document = current.model_dump() | change(current, row)
            try:
                updated = TrainingJob.model_validate(document)
            except ValidationError as exc:
                raise InvalidTransitionError(f"{job_id}: {exc}") from exc
            connection.execute(
                jobs_table.update()
                .where(jobs_table.c.job_id == job_id)
                .values(**self._row(updated))
            )
        return updated

    @staticmethod
    def _row(job: TrainingJob) -> dict:
        document = job.model_dump()
        for key in ("params", "error", "progress"):
            document[key] = _dump(document[key])
        return document

    @staticmethod
    def _job(row: RowMapping) -> TrainingJob:
        return TrainingJob.model_validate(
            {
                "job_id": row["job_id"],
                "status": row["status"],
                "dataset_version": row["dataset_version"],
                "manifest_hash": row["manifest_hash"],
                "experiment_id": row["experiment_id"],
                "run_id": row["run_id"],
                "checkpoint": row["checkpoint"],
                "params": _load(row["params"]),
                "created_at": row["created_at"],
                "started_at": row["started_at"],
                "finished_at": row["finished_at"],
                "error": _load(row["error"]),
                "progress": _load(row["progress"]),
            }
        )
