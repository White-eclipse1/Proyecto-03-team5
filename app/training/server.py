"""APP-03 — API de jobs de entrenamiento, servida en `/api/ml/` por nginx.

    GET  /training/jobs                  TrainingJobsResponse (más reciente primero)
    POST /training/jobs                  TrainingJobRequest → 202 TrainingJob (queued)
    GET  /training/jobs/{job_id}         TrainingJob
    GET  /training/jobs/{job_id}/logs    TrainingLogsResponse (?after=<seq>)
    GET  /health

El POST valida el contrato y las reglas del release (`training_request_rejection`) y
solo inserta el job en la cola de MariaDB: el entrenamiento lo corre el worker de
OPS-04 en otro proceso, nunca dentro del request. Los errores siguen
`ErrorResponse`. Un release desconocido responde 422 (no 404), porque el portal
interpreta un 404 del POST como "servicio no conectado".

Desde `app/`:

    uv run python -m training.server
"""

import json
import logging
from pathlib import Path

import uvicorn
from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from presentation.ml_contracts import (
    ContractError,
    ErrorResponse,
    TrainingJobRequest,
    TrainingJobsResponse,
    TrainingLogsResponse,
    training_request_rejection,
)
from storage.db import get_engine
from storage.settings import Settings
from training.queue import TrainingJobQueue
from training.releases import load_release, published_releases

logger = logging.getLogger("ml-api")

PUBLIC_PREFIX = "/api/ml"


def _error(status_code: int, code: str, message: str, *, retryable: bool = False) -> JSONResponse:
    body = ErrorResponse(
        schema_version="1.0",
        error=ContractError(code=code, message=message, retryable=retryable),
    )
    return JSONResponse(body.model_dump(), status_code=status_code)


def _describe(exc: ValidationError) -> str:
    problems = [
        f"{'.'.join(str(part) for part in error['loc']) or 'body'}: {error['msg']}"
        for error in exc.errors()
    ]
    return "Solicitud inválida: " + "; ".join(problems)


def create_app(*, queue: TrainingJobQueue, reports_dir: Path) -> Starlette:
    async def list_jobs(_request: Request) -> JSONResponse:
        response = TrainingJobsResponse(schema_version="1.0", jobs=queue.list_jobs())
        return JSONResponse(response.model_dump())

    async def create_job(request: Request) -> JSONResponse:
        try:
            document = json.loads(await request.body())
        except ValueError:
            return _error(400, "invalid_json", "El cuerpo no es JSON válido.")
        try:
            job_request = TrainingJobRequest.model_validate(document)
        except ValidationError as exc:
            return _error(422, "invalid_request", _describe(exc))

        version = job_request.dataset_version
        if version not in published_releases(reports_dir):
            return _error(422, "release_not_found", f"El release {version} no está publicado.")
        release = load_release(reports_dir, version)
        if release.quality_problem is not None:
            return _error(409, "training_blocked", release.quality_problem)
        reason = training_request_rejection(
            job_request, release.quality_status, release.provenance, release.manifest
        )
        if reason is not None:
            return _error(409, "training_blocked", reason)

        job = queue.enqueue(job_request)
        logger.info("Job %s en cola para %s", job.job_id, version)
        return JSONResponse(
            job.model_dump(),
            status_code=202,
            headers={"Location": f"{PUBLIC_PREFIX}/training/jobs/{job.job_id}"},
        )

    async def get_job(request: Request) -> JSONResponse:
        job = queue.get(request.path_params["job_id"])
        if job is None:
            return _error(404, "job_not_found", "No existe ese job de entrenamiento.")
        return JSONResponse(job.model_dump())

    async def get_logs(request: Request) -> JSONResponse:
        job_id = request.path_params["job_id"]
        raw_after = request.query_params.get("after", "0")
        if not raw_after.isdigit():
            return _error(400, "invalid_request", "after debe ser un entero >= 0.")
        if queue.get(job_id) is None:
            return _error(404, "job_not_found", "No existe ese job de entrenamiento.")
        response = TrainingLogsResponse(
            schema_version="1.0",
            job_id=job_id,
            entries=queue.logs(job_id, after_seq=int(raw_after)),
        )
        return JSONResponse(response.model_dump())

    async def health(_request: Request) -> JSONResponse:
        try:
            queue.ping()
        except Exception:
            logger.exception("Sin acceso a la base de jobs")
            return JSONResponse({"status": "unavailable"}, status_code=503)
        return JSONResponse({"status": "ok"})

    return Starlette(
        routes=[
            Route("/health", health),
            Route("/training/jobs", list_jobs, methods=["GET"]),
            Route("/training/jobs", create_job, methods=["POST"]),
            Route("/training/jobs/{job_id}", get_job, methods=["GET"]),
            Route("/training/jobs/{job_id}/logs", get_logs, methods=["GET"]),
        ]
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = Settings()
    queue = TrainingJobQueue(get_engine())
    queue.create_tables()
    app = create_app(queue=queue, reports_dir=settings.reports_dir)
    uvicorn.run(app, host=settings.ml_api_host, port=settings.ml_api_port)


if __name__ == "__main__":
    main()
