"""APP-03 — API de jobs de entrenamiento, servida en `/api/ml/` por nginx.

    GET  /training/jobs                  TrainingJobsResponse (más reciente primero)
    POST /training/jobs                  TrainingJobRequest → 202 TrainingJob (queued)
    GET  /training/jobs/{job_id}         TrainingJob
    GET  /training/jobs/{job_id}/logs    TrainingLogsResponse (?after=<seq>)
    GET  /runs                           RunsResponse: runs de MLflow (APP-04)
    GET  /runs/{run_id}/curves           RunCurvesResponse: historial por época (APP-04)
    GET  /evaluation                     EvaluationOverview: candidato y test (APP-05)
    GET  /evaluation/predictions.csv     predicciones por recorte de ML-09 (APP-05)
    GET  /crops/{crop_id}                imagen del recorte (APP-05)
    POST /inference                      InferenceRequest (recorte) → InferenceResponse (APP-07)
    POST /inference/upload               multipart model_name, model_version, file → ídem
    GET  /health

El POST valida el contrato y las reglas del release (`training_request_rejection`) y
solo inserta el job en la cola de MariaDB: el entrenamiento lo corre el worker de
OPS-04 en otro proceso, nunca dentro del request. Los errores siguen
`ErrorResponse`. Un release desconocido responde 422 (no 404), porque el portal
interpreta un 404 del POST como "servicio no conectado".

Los runs se leen de MLflow en cada petición (sin caché). Si MLflow no responde, 503
`mlflow_unavailable` (reintentable); si el servicio no tiene `MLFLOW_TRACKING_URI`,
503 `mlflow_not_configured`.

Desde `app/`:

    uv run python -m training.server
"""

import json
import logging
import re
import time
from pathlib import Path
from uuid import uuid4

import uvicorn
from mlflow.exceptions import MlflowException
from PIL import Image
from pydantic import BaseModel, ConfigDict, ValidationError
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from presentation.ml_contracts import (
    ContractError,
    CropSelection,
    ErrorResponse,
    Identifier,
    InferenceRequest,
    InferenceResponse,
    ModelVersion,
    RunsResponse,
    TrainingJobRequest,
    TrainingJobsResponse,
    TrainingLogsResponse,
    UploadedImage,
    training_request_rejection,
)
from storage.db import get_engine
from storage.settings import Settings, TrackingSettings
from tracking.client import tracking_client
from training.evaluation_view import crop_path, load_evaluation_view
from training.experiments import TrackingClient, list_runs, run_curves
from training.inference import (
    MAX_IMAGE_PIXELS,
    MAX_UPLOAD_BYTES,
    InferenceRejected,
    ModelResolver,
    PackageRegistryResolver,
    classify,
    read_crop,
    read_upload,
)
from training.queue import TrainingJobQueue
from training.releases import load_release, published_releases

logger = logging.getLogger("ml-api")

PUBLIC_PREFIX = "/api/ml"
RUN_ID = re.compile(r"^[0-9a-f]{32}$")
CROP_ID = re.compile(r"^img[0-9]+-ann[0-9]+$")


def _error(status_code: int, code: str, message: str, *, retryable: bool = False) -> JSONResponse:
    body = ErrorResponse(
        schema_version="1.0",
        error=ContractError(code=code, message=message, retryable=retryable),
    )
    return JSONResponse(body.model_dump(), status_code=status_code)


class _UploadFields(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model_name: Identifier
    model_version: ModelVersion


def _describe(exc: ValidationError) -> str:
    problems = [
        f"{'.'.join(str(part) for part in error['loc']) or 'body'}: {error['msg']}"
        for error in exc.errors()
    ]
    return "Solicitud inválida: " + "; ".join(problems)


def create_app(
    *,
    queue: TrainingJobQueue,
    reports_dir: Path,
    tracking: TrackingClient | None = None,
    crops_dir: Path | None = None,
    models: ModelResolver | None = None,
    max_upload_bytes: int = MAX_UPLOAD_BYTES,
    max_image_pixels: int = MAX_IMAGE_PIXELS,
) -> Starlette:
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

    def mlflow_error() -> JSONResponse | None:
        if tracking is None:
            return _error(
                503,
                "mlflow_not_configured",
                "El servicio no tiene MLFLOW_TRACKING_URI: no puede leer los runs.",
            )
        return None

    def mlflow_unavailable() -> JSONResponse:
        logger.exception("MLflow no respondió")
        return _error(503, "mlflow_unavailable", "MLflow no respondió.", retryable=True)

    async def get_runs(_request: Request) -> JSONResponse:
        if (error := mlflow_error()) is not None:
            return error
        try:
            runs = list_runs(tracking)
        except (MlflowException, OSError):
            return mlflow_unavailable()
        return JSONResponse(RunsResponse(schema_version="1.0", runs=runs).model_dump())

    async def get_run_curves(request: Request) -> JSONResponse:
        run_id = request.path_params["run_id"]
        if not RUN_ID.fullmatch(run_id):
            return _error(400, "invalid_request", "run_id son 32 hex en minúsculas.")
        if (error := mlflow_error()) is not None:
            return error
        try:
            curves = run_curves(tracking, run_id)
        except (MlflowException, OSError):
            return mlflow_unavailable()
        if curves is None:
            return _error(404, "run_not_found", "No existe ese run de entrenamiento en MLflow.")
        return JSONResponse(curves.model_dump())

    async def get_evaluation(_request: Request) -> JSONResponse:
        return JSONResponse(load_evaluation_view(reports_dir).overview.model_dump())

    async def get_predictions_csv(_request: Request):
        view = load_evaluation_view(reports_dir)
        if view.overview.state != "evaluated" or view.predictions_csv is None:
            return _error(
                404,
                "predictions_not_available",
                "Las predicciones de test se publican solo con la evaluación final del "
                "candidato congelado.",
            )
        return FileResponse(
            view.predictions_csv,
            media_type="text/csv",
            filename=view.predictions_csv.name,
        )

    async def get_crop(request: Request):
        crop_id = request.path_params["crop_id"]
        if not CROP_ID.fullmatch(crop_id):
            return _error(400, "invalid_request", "crop_id tiene la forma img<id>-ann<id>.")
        if crops_dir is None:
            return _error(503, "crops_not_configured", "El servicio no tiene CROPS_DIR.")
        path = crop_path(reports_dir, crops_dir, crop_id)
        if path is None:
            return _error(404, "crop_not_found", "No existe ese recorte.")
        return FileResponse(path, media_type="image/png")

    def infer(
        model_name: str,
        model_version: str,
        image: Image.Image,
        *,
        crop: CropSelection | None = None,
        upload: UploadedImage | None = None,
    ) -> InferenceResponse:
        if models is None:
            raise InferenceRejected(
                503, "inference_not_configured", "El servicio no tiene un Model Registry."
            )
        loaded = models.load(model_name, model_version)
        started = time.perf_counter()
        predicted, probabilities = classify(loaded.model, image)
        latency_ms = (time.perf_counter() - started) * 1000
        return InferenceResponse(
            schema_version="1.0",
            request_id=f"inf-{uuid4().hex[:16]}",
            model_name=loaded.model_name,
            model_version=loaded.model_version,
            run_id=loaded.run_id,
            checkpoint=loaded.checkpoint,
            checkpoint_sha256=loaded.checkpoint_sha256,
            dataset_version=loaded.dataset_version,
            image_size=loaded.model.config.image_size,
            source="crop" if crop is not None else "upload",
            crop=crop,
            upload=upload,
            predicted_class=predicted,
            probabilities=probabilities,
            latency_ms=latency_ms,
        )

    def rejected(exc: InferenceRejected) -> JSONResponse:
        return _error(exc.status, exc.code, exc.message, retryable=exc.retryable)

    async def infer_crop(request: Request) -> JSONResponse:
        try:
            document = json.loads(await request.body())
        except ValueError:
            return _error(400, "invalid_json", "El cuerpo no es JSON válido.")
        try:
            body = InferenceRequest.model_validate(document)
        except ValidationError as exc:
            return _error(422, "invalid_request", _describe(exc))

        def run() -> InferenceResponse:
            image = read_crop(body.crop, reports_dir, crops_dir)
            return infer(body.model_name, body.model_version, image, crop=body.crop)

        try:
            response = await run_in_threadpool(run)
        except InferenceRejected as exc:
            return rejected(exc)
        return JSONResponse(response.model_dump())

    async def infer_upload(request: Request) -> JSONResponse:
        declared = request.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > max_upload_bytes + 65536:
            return _error(
                413,
                "image_too_large",
                f"El archivo supera el máximo de {max_upload_bytes:,} bytes.",
            )
        async with request.form(max_files=1, max_fields=10) as form:
            try:
                fields = _UploadFields.model_validate(dict(form))
            except ValidationError as exc:
                return _error(422, "invalid_request", _describe(exc))
            file = form.get("file")
            if file is None or isinstance(file, str):
                return _error(422, "invalid_request", "Falta el archivo de imagen (campo file).")
            data = await file.read(max_upload_bytes + 1)
            filename = file.filename

        def run() -> InferenceResponse:
            uploaded, image = read_upload(
                filename, data, max_bytes=max_upload_bytes, max_pixels=max_image_pixels
            )
            return infer(fields.model_name, fields.model_version, image, upload=uploaded)

        try:
            response = await run_in_threadpool(run)
        except InferenceRejected as exc:
            return rejected(exc)
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
            Route("/runs", get_runs, methods=["GET"]),
            Route("/runs/{run_id}/curves", get_run_curves, methods=["GET"]),
            Route("/evaluation", get_evaluation, methods=["GET"]),
            Route("/evaluation/predictions.csv", get_predictions_csv, methods=["GET"]),
            Route("/crops/{crop_id:path}", get_crop, methods=["GET"]),
            Route("/inference", infer_crop, methods=["POST"]),
            Route("/inference/upload", infer_upload, methods=["POST"]),
        ]
    )


def _tracking_or_none() -> TrackingClient | None:
    """Sin MLFLOW_TRACKING_URI la API de jobs sigue funcionando; /runs responde 503."""
    try:
        return tracking_client(TrackingSettings())
    except ValidationError:
        logger.warning("Sin MLFLOW_TRACKING_URI válido: /runs responderá mlflow_not_configured")
        return None


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = Settings()
    queue = TrainingJobQueue(get_engine())
    queue.create_tables()
    tracking = _tracking_or_none()
    app = create_app(
        queue=queue,
        reports_dir=settings.reports_dir,
        tracking=tracking,
        crops_dir=settings.crops_dir,
        # APP-07: model_version (SemVer) del registro de OPS-06 → paquete de data/models.
        models=PackageRegistryResolver(
            settings.reports_dir / "models" / "registry.json",
            repo_root=settings.reports_dir.parent,
        ),
    )
    uvicorn.run(app, host=settings.ml_api_host, port=settings.ml_api_port)


if __name__ == "__main__":
    main()
