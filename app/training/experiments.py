"""APP-04 — de la API de MLflow a los contratos de la pantalla Experiments.

Se lee MLflow en cada petición (sin caché): un cambio en un run se ve en el portal
la siguiente vez que se consulta. Solo se listan runs con procedencia de
entrenamiento válida (ver `tracking/run_schema.py`).
"""

import logging
import math
import sys
from datetime import UTC, datetime
from typing import Protocol

from mlflow.entities import Run
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode
from pydantic import ValidationError

from presentation.ml_contracts import CurvePoint, ExperimentRun, RunCurvesResponse
from tracking.run_schema import TAG_DATASET_VERSION, TAG_GIT_COMMIT, TAG_MANIFEST_HASH

logger = logging.getLogger("ml-api")

PAGE_SIZE = 1000


class TrackingClient(Protocol):
    """Lo que se usa de `mlflow.tracking.MlflowClient`."""

    def search_experiments(self, **kwargs): ...
    def search_runs(self, experiment_ids, **kwargs): ...
    def get_run(self, run_id): ...
    def get_metric_history(self, run_id, key): ...


def _timestamp(milliseconds: int | None) -> str | None:
    if milliseconds is None:
        return None
    moment = datetime.fromtimestamp(milliseconds / 1000, tz=UTC)
    text = moment.strftime("%Y-%m-%dT%H:%M:%S")
    return f"{text}Z" if moment.microsecond == 0 else f"{text}.{moment.microsecond:06d}Z"


def finite_or_none(value: float) -> float | None:
    """El valor, o `None` si no es finito.

    MLflow conserva NaN, pero su store SQL guarda ±inf como ±sys.float_info.max
    (1.797e308): también es un valor no finito y mostrarlo como número sería inventarlo.
    """
    number = float(value)
    if not math.isfinite(number) or abs(number) >= sys.float_info.max:
        return None
    return number


def run_to_contract(run: Run) -> ExperimentRun | None:
    """El run como `ExperimentRun`, o `None` si no es un entrenamiento trazable."""
    tags = run.data.tags
    if TAG_DATASET_VERSION not in tags or TAG_MANIFEST_HASH not in tags:
        return None
    try:
        return ExperimentRun(
            run_id=run.info.run_id,
            experiment_id=run.info.experiment_id,
            run_name=run.info.run_name or run.info.run_id,
            status=run.info.status,
            start_time=_timestamp(run.info.start_time),
            end_time=_timestamp(run.info.end_time),
            dataset_version=tags[TAG_DATASET_VERSION],
            manifest_hash=tags[TAG_MANIFEST_HASH],
            git_commit=tags.get(TAG_GIT_COMMIT),
            params=dict(run.data.params),
            metrics={name: finite_or_none(value) for name, value in run.data.metrics.items()},
        )
    except ValidationError as exc:
        logger.warning("Run %s omitido: no cumple ExperimentRun (%s)", run.info.run_id, exc)
        return None


def list_runs(client: TrackingClient) -> list[ExperimentRun]:
    """Runs de entrenamiento de todos los experimentos activos, más recientes primero."""
    experiment_ids = [e.experiment_id for e in client.search_experiments(max_results=PAGE_SIZE)]
    if not experiment_ids:
        return []
    started: list[tuple[int, ExperimentRun]] = []
    page_token = None
    while True:
        page = client.search_runs(
            experiment_ids,
            max_results=PAGE_SIZE,
            order_by=["attributes.start_time DESC"],
            page_token=page_token,
        )
        for run in page:
            mapped = run_to_contract(run)
            if mapped is not None:
                started.append((run.info.start_time, mapped))
        page_token = getattr(page, "token", None)
        if not page_token:
            break
    # Por los milisegundos de MLflow, no por el texto ISO ("…:20Z" > "…:20.5Z"); un
    # empate exacto se desempata por run_id, igual que la pantalla.
    return [run for _, run in sorted(started, key=lambda item: (-item[0], item[1].run_id))]


def _is_missing(exc: MlflowException) -> bool:
    return exc.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def run_curves(client: TrackingClient, run_id: str) -> RunCurvesResponse | None:
    """Historial por step de cada métrica del run; `None` si no es un entrenamiento."""
    try:
        run = client.get_run(run_id)
    except MlflowException as exc:
        if _is_missing(exc):
            return None
        raise
    if run_to_contract(run) is None:
        return None

    curves: dict[str, list[CurvePoint]] = {}
    for name in sorted(run.data.metrics):
        by_step: dict[int, float | None] = {}
        history = sorted(client.get_metric_history(run_id, name), key=lambda m: m.timestamp)
        for metric in history:
            # El último registro de un step gana; un NaN/inf queda como punto None.
            by_step[metric.step] = finite_or_none(metric.value)
        curves[name] = [CurvePoint(step=step, value=by_step[step]) for step in sorted(by_step)]
    return RunCurvesResponse(schema_version="1.0", run_id=run_id, curves=curves)
