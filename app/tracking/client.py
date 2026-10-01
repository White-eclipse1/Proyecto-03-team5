"""OPS-03 — cliente del servidor MLflow persistente.

Punto único para que el worker (OPS-04) y el loop de entrenamiento (ML-04)
obtengan un cliente apuntando al servidor configurado en `MLFLOW_TRACKING_URI`.
Los artefactos (checkpoints, curvas) se suben a través del propio servidor
(`mlflow-artifacts:`), así que el cliente no necesita credenciales de MinIO/S3.

MLflow 3 intenta subir/descargar directo al almacenamiento con URLs prefirmadas
cuando el servidor usa S3; esas URLs apuntan a `minio:9000`, un nombre que solo
existe dentro de Compose. `tracking_client()` desactiva ese atajo (salvo que el
proceso ya lo haya configurado) para que todo pase por el servidor.
"""

import urllib.error
import urllib.request

import mlflow
from mlflow.environment_variables import (
    MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD,
    MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD,
)
from mlflow.tracking import MlflowClient

from storage.settings import TrackingSettings


class TrackingServerUnavailableError(RuntimeError):
    """El servidor MLflow no respondió a `/health`."""


def tracking_client(settings: TrackingSettings | None = None) -> MlflowClient:
    """Configura `mlflow` (API fluida) y devuelve un cliente del mismo servidor."""
    uri = (settings or TrackingSettings()).mlflow_tracking_uri
    for direct_storage_access in (
        MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD,
        MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD,
    ):
        if not direct_storage_access.is_set():
            direct_storage_access.set(False)
    mlflow.set_tracking_uri(uri)
    return MlflowClient(tracking_uri=uri)


def check_server(settings: TrackingSettings | None = None, *, timeout_s: float = 5) -> str:
    """Comprueba `/health` sin los reintentos largos del cliente; devuelve la URI."""
    uri = (settings or TrackingSettings()).mlflow_tracking_uri
    try:
        with urllib.request.urlopen(f"{uri}/health", timeout=timeout_s) as response:
            if response.status == 200:
                return uri
            detail = f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        detail = f"HTTP {exc.code}"
    except OSError as exc:
        detail = str(exc)
    raise TrackingServerUnavailableError(f"MLflow no responde en {uri}/health: {detail}")
