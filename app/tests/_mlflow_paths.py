"""Rutas locales de los artefactos de un run en un MLflow de archivos (tests)."""

from pathlib import Path

from mlflow.tracking import MlflowClient
from mlflow.utils.file_utils import local_file_uri_to_path


def artifact_dir_from_uri(uri: str) -> Path:
    """`file://` → ruta del SO: unidad de Windows (`C:\\...`) y `%20` decodificado."""
    return Path(local_file_uri_to_path(uri))


def artifact_dir(client: MlflowClient, run_id: str) -> Path:
    return artifact_dir_from_uri(client.get_run(run_id).info.artifact_uri)
