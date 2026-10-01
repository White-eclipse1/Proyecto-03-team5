"""Conversión de URIs `file://` de artefactos de MLflow a rutas locales en cualquier SO.

Los tests del snapshot y del reporte de ML-07 borran o modifican artefactos en el
almacén local de MLflow. `uri.removeprefix("file://")` deja `/C:/...` en Windows y
no decodifica `%20`; el helper usa la conversión de MLflow (`url2pathname`).
"""

import nturl2path
import urllib.request

import mlflow.utils.file_utils as mlflow_file_utils

from tests._mlflow_paths import artifact_dir_from_uri


def test_posix_uri_with_spaces_and_parentheses_round_trips(tmp_path):
    directory = tmp_path / "iCloud Drive (archivado)" / "mlruns" / "1" / "abc" / "artifacts"
    directory.mkdir(parents=True)

    assert artifact_dir_from_uri(directory.as_uri()) == directory


def test_windows_drive_uri_becomes_a_drive_path(monkeypatch):
    # Simula Windows: la misma conversión que hace Python allí (nturl2path).
    monkeypatch.setattr(urllib.request, "url2pathname", nturl2path.url2pathname)
    monkeypatch.setattr(mlflow_file_utils, "is_windows", lambda: True)

    path = artifact_dir_from_uri("file:///C:/Users/Stephy%20B/mlruns/1/abc/artifacts")

    assert str(path).replace("/", "\\") == "C:\\Users\\Stephy B\\mlruns\\1\\abc\\artifacts"
    assert not str(path).startswith(("/C:", "\\C:"))
