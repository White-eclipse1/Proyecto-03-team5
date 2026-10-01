"""OPS-03: contrato del servicio MLflow persistente.

Estas pruebas no necesitan Docker: revisan la configuración (docker-compose.yml,
la imagen `mlflow-server/` y `TrackingSettings`). La prueba de persistencia real,
con un run que sobrevive al reinicio, vive en `test_mlflow_persistence.py`.
"""

import importlib.util
import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from storage.settings import TrackingSettings

ROOT = Path(__file__).resolve().parents[2]
SERVER_DIR = ROOT / "mlflow-server"

# Contrato acordado con OPS-04 (worker) en el issue #8.
TRACKING_URI_IN_COMPOSE = "http://mlflow:5000"


def _compose() -> dict:
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


def _mlflow_service() -> dict:
    services = _compose()["services"]
    assert "mlflow" in services, "docker-compose.yml no define el servicio mlflow"
    return services["mlflow"]


def _load_entrypoint():
    spec = importlib.util.spec_from_file_location(
        "mlflow_server_entrypoint", SERVER_DIR / "entrypoint.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- TrackingSettings: la URI es configurable y apunta a un servidor ---------------


def test_tracking_uri_comes_from_environment(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    assert TrackingSettings().mlflow_tracking_uri == "http://mlflow:5000"


def test_tracking_uri_can_point_elsewhere(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.org/")
    assert TrackingSettings().mlflow_tracking_uri == "https://mlflow.example.org"


def test_missing_tracking_uri_is_rejected_naming_the_field(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(ValidationError, match="mlflow_tracking_uri"):
        TrackingSettings(_env_file=None)


@pytest.mark.parametrize(
    "uri",
    ["", "   ", "file:///tmp/mlruns", "sqlite:///mlflow.db", "./mlruns", "mlflow:5000"],
)
def test_tracking_uri_must_be_an_http_server(monkeypatch, uri):
    """Un file store o sqlite local no sobrevive al contenedor: solo se acepta el servidor."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    with pytest.raises(ValidationError, match="mlflow_tracking_uri"):
        TrackingSettings(_env_file=None)


# --- docker-compose.yml: servicio, persistencia y contrato con el worker ----------


def test_pipeline_containers_receive_the_agreed_tracking_uri():
    compose = _compose()
    assert compose["x-pipeline-env"]["MLFLOW_TRACKING_URI"] == TRACKING_URI_IN_COMPOSE
    assert compose["services"]["app"]["environment"]["MLFLOW_TRACKING_URI"] == (
        TRACKING_URI_IN_COMPOSE
    )


def test_mlflow_service_builds_the_pinned_server_image():
    service = _mlflow_service()
    assert service["build"] == "./mlflow-server"
    assert (SERVER_DIR / "Dockerfile").is_file()


def test_runs_live_in_mariadb_not_inside_the_container():
    env = _mlflow_service()["environment"]
    uri = env["MLFLOW_BACKEND_STORE_URI"]
    assert uri.startswith("mysql+pymysql://root:")
    assert uri.endswith("@mariadb:3306/mlflow")
    assert "${MARIADB_ROOT_PASSWORD:?" in uri


def test_artifacts_live_in_minio_behind_the_tracking_server():
    env = _mlflow_service()["environment"]
    assert env["MLFLOW_ARTIFACTS_DESTINATION"] == "s3://mlflow"
    assert env["MLFLOW_S3_ENDPOINT_URL"] == "http://minio:9000"
    assert env["AWS_ACCESS_KEY_ID"].startswith("${MINIO_ROOT_USER:?")
    assert env["AWS_SECRET_ACCESS_KEY"].startswith("${MINIO_ROOT_PASSWORD:?")


def test_mlflow_state_is_backed_by_named_volumes():
    compose = _compose()
    service = _mlflow_service()
    # El contenedor de MLflow no guarda estado: todo vive en MariaDB y MinIO.
    assert "volumes" not in service
    assert "mariadb_data:/var/lib/mysql" in compose["services"]["mariadb"]["volumes"]
    assert "minio_data:/data" in compose["services"]["minio"]["volumes"]
    assert {"mariadb_data", "minio_data"} <= set(compose["volumes"])


def test_mlflow_waits_for_its_stores_and_reports_health():
    service = _mlflow_service()
    assert service["depends_on"]["mariadb"]["condition"] == "service_healthy"
    assert "minio" in service["depends_on"]
    assert "/health" in " ".join(service["healthcheck"]["test"])


def test_mlflow_ui_is_published_only_on_loopback():
    assert _mlflow_service()["ports"] == ["127.0.0.1:${MLFLOW_PORT:-5000}:5000"]


def test_compose_does_not_version_mlflow_credentials():
    text = yaml.safe_dump(_mlflow_service())
    assert not re.search(r"(?i)(secret|password)[^$\n]*:\s*[^$\s'\"][^\n]*$", text, re.M)


# --- mlflow-server/: versiones fijas y comando del servidor ------------------------


def test_server_requirements_are_fully_pinned():
    lines = [
        line.split("#")[0].strip()
        for line in (SERVER_DIR / "requirements.txt").read_text(encoding="utf-8").splitlines()
    ]
    requirements = [line for line in lines if line and not line.startswith("--")]
    assert requirements, "mlflow-server/requirements.txt está vacío"
    unpinned = [line for line in requirements if "==" not in line]
    assert not unpinned, f"Dependencias sin versión fija: {unpinned}"
    assert any(line.startswith("mlflow==") for line in requirements)


SERVER_ENV = {
    "MLFLOW_BACKEND_STORE_URI": "mysql+pymysql://root:s3cr3t@mariadb:3306/mlflow",
    "MLFLOW_ARTIFACTS_DESTINATION": "s3://mlflow",
    "MLFLOW_S3_ENDPOINT_URL": "http://minio:9000",
    "MLFLOW_ALLOWED_HOSTS": "mlflow,mlflow:*,localhost,localhost:*",
}


def test_server_command_serves_artifacts_from_the_persistent_stores():
    command = _load_entrypoint().server_command(SERVER_ENV)

    assert command[:2] == ["mlflow", "server"]
    assert command[command.index("--host") + 1] == "0.0.0.0"
    assert command[command.index("--port") + 1] == "5000"
    assert (
        command[command.index("--backend-store-uri") + 1]
        == (SERVER_ENV["MLFLOW_BACKEND_STORE_URI"])
    )
    assert command[command.index("--artifacts-destination") + 1] == "s3://mlflow"
    assert "--serve-artifacts" in command


def test_server_accepts_the_compose_hostname():
    """MLflow 3 rechaza Host headers que no sean localhost/IP privada: el worker usa `mlflow`."""
    command = _load_entrypoint().server_command(SERVER_ENV)
    allowed = command[command.index("--allowed-hosts") + 1].split(",")
    assert {"mlflow", "mlflow:*"} <= set(allowed)


@pytest.mark.parametrize("missing", list(SERVER_ENV))
def test_server_refuses_to_start_without_its_configuration(missing):
    env = {key: value for key, value in SERVER_ENV.items() if key != missing}
    with pytest.raises(SystemExit, match=missing):
        _load_entrypoint().server_command(env)


def test_server_logs_never_show_the_database_password():
    redacted = _load_entrypoint().redact(SERVER_ENV["MLFLOW_BACKEND_STORE_URI"])
    assert "s3cr3t" not in redacted
    assert redacted.endswith("@mariadb:3306/mlflow")
