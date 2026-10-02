"""APP-03: el servicio `ml-api` está conectado al stack del portal.

`docker compose up` lo levanta junto al resto, nginx le manda `/api/ml/` (antes
iba al backend de Node, que no implementa esas rutas) y su imagen incluye el
paquete `training/`.
"""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _compose() -> dict:
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


def test_ml_api_service_runs_the_training_server():
    service = _compose()["services"]["ml-api"]

    assert service["build"] == "./app"
    assert service["command"] == ["python", "-m", "training.server"]
    assert service["environment"]["ML_API_HOST"] == "0.0.0.0"
    assert service["environment"]["DATABASE_URL"].endswith("@mariadb:3306/image_repo")


def test_ml_api_reads_release_files_without_writing_them():
    assert "./reports:/app/reports:ro" in _compose()["services"]["ml-api"]["volumes"]


def test_ml_api_waits_for_mariadb_and_is_not_published():
    service = _compose()["services"]["ml-api"]
    assert service["depends_on"]["mariadb"]["condition"] == "service_healthy"
    assert "ports" not in service, "solo nginx (frontend) debe alcanzar la API"


def test_frontend_waits_for_ml_api():
    assert "ml-api" in _compose()["services"]["frontend"]["depends_on"]


def test_nginx_routes_api_ml_to_the_training_service():
    nginx = (ROOT / "frontend" / "docker" / "nginx.conf").read_text(encoding="utf-8")
    block = re.search(r"location /api/ml/ \{(.*?)\}", nginx, re.S)

    assert block, "nginx.conf no tiene location /api/ml/"
    assert "proxy_pass http://ml-api:8001/;" in block.group(1)


def test_app_image_ships_the_training_package():
    dockerfile = (ROOT / "app" / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY training/ ./training/" in dockerfile


def test_ml_api_reaches_mlflow_and_fails_fast():
    """APP-04: /runs lee MLflow; si no responde, 503 en segundos y no tras minutos de reintentos."""
    env = _compose()["services"]["ml-api"]["environment"]
    assert env["MLFLOW_TRACKING_URI"] == "http://mlflow:5000"
    assert int(env["MLFLOW_HTTP_REQUEST_MAX_RETRIES"]) <= 2
    assert int(env["MLFLOW_HTTP_REQUEST_TIMEOUT"]) <= 15


def test_ci_runs_the_experiments_integration_test_against_mlflow():
    ci = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
    commands = " ".join(step.get("run", "") for step in ci["jobs"]["mlflow"]["steps"])
    assert "tests/test_experiments_mlflow_integration.py" in commands


def test_ml_api_serves_crops_read_only_for_the_evaluation_examples():
    """APP-05: los ejemplos de Evaluation muestran el recorte de ML-01 (`data/crops`, DVC)."""
    service = _compose()["services"]["ml-api"]
    assert "./data/crops:/app/data/crops:ro" in service["volumes"]
    assert service["environment"]["CROPS_DIR"] == "/app/data/crops"


def test_app_image_ships_the_model_code_for_inference():
    """APP-07: ml-api carga checkpoints (classification/) con las clases de ML-01 (crops/)."""
    dockerfile = (ROOT / "app" / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY classification/ ./classification/" in dockerfile
    assert "COPY crops/ ./crops/" in dockerfile


def test_ml_api_serves_the_ops06_model_packages_read_only():
    """APP-07: las versiones salen de reports/models/registry.json y data/models (OPS-06, DVC)."""
    service = _compose()["services"]["ml-api"]
    assert "./data/models:/app/data/models:ro" in service["volumes"]
    assert "./reports:/app/reports:ro" in service["volumes"]


def test_inference_does_not_depend_on_the_mlflow_model_registry():
    """OPS-06 no registra versiones en MLflow: no queda prueba ni paso de CI de ese registry."""
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "test_inference_mlflow_integration.py" not in ci
    assert not (ROOT / "app" / "tests" / "test_inference_mlflow_integration.py").exists()
