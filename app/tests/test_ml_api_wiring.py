"""APP-03: el servicio `ml-api` está conectado al stack del portal.

`docker compose up` lo levanta junto al resto, nginx le manda `/api/ml/` (antes
iba al backend de Node, que no implementa esas rutas) y su imagen incluye el
paquete `training/`.
"""

import re
from pathlib import Path

import pytest
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


def test_every_endpoint_the_portal_calls_is_served_by_ml_api():
    """APP-09: ninguna pantalla de modelos llama a una ruta sin backend (placeholder)."""
    data_source = (ROOT / "frontend" / "src" / "ml" / "dataSource.ts").read_text(encoding="utf-8")
    block = re.search(r"export const ML_ENDPOINTS = \{(.*?)\} as const;", data_source, re.S)
    assert block, "dataSource.ts no tiene ML_ENDPOINTS"
    endpoints = re.findall(r'"/ml(/[^"]*)"', block.group(1))
    server = (ROOT / "app" / "training" / "server.py").read_text(encoding="utf-8")
    routes = set(re.findall(r'Route\("([^"]+)"', server))
    prefixes = {route.split("{")[0].rstrip("/") for route in routes}

    assert endpoints
    for endpoint in endpoints:
        assert endpoint in routes or endpoint in prefixes, f"{endpoint} no está en ml-api"


@pytest.mark.parametrize("location", ["/api/ml/", "/api/"])
def test_nginx_accepts_the_uploads_that_inference_allows(location):
    """APP-09 (bug): nginx cortaba con 413 (1 MB por defecto) imágenes válidas de <10 MB.

    `/api/` (backend de Node) recibe la misma imagen en "Enviar a cola de anotación"
    (`POST /api/images/from-inference`), más el campo `metadata`: con 10m cortaba una
    imagen de exactamente 10 MiB que Inference sí acepta (revisión del PR #63).
    """
    from training.inference import MAX_UPLOAD_BYTES

    nginx = (ROOT / "frontend" / "docker" / "nginx.conf").read_text(encoding="utf-8")
    block = re.search(rf"location {re.escape(location)} \{{(.*?)\}}", nginx, re.S)
    limit = re.search(r"client_max_body_size\s+(\d+)m;", block.group(1)) if block else None

    assert limit, f"location {location} debe fijar client_max_body_size"
    # El multipart agrega cabeceras y campos (model_name/model_version o metadata).
    assert int(limit.group(1)) * 1024 * 1024 > MAX_UPLOAD_BYTES


def test_ml_api_can_receive_aws_credentials_without_secrets_in_the_repo():
    """Revisión del PR #63: Models confirma en S3 lo publicado; las credenciales son
    opcionales y salen del entorno de quien levanta el stack (vacías por defecto)."""
    env = _compose()["services"]["ml-api"]["environment"]
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        assert env[name] == f"${{{name}:-}}", f"{name} viene del entorno, vacío por defecto"
    assert env["AWS_EC2_METADATA_DISABLED"] == "${AWS_EC2_METADATA_DISABLED:-true}"


def test_ml_api_confirms_publications_against_the_real_s3():
    server = (ROOT / "app" / "training" / "server.py").read_text(encoding="utf-8")
    assert "s3_verifier=BotoS3Verifier()" in server


def test_backend_verifies_annotation_queue_traceability_against_the_reports():
    """APP-09: la cola de APP-08 compara la trazabilidad con registry.json y crops.json."""
    service = _compose()["services"]["backend"]
    assert "./reports:/reports:ro" in service["volumes"]
    assert service["environment"]["REPORTS_DIR"] == "/reports"


def test_readme_explains_how_a_clean_clone_gets_the_p3_data():
    """APP-10: siguiendo el README, el portal tiene recortes, modelo y las corridas de MLflow."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for step in (
        "dvc pull -r prod crops",
        "dvc pull -r prod data/models.dvc",
        "dvc pull -r prod data/mlflow-snapshot.dvc",
        "python -m tracking.snapshot restore",
        "aws configure export-credentials",
        "tests/test_app10_portal_smoke.py",
    ):
        assert step in readme, f"Falta en el README: {step}"


def test_the_clean_clone_section_alone_starts_the_whole_stack():
    """OPS-10 (M1): sin `data/raw`, el servicio `app` (compuerta P2) falla y `up --wait` también.

    La sección del clon limpio debe bastar por sí sola, en una misma shell: dataset P2,
    datos de P3, `GIT_COMMIT` y el restore de MLflow.
    """
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    start = readme.index("### Proyecto 3 completo en un clon limpio")
    section = readme[start : readme.index("\n### ", start + 1)]
    for step in (
        "dvc pull -r prod data/raw/images.dvc data/raw/annotations.dvc",
        "dvc pull -r prod crops",
        "dvc pull -r prod data/models.dvc",
        "dvc pull -r prod data/mlflow-snapshot.dvc",
        'export GIT_COMMIT="$(git rev-parse HEAD)"',
        "docker compose up -d --build --wait",
        "python -m tracking.snapshot restore",
        "misma terminal",
    ):
        assert step in section, f"Falta en la sección del clon limpio: {step}"
    assert section.index("data/raw/images.dvc") < section.index(
        "docker compose up -d --build --wait"
    )
