"""Healthchecks de docker-compose.yml: nadie recibe tráfico antes de estar listo.

- MariaDB: la imagen oficial arranca primero un servidor temporal de inicialización,
  solo por socket (sin red), y luego lo reinicia. `mariadb-admin ping -h localhost`
  usa el socket, así que daba "sano" con el servidor temporal y la primera consulta de
  otro servicio caía en el reinicio ("Lost connection to MySQL server during query",
  visto en el job de CI de la cola). Haciendo el ping por TCP solo pasa con el servidor
  real.
- ml-api: nginx (frontend) le mandaba peticiones antes de que uvicorn escuchara, y el
  portal recibía 502 unos segundos.
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _services() -> dict:
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))["services"]


def _test_command(service: dict) -> str:
    return " ".join(service["healthcheck"]["test"])


def test_mariadb_is_healthy_only_when_it_accepts_tcp_connections():
    command = _test_command(_services()["mariadb"])
    assert "mariadb-admin ping" in command
    assert "--protocol=tcp" in command
    assert "-h 127.0.0.1" in command
    assert "-h localhost" not in command, "localhost usa el socket del servidor temporal"


def test_mariadb_healthcheck_does_not_version_the_password():
    command = _test_command(_services()["mariadb"])
    assert "$${MARIADB_ROOT_PASSWORD}" in command


def test_ml_api_reports_health_from_its_own_endpoint():
    healthcheck = _services()["ml-api"]["healthcheck"]
    command = " ".join(healthcheck["test"])
    assert "http://localhost:8001/health" in command
    assert healthcheck.get("start_period")


def test_frontend_waits_until_ml_api_is_healthy():
    depends_on = _services()["frontend"]["depends_on"]
    assert depends_on["ml-api"]["condition"] == "service_healthy"
    # El resto sigue igual que antes: solo hace falta que arranquen.
    assert depends_on["backend"]["condition"] == "service_started"
    assert depends_on["copilot"]["condition"] == "service_started"
