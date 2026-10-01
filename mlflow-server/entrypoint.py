"""OPS-03 — arranque del servidor MLflow persistente.

Antes de lanzar `mlflow server`:

1. Crea la base `mlflow` en MariaDB si no existe. `MARIADB_DATABASE` solo crea
   `image_repo`, y los scripts de `docker-entrypoint-initdb.d` no corren sobre un
   volumen que ya existe, así que esto también sirve para clones con datos previos.
2. Crea el bucket de artefactos en MinIO si no existe.

Los runs y las métricas viven en MariaDB (volumen `mariadb_data`) y los artefactos
en MinIO (volumen `minio_data`); el contenedor de MLflow no guarda estado propio.

Variables (ver docker-compose.yml):
- MLFLOW_BACKEND_STORE_URI       mysql+pymysql://usuario:clave@mariadb:3306/mlflow
- MLFLOW_ARTIFACTS_DESTINATION   s3://<bucket>
- MLFLOW_S3_ENDPOINT_URL         http://minio:9000
- MLFLOW_ALLOWED_HOSTS           Host headers aceptados (MLflow 3 solo acepta
                                 localhost/IPs privadas por defecto; `mlflow`
                                 es el nombre del servicio en Compose)
- AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY   credenciales locales de MinIO
"""

import os
import sys
import time
from collections.abc import Mapping
from urllib.parse import urlsplit

REQUIRED = (
    "MLFLOW_BACKEND_STORE_URI",
    "MLFLOW_ARTIFACTS_DESTINATION",
    "MLFLOW_S3_ENDPOINT_URL",
    "MLFLOW_ALLOWED_HOSTS",
)
PORT = "5000"
RETRY_SECONDS = 90


def _require(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise SystemExit(f"Falta la variable {name} para arrancar MLflow")
    return value


def server_command(env: Mapping[str, str]) -> list[str]:
    values = {name: _require(env, name) for name in REQUIRED}
    return [
        "mlflow",
        "server",
        "--host",
        "0.0.0.0",
        "--port",
        PORT,
        "--backend-store-uri",
        values["MLFLOW_BACKEND_STORE_URI"],
        "--artifacts-destination",
        values["MLFLOW_ARTIFACTS_DESTINATION"],
        "--serve-artifacts",
        "--allowed-hosts",
        values["MLFLOW_ALLOWED_HOSTS"],
    ]


def redact(uri: str) -> str:
    """La URI con la contraseña oculta, para los logs."""
    parts = urlsplit(uri)
    if parts.password is None:
        return uri
    netloc = parts.netloc.replace(f":{parts.password}@", ":***@", 1)
    return parts._replace(netloc=netloc).geturl()


def _retry(action, description: str) -> None:
    deadline = time.monotonic() + RETRY_SECONDS
    while True:
        try:
            action()
            return
        except Exception as exc:
            if time.monotonic() > deadline:
                raise SystemExit(f"No se pudo {description}: {exc}") from exc
            print(f"Esperando para {description}: {exc}", flush=True)
            time.sleep(3)


def ensure_database(backend_store_uri: str) -> None:
    import pymysql
    from sqlalchemy.engine import make_url

    url = make_url(backend_store_uri)

    def create() -> None:
        connection = pymysql.connect(
            host=url.host,
            port=url.port or 3306,
            user=url.username,
            password=url.password or "",
        )
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"CREATE DATABASE IF NOT EXISTS `{url.database}` "
                    "CHARACTER SET utf8mb4 COLLATE utf8mb4_bin"
                )
        finally:
            connection.close()

    _retry(create, f"crear la base {url.database} en {redact(backend_store_uri)}")


def ensure_bucket(destination: str, endpoint_url: str) -> None:
    import boto3
    from botocore.exceptions import ClientError

    bucket = urlsplit(destination).netloc
    s3 = boto3.client("s3", endpoint_url=endpoint_url)

    def create() -> None:
        try:
            s3.head_bucket(Bucket=bucket)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") not in {"404", "NoSuchBucket"}:
                raise
            s3.create_bucket(Bucket=bucket)

    _retry(create, f"crear el bucket {bucket} en {endpoint_url}")


def main() -> None:
    command = server_command(os.environ)
    ensure_database(os.environ["MLFLOW_BACKEND_STORE_URI"])
    ensure_bucket(os.environ["MLFLOW_ARTIFACTS_DESTINATION"], os.environ["MLFLOW_S3_ENDPOINT_URL"])
    print(
        "Arrancando MLflow: runs en "
        f"{redact(os.environ['MLFLOW_BACKEND_STORE_URI'])}, artefactos en "
        f"{os.environ['MLFLOW_ARTIFACTS_DESTINATION']}",
        flush=True,
    )
    sys.stdout.flush()
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
