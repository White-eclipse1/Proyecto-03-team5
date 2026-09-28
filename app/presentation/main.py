import logging
import time

from sqlalchemy import text

from presentation import gate
from storage.db import get_engine
from storage.object_store import get_bucket_name, get_minio_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("dataset-quality-pipeline")


def _wait_for_dependencies(retries: int = 10, delay_seconds: float = 3.0) -> None:
    engine = get_engine()
    minio_client = get_minio_client()
    bucket = get_bucket_name()

    for attempt in range(1, retries + 1):
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            minio_client.bucket_exists(bucket)
            logger.info("Conectado a MariaDB y a MinIO (bucket=%s).", bucket)
            return
        except Exception as error:  # reintento deliberado en el arranque
            logger.warning(
                "Intento %s/%s: dependencias aún no listas (%s).", attempt, retries, error
            )
            time.sleep(delay_seconds)

    raise RuntimeError("No se pudo conectar a MariaDB/MinIO tras varios intentos.")


def main() -> None:
    logger.info("dataset-quality-pipeline: arrancando.")
    _wait_for_dependencies()

    # Un error al correr la compuerta (dataset roto, config inválida, storage
    # caído) no se traga: el contenedor debe morir y dejar que `restart:
    # on-failure` de docker-compose lo reintente, para que sea visible desde
    # afuera (docker compose ps, reinicios) en vez de quedar "vivo" pero sin
    # haber corrido nunca el gate.
    report = gate.run()

    if report.status == "failed":
        logger.error(
            "Compuerta de calidad BLOQUEADA (status=failed): al menos un check fail no pasó."
        )
    elif report.status == "warning":
        logger.warning("Compuerta de calidad con advertencias (status=warning).")
    else:
        logger.info("Compuerta de calidad OK (status=%s).", report.status)

    logger.info("Listo. Splits y versionado (tiers 4-5) se implementan en frentes posteriores.")

    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
