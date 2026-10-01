# Servidor MLflow (OPS-03)

Imagen del servicio `mlflow` de `docker-compose.yml`.

- **Runs, parámetros y métricas:** base `mlflow` en la MariaDB del proyecto
  (volumen `mariadb_data`). `entrypoint.py` la crea si no existe.
- **Artefactos (checkpoints, curvas):** bucket `mlflow` en el MinIO del proyecto
  (volumen `minio_data`), servidos por el propio MLflow (`--serve-artifacts`). Los
  clientes solo necesitan `MLFLOW_TRACKING_URI`, no credenciales de MinIO.

## Actualizar dependencias

`requirements.in` lista las dependencias directas; `requirements.txt` fija todas las
versiones para Linux/Python 3.12. Después de cambiar `requirements.in`:

```bash
uv pip compile requirements.in --python-version 3.12 --python-platform linux \
  --no-header -o requirements.txt
```

Mantén la versión de `mlflow-skinny` en `app/pyproject.toml` alineada con la de
`mlflow` aquí.
