"""Única capa autorizada a tocar red, entorno y credenciales (MariaDB, MinIO/S3).

El resto de las capas (ingestion, analyzers, policies, splits, presentation) reciben
los datos ya cargados como argumentos; nunca importan `os`, `boto3`, `minio` ni
`sqlalchemy.create_engine` directamente.
"""
