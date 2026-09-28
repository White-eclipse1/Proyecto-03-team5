from minio import Minio

from storage.settings import Settings


def get_minio_client() -> Minio:
    settings = Settings()
    return Minio(
        f"{settings.minio_endpoint}:{settings.minio_port}",
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_use_ssl,
    )


def get_bucket_name() -> str:
    return Settings().minio_bucket
