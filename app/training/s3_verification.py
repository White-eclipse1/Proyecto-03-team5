"""APP-06/09 — confirma en S3 los objetos que Models marcaría como publicados.

`reports/models/s3_publications.json` (OPS-07) dice qué se subió, pero el archivo puede
cambiar o el objeto borrarse después. Antes de declarar `published`, `ml-api` hace
`head_object` de cada objeto con `ChecksumMode=ENABLED` (y el `VersionId` registrado)
y compara el `ChecksumSHA256` de S3 con el sha256 registrado (`models_view`).

- `head` devuelve `S3Head` (checksum en hexadecimal, o `None` si S3 no lo tiene) o
  `None` si el objeto no existe (404, `NoSuchKey`, `NoSuchVersion`).
- Si S3 no se puede consultar (sin credenciales, acceso denegado, token vencido, sin
  red) lanza `S3Unverifiable` con el motivo, sin keys: Models muestra "No verificable".
- Las respuestas (también `S3Unverifiable`) se guardan `CACHE_SECONDS` por
  (bucket, key, version_id), para no consultar S3 en cada petición de la pantalla.

Las credenciales salen de la cadena por defecto de AWS (variables de entorno,
`AWS_PROFILE`, rol); este módulo no lee ni guarda access keys.
"""

import base64
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

CACHE_SECONDS = 60.0
NOT_FOUND = {"404", "NoSuchKey", "NotFound", "NoSuchVersion"}
DENIED = {"403", "AccessDenied", "Forbidden"}


class S3Unverifiable(Exception):
    """S3 no se pudo consultar; el mensaje explica por qué (sin keys)."""


@dataclass(frozen=True)
class S3Head:
    checksum_sha256: str | None


class S3Verifier(Protocol):
    def head(
        self, *, bucket: str, region: str, key: str, version_id: str | None
    ) -> S3Head | None: ...


def _default_client(region: str):
    import boto3
    from botocore.config import Config

    config = Config(connect_timeout=3, read_timeout=5, retries={"max_attempts": 2})
    return boto3.Session().client("s3", region_name=region, config=config)


def _unverifiable(error: Exception) -> S3Unverifiable:
    if isinstance(error, NoCredentialsError):
        return S3Unverifiable("ml-api no tiene credenciales de AWS.")
    if isinstance(error, ClientError):
        code = error.response.get("Error", {}).get("Code", "")
        if code in DENIED:
            return S3Unverifiable("S3 negó el acceso a las credenciales de ml-api.")
        return S3Unverifiable(f"S3 respondió con un error ({code or 'desconocido'}).")
    return S3Unverifiable(f"S3 no respondió ({type(error).__name__}).")


class BotoS3Verifier:
    """`S3Verifier` con boto3: un cliente por región y caché corta por objeto."""

    def __init__(
        self,
        client_for_region: Callable[[str], Any] = _default_client,
        *,
        ttl: float = CACHE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._client_for_region = client_for_region
        self._ttl = ttl
        self._clock = clock
        self._clients: dict[str, Any] = {}
        self._cache: dict[tuple[str, str, str | None], tuple[float, object]] = {}
        self._lock = threading.Lock()

    def _client(self, region: str):
        with self._lock:
            if region not in self._clients:
                self._clients[region] = self._client_for_region(region)
            return self._clients[region]

    def _ask(self, bucket: str, region: str, key: str, version_id: str | None) -> S3Head | None:
        kwargs = {"Bucket": bucket, "Key": key, "ChecksumMode": "ENABLED"}
        if version_id:
            kwargs["VersionId"] = version_id
        try:
            head = self._client(region).head_object(**kwargs)
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") in NOT_FOUND:
                return None
            raise _unverifiable(error) from error
        except BotoCoreError as error:
            raise _unverifiable(error) from error
        checksum = head.get("ChecksumSHA256")
        return S3Head(checksum_sha256=base64.b64decode(checksum).hex() if checksum else None)

    def head(self, *, bucket: str, region: str, key: str, version_id: str | None) -> S3Head | None:
        cache_key = (bucket, key, version_id)
        with self._lock:
            cached = self._cache.get(cache_key)
        if cached is not None and self._clock() - cached[0] < self._ttl:
            answer = cached[1]
        else:
            try:
                answer = self._ask(bucket, region, key, version_id)
            except S3Unverifiable as error:
                answer = error
            with self._lock:
                self._cache[cache_key] = (self._clock(), answer)
        if isinstance(answer, S3Unverifiable):
            raise S3Unverifiable(str(answer))
        return answer
