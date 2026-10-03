"""APP-06 — lo que la pantalla Models muestra de cada versión del modelo.

Lee lo que dejan OPS-06 y OPS-07 en `reports/models/`:

- `registry.json` (`classification.registry`): versiones SemVer con run, checkpoint y
  su sha256, release, manifest, arquitectura, métricas de test, model card y el
  paquete (`data/models/<modelo>/<versión>/`, `files` con el sha256 de cada archivo).
- `s3_publications.json`: por versión, bucket, región y, por archivo, key, sha256
  local, `ChecksumSHA256` que calculó S3 y sha256 de la descarga de verificación.

Una versión es `servable` si este servidor tiene cada archivo del paquete con su
sha256. Su publicación es `published` solo si cada archivo del paquete está en S3
con el sha256 registrado, confirmado por S3 y por la descarga en OPS-07 **y** ahora
mismo por S3 (`head_object` de cada objeto con su `ChecksumSHA256`, ver
`training.s3_verification`). Si algo no cuadra es `inconsistent`; si S3 no se puede
consultar (sin credenciales o sin red) es `unverifiable`. En ambos casos se explica
por qué y no se muestran keys como publicadas.
"""

import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from classification.registry import ModelPackage, ModelRegistry
from presentation.ml_contracts import (
    PACKAGE_CHECKPOINT,
    ModelCardSummary,
    ModelFile,
    ModelPublication,
    ModelsResponse,
    ModelTestMetrics,
    RegisteredModelVersion,
    S3Object,
)
from training.s3_verification import S3Unverifiable, S3Verifier

REGISTRY_FILE = Path("models") / "registry.json"
PUBLICATIONS_FILE = Path("models") / "s3_publications.json"


class RegistryUnavailable(Exception):
    pass


class FileRejected(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


_digests: dict[tuple[Path, int, int], str] = {}


def _sha256(path: Path) -> str:
    """sha256 del archivo; se recalcula solo si cambia su tamaño o fecha."""
    stat = path.stat()
    key = (path, stat.st_mtime_ns, stat.st_size)
    if key not in _digests:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        _digests[key] = digest.hexdigest()
    return _digests[key]


def load_registry(reports_dir: Path) -> list[ModelPackage]:
    path = reports_dir / REGISTRY_FILE
    if not path.is_file():
        return []
    try:
        return ModelRegistry.model_validate_json(path.read_text(encoding="utf-8")).models
    except (OSError, ValidationError) as exc:
        raise RegistryUnavailable(str(exc)) from exc


def _expected_files(package: ModelPackage) -> dict[str, str]:
    return package.files or {PACKAGE_CHECKPOINT: package.checkpoint_sha256}


def _package_file(package: ModelPackage, repo_root: Path, name: str) -> Path | None:
    if not package.package_path or not package.files or name not in package.files:
        return None
    return repo_root / package.package_path / name


def _files(package: ModelPackage, repo_root: Path) -> list[ModelFile]:
    files = []
    for name, sha256 in _expected_files(package).items():
        path = _package_file(package, repo_root, name)
        available = path is not None and path.is_file() and _sha256(path) == sha256
        files.append(ModelFile(name=name, sha256=sha256, available=available))
    return files


def _load_publications(reports_dir: Path) -> tuple[dict[tuple[str, str], dict], str | None]:
    path = reports_dir / PUBLICATIONS_FILE
    if not path.is_file():
        return {}, None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        records = {
            (record["model_name"], record["model_version"]): record
            for record in document["publications"]
        }
    except (OSError, ValueError, KeyError, TypeError):
        return {}, "No se pudo leer reports/models/s3_publications.json."
    return records, None


def _publication_problem(package: ModelPackage, record: dict) -> str | None:
    if record.get("run_id") != package.run_id:
        return f"La publicación es de otro run ({record.get('run_id')}), no de {package.run_id}."
    if record.get("checkpoint_sha256") != package.checkpoint_sha256:
        return "El checkpoint publicado no es el registrado (sha256 distinto)."
    if not all(record.get(field) for field in ("bucket", "region", "published_at")):
        return "La publicación no dice en qué bucket, región o fecha se subió."
    published = record.get("files")
    if not isinstance(published, dict):
        return "La publicación no lista sus archivos."
    for name, sha256 in _expected_files(package).items():
        if name not in published:
            return f"{name} no está en S3."
        if published[name].get("sha256") != sha256:
            return f"{name}: el sha256 publicado no es el registrado."
    for name, entry in published.items():
        if entry.get("s3_checksum_sha256") != entry.get("sha256"):
            return f"{name}: el ChecksumSHA256 de S3 no coincide con el archivo."
        if entry.get("download_sha256") != entry.get("sha256"):
            return f"{name}: la descarga de verificación no coincide con el archivo."
    return None


def _s3_problem(record: dict, s3: S3Verifier | None) -> tuple[str, str] | None:
    """(estado, motivo) si S3 no confirma cada objeto con su sha256; None si lo confirma."""
    if s3 is None:
        return "unverifiable", "ml-api no tiene cómo consultar S3 para confirmar la publicación."
    for name, entry in record["files"].items():
        if not isinstance(entry.get("key"), str) or not entry["key"]:
            return "inconsistent", "La publicación tiene campos incompletos."
        try:
            head = s3.head(
                bucket=record["bucket"],
                region=record["region"],
                key=entry["key"],
                version_id=entry.get("version_id"),
            )
        except S3Unverifiable as error:
            return "unverifiable", f"No se pudo consultar S3: {error}"
        if head is None:
            return "inconsistent", f"{name} no existe en S3 en la key y versión registradas."
        if head.checksum_sha256 is None:
            return "inconsistent", f"{name}: S3 no devuelve su ChecksumSHA256."
        if head.checksum_sha256 != entry["sha256"]:
            return "inconsistent", f"{name}: el ChecksumSHA256 de S3 no es el sha256 registrado."
    return None


def _publication(
    package: ModelPackage, records: dict, problem: str | None, s3: S3Verifier | None
) -> ModelPublication:
    def unpublished(status: str, reason: str | None = None) -> ModelPublication:
        return ModelPublication(
            status=status, bucket=None, region=None, published_at=None, objects=[], problem=reason
        )

    if problem is not None:
        return unpublished("inconsistent", problem)
    record = records.get((package.model_name, str(package.model_version)))
    if record is None:
        return unpublished("not_published")
    problem = _publication_problem(package, record)
    if problem is not None:
        return unpublished("inconsistent", problem)
    found = _s3_problem(record, s3)
    if found is not None:
        return unpublished(*found)
    try:
        return ModelPublication(
            status="published",
            bucket=record["bucket"],
            region=record["region"],
            published_at=record["published_at"],
            objects=[
                S3Object(
                    name=name,
                    key=entry["key"],
                    version_id=entry.get("version_id"),
                    sha256=entry["sha256"],
                )
                for name, entry in record["files"].items()
            ],
            problem=None,
        )
    except (KeyError, ValidationError):
        return unpublished("inconsistent", "La publicación tiene campos incompletos.")


def _version(
    package: ModelPackage, repo_root: Path, records, problem, s3: S3Verifier | None
) -> RegisteredModelVersion:
    files = _files(package, repo_root)
    return RegisteredModelVersion(
        model_name=package.model_name,
        model_version=str(package.model_version),
        run_id=package.run_id,
        checkpoint=package.checkpoint,
        checkpoint_sha256=package.checkpoint_sha256,
        dataset_version=package.dataset_version,
        manifest_hash=package.manifest_hash,
        architecture=package.architecture.name,
        image_size=package.architecture.image_size,
        test_metrics=(
            ModelTestMetrics(
                accuracy_top1=package.metrics["accuracy_top1"],
                f1_macro=package.metrics["f1_macro"],
            )
            if "accuracy_top1" in package.metrics
            else None  # versión anterior: no se evaluó en test (OPS-10)
        ),
        model_card=ModelCardSummary(
            purpose=package.model_card.purpose,
            limitations=package.model_card.limitations,
            pretrained_weights=package.model_card.pretrained_weights,
        ),
        files=files,
        servable=all(file.available for file in files),
        publication=_publication(package, records, problem, s3),
    )


def list_models(reports_dir: Path, repo_root: Path, s3: S3Verifier | None = None) -> ModelsResponse:
    records, problem = _load_publications(reports_dir)
    return ModelsResponse(
        schema_version="1.0",
        models=[
            _version(package, repo_root, records, problem, s3)
            for package in load_registry(reports_dir)
        ],
    )


def find_model(
    reports_dir: Path, repo_root: Path, version: str, s3: S3Verifier | None = None
) -> RegisteredModelVersion | None:
    for model in list_models(reports_dir, repo_root, s3).models:
        if model.model_version == version:
            return model
    return None


def package_file(reports_dir: Path, repo_root: Path, version: str, name: str) -> Path:
    """Un archivo del paquete, solo si está registrado y tiene su sha256."""
    package = next((p for p in load_registry(reports_dir) if str(p.model_version) == version), None)
    path = _package_file(package, repo_root, name) if package is not None else None
    if package is None or path is None:
        raise FileRejected(404, "file_not_found", f"{name} no es un archivo del paquete {version}.")
    if not path.is_file():
        raise FileRejected(
            404,
            "file_not_available",
            f"Falta {name} de {version} en este servidor; ejecuta `dvc pull data/models.dvc`.",
        )
    if _sha256(path) != package.files[name]:
        raise FileRejected(
            409, "file_not_servable", f"{name} de {version} no tiene el sha256 registrado."
        )
    return path
