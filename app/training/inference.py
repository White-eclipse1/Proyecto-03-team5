"""APP-07 — inferencia dog/cat con el modelo real, para `ml-api`.

- `read_upload` valida la imagen subida por su **contenido** (PNG o JPEG que Pillow
  pueda decodificar), su tamaño en bytes y en pixeles. El nombre del archivo y el
  tipo que declara el navegador no deciden nada.
- `read_crop` toma un recorte de ML-01 por `image_id` + `annotation_id` del release,
  y verifica su sha256 contra `reports/crops.json`.
- `PackageRegistryResolver` resuelve `model_name` + `model_version` (SemVer) en el
  registro de OPS-06 (`reports/models/registry.json`, `classification.registry`): el
  paquete de `data/models/` con el sha256 de cada archivo verificado, y la
  arquitectura y el preprocesamiento del checkpoint iguales a los registrados.
  Guarda en memoria los últimos modelos cargados.
- `classify` aplica `preprocess_image` (el preprocesamiento de validation/test de
  ML-02) y `predict_proba` (modo eval, sin gradiente).

Los errores son `InferenceRejected` con el código y el estado HTTP que `ml-api`
devuelve como `ErrorResponse`.
"""

import hashlib
import io
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from classification.model import DogCatResNet18, load_checkpoint, predict_proba
from classification.registry import resolve_checkpoint, resolve_model
from classification.transforms import preprocess_image
from presentation.ml_contracts import CropSelection, UploadedImage

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
FORMATS = {"PNG": "image/png", "JPEG": "image/jpeg"}


class InferenceRejected(Exception):
    def __init__(self, status: int, code: str, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.status, self.code, self.message, self.retryable = status, code, message, retryable


@dataclass(frozen=True)
class LoadedModel:
    model_name: str
    model_version: str
    run_id: str
    checkpoint: str
    checkpoint_sha256: str
    dataset_version: str
    model: DogCatResNet18


class ModelResolver(Protocol):
    def load(self, model_name: str, model_version: str) -> LoadedModel: ...


def _decode(data: bytes, max_pixels: int) -> Image.Image:
    try:
        with Image.open(io.BytesIO(data)) as probe:
            image_format = probe.format
            width, height = probe.size
    except UnidentifiedImageError as exc:
        raise InferenceRejected(
            415, "unsupported_image_type", "El archivo no es una imagen PNG o JPEG."
        ) from exc
    if image_format not in FORMATS:
        raise InferenceRejected(
            415,
            "unsupported_image_type",
            f"El archivo es {image_format}; solo se aceptan imágenes PNG o JPEG.",
        )
    if width * height > max_pixels:
        raise InferenceRejected(
            413,
            "image_too_large",
            f"La imagen tiene {width}x{height} pixeles; el máximo es {max_pixels:,} pixeles.",
        )
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (OSError, SyntaxError, ValueError) as exc:
        raise InferenceRejected(
            422, "invalid_image", "La imagen está dañada o incompleta y no se pudo leer."
        ) from exc
    return image


def read_upload(
    filename: str | None,
    data: bytes,
    *,
    max_bytes: int = MAX_UPLOAD_BYTES,
    max_pixels: int = MAX_IMAGE_PIXELS,
) -> tuple[UploadedImage, Image.Image]:
    if not data:
        raise InferenceRejected(422, "invalid_image", "El archivo está vacío.")
    if len(data) > max_bytes:
        raise InferenceRejected(
            413,
            "image_too_large",
            f"El archivo pesa {len(data):,} bytes; el máximo es {max_bytes:,} bytes.",
        )
    image = _decode(data, max_pixels)
    name = PurePosixPath((filename or "").replace("\\", "/")).name or "imagen"
    uploaded = UploadedImage(
        filename=name,
        content_type=FORMATS[image.format],
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        width=image.width,
        height=image.height,
    )
    return uploaded, image


def read_crop(selection: CropSelection, reports_dir: Path, crops_dir: Path | None) -> Image.Image:
    """El PNG del recorte de ML-01, solo si su sha256 es el de `crops.json`."""
    if crops_dir is None:
        raise InferenceRejected(503, "crops_not_configured", "El servicio no tiene CROPS_DIR.")
    try:
        report = json.loads((reports_dir / "crops.json").read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ValueError("crops.json no es un objeto")
    except (OSError, ValueError) as exc:
        raise InferenceRejected(
            503, "crops_not_configured", "No se pudo leer reports/crops.json."
        ) from exc
    entry = None
    if report.get("dataset_version") == selection.dataset_version:
        entry = next(
            (
                crop
                for crop in report.get("crops", [])
                if crop.get("image_id") == selection.image_id
                and crop.get("annotation_id") == selection.annotation_id
            ),
            None,
        )
    if entry is None:
        raise InferenceRejected(
            422,
            "crop_not_found",
            f"No hay recorte img {selection.image_id} · ann {selection.annotation_id} en el "
            f"release {selection.dataset_version}.",
        )
    root = crops_dir.resolve()
    path = (root / entry["crop_path"]).resolve()
    data = path.read_bytes() if root in path.parents and path.is_file() else None
    if data is None or hashlib.sha256(data).hexdigest() != entry.get("sha256"):
        raise InferenceRejected(
            422,
            "crop_not_available",
            f"El recorte {entry.get('crop_id')} falta o no es el registrado en crops.json.",
        )
    return _decode(data, MAX_IMAGE_PIXELS)


def classify(model: DogCatResNet18, image: Image.Image) -> tuple[str, dict[str, float]]:
    tensor = preprocess_image(image, model.config.image_size).unsqueeze(0)
    row = predict_proba(model, tensor)[0].tolist()
    index_to_class = {index: name for name, index in model.class_map.items()}
    probabilities = {index_to_class[index]: value for index, value in enumerate(row)}
    return max(probabilities, key=probabilities.__getitem__), probabilities


class PackageRegistryResolver:
    """`model_version` del registro de OPS-06 → paquete verificado → modelo cargado.

    `registry_path` es `reports/models/registry.json` y `repo_root` la carpeta contra
    la que se resuelve `package_path` (`data/models/<modelo>/<versión>`).
    """

    def __init__(self, registry_path: Path, *, repo_root: Path, cache_size: int = 4):
        self._registry_path = registry_path
        self._repo_root = repo_root
        self._cache: OrderedDict[tuple[str, str], LoadedModel] = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()

    def load(self, model_name: str, model_version: str) -> LoadedModel:
        key = (model_name, model_version)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            loaded = self._load(model_name, model_version)
            self._cache[key] = loaded
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
            return loaded

    def _load(self, model_name: str, model_version: str) -> LoadedModel:
        label = f"{model_name} {model_version}"
        try:
            package = resolve_model(model_version, self._registry_path)
            if package.model_name != model_name:
                raise KeyError(model_version)
            checkpoint = resolve_checkpoint(
                model_version, registry_path=self._registry_path, repo_root=self._repo_root
            )
            model = load_checkpoint(checkpoint)
        except KeyError as exc:
            raise InferenceRejected(
                422, "model_not_found", f"No existe {label} en el registro de modelos."
            ) from exc
        except FileNotFoundError as exc:
            raise InferenceRejected(
                503,
                "model_package_missing",
                f"Falta el paquete de {label} en este servidor; ejecuta "
                "`dvc pull data/models.dvc`.",
            ) from exc
        except (OSError, ValidationError) as exc:
            raise InferenceRejected(
                503, "registry_unavailable", "No se pudo leer el registro de modelos."
            ) from exc
        except ValueError as exc:
            raise InferenceRejected(
                409, "model_not_servable", f"{label} no se puede servir: {exc}"
            ) from exc
        config = model.config
        expected = package.architecture
        mismatched = [
            name
            for name, served, registered in (
                ("image_size", config.image_size, package.preprocessing.image_size),
                ("hidden_layers", config.hidden_layers, expected.hidden_layers),
            )
            if served != registered
        ]
        if mismatched:
            raise InferenceRejected(
                409,
                "model_not_servable",
                f"El checkpoint de {label} no coincide con lo registrado: {', '.join(mismatched)}.",
            )
        return LoadedModel(
            model_name=package.model_name,
            model_version=str(package.model_version),
            run_id=package.run_id,
            checkpoint=package.checkpoint,
            checkpoint_sha256=package.checkpoint_sha256,
            dataset_version=package.dataset_version,
            model=model,
        )
