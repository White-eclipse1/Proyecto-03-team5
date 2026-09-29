"""ML-01 — procedencia de los crops: release aprobado de P2 y huella DVC.

Los crops no salen del dataset de trabajo (`local-dev`) sino del release
fijado en `crops/crops.yaml`. Antes de extraer se comprueba que:

- el release existe en `reports/versions.json` y su `quality.json` congelado
  corresponde a esa versión y no está en `failed`;
- las anotaciones en disco son las que describió ese reporte congelado (mismas
  imágenes por clase y mismo total de anotaciones);
- `data/raw/images` y `data/raw/annotations` tienen exactamente el md5 de
  directorio que registran sus `.dvc` (mismo cálculo que DVC 3).

Después de extraer, `assert_min_images_per_class` exige el mínimo de imágenes
originales distintas por clase.
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field

from crops.classes import CLASS_NAMES, resolve_category_classes
from crops.models import CropReport, DvcFingerprint, Identifier
from presentation.contracts import DatasetRelease, QualityReport, VersionsReport


class CropsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    dataset_version: Identifier
    min_images_per_class: Annotated[int, Field(gt=0)]


def load_crops_config(path: Path | None = None) -> CropsConfig:
    path = path if path is not None else Path(__file__).with_name("crops.yaml")
    return CropsConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def load_approved_release(
    reports_dir: Path, dataset_version: str
) -> tuple[DatasetRelease, QualityReport]:
    """Devuelve la entrada del catálogo y su `quality.json` congelado, ya validados."""
    catalog = VersionsReport.model_validate_json(
        (reports_dir / "versions.json").read_text(encoding="utf-8")
    )
    release = next(
        (item for item in catalog.releases if item.dataset_version == dataset_version), None
    )
    if release is None:
        raise ValueError(f"El release {dataset_version} no está en versions.json")

    quality = QualityReport.model_validate_json(
        (reports_dir / release.quality_file).read_text(encoding="utf-8")
    )
    if quality.dataset_version != dataset_version:
        raise ValueError(
            f"{release.quality_file} describe {quality.dataset_version}, no {dataset_version}"
        )
    if quality.status == "failed":
        raise ValueError(f"El release {dataset_version} tiene el quality gate en failed")
    return release, quality


def _check_details(quality: QualityReport, name: str) -> dict:
    for check in quality.checks:
        if check.check_name == name:
            return check.details
    raise ValueError(f"El reporte congelado de {quality.dataset_version} no tiene el check {name}")


def assert_dataset_matches_release(coco: dict, quality: QualityReport) -> None:
    """Rechaza anotaciones en disco distintas de las que congeló el release."""
    frozen = {
        entry["category_name"]: set(entry["image_ids"])
        for entry in _check_details(quality, "max_imbalance_ratio")["images_per_category"]
    }
    class_by_category = resolve_category_classes(coco["categories"])
    observed: dict[str, set[int]] = defaultdict(set)
    for annotation in coco["annotations"]:
        class_name = class_by_category.get(annotation["category_id"])
        if class_name is not None:
            observed[class_name].add(annotation["image_id"])

    for class_name in CLASS_NAMES:
        if frozen.get(class_name) != observed[class_name]:
            raise ValueError(
                f"Las imágenes de '{class_name}' no coinciden con el release "
                f"{quality.dataset_version}: {len(observed[class_name])} en disco, "
                f"{len(frozen.get(class_name, ()))} en el reporte congelado"
            )

    total = _check_details(quality, "degenerate_boxes")["total_annotations"]
    if total != len(coco["annotations"]):
        raise ValueError(
            f"El release {quality.dataset_version} tiene {total} anotaciones y en disco "
            f"hay {len(coco['annotations'])}"
        )


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def dvc_fingerprint(dvc_file: Path, *, relative_to: Path) -> DvcFingerprint:
    """Lee la salida de `dvc_file` y comprueba que el workspace tenga ese mismo hash.

    Para directorios DVC 3 calcula el md5 del JSON `[{"md5", "relpath"}, ...]`
    ordenado por ruta; aquí se recalcula igual para no confiar solo en el `.dvc`.
    """
    [out] = yaml.safe_load(dvc_file.read_text(encoding="utf-8"))["outs"]
    target = dvc_file.parent / out["path"]
    if out["md5"].endswith(".dir"):
        files = sorted(
            (path for path in target.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(target).parts,
        )
        entries = [
            {"md5": _md5(path), "relpath": path.relative_to(target).as_posix()} for path in files
        ]
        digest = hashlib.md5(json.dumps(entries, sort_keys=True).encode()).hexdigest() + ".dir"
        size, nfiles = sum(path.stat().st_size for path in files), len(files)
    else:
        digest, size, nfiles = _md5(target), target.stat().st_size, 1

    if (digest, size, nfiles) != (out["md5"], out["size"], out.get("nfiles", 1)):
        raise ValueError(
            f"{target} no coincide con {dvc_file.name} ({digest} != {out['md5']}); "
            "corre `dvc pull` o `dvc checkout`"
        )
    return DvcFingerprint(
        path=target.relative_to(relative_to).as_posix(), md5=digest, size=size, nfiles=nfiles
    )


def assert_min_images_per_class(report: CropReport, minimum: int) -> None:
    """Exige `minimum` imágenes originales distintas con crop aceptado para dog y cat."""
    short = {
        name: count for name, count in report.summary.images_per_class.items() if count < minimum
    }
    if short:
        raise ValueError(
            f"Clases por debajo de {minimum} imágenes originales en {report.dataset_version}: "
            f"{short}"
        )
