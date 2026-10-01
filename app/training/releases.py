"""APP-03 — lo que la API revisa de un release antes de aceptar un job.

Lee los mismos archivos que la pantalla Training (`reports/versions.json` y
`reports/releases/<v>/{quality,provenance,manifest}.json`), pero en el servidor: el
POST no confía en lo que el navegador haya validado.

Un `provenance.json` o `manifest.json` que falta o no cumple su contrato cuenta como
ausente; un `quality.json` ausente, fuera del contrato `QualityReport` o de otra
versión bloquea el job con su motivo. Después, la regla de
`training_request_rejection` (ml_contracts.py) decide si el job se bloquea.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from presentation.contracts import QualityReport
from presentation.ml_contracts import ReleaseProvenance, TrainingManifest


@dataclass(frozen=True)
class ReleaseFiles:
    # `quality_problem` explica por qué no se pudo comprobar la compuerta (reporte
    # ausente, ilegible, fuera del contrato o de otra versión); entonces
    # `quality_status` es None y el job se rechaza.
    quality_status: str | None
    quality_problem: str | None
    provenance: ReleaseProvenance | None
    manifest: TrainingManifest | None


def _read_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def published_releases(reports_dir: Path) -> set[str]:
    catalog = _read_json(reports_dir / "versions.json")
    if not isinstance(catalog, dict) or not isinstance(catalog.get("releases"), list):
        return set()
    return {
        release["dataset_version"]
        for release in catalog["releases"]
        if isinstance(release, dict) and isinstance(release.get("dataset_version"), str)
    }


def _validated(model, path: Path):
    document = _read_json(path)
    if document is None:
        return None
    try:
        return model.model_validate(document)
    except ValidationError:
        return None


def _quality(release_dir: Path, version: str) -> tuple[str | None, str | None]:
    """(status, None) si `quality.json` es un QualityReport de `version`; si no, (None, motivo)."""
    document = _read_json(release_dir / "quality.json")
    if document is None:
        return None, (
            f"El release {version} no tiene un quality.json legible; "
            "no se puede comprobar el Quality Gate."
        )
    try:
        report = QualityReport.model_validate(document)
    except ValidationError as exc:
        first = exc.errors()[0]
        field = ".".join(str(part) for part in first["loc"]) or "documento"
        return None, (
            f"El quality.json del release {version} no cumple el contrato QualityReport "
            f"({field}: {first['msg']}); no se puede comprobar el Quality Gate."
        )
    if report.dataset_version != version:
        return None, (
            f"El quality.json corresponde a {report.dataset_version}, no a {version}; "
            "no se puede comprobar el Quality Gate."
        )
    return report.status, None


def load_release(reports_dir: Path, version: str) -> ReleaseFiles:
    release_dir = reports_dir / "releases" / version
    status, problem = _quality(release_dir, version)
    return ReleaseFiles(
        quality_status=status,
        quality_problem=problem,
        provenance=_validated(ReleaseProvenance, release_dir / "provenance.json"),
        manifest=_validated(TrainingManifest, release_dir / "manifest.json"),
    )
