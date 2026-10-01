"""APP-03 — lo que la API revisa de un release antes de aceptar un job.

Lee los mismos archivos que la pantalla Training (`reports/versions.json` y
`reports/releases/<v>/{quality,provenance,manifest}.json`), pero en el servidor: el
POST no confía en lo que el navegador haya validado.

Un archivo que falta o no cumple su contrato cuenta como ausente, y la regla de
`training_request_rejection` (ml_contracts.py) decide si el job se bloquea.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from presentation.ml_contracts import ReleaseProvenance, TrainingManifest


@dataclass(frozen=True)
class ReleaseFiles:
    quality_status: str
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


def load_release(reports_dir: Path, version: str) -> ReleaseFiles:
    release_dir = reports_dir / "releases" / version
    quality = _read_json(release_dir / "quality.json")
    status = quality.get("status") if isinstance(quality, dict) else None
    return ReleaseFiles(
        # Sin un quality.json legible no hay forma de comprobar la compuerta.
        quality_status=status if isinstance(status, str) else "failed",
        provenance=_validated(ReleaseProvenance, release_dir / "provenance.json"),
        manifest=_validated(TrainingManifest, release_dir / "manifest.json"),
    )
