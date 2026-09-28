"""Los reportes REALES commiteados en `reports/` cumplen los contratos v1.0 (P2-47).

`test_json_contracts.py` solo valida los ejemplos ficticios de `presentation/examples/`.
Esto valida lo que el frontend y el Copilot leen de verdad: un reporte roto, o una
`dataset_version` que no coincide con el catálogo, rompe la UI en tiempo de ejecución
y sin este test nada lo detectaría antes del merge.
"""

from pathlib import Path

import pytest

from presentation.contracts import QualityReport, SplitsReport, VersionsReport
from presentation.projections_contracts import ProjectionsReport

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


def load(relative: str) -> str:
    path = REPORTS_DIR / relative
    assert path.is_file(), f"falta el reporte commiteado reports/{relative}"
    return path.read_text(encoding="utf-8")


def catalog_releases() -> list:
    """Vacío si el catálogo no existe: `test_committed_catalog...` lo reporta con claridad."""
    if not (REPORTS_DIR / "versions.json").is_file():
        return []
    return VersionsReport.model_validate_json(load("versions.json")).releases


def test_committed_working_quality_report_matches_the_contract():
    QualityReport.model_validate_json(load("quality.json"))


def test_committed_projections_report_matches_the_contract():
    ProjectionsReport.model_validate_json(load("projections.json"))


def test_committed_catalog_matches_the_contract():
    VersionsReport.model_validate_json(load("versions.json"))


@pytest.mark.parametrize("release", catalog_releases(), ids=lambda r: r.dataset_version)
def test_every_release_has_valid_reports_for_its_own_version(release):
    quality = QualityReport.model_validate_json(load(release.quality_file))
    splits = SplitsReport.model_validate_json(load(release.splits_file))

    assert quality.dataset_version == release.dataset_version
    assert splits.dataset_version == release.dataset_version


def test_every_release_folder_is_listed_in_the_catalog():
    folders = {path.name for path in (REPORTS_DIR / "releases").iterdir() if path.is_dir()}
    listed = {release.dataset_version for release in catalog_releases()}

    assert folders == listed
