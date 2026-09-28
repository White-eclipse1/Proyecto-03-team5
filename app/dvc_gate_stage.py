"""DVC stage that blocks downstream stages when the report is failed."""

import logging
import os
from pathlib import Path

from presentation.contracts import QualityReport

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", REPO_ROOT / "reports"))
REPORT_PATH = REPORTS_DIR / "quality.json"
PASS_MARKER = REPORTS_DIR / ".quality_gate.passed"
logger = logging.getLogger("dvc-quality-gate")


def read_quality_report() -> QualityReport:
    """Load and validate the report shared by the downstream stages."""
    if not REPORT_PATH.exists():
        raise RuntimeError(f"No existe el reporte de calidad: {REPORT_PATH}")
    try:
        return QualityReport.model_validate_json(REPORT_PATH.read_text(encoding="utf-8"))
    except ValueError as error:
        raise RuntimeError(f"El reporte de calidad no cumple el contrato: {REPORT_PATH}") from error


def assert_quality_gate_passed() -> None:
    """Refuse downstream work unless the current report and marker both pass."""
    report = read_quality_report()
    if report.status == "failed":
        raise RuntimeError("La compuerta de calidad está failed; no se puede generar el split.")
    if not PASS_MARKER.exists():
        raise RuntimeError("Falta el marcador de la compuerta de calidad aprobada.")


def enforce_quality_gate() -> int:
    """Return a non-zero status for failed reports and mark accepted reports."""
    PASS_MARKER.unlink(missing_ok=True)
    try:
        report = read_quality_report()
    except RuntimeError as error:
        logger.error("%s", error)
        return 1

    if report.status == "failed":
        logger.error("Compuerta de calidad BLOQUEADA: el reporte tiene status=failed.")
        return 1

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PASS_MARKER.write_text("passed\n", encoding="utf-8")
    logger.info("Compuerta de calidad OK (status=%s).", report.status)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(enforce_quality_gate())
