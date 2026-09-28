"""Tier 6 — servidor MCP de solo lectura para el Copilot (P2-34).

Expone herramientas de consulta sobre el dataset real y sus resultados de
calidad. Cada herramienta se anota `read_only_hint=True` (`ToolAnnotations`
del protocolo MCP) y ninguna toca MariaDB/MinIO ni escribe nada a disco —
solo lee `data/raw/annotations/` (vía `ingestion.loader`, igual que
`gate.py`) y los reportes que ya escribió `presentation/gate.py` en
`REPORTS_DIR`. Ningún handler llama a `storage/` (el único tier autorizado
a tocar red/DB — ver `app/tests/test_architecture.py`), así que no hay
manera de que uno de estos handlers dispare una escritura a la base de
datos o al almacén de objetos: la capa que sabe hacer eso ni siquiera está
importada aquí (verificado por `tests/test_mcp_server.py`, que además
grepea este archivo contra los verbos SQL de escritura prohibidos).

Los splits solo existen dentro de un release cortado por `presentation/release.py`
(`versions.json` + `releases/<version>/{quality,splits}.json`, P2-45); no hay
un `splits.json` vigente fuera de un release. Cuando un reporte no existe, las
herramientas devuelven `available: false` en vez de fallar o inventar datos —
mismo principio que ya usa el frontend (`frontend/src/pipeline/dataSource.ts`).
"""

import json
import re
from pathlib import Path
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from ingestion.loader import load_dataset
from ingestion.models import CocoDataset
from presentation.contracts import DatasetRelease, VersionsReport
from storage.settings import Settings

SEMVER = re.compile(r"v(\d+)\.(\d+)\.(\d+)")

READ_ONLY = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)


def _dataset_summary(dataset: CocoDataset) -> dict[str, Any]:
    """Pura: cuenta imágenes distintas por categoría (igual que analyzers/imbalance.py)."""
    images_by_category: dict[int, set[int]] = {
        category.id: set() for category in dataset.categories
    }
    for annotation in dataset.annotations:
        images_by_category[annotation.category_id].add(annotation.image_id)

    return {
        "total_images": len(dataset.images),
        "total_annotations": len(dataset.annotations),
        "images_per_category": {
            category.name: len(images_by_category[category.id]) for category in dataset.categories
        },
    }


def _read_report(reports_dir: Path, filename: str) -> dict[str, Any]:
    """Pura dado el filesystem: no escribe nada, solo intenta leer un JSON ya generado."""
    path = reports_dir / filename
    if not path.exists():
        return {"available": False, "reason": f"{filename} todavía no existe"}
    return {"available": True, "data": json.loads(path.read_text(encoding="utf-8"))}


def _latest_release(releases: list[DatasetRelease]) -> DatasetRelease | None:
    """El de mayor versión semántica `vMAJOR.MINOR.PATCH` (la única que produce `cut_release`).

    El contrato de `versions.json` no ordena el catálogo ni marca un "último", así que
    no se usa la posición en la lista. Los identificadores que no son semver no compiten.
    """
    versioned = [
        (tuple(int(part) for part in match.groups()), release)
        for release in releases
        if (match := SEMVER.fullmatch(release.dataset_version))
    ]
    return max(versioned, key=lambda pair: pair[0])[1] if versioned else None


def _read_release_report(
    reports_dir: Path, kind: Literal["quality", "splits"], dataset_version: str | None
) -> dict[str, Any]:
    """Lee el reporte congelado de un release; sin versión usa el de mayor versión semántica.

    `VersionsReport` ya rechaza rutas con `..` o absolutas, por eso no se vuelve a
    validar aquí. Si ningún release tiene versión semántica no se adivina cuál es el
    último: se pide indicar `dataset_version`.
    """
    catalog_path = reports_dir / "versions.json"
    if not catalog_path.exists():
        return {"available": False, "reason": "versions.json todavía no existe: no hay releases"}
    releases = VersionsReport.model_validate_json(catalog_path.read_text(encoding="utf-8")).releases
    if not releases:
        return {"available": False, "reason": "el catálogo no tiene ningún release cortado"}

    if dataset_version is None:
        release = _latest_release(releases)
        if release is None:
            names = ", ".join(r.dataset_version for r in releases)
            return {
                "available": False,
                "reason": f"ningún release tiene versión vX.Y.Z; indica dataset_version ({names})",
            }
    else:
        release = next((r for r in releases if r.dataset_version == dataset_version), None)
        if release is None:
            return {"available": False, "reason": f"no existe el release '{dataset_version}'"}
    return _read_report(
        reports_dir, release.quality_file if kind == "quality" else release.splits_file
    )


def _find_check(quality_report: dict[str, Any], check_name: str) -> dict[str, Any] | None:
    return next(
        (check for check in quality_report["checks"] if check["check_name"] == check_name), None
    )


def build_server(settings: Settings | None = None) -> MCPServer:
    settings = settings if settings is not None else Settings()
    server = MCPServer(
        name="dataset-quality-copilot",
        instructions=(
            "Consultas de solo lectura sobre el dataset de calidad (Fase 2) y sus "
            "resultados: dataset real, reporte de calidad, splits y versiones."
        ),
    )

    @server.tool(
        name="get_dataset_summary",
        description=(
            "Resumen del dataset real (copia de trabajo): total de imágenes, total de "
            "anotaciones y conteo de imágenes distintas por categoría, con su "
            "`dataset_version`."
        ),
        annotations=READ_ONLY,
    )
    def get_dataset_summary() -> dict[str, Any]:
        annotations_dir = settings.dataset_dir / "annotations"
        if not annotations_dir.exists():
            return {
                "available": False,
                "reason": (
                    "data/raw/annotations no existe en este entorno: "
                    "falta traer el dataset (dvc pull)"
                ),
            }
        dataset = load_dataset(annotations_dir)
        return {
            "available": True,
            "dataset_version": settings.dataset_version,
            **_dataset_summary(dataset),
        }

    @server.tool(
        name="get_quality_report",
        description=(
            "Reporte de calidad real generado por la compuerta: status global y cada "
            "check con su valor, umbral-acción y detalle. Sin `dataset_version` devuelve "
            "el reporte vigente (copia de trabajo, su dataset_version suele ser "
            "'local-dev'); con `dataset_version` (p. ej. 'v0.1.0') devuelve el reporte "
            "congelado de ese release."
        ),
        annotations=READ_ONLY,
    )
    def get_quality_report(dataset_version: str | None = None) -> dict[str, Any]:
        if dataset_version is None:
            return _read_report(settings.reports_dir, "quality.json")
        return _read_release_report(settings.reports_dir, "quality", dataset_version)

    @server.tool(
        name="get_check_result",
        description=(
            "Resultado de un check de calidad específico por nombre (p. ej. 'spatial_bias')."
        ),
        annotations=READ_ONLY,
    )
    def get_check_result(check_name: str) -> dict[str, Any]:
        report = _read_report(settings.reports_dir, "quality.json")
        if not report["available"]:
            return report
        match = _find_check(report["data"], check_name)
        if match is None:
            return {"available": False, "reason": f"no existe un check llamado '{check_name}'"}
        return {
            "available": True,
            "dataset_version": report["data"]["dataset_version"],
            "data": match,
        }

    @server.tool(
        name="get_splits_report",
        description=(
            "Reporte de splits (train/validation/test) de un release cortado. Sin "
            "`dataset_version` usa el release de mayor versión semántica (vX.Y.Z); "
            "`data.dataset_version` dice de cuál es."
        ),
        annotations=READ_ONLY,
    )
    def get_splits_report(dataset_version: str | None = None) -> dict[str, Any]:
        return _read_release_report(settings.reports_dir, "splits", dataset_version)

    @server.tool(
        name="get_versions_report",
        description="Catálogo de releases publicadas del dataset (versions.json), si ya existe.",
        annotations=READ_ONLY,
    )
    def get_versions_report() -> dict[str, Any]:
        return _read_report(settings.reports_dir, "versions.json")

    return server


def main() -> None:
    server = build_server()
    server.run()


if __name__ == "__main__":
    main()
