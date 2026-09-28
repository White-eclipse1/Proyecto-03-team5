"""Tier 6 — corte y diff de releases versionados (P2-45).

Un release congela un `dataset_version` semver (`vMAJOR.MINOR.PATCH`) junto
con su `quality.json` y `splits.json` bajo
`REPORTS_DIR/releases/<version>/`, y agrega la entrada al catálogo
`REPORTS_DIR/versions.json` (`VersionsReport`, contrato v1.0 de P2-12).

`cut_release`/`diff_releases` reciben `dataset_dir`/`reports_dir`/`policy`
directos, no un `Settings` completo — mismo patrón que
`gate.build_quality_report`, para poder probarlos sin declarar
`DATABASE_URL`/`MINIO_*` (campos requeridos por `Settings` mas nunca
usados aquí). `main()` es el único punto que construye `Settings()`.

No recalcula el "content hash" del dataset entre DEV/PROD: ese hash ya es
el md5 en `data/raw/annotations.dvc`/`data/raw/images.dvc` que DVC calcula
al hacer `dvc add` — es el mismo valor sin importar el remote, por
construcción. Verificar que ambos remotes lo tengan de verdad es
`dvc status -r dev`/`dvc status -r prod` reportando "in sync" (ver
README.md, sección P2-45) — no hace falta reimplementarlo aquí.
"""

import argparse
import json
import logging
import re
from pathlib import Path

from policies.models import QualityPolicy
from presentation.contracts import DatasetRelease, VersionsReport
from presentation.gate import evaluate_dataset
from presentation.splits import build_splits_report
from splits.models import SplitsConfig, load_splits_config

logger = logging.getLogger("dataset-release")

SEMVER_PATTERN = re.compile(r"^v\d+\.\d+\.\d+$")


def _load_catalog(catalog_path: Path) -> VersionsReport:
    if not catalog_path.exists():
        return VersionsReport(schema_version="1.0", releases=[])
    return VersionsReport.model_validate_json(catalog_path.read_text(encoding="utf-8"))


def cut_release(
    version: str,
    *,
    dataset_dir: Path,
    reports_dir: Path,
    policy: QualityPolicy,
    splits_config: SplitsConfig | None = None,
) -> DatasetRelease:
    """Corre el gate y los splits reales contra el dataset actual y los congela.

    Rechaza un `version` ya existente en el catálogo: un release es
    inmutable una vez cortado, no se sobreescribe.
    """
    if not SEMVER_PATTERN.fullmatch(version):
        raise ValueError(f"version debe seguir vMAJOR.MINOR.PATCH, recibido: {version!r}")

    catalog_path = reports_dir / "versions.json"
    catalog = _load_catalog(catalog_path)
    if any(release.dataset_version == version for release in catalog.releases):
        raise ValueError(f"el release {version!r} ya existe en el catálogo")

    # One configuration snapshot per operation; custom callers can inject it.
    splits_config = splits_config if splits_config is not None else load_splits_config()
    quality_report, split_result = evaluate_dataset(
        dataset_dir=dataset_dir,
        policy=policy,
        dataset_version=version,
        splits_config=splits_config,
    )
    if quality_report.status == "failed":
        raise ValueError("No se puede cortar el release: la compuerta de calidad está en failed.")
    splits_report = build_splits_report(split_result, dataset_version=version)

    release_dir = reports_dir / "releases" / version
    release_dir.mkdir(parents=True, exist_ok=True)
    (release_dir / "quality.json").write_text(
        quality_report.model_dump_json(indent=2), encoding="utf-8"
    )
    (release_dir / "splits.json").write_text(
        splits_report.model_dump_json(indent=2), encoding="utf-8"
    )

    release = DatasetRelease(
        dataset_version=version,
        quality_file=f"releases/{version}/quality.json",
        splits_file=f"releases/{version}/splits.json",
    )
    catalog = VersionsReport(schema_version="1.0", releases=[*catalog.releases, release])
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(catalog.model_dump_json(indent=2), encoding="utf-8")

    logger.info("Release %s cortado en %s", version, release_dir)
    return release


def _category_counts(quality_report_dict: dict) -> dict[str, int]:
    check = next(
        c for c in quality_report_dict["checks"] if c["check_name"] == "max_imbalance_ratio"
    )
    return {
        category["category_name"]: category["image_count"]
        for category in check["details"]["images_per_category"]
    }


def _quality_check(quality_report_dict: dict, name: str) -> dict:
    return next(check for check in quality_report_dict["checks"] if check["check_name"] == name)


def _annotation_count(quality_report_dict: dict) -> int:
    return int(
        _quality_check(quality_report_dict, "degenerate_boxes")["details"]["total_annotations"]
    )


def _classes_crossing_minimum(quality_a: dict, quality_b: dict) -> dict[str, list[str]]:
    min_check_a = _quality_check(quality_a, "min_images_per_class")
    min_check_b = _quality_check(quality_b, "min_images_per_class")
    threshold_a = min_check_a["details"]["criterion"]["threshold"]
    threshold_b = min_check_b["details"]["criterion"]["threshold"]
    if threshold_a != threshold_b:
        raise ValueError("No se puede comparar releases con mínimos de imágenes distintos")

    counts_a, counts_b = _category_counts(quality_a), _category_counts(quality_b)
    categories = sorted(set(counts_a) | set(counts_b))
    entered, left = [], []
    for category in categories:
        before = counts_a.get(category, 0) >= threshold_a
        after = counts_b.get(category, 0) >= threshold_b
        if not before and after:
            entered.append(category)
        elif before and not after:
            left.append(category)
    return {"entered": entered, "left": left}


def diff_releases(version_a: str, version_b: str, *, reports_dir: Path) -> dict:
    """Compara dos releases ya cortados: conteo por categoría y status de cada check.

    Ambos deben existir en el catálogo; no corre el gate de nuevo, lee los
    JSON ya congelados — un diff no debe poder ver un dataset distinto al
    que el release realmente describió.
    """
    catalog = _load_catalog(reports_dir / "versions.json")
    by_version = {release.dataset_version: release for release in catalog.releases}
    for version in (version_a, version_b):
        if version not in by_version:
            raise ValueError(f"el release {version!r} no existe en el catálogo")

    def _load_quality(version: str) -> dict:
        path = reports_dir / by_version[version].quality_file
        return json.loads(path.read_text(encoding="utf-8"))

    quality_a, quality_b = _load_quality(version_a), _load_quality(version_b)
    counts_a, counts_b = _category_counts(quality_a), _category_counts(quality_b)
    categories = sorted(set(counts_a) | set(counts_b))

    checks_a = {c["check_name"]: c for c in quality_a["checks"]}
    checks_b = {c["check_name"]: c for c in quality_b["checks"]}
    check_names = sorted(set(checks_a) | set(checks_b))

    check_diff = {}
    for name in check_names:
        check_a, check_b = checks_a.get(name), checks_b.get(name)
        metric_a = check_a["metric_value"] if check_a is not None else None
        metric_b = check_b["metric_value"] if check_b is not None else None
        check_diff[name] = {
            "from": check_a["passed"] if check_a is not None else None,
            "to": check_b["passed"] if check_b is not None else None,
            "metric_from": metric_a,
            "metric_to": metric_b,
            "metric_delta": metric_b - metric_a
            if isinstance(metric_a, (int, float)) and isinstance(metric_b, (int, float))
            else None,
        }

    return {
        "from": version_a,
        "to": version_b,
        "status": {"from": quality_a["status"], "to": quality_b["status"]},
        "images_per_category": {
            name: {"from": counts_a.get(name, 0), "to": counts_b.get(name, 0)}
            for name in categories
        },
        "image_deltas": {
            name: counts_b.get(name, 0) - counts_a.get(name, 0) for name in categories
        },
        "annotations": {
            "from": _annotation_count(quality_a),
            "to": _annotation_count(quality_b),
            "delta": _annotation_count(quality_b) - _annotation_count(quality_a),
        },
        "classes_crossing_minimum": _classes_crossing_minimum(quality_a, quality_b),
        "checks": check_diff,
    }


def main(argv: list[str] | None = None) -> None:
    """Sin exit code propio: un fallo real se señala con una excepción (no
    con un entero), así que no hay dos ramas de éxito devolviendo el mismo
    valor por separado."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="presentation.release")
    subparsers = parser.add_subparsers(dest="action", required=True)

    cut_parser = subparsers.add_parser("cut", help="Corta un release nuevo del dataset actual.")
    cut_parser.add_argument("version", help="vMAJOR.MINOR.PATCH")

    diff_parser = subparsers.add_parser("diff", help="Compara dos releases ya cortados.")
    diff_parser.add_argument("version_a")
    diff_parser.add_argument("version_b")

    args = parser.parse_args(argv)

    from storage.settings import Settings

    settings = Settings()

    if args.action == "cut":
        cut_release(
            args.version,
            dataset_dir=settings.dataset_dir,
            reports_dir=settings.reports_dir,
            policy=settings.quality,
            splits_config=load_splits_config(),
        )
    else:
        result = diff_releases(args.version_a, args.version_b, reports_dir=settings.reports_dir)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
