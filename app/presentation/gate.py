"""Tier 3 — la compuerta de calidad end-to-end (P2-22/23/24).

Corre los 7 checks de calidad contra el dataset real, arma el
`QualityReport` (contrato v1.0 de P2-12) y lo escribe a disco. Si algún
check con `action: fail` no pasa, `main()` devuelve 1 — pensado para
encadenarse como paso previo obligatorio de la siguiente etapa del
pipeline (`<comando del gate>; echo "exit=$?"` debe imprimir un código
distinto de 0).

P2-53: los siete checks comparten COCO, bytes, pares pHash y splits reales.
"""

import logging
import sys
from pathlib import Path

from analyzers.cross_split_leakage import LeakageConfig, analyze_cross_split_leakage
from analyzers.duplicates import DuplicateConfig, analyze_duplicates
from analyzers.imbalance import ImbalanceConfig, analyze_imbalance
from analyzers.invalid_boxes import InvalidBoxConfig, analyze_invalid_boxes
from analyzers.small_objects import SmallObjectConfig, analyze_small_objects
from analyzers.spatial_bias import SpatialBiasConfig, analyze_spatial_bias
from ingestion.loader import load_dataset, load_image_bytes
from policies.models import QualityPolicy
from presentation.contracts import QualityCheck, QualityReport
from splits.models import SplitsConfig, load_splits_config
from splits.stratified import SplitResult, split_dataset
from storage.settings import Settings

logger = logging.getLogger("quality-gate")


def _min_images_per_class_check(imbalance_result, policy: QualityPolicy) -> QualityCheck:
    """`min_images_per_class` no es su propio analizador (ver imbalance.py);
    se deriva de `details.classes_below_minimum`, que ya usa ese umbral."""
    classes_below = imbalance_result.details["classes_below_minimum"]
    counts = [c["image_count"] for c in imbalance_result.details["images_per_category"]]
    return QualityCheck(
        check_name="min_images_per_class",
        passed=len(classes_below) == 0,
        metric_value=float(min(counts, default=0)),
        details={"classes_below_minimum": classes_below},
        action=policy.min_images_per_class.action,
    )


def evaluate_dataset(
    *,
    dataset_dir: Path,
    policy: QualityPolicy,
    dataset_version: str,
    splits_config: SplitsConfig | None = None,
) -> tuple[QualityReport, SplitResult]:
    """Ejecuta y describe los checks desde una única política ya cargada."""
    splits_config = splits_config if splits_config is not None else load_splits_config()
    coco = load_dataset(dataset_dir / "annotations")
    coco_dict = coco.model_dump()
    images_dir = dataset_dir / "images"

    imbalance_config = ImbalanceConfig(
        min_images_per_class=policy.min_images_per_class.threshold,
        threshold=policy.max_imbalance_ratio.threshold,
    )
    small_config = SmallObjectConfig(
        width_px=policy.max_small_object_ratio.width_px,
        height_px=policy.max_small_object_ratio.height_px,
        threshold=policy.max_small_object_ratio.threshold,
    )
    invalid_config = InvalidBoxConfig(threshold=policy.degenerate_boxes.threshold)
    duplicate_config = DuplicateConfig(threshold=policy.duplicate_similarity_threshold.threshold)
    spatial_config = SpatialBiasConfig(min_std_dev=policy.min_spatial_dispersion.threshold)

    imbalance_result = analyze_imbalance(coco_dict, imbalance_config)
    small_objects_result = analyze_small_objects(coco_dict, small_config)
    invalid_boxes_result = analyze_invalid_boxes(coco_dict, invalid_config)
    image_contents = load_image_bytes(coco, images_dir)
    duplicates_result = analyze_duplicates(image_contents, duplicate_config)
    split_result = split_dataset(
        coco,
        splits_config,
        image_contents=image_contents,
        duplicate_pairs=duplicates_result.details["image_pairs"],
    )
    leakage_config = LeakageConfig(
        threshold=policy.cross_split_leakage.threshold,
        similarity_threshold=duplicate_config.threshold,
    )
    leakage_result = analyze_cross_split_leakage(
        split_result.assignments,
        duplicates_result.details["image_pairs"],
        leakage_config,
        image_ids=[image.id for image in coco.images],
    )
    leakage_result.details["splits_config"] = splits_config.model_dump()
    spatial_bias_result = analyze_spatial_bias(coco_dict, spatial_config)

    checks = [
        _min_images_per_class_check(imbalance_result, policy),
        QualityCheck(**imbalance_result.model_dump(), action=policy.max_imbalance_ratio.action),
        QualityCheck(
            **small_objects_result.model_dump(), action=policy.max_small_object_ratio.action
        ),
        QualityCheck(**invalid_boxes_result.model_dump(), action=policy.degenerate_boxes.action),
        # analyze_duplicates() usa "duplicate_images" como check_name interno
        # (ver app/tests/test_duplicates.py); en el reporte se renombra al
        # nombre de la política que en verdad evalúa, sin tocar el analizador.
        QualityCheck(
            **{**duplicates_result.model_dump(), "check_name": "duplicate_similarity_threshold"},
            action=policy.duplicate_similarity_threshold.action,
        ),
        QualityCheck(
            **spatial_bias_result.model_dump(), action=policy.min_spatial_dispersion.action
        ),
        QualityCheck(**leakage_result.model_dump(), action=policy.cross_split_leakage.action),
    ]

    # Los criterios describen el cumplimiento; no recalculan `passed`.
    criteria = [
        {"threshold": imbalance_config.min_images_per_class, "operator": ">="},
        {
            "threshold": imbalance_config.threshold,
            "operator": "<=",
            "requires_ratio_defined": True,
        },
        {"threshold": small_config.threshold, "operator": "<="},
        {"threshold": invalid_config.threshold, "operator": "<="},
        # Cero pares es el criterio existente, no el umbral de similitud pHash.
        {"threshold": 0, "operator": "=="},
        {"threshold": spatial_config.min_std_dev, "operator": ">="},
        {"threshold": leakage_config.threshold, "operator": "<="},
    ]
    for check, criterion in zip(checks, criteria, strict=True):
        check.details["criterion"] = {"metric": "metric_value", **criterion}
    small_objects_check = checks[2]
    small_objects_check.details["small_box_detection"] = {
        "width_px": small_config.width_px,
        "height_px": small_config.height_px,
        "operator": "<",
        "combination": "and",
    }
    checks[4].details["similarity_operator"] = ">="
    checks[4].details["similarity_formula"] = "1 - hamming_distance / hash_bits"

    if any(check.action == "fail" and not check.passed for check in checks):
        status = "failed"
    elif any(not check.passed for check in checks):
        status = "warning"
    else:
        status = "passed"

    report = QualityReport(
        schema_version="1.0",
        dataset_version=dataset_version,
        status=status,
        checks=checks,
    )

    return report, split_result


def build_quality_report(
    *,
    dataset_dir: Path,
    policy: QualityPolicy,
    dataset_version: str,
    splits_config: SplitsConfig | None = None,
) -> QualityReport:
    """Compatible report-only API; release consumes evaluate_dataset's shared split."""
    report, _ = evaluate_dataset(
        dataset_dir=dataset_dir,
        policy=policy,
        dataset_version=dataset_version,
        splits_config=splits_config,
    )
    return report


def run(settings: Settings | None = None) -> QualityReport:
    """Corre la compuerta y escribe `quality.json`; no decide el exit code (ver main())."""
    settings = settings if settings is not None else Settings()
    report = build_quality_report(
        dataset_dir=settings.dataset_dir,
        policy=settings.quality,
        dataset_version=settings.dataset_version,
    )

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = settings.reports_dir / "quality.json"
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    logger.info("quality.json escrito en %s (status=%s).", report_path, report.status)

    for check in report.checks:
        outcome = "PASS" if check.passed else "FAIL"
        logger.info("  %-32s %-4s action=%s", check.check_name, outcome, check.action)

    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = run()

    if report.status == "failed":
        logger.error("Compuerta de calidad BLOQUEADA: al menos un check fail no pasó.")
        return 1

    logger.info("Compuerta de calidad OK (status=%s).", report.status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
