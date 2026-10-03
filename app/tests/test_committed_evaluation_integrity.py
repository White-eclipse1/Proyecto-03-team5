"""OPS-10 (rúbrica 7.1): la evaluación final versionada es consistente consigo misma.

Si alguien cambia una predicción del CSV o del JSON de ML-09 (o del reporte de ML-10),
estas pruebas fallan: CSV y JSON coinciden muestra por muestra, la matriz y las métricas
del JSON se recalculan desde el CSV, y el reporte de ML-10 tiene las mismas cifras.
"""

import json
from pathlib import Path

import pytest

from classification.quality_audit import flat_metrics, read_predictions, recompute
from presentation.ml_contracts import Evaluation

REPORTS = Path(__file__).resolve().parents[2] / "reports"
TEST_DIR = REPORTS / "evaluations" / "test"


@pytest.fixture(scope="module")
def committed():
    [evaluation_path] = sorted(TEST_DIR.glob("*.json"))
    evaluation = Evaluation.model_validate_json(evaluation_path.read_text(encoding="utf-8"))
    predictions = read_predictions(evaluation_path.with_suffix(".predictions.csv"))
    return evaluation, predictions


def test_csv_and_json_predictions_match_sample_by_sample(committed):
    evaluation, predictions = committed
    from_json = {
        p.annotation_id: (p.image_id, p.true_class, p.predicted_class, p.probabilities)
        for p in evaluation.predictions
    }
    from_csv = {
        p.annotation_id: (p.image_id, p.true_class, p.predicted_class, p.probabilities)
        for p in predictions
    }

    assert from_csv == from_json


def test_matrix_and_metrics_are_recomputed_from_the_csv(committed):
    evaluation, predictions = committed
    metrics = recompute(predictions, evaluation.class_names)

    assert metrics["confusion_matrix"] == evaluation.confusion_matrix
    assert metrics["accuracy"] == evaluation.metrics.accuracy_top1
    assert metrics["f1_macro"] == evaluation.metrics.f1_macro


def test_ml10_report_has_the_figures_recomputed_from_the_csv(committed):
    _, predictions = committed
    report = json.loads((REPORTS / "quality" / "ml10_quality_audit.json").read_text("utf-8"))

    assert report["verified"] is True
    assert flat_metrics(recompute(predictions)) == flat_metrics(report["metrics"])
