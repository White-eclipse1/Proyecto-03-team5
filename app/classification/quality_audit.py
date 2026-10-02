"""ML-10 — auditoría final de calidad del modelo y análisis de errores.

Verifica de forma **independiente** la evaluación final de ML-09:

1. Lee solo el archivo de predicciones (`<id>.predictions.csv`) y rechaza filas
   inconsistentes: `crop_id` repetido o distinto de `img<image_id>-ann<annotation_id>`,
   clases desconocidas, probabilidades fuera de [0, 1] o que no suman 1, clase
   predicha que no es el argmax, columna `correct` incoherente.
2. Recalcula desde esas filas, sin reutilizar el código de métricas de ML-09:
   matriz de confusión, accuracy (fracción exacta), meta 0.85 con enteros, F1 macro,
   precision/recall/F1/support por clase, baseline de clase mayoritaria sobre el
   mismo test, confusión más frecuente, clase más confundida y si el accuracy oculta
   un recall por clase bajo 0.85.
3. Comprueba que las predicciones son exactamente el split `test` del manifiesto
   congelado y que el candidato no cambió después de consultar el test.
4. Compara cada cifra con MLflow (métricas `test_*`, tag de la meta y matriz del
   artefacto), con la API (`GET /api/ml/evaluation`, también muestra por muestra) y
   con el portal (la pantalla Evaluation renderizada en Chrome headless).
5. Elige ejemplos reales del test (todos los errores y, por clase, el acierto más y
   el menos confiado) y verifica que la API sirve su recorte.

No escribe en MLflow ni toca la evaluación de ML-09: solo escribe el reporte.

    uv run python -m classification.quality_audit
"""

import csv
import json
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from html.parser import HTMLParser
from pathlib import Path

from mlflow.tracking import MlflowClient

from crops.classes import CLASS_NAMES

TARGET_ACCURACY = Fraction(85, 100)
PROBABILITY_TOLERANCE = 1e-6
# `EXAMPLES_PAGE_SIZE` de frontend/src/ml/pages/Evaluation.tsx: errores visibles al abrir.
PORTAL_PAGE_SIZE = 12
REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "reports"
AUDIT_PATH = REPORTS_DIR / "quality" / "ml10_quality_audit.json"
CANDIDATE_PATH = REPORTS_DIR / "candidates" / "ml08_candidate.json"
TEST_EVALUATIONS_DIR = REPORTS_DIR / "evaluations" / "test"
DEFAULT_API_URL = "http://localhost:8080/api/ml"
DEFAULT_PORTAL_URL = "http://localhost:8080/ml/evaluation"
DEFAULT_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


class PredictionsFileError(ValueError):
    """El archivo de predicciones no es consistente: no se calcula nada con él."""


@dataclass(frozen=True)
class Prediction:
    crop_id: str
    image_id: int
    annotation_id: int
    true_class: str
    predicted_class: str
    probabilities: dict[str, float]
    correct: bool

    @property
    def confidence(self) -> float:
        return self.probabilities[self.predicted_class]


def meets_target(correct: int, total: int, target: Fraction = TARGET_ACCURACY) -> bool:
    """`correct / total >= target` con enteros: sin dividir ni redondear."""
    return correct * target.denominator >= target.numerator * total


# --- Agent Test: solo el archivo de predicciones ----------------------------------------


def _check_row(row: dict, where: str, class_names) -> tuple[list[str], Prediction | None]:
    try:
        image_id, annotation_id = int(row["image_id"]), int(row["annotation_id"])
    except ValueError:
        return [f"{where}: image_id y annotation_id deben ser enteros"], None
    problems = []
    if row["crop_id"] != f"img{image_id}-ann{annotation_id}":
        problems.append(f"{where}: crop_id no corresponde a image_id {image_id} y annotation_id")
    unknown = sorted({row["true_class"], row["predicted_class"]} - set(class_names))
    if unknown:
        return [*problems, f"{where}: clases fuera de {list(class_names)}: {unknown}"], None
    try:
        probabilities = {name: float(row[f"p_{name}"]) for name in class_names}
    except ValueError:
        return [*problems, f"{where}: probabilidades no numéricas"], None
    if any(not 0.0 <= value <= 1.0 for value in probabilities.values()):
        problems.append(f"{where}: las probabilidades deben estar entre 0 y 1")
    elif abs(sum(probabilities.values()) - 1.0) > PROBABILITY_TOLERANCE:
        problems.append(f"{where}: las probabilidades suman {sum(probabilities.values())!r}")
    argmax = max(class_names, key=probabilities.__getitem__)
    if row["predicted_class"] != argmax:
        problems.append(f"{where}: predicted_class {row['predicted_class']} no es el argmax")
    correct = row["true_class"] == row["predicted_class"]
    if row["correct"] != str(correct):
        problems.append(f"{where}: correct={row['correct']} no coincide con real y predicha")
    prediction = Prediction(
        crop_id=row["crop_id"],
        image_id=image_id,
        annotation_id=annotation_id,
        true_class=row["true_class"],
        predicted_class=row["predicted_class"],
        probabilities=probabilities,
        correct=correct,
    )
    return problems, prediction


def read_predictions(path: Path, class_names=CLASS_NAMES) -> list[Prediction]:
    """Las filas del CSV de predicciones, validadas. Ninguna otra fuente."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    required = [
        "crop_id",
        "image_id",
        "annotation_id",
        "true_class",
        "predicted_class",
        *(f"p_{name}" for name in class_names),
        "correct",
    ]
    missing = [column for column in required if column not in (reader.fieldnames or [])]
    if missing:
        raise PredictionsFileError(f"Faltan columnas en {path.name}: {missing}")
    if not rows:
        raise PredictionsFileError(f"{path.name} está vacío: no hay predicciones")
    problems, predictions, seen = [], [], set()
    for line, row in enumerate(rows, start=2):
        where = f"línea {line} ({row['crop_id']})"
        if row["crop_id"] in seen:
            problems.append(f"{where}: crop_id repetido")
        seen.add(row["crop_id"])
        found, prediction = _check_row(row, where, class_names)
        problems += found
        if prediction is not None:
            predictions.append(prediction)
    if problems:
        raise PredictionsFileError(f"{path.name} inconsistente: " + "; ".join(problems))
    return predictions


def recompute(predictions: list[Prediction], class_names=CLASS_NAMES) -> dict:
    """Todas las métricas de test desde las predicciones (filas: real, columnas: predicha)."""
    names = list(class_names)
    position = {name: index for index, name in enumerate(names)}
    matrix = [[0] * len(names) for _ in names]
    for prediction in predictions:
        matrix[position[prediction.true_class]][position[prediction.predicted_class]] += 1
    total = len(predictions)
    correct = sum(matrix[i][i] for i in range(len(names)))

    per_class = []
    for index, name in enumerate(names):
        hits = matrix[index][index]
        support = sum(matrix[index])
        predicted = sum(row[index] for row in matrix)
        precision = hits / predicted if predicted else 0.0
        recall = hits / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(
            {
                "class_name": name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "hits": hits,
                "errors": support - hits,
                "low_recall": support > 0 and not meets_target(hits, support),
            }
        )

    majority = max(entry["support"] for entry in per_class)
    pairs = [
        {"true_class": names[row], "predicted_class": names[column], "count": count}
        for row in range(len(names))
        for column, count in enumerate(matrix[row])
        if row != column and count > 0
    ]
    top_pair = max((pair["count"] for pair in pairs), default=0)
    with_errors = [entry for entry in per_class if entry["errors"] > 0]
    lowest_recall = min((entry["recall"] for entry in with_errors), default=None)
    low_recall = [entry["class_name"] for entry in per_class if entry["low_recall"]]
    target_met = meets_target(correct, total)
    accuracy = correct / total
    baseline_accuracy = majority / total
    return {
        "class_names": names,
        "confusion_matrix": matrix,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "accuracy_fraction": f"{correct}/{total}",
        "meets_target": target_met,
        "target": f"{TARGET_ACCURACY.numerator}/{TARGET_ACCURACY.denominator}",
        "f1_macro": sum(entry["f1"] for entry in per_class) / len(names),
        "per_class": per_class,
        "baseline": {
            "classes": [e["class_name"] for e in per_class if e["support"] == majority],
            "correct": majority,
            "total": total,
            "accuracy": baseline_accuracy,
        },
        "accuracy_over_baseline": accuracy - baseline_accuracy,
        "most_confused_pairs": [pair for pair in pairs if pair["count"] == top_pair],
        "most_confused_classes": [
            entry["class_name"] for entry in with_errors if entry["recall"] == lowest_recall
        ],
        "low_recall_classes": low_recall,
        "accuracy_hides_low_recall": target_met and bool(low_recall),
        "error_crop_ids": [p.crop_id for p in predictions if not p.correct],
    }


def flat_metrics(metrics: dict) -> dict[str, float]:
    """Cada cifra comparable por nombre: las de MLflow `test_<nombre>` y la matriz."""
    flat = {
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "correct": float(metrics["correct"]),
        "total": float(metrics["total"]),
    }
    for entry in metrics["per_class"]:
        name = entry["class_name"]
        for field in ("precision", "recall", "f1"):
            flat[f"{field}_{name}"] = entry[field]
        flat[f"support_{name}"] = float(entry["support"])
    flat.update(_matrix_figures(metrics["class_names"], metrics["confusion_matrix"], float))
    return flat


def _matrix_figures(class_names, matrix, kind) -> dict:
    return {
        f"confusion_{true}_{predicted}": kind(matrix[row][column])
        for row, true in enumerate(class_names)
        for column, predicted in enumerate(class_names)
    }


def compare_figures(source: str, expected: dict, reported: dict) -> list[dict]:
    """Cifra por cifra; una ausente en la fuente cuenta como distinta."""
    return [
        {
            "source": source,
            "metric": name,
            "recomputed": value,
            "reported": reported.get(name),
            "equal": name in reported and reported[name] == value,
        }
        for name, value in expected.items()
    ]


# --- Ejemplos y split congelado ---------------------------------------------------------


def _example(prediction: Prediction, kind: str) -> dict:
    return {
        "kind": kind,
        "crop_id": prediction.crop_id,
        "image_id": prediction.image_id,
        "annotation_id": prediction.annotation_id,
        "true_class": prediction.true_class,
        "predicted_class": prediction.predicted_class,
        "confidence": prediction.confidence,
        "probabilities": prediction.probabilities,
    }


def choose_examples(
    predictions: list[Prediction], hits_per_class: int = 2, class_names=CLASS_NAMES
) -> dict:
    """Todos los errores (primero el más confiado) y, por clase, aciertos extremos."""
    by_confidence = sorted(predictions, key=lambda p: (-p.confidence, p.crop_id))
    errors = [_example(p, "error") for p in by_confidence if not p.correct]
    hits = []
    for name in class_names:
        ranked = [p for p in by_confidence if p.correct and p.true_class == name]
        top = (hits_per_class + 1) // 2
        chosen = ranked[:top] + ranked[top:][-(hits_per_class - top) :]
        unique = {p.crop_id: p for p in chosen}.values()
        hits += [_example(p, "hit") for p in unique]
    return {"errors": errors, "hits": hits}


def frozen_split_problems(predictions: list[Prediction], records: list[dict]) -> list[str]:
    """Las predicciones son exactamente el split test del manifiesto, con su clase real."""
    by_crop = {record["crop_id"]: record for record in records}
    problems = []
    for prediction in predictions:
        record = by_crop.get(prediction.crop_id)
        if record is None:
            problems.append(f"{prediction.crop_id} no está en el manifiesto")
            continue
        if record["split"] != "test":
            problems.append(f"{prediction.crop_id} no es del split test")
            continue
        if record["class"] != prediction.true_class:
            problems.append(
                f"{prediction.crop_id}: clase real {prediction.true_class}, "
                f"en el manifiesto {record['class']}"
            )
        if record["source_image_id"] != prediction.image_id:
            problems.append(
                f"{prediction.crop_id}: image_id {prediction.image_id} distinto de "
                f"source_image_id {record['source_image_id']}"
            )
    test_ids = {crop_id for crop_id, record in by_crop.items() if record["split"] == "test"}
    missing = sorted(test_ids - {p.crop_id for p in predictions})
    if missing:
        problems.append(f"faltan predicciones de test: {missing}")
    return problems


def _instant(timestamp: str) -> datetime:
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


def frozen_problems(candidate: dict, evaluation: dict, run_tags: dict[str, dict]) -> list[str]:
    """El candidato evaluado es el congelado antes del test y nadie lo cambió después."""
    problems = [
        f"la evaluación tiene otro {field} que el candidato congelado"
        for field in ("run_id", "checkpoint", "dataset_version", "manifest_hash")
        if evaluation[field] != candidate[field]
    ]
    if _instant(evaluation["created_at"]) <= _instant(candidate["frozen_at"]):
        problems.append("la evaluación de test es de antes de congelar el candidato")
    tags = run_tags.get(candidate["run_id"], {})
    if tags.get("candidate") != "true":
        problems.append("el run congelado ya no tiene candidate=true en MLflow")
    frozen_tag = tags.get("candidate_frozen_at")
    if frozen_tag is None or _instant(frozen_tag) != _instant(candidate["frozen_at"]):
        problems.append(f"candidate_frozen_at={frozen_tag} distinto del frozen_at congelado")
    if "candidate_replaced_by" in tags:
        problems.append(f"el candidato fue reemplazado por {tags['candidate_replaced_by']}")
    others = sorted(
        run_id
        for run_id, other in run_tags.items()
        if run_id != candidate["run_id"] and other.get("candidate") == "true"
    )
    if others:
        problems.append(f"hay otro run marcado como candidato: {others}")
    if tags.get("test_evaluation_id") != evaluation["evaluation_id"]:
        problems.append(
            f"test_evaluation_id del run ({tags.get('test_evaluation_id')}) no es "
            f"{evaluation['evaluation_id']}"
        )
    return problems


# --- MLflow, API y portal -----------------------------------------------------------------


def mlflow_figures(client: MlflowClient, run_id: str) -> dict:
    """Métricas `test_*`, el tag de la meta y la matriz del artefacto de evaluación."""
    run = client.get_run(run_id)
    figures: dict = {
        name.removeprefix("test_"): value
        for name, value in run.data.metrics.items()
        if name.startswith("test_")
    }
    tags = run.data.tags
    if "test_accuracy_meets_target" in tags:
        figures["meets_target"] = tags["test_accuracy_meets_target"]
    evaluation_id = tags.get("test_evaluation_id")
    if evaluation_id:
        with tempfile.TemporaryDirectory() as tmp:
            local = client.download_artifacts(run_id, f"evaluation/test/{evaluation_id}.json", tmp)
            logged = json.loads(Path(local).read_text(encoding="utf-8"))
        names = logged.get("class_names", list(CLASS_NAMES))
        figures.update(_matrix_figures(names, logged["confusion_matrix"], float))
    return figures


def api_figures(overview: dict) -> dict:
    """Las cifras de `GET /api/ml/evaluation`; vacío si no publica la evaluación final."""
    evaluation = overview.get("evaluation")
    if overview.get("state") != "evaluated" or evaluation is None:
        return {}
    names, matrix = evaluation["class_names"], evaluation["confusion_matrix"]
    figures = {
        "accuracy": evaluation["metrics"]["accuracy_top1"],
        "f1_macro": evaluation["metrics"]["f1_macro"],
        "correct": float(sum(matrix[i][i] for i in range(len(names)))),
        "total": float(sum(map(sum, matrix))),
    }
    for entry in evaluation["per_class"]:
        for field in ("precision", "recall", "f1"):
            figures[f"{field}_{entry['class_name']}"] = entry[field]
        figures[f"support_{entry['class_name']}"] = float(entry["support"])
    figures.update(_matrix_figures(names, matrix, float))
    return figures


def api_prediction_differences(predictions: list[Prediction], overview: dict) -> list[str]:
    """Cada predicción de la API contra la del CSV, por `annotation_id`."""
    evaluation = overview.get("evaluation")
    if overview.get("state") != "evaluated" or evaluation is None:
        return [f"la API no publica la evaluación final (state={overview.get('state')})"]
    fields = ("image_id", "true_class", "predicted_class", "probabilities")
    api = {p["annotation_id"]: {f: p[f] for f in fields} for p in evaluation["predictions"]}
    local = {p.annotation_id: {f: getattr(p, f) for f in fields} for p in predictions}
    differences = []
    for key in sorted(api.keys() | local.keys()):
        if key not in api:
            differences.append(f"annotation_id {key}: falta en la API")
        elif key not in local:
            differences.append(f"annotation_id {key}: sobra en la API")
        else:
            changed = [f for f in fields if api[key][f] != local[key][f]]
            if changed:
                differences.append(f"annotation_id {key}: distinto en {', '.join(changed)}")
    return differences


class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tokens: list[str] = []
        self._hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._hidden += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._hidden -= 1

    def handle_data(self, data):
        if not self._hidden and data.strip():
            self.tokens.append(" ".join(data.split()))


def _after(tokens: list[str], label: str, count: int = 1) -> list[str] | None:
    if label not in tokens:
        return None
    start = tokens.index(label) + 1
    found = tokens[start : start + count]
    return found if len(found) == count else None


def _matching(tokens: list[str], pattern: str) -> list[re.Match]:
    return [m for m in (re.fullmatch(pattern, token) for token in tokens) if m]


def portal_figures(html: str, class_names=CLASS_NAMES) -> dict[str, str]:
    """Los textos de la pantalla Evaluation renderizada, tal como los ve el usuario."""
    parser = _Text()
    parser.feed(html)
    tokens = parser.tokens
    names = list(class_names)
    figures: dict[str, str] = {}
    for key, label in (
        ("accuracy", "Accuracy top-1 (test)"),
        ("f1_macro", "F1 macro (test)"),
        ("baseline_accuracy", "Baseline de clase mayoritaria"),
    ):
        if value := _after(tokens, label):
            figures[key] = value[0]
    for match in _matching(tokens, r"(\d+) de (\d+) recortes de test"):
        figures["correct"], figures["total"] = match.groups()
    for match in _matching(tokens, r"Meta 0\.85: (.+)"):
        figures["meets_target"] = match.group(1)
    for match in _matching(tokens, r"Predecir siempre «(.+)» en el mismo test"):
        figures["baseline_class"] = match.group(1)

    header = ["Clase", "Precision", "Recall", "F1", "Support"]
    starts = [i for i in range(len(tokens)) if tokens[i : i + len(header)] == header]
    if starts:
        rows = tokens[starts[0] + len(header) : starts[0] + len(header) + 5 * len(names)]
        for index in range(0, len(rows) - 4, 5):
            name, precision, recall, f1, support = rows[index : index + 5]
            if name in names:
                figures.update(
                    {
                        f"precision_{name}": precision,
                        f"recall_{name}": recall,
                        f"f1_{name}": f1,
                        f"support_{name}": support,
                    }
                )

    width = len(names) + 2  # nombre de la fila, conteos y total
    matrix = _after(tokens, "Real \\ Predicha", len(names) + 1 + width * len(names) + width)
    if matrix and matrix[: len(names) + 1] == [*names, "Total real"]:
        body = matrix[len(names) + 1 :]
        for row, true in enumerate(names):
            cells = body[row * width : (row + 1) * width]
            if cells[0] != true:
                break
            for column, predicted in enumerate(names):
                figures[f"confusion_{true}_{predicted}"] = cells[1 + column]
            figures[f"row_total_{true}"] = cells[-1]
        totals = body[len(names) * width :]
        if totals[0] == "Total predicho":
            for column, predicted in enumerate(names):
                figures[f"column_total_{predicted}"] = totals[1 + column]
            figures["matrix_total"] = totals[-1]
    for match in _matching(tokens, r"La matriz suma (\d+) = (\d+) predicciones de test\."):
        figures["predictions_count"] = match.group(2)

    if "Interpretación de errores" in tokens:
        confused = _matching(
            tokens, r"Confusión más frecuente \(real → predicha\): (\S+) → (\S+): (\d+) recortes\."
        )
        if confused:
            true, predicted, count = confused[0].groups()
            figures["most_confused"] = f"{true} → {predicted}: {count}"
        elif "Sin confusiones entre clases." in tokens:
            figures["most_confused"] = "ninguna"
        low = _matching(tokens, r"El recall de (\S+) es [\d.]+, por debajo de 0\.85\.")
        figures["low_recall_classes"] = ", ".join(m.group(1) for m in low)
        hides = any(t.startswith("El accuracy global oculta recall bajo") for t in tokens)
        figures["accuracy_hides_low_recall"] = "sí" if hides else "no"
    for key, pattern in (
        ("errors_shown", r"Incorrectos \((\d+)\)"),
        ("hits_shown", r"Correctos \((\d+)\)"),
    ):
        if found := _matching(tokens, pattern):
            figures[key] = found[0].group(1)
    if "Descargar predicciones (CSV)" in tokens:
        shown = tokens[tokens.index("Descargar predicciones (CSV)") + 1 :]
        figures["error_examples"] = ", ".join(t for t in shown if re.fullmatch(r"img\d+-ann\d+", t))
    return figures


def _fixed4(value: float) -> str:
    """`Number.prototype.toFixed(4)`: valor binario exacto, empate hacia arriba."""
    return str(Decimal(value).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def expected_portal_figures(metrics: dict) -> dict[str, str]:
    """Lo que la pantalla Evaluation debe mostrar para estas métricas recalculadas."""
    names, matrix = metrics["class_names"], metrics["confusion_matrix"]
    pairs = metrics["most_confused_pairs"]
    figures = {
        "accuracy": _fixed4(metrics["accuracy"]),
        "correct": str(metrics["correct"]),
        "total": str(metrics["total"]),
        "meets_target": "alcanzada" if metrics["meets_target"] else "no alcanzada",
        "f1_macro": _fixed4(metrics["f1_macro"]),
        "baseline_accuracy": _fixed4(metrics["baseline"]["accuracy"]),
        "baseline_class": metrics["baseline"]["classes"][0],
    }
    for entry in metrics["per_class"]:
        name = entry["class_name"]
        for field in ("precision", "recall", "f1"):
            figures[f"{field}_{name}"] = _fixed4(entry[field])
        figures[f"support_{name}"] = str(entry["support"])
    figures.update(_matrix_figures(names, matrix, str))
    for row, true in enumerate(names):
        figures[f"row_total_{true}"] = str(sum(matrix[row]))
    for column, predicted in enumerate(names):
        figures[f"column_total_{predicted}"] = str(sum(row[column] for row in matrix))
    errors = metrics["error_crop_ids"]
    figures.update(
        {
            "matrix_total": str(metrics["total"]),
            "predictions_count": str(metrics["total"]),
            "most_confused": (
                f"{pairs[0]['true_class']} → {pairs[0]['predicted_class']}: {pairs[0]['count']}"
                if pairs
                else "ninguna"
            ),
            "low_recall_classes": ", ".join(metrics["low_recall_classes"]),
            "accuracy_hides_low_recall": "sí" if metrics["accuracy_hides_low_recall"] else "no",
            "errors_shown": str(len(errors)),
            "hits_shown": str(metrics["correct"]),
            "error_examples": ", ".join(errors[:PORTAL_PAGE_SIZE]),
        }
    )
    return figures
