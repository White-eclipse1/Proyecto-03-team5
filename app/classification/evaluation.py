"""ML-09 — evaluación final del candidato congelado sobre el test, auditable.

`evaluate_frozen_candidate` es la **única** evaluación final del test:

1. Exige el candidato congelado de ML-08 (`require_frozen_candidate`). Descarga
   su checkpoint de MLflow y verifica que el sha256 sea el registrado al congelar.
2. Carga **solo** el split `test` del manifiesto P3 con el `image_size` del
   checkpoint. Exige que el manifiesto sea el del candidato y que el
   preprocesamiento no tenga pasos aleatorios.
3. Predice en modo eval y sin gradiente: no hay optimizador, ni entrenamiento, ni
   ajuste de umbrales o hiperparámetros.
4. Escribe en `reports/evaluations/test/`:
   - `<id>.json`: contrato `Evaluation` de APP-01, que valida la matriz y las
     métricas;
   - `<id>.predictions.csv`: `crop_id`, `image_id`, `annotation_id`, clase real,
     clase predicha, probabilidades y acierto, por crop.
5. Registra en el run del candidato las métricas `test_*`, los tags
   (`test_accuracy_meets_target` compara con 0.85 **sin redondear**) y los dos
   archivos como artefactos.

Si ya existe una evaluación de test, no se repite. `audit_evaluation` vuelve a
inferir con el mismo checkpoint y manifiesto y compara, sin escribir nada, el
registro completo de cada muestra en el JSON y en el CSV, la cabecera, la matriz y
las métricas.
`verify_against_mlflow` recalcula accuracy y F1 desde el CSV y los compara con
MLflow (Agent Test).

    uv run python -m classification.evaluation evaluate
    uv run python -m classification.evaluation audit
    uv run python -m classification.evaluation verify
"""

import argparse
import csv
import hashlib
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient

from classification.dataset import build_dataloader, load_split_with_image_size
from classification.model import DogCatResNet18, load_checkpoint
from classification.selection import (
    CANDIDATE_PATH,
    EVALUATIONS_DIR,
    CandidateSelection,
    _evaluations,
    require_frozen_candidate,
)
from classification.training import CHECKPOINT_ARTIFACT, DataPaths
from classification.transforms import random_transform_names
from crops.classes import CLASS_NAMES
from crops.models import CropReport
from presentation.ml_contracts import (
    ClassMetrics,
    CropPrediction,
    Evaluation,
    EvaluationMetrics,
    f1_score,
    ratio,
)

TARGET_ACCURACY = 0.85
TEST_DIR = "test"
CSV_FIELDS = (
    "crop_id",
    "image_id",
    "annotation_id",
    "true_class",
    "predicted_class",
    *(f"p_{name}" for name in CLASS_NAMES),
    "correct",
)
# Cabecera de `Evaluation` que la auditoría recalcula (no evaluation_id ni created_at).
AUDITED_FIELDS = (
    "run_id",
    "checkpoint",
    "dataset_version",
    "manifest_hash",
    "split",
    "class_names",
    "per_class",
    "confusion_matrix",
    "metrics",
)


@dataclass(frozen=True)
class EvaluationRecord:
    evaluation: Evaluation
    evaluation_path: Path
    predictions_path: Path
    correct: int
    total: int
    meets_target: bool


def compute_metrics(true: list[str], predicted: list[str], class_names: list[str]) -> dict:
    """Matriz (filas = real, columnas = predicha) y métricas, como las recalcula el contrato."""
    if len(true) != len(predicted):
        raise ValueError(f"Distinta longitud: {len(true)} reales y {len(predicted)} predichas")
    position = {name: index for index, name in enumerate(class_names)}
    unknown = sorted({label for label in (*true, *predicted) if label not in position})
    if unknown:
        raise ValueError(f"Clases fuera de {class_names}: {unknown}")
    matrix = [[0] * len(class_names) for _ in class_names]
    for real, guess in zip(true, predicted, strict=True):
        matrix[position[real]][position[guess]] += 1
    per_class = []
    for index, name in enumerate(class_names):
        hits = matrix[index][index]
        precision = ratio(hits, sum(row[index] for row in matrix))
        recall = ratio(hits, sum(matrix[index]))
        per_class.append(
            {
                "class_name": name,
                "precision": precision,
                "recall": recall,
                "f1": f1_score(precision, recall),
                "support": sum(matrix[index]),
            }
        )
    correct = sum(matrix[index][index] for index in range(len(class_names)))
    total = len(true)
    return {
        "confusion_matrix": matrix,
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "f1_macro": sum(entry["f1"] for entry in per_class) / len(class_names),
        "per_class": per_class,
    }


def meets_target(accuracy: float, target: float = TARGET_ACCURACY) -> bool:
    """Comparación exacta con la meta: sin redondear la accuracy antes."""
    return accuracy >= target


def _verified_model(client: MlflowClient, candidate: CandidateSelection) -> DogCatResNet18:
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(client.download_artifacts(candidate.run_id, CHECKPOINT_ARTIFACT, tmp))
        digest = hashlib.sha256(local.read_bytes()).hexdigest()
        if digest != candidate.checkpoint_sha256:
            raise ValueError(
                f"El checkpoint de {candidate.run_id} tiene sha256 {digest}, no el congelado "
                f"{candidate.checkpoint_sha256}"
            )
        return load_checkpoint(local)


def _predict(client: MlflowClient, candidate: CandidateSelection, data: DataPaths) -> list[dict]:
    model = _verified_model(client, candidate)
    test_set = load_split_with_image_size(
        data.manifest,
        data.crop_report,
        crops_dir=data.crops_dir,
        split="test",
        image_size=model.config.image_size,
    )
    if test_set.manifest_hash != candidate.manifest_hash:
        raise ValueError(
            f"manifest_hash del test ({test_set.manifest_hash}) distinto del candidato "
            f"({candidate.manifest_hash})"
        )
    if test_set.dataset_version != candidate.dataset_version:
        raise ValueError(f"El test es del release {test_set.dataset_version}, no del candidato")
    random_steps = random_transform_names(test_set.transform)
    if random_steps:
        raise ValueError(f"El test no debe tener transforms aleatorios: {random_steps}")

    crops = {
        crop.crop_id: crop
        for crop in CropReport.model_validate_json(
            data.crop_report.read_text(encoding="utf-8")
        ).crops
    }
    index_to_class = {index: name for name, index in model.class_map.items()}
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in build_dataloader(test_set, batch_size=32, seed=0):
            probabilities = torch.softmax(model(batch["image"]), dim=1)
            for crop_id, true_class, row in zip(
                batch["crop_id"], batch["class_name"], probabilities.tolist(), strict=True
            ):
                predicted = index_to_class[max(range(len(row)), key=row.__getitem__)]
                crop = crops[crop_id]
                rows.append(
                    {
                        "crop_id": crop_id,
                        "image_id": crop.image_id,
                        "annotation_id": crop.annotation_id,
                        "true_class": true_class,
                        "predicted_class": predicted,
                        "probabilities": {index_to_class[i]: p for i, p in enumerate(row)},
                    }
                )
    return rows


def _build_evaluation(
    candidate: CandidateSelection, rows: list[dict], created_at: datetime
) -> tuple[Evaluation, dict]:
    class_names = list(CLASS_NAMES)
    metrics = compute_metrics(
        [row["true_class"] for row in rows], [row["predicted_class"] for row in rows], class_names
    )
    evaluation = Evaluation(
        evaluation_id=f"test-{candidate.entry}-{created_at:%Y%m%dT%H%M%SZ}",
        run_id=candidate.run_id,
        checkpoint=candidate.checkpoint,
        model_name=None,
        model_version=None,
        dataset_version=candidate.dataset_version,
        manifest_hash=candidate.manifest_hash,
        split="test",
        class_names=class_names,
        metrics=EvaluationMetrics(accuracy_top1=metrics["accuracy"], f1_macro=metrics["f1_macro"]),
        per_class=[ClassMetrics(**entry) for entry in metrics["per_class"]],
        confusion_matrix=metrics["confusion_matrix"],
        predictions=[
            CropPrediction(
                image_id=row["image_id"],
                annotation_id=row["annotation_id"],
                true_class=row["true_class"],
                predicted_class=row["predicted_class"],
                probabilities=row["probabilities"],
            )
            for row in rows
        ],
        created_at=created_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
    )
    return evaluation, metrics


def _csv_row(row: dict) -> dict[str, str]:
    return {
        "crop_id": row["crop_id"],
        "image_id": str(row["image_id"]),
        "annotation_id": str(row["annotation_id"]),
        "true_class": row["true_class"],
        "predicted_class": row["predicted_class"],
        **{f"p_{name}": repr(row["probabilities"][name]) for name in CLASS_NAMES},
        "correct": str(row["true_class"] == row["predicted_class"]),
    }


def _write_predictions_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(_csv_row(row))


def _mlflow_metrics(metrics: dict) -> dict[str, float]:
    values = {
        "test_accuracy": metrics["accuracy"],
        "test_f1_macro": metrics["f1_macro"],
        "test_correct": float(metrics["correct"]),
        "test_total": float(metrics["total"]),
    }
    for entry in metrics["per_class"]:
        name = entry["class_name"]
        values[f"test_precision_{name}"] = entry["precision"]
        values[f"test_recall_{name}"] = entry["recall"]
        values[f"test_f1_{name}"] = entry["f1"]
        values[f"test_support_{name}"] = float(entry["support"])
    return values


def evaluate_frozen_candidate(
    *,
    client: MlflowClient,
    data: DataPaths,
    candidate_path: Path = CANDIDATE_PATH,
    evaluations_dir: Path = EVALUATIONS_DIR,
) -> EvaluationRecord:
    candidate = require_frozen_candidate(candidate_path)
    existing = _evaluations(evaluations_dir, "test")
    if existing:
        raise RuntimeError(
            f"Ya existe la evaluación final de test ({existing[0]['path']}): el test se evalúa "
            "una sola vez; usa audit_evaluation para la auditoría"
        )
    rows = _predict(client, candidate, data)
    created_at = datetime.now(UTC)
    evaluation, metrics = _build_evaluation(candidate, rows, created_at)
    target_met = meets_target(metrics["accuracy"])

    out_dir = evaluations_dir / TEST_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluation_path = out_dir / f"{evaluation.evaluation_id}.json"
    predictions_path = out_dir / f"{evaluation.evaluation_id}.predictions.csv"
    evaluation_path.write_text(evaluation.model_dump_json(indent=2) + "\n", encoding="utf-8")
    _write_predictions_csv(predictions_path, rows)

    timestamp = int(time.time() * 1000)
    client.log_batch(
        candidate.run_id,
        metrics=[
            Metric(name, value, timestamp, 0) for name, value in _mlflow_metrics(metrics).items()
        ],
    )
    for key, value in {
        "test_evaluation_id": evaluation.evaluation_id,
        "test_evaluated_at": evaluation.created_at,
        "test_target_accuracy": str(TARGET_ACCURACY),
        "test_accuracy_meets_target": str(target_met),
    }.items():
        client.set_tag(candidate.run_id, key, value)
    for path in (evaluation_path, predictions_path):
        client.log_artifact(candidate.run_id, str(path), f"evaluation/{TEST_DIR}")
    return EvaluationRecord(
        evaluation=evaluation,
        evaluation_path=evaluation_path,
        predictions_path=predictions_path,
        correct=metrics["correct"],
        total=metrics["total"],
        meets_target=target_met,
    )


def _record_differences(
    old: dict, new: dict, fields, *, label: str, where: str, changed_in: str
) -> list[str]:
    """Compara registro por registro (todos los campos) dos tablas indexadas por id."""
    differences = [f"{label} {key}: falta en {where}" for key in sorted(new.keys() - old.keys())]
    differences += [f"{label} {key}: sobra en {where}" for key in sorted(old.keys() - new.keys())]
    for key in sorted(old.keys() & new.keys()):
        changed = [field for field in fields if old[key].get(field) != new[key][field]]
        if changed:
            differences.append(f"{label} {key}: {changed_in} {', '.join(changed)}")
    return differences


def _prediction_differences(stored: Evaluation, recomputed: Evaluation) -> list[str]:
    old = {p.annotation_id: p.model_dump() for p in stored.predictions}
    new = {p.annotation_id: p.model_dump() for p in recomputed.predictions}
    # El contrato `Evaluation` ya rechaza un annotation_id repetido.
    return _record_differences(
        old,
        new,
        CropPrediction.model_fields,
        label="annotation_id",
        where="la evaluación",
        changed_in="distinto en",
    )


def _csv_differences(path: Path, rows: list[dict]) -> list[str]:
    if not path.is_file():
        return [f"falta el CSV de predicciones {path.name}"]
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        stored_rows = list(reader)
    differences = []
    if reader.fieldnames != list(CSV_FIELDS):
        differences.append(f"columnas del CSV {reader.fieldnames} distintas de {list(CSV_FIELDS)}")
    old = {row.get("crop_id"): row for row in stored_rows}
    if len(old) != len(stored_rows):
        differences.append("crop_id repetido en el CSV de predicciones")
    new = {row["crop_id"]: _csv_row(row) for row in rows}
    return differences + _record_differences(
        old,
        new,
        CSV_FIELDS,
        label="crop_id",
        where="el CSV de predicciones",
        changed_in="fila del CSV distinta en",
    )


def audit_evaluation(
    *,
    client: MlflowClient,
    data: DataPaths,
    candidate_path: Path = CANDIDATE_PATH,
    evaluations_dir: Path = EVALUATIONS_DIR,
) -> list[str]:
    """Vuelve a inferir el test con el mismo checkpoint y manifiesto; no escribe nada.

    Compara el registro completo de cada muestra (JSON por `annotation_id` y CSV por
    `crop_id`: ids, clase real, clase predicha, probabilidades y acierto), la cabecera
    de la evaluación, la matriz y las métricas. Que cuadren la matriz y las métricas
    no basta: dos etiquetas reales intercambiadas pueden dejarlas iguales.
    """
    candidate = require_frozen_candidate(candidate_path)
    stored_files = [
        e for e in _evaluations(evaluations_dir, "test") if e["run_id"] == candidate.run_id
    ]
    if len(stored_files) != 1:
        raise RuntimeError(f"Se esperaba una evaluación de test del candidato: {len(stored_files)}")
    evaluation_path = Path(stored_files[0]["path"])
    stored = Evaluation.model_validate_json(evaluation_path.read_text(encoding="utf-8"))
    rows = _predict(client, candidate, data)
    recomputed, _ = _build_evaluation(candidate, rows, datetime.now(UTC))

    differences = _prediction_differences(stored, recomputed)
    differences += _csv_differences(evaluation_path.with_suffix(".predictions.csv"), rows)
    for field in AUDITED_FIELDS:
        if getattr(stored, field) != getattr(recomputed, field):
            differences.append(f"{field} distinto del recalculado")
    return differences


def verify_against_mlflow(client: MlflowClient, predictions_path: Path, run_id: str) -> dict:
    """Agent Test: recalcula las métricas desde el CSV y las compara con las de MLflow."""
    with predictions_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    metrics = compute_metrics(
        [row["true_class"] for row in rows],
        [row["predicted_class"] for row in rows],
        list(CLASS_NAMES),
    )
    logged = client.get_run(run_id).data.metrics
    return {
        name: {
            "from_file": value,
            "from_mlflow": logged.get(name),
            "equal": logged.get(name) is not None and abs(logged[name] - value) <= 1e-12,
        }
        for name, value in _mlflow_metrics(metrics).items()
    }


def main(argv: list[str] | None = None) -> int:
    from tracking.client import tracking_client

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["evaluate", "audit", "verify"])
    parser.add_argument("--candidate", type=Path, default=CANDIDATE_PATH)
    parser.add_argument("--evaluations", type=Path, default=EVALUATIONS_DIR)
    args = parser.parse_args(argv)
    client = tracking_client()
    candidate = require_frozen_candidate(args.candidate)
    data = DataPaths.for_release(candidate.dataset_version)

    if args.command == "evaluate":
        record = evaluate_frozen_candidate(
            client=client,
            data=data,
            candidate_path=args.candidate,
            evaluations_dir=args.evaluations,
        )
        evaluation = record.evaluation
        print(f"Evaluación {evaluation.evaluation_id} de {candidate.entry} ({candidate.run_id})")
        print(f"accuracy = {record.correct}/{record.total} = {evaluation.metrics.accuracy_top1!r}")
        print(f"¿>= {TARGET_ACCURACY} sin redondear?: {record.meets_target}")
        print(f"f1_macro = {evaluation.metrics.f1_macro!r}")
        print(f"confusion_matrix (filas=real, columnas=predicha {evaluation.class_names}):")
        for name, row in zip(evaluation.class_names, evaluation.confusion_matrix, strict=True):
            print(f"  {name:4s} {row}")
        for entry in evaluation.per_class:
            print(
                f"  {entry.class_name}: precision={entry.precision:.4f} recall={entry.recall:.4f} "
                f"f1={entry.f1:.4f} support={entry.support}"
            )
        print(f"Archivos: {record.evaluation_path} y {record.predictions_path}")
        return 0
    if args.command == "audit":
        differences = audit_evaluation(
            client=client,
            data=data,
            candidate_path=args.candidate,
            evaluations_dir=args.evaluations,
        )
        print("Auditoría: sin diferencias" if not differences else "\n".join(differences))
        return 0 if not differences else 1
    stored = [e for e in _evaluations(args.evaluations, "test") if e["run_id"] == candidate.run_id]
    predictions = Path(stored[0]["path"]).with_suffix(".predictions.csv")
    comparison = verify_against_mlflow(client, predictions, candidate.run_id)
    for name, row in comparison.items():
        print(
            f"{name:24s} archivo={row['from_file']!r:22s} "
            f"mlflow={row['from_mlflow']!r:22s} {'OK' if row['equal'] else 'DISTINTO'}"
        )
    return 0 if all(row["equal"] for row in comparison.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
