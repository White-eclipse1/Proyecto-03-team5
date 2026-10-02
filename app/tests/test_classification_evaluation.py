"""ML-09: evaluación final del candidato congelado sobre el test, auditable y reproducible."""

import csv
import json
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

import pytest
import torch
from mlflow.tracking import MlflowClient

import classification.dataset as dataset_module
import classification.evaluation as evaluation_module
from classification.evaluation import (
    TARGET_ACCURACY,
    audit_evaluation,
    compute_metrics,
    evaluate_frozen_candidate,
    meets_target,
    register_evaluation,
    verify_against_mlflow,
)
from classification.selection import (
    CandidateLockedError,
    CandidateSelection,
    RankedRun,
    TieBreaker,
    _checkpoint_sha256,
    early_test_evaluations,
    freeze_candidate,
)
from classification.training import DataPaths, run_training
from classification.transforms import train_transform
from presentation.ml_contracts import Evaluation, TrainingParams
from tests._classification_fixtures import write_controlled_release

COMMIT = "0123456789abcdef0123456789abcdef01234567"


# --- Métricas con predicciones conocidas (TDD del issue) ------------------------------


def test_metrics_from_known_predictions():
    true = ["dog", "dog", "dog", "cat", "cat"]
    pred = ["dog", "cat", "dog", "cat", "dog"]

    metrics = compute_metrics(true, pred, ["dog", "cat"])

    # Filas = clase real, columnas = clase predicha.
    assert metrics["confusion_matrix"] == [[2, 1], [1, 1]]
    assert (metrics["correct"], metrics["total"]) == (3, 5)
    assert metrics["accuracy"] == 3 / 5
    dog, cat = metrics["per_class"]
    assert (dog["class_name"], dog["support"]) == ("dog", 3)
    assert dog["precision"] == pytest.approx(2 / 3)
    assert dog["recall"] == pytest.approx(2 / 3)
    assert dog["f1"] == pytest.approx(2 / 3)
    assert (cat["precision"], cat["recall"], cat["f1"], cat["support"]) == (0.5, 0.5, 0.5, 2)
    assert metrics["f1_macro"] == pytest.approx((2 / 3 + 0.5) / 2)


def test_accuracy_is_the_exact_fraction_without_rounding():
    metrics = compute_metrics(["dog", "cat", "cat"], ["dog", "cat", "dog"], ["dog", "cat"])

    assert (metrics["correct"], metrics["total"]) == (2, 3)
    assert metrics["accuracy"] == 2 / 3  # no 0.67 ni 0.667


def test_a_class_that_is_never_predicted_has_zero_precision_without_error():
    metrics = compute_metrics(["dog", "cat", "cat"], ["dog", "dog", "dog"], ["dog", "cat"])

    assert metrics["confusion_matrix"] == [[1, 0], [2, 0]]
    cat = metrics["per_class"][1]
    assert (cat["precision"], cat["recall"], cat["f1"], cat["support"]) == (0.0, 0.0, 0.0, 2)
    assert metrics["per_class"][0]["precision"] == pytest.approx(1 / 3)


def test_confusion_matrix_adds_up_to_the_number_of_predictions():
    true = ["dog", "cat"] * 7 + ["dog"]
    pred = ["cat", "cat", "dog"] * 5

    metrics = compute_metrics(true, pred, ["dog", "cat"])

    assert sum(map(sum, metrics["confusion_matrix"])) == metrics["total"] == 15


def test_metrics_reject_unknown_labels_and_length_mismatch():
    with pytest.raises(ValueError, match="person"):
        compute_metrics(["dog"], ["person"], ["dog", "cat"])
    with pytest.raises(ValueError, match="longitud"):
        compute_metrics(["dog", "cat"], ["dog"], ["dog", "cat"])


@pytest.mark.parametrize(
    ("correct", "total", "expected"),
    [
        (17, 20, True),  # exactamente 0.85
        (16, 20, False),
        (68, 71, True),  # resultado real de r03-sgd
        (84, 99, False),  # 0.8484…
        (8499, 10_000, False),  # redondeado a 2 decimales sería 0.85
        (85 * 10**15, 10**17, True),
        # Un acierto menos que 0.85 exacto: en float, la división redondea a 0.85.
        (85 * 10**15 - 1, 10**17, False),
    ],
)
def test_target_is_compared_with_integers_without_rounding(correct, total, expected):
    assert Fraction(85, 100) == TARGET_ACCURACY
    assert meets_target(correct, total) is expected


def test_float_division_cannot_tell_the_limit_case_apart():
    # Por qué la comparación es con enteros: el float de este caso es 0.85.
    assert (85 * 10**15 - 1) / 10**17 >= 0.85


def test_target_requires_at_least_one_test_crop():
    with pytest.raises(ValueError, match="total"):
        meets_target(0, 0)


# --- Evaluación del candidato congelado (extremo a extremo con dataset controlado) -------


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    return MlflowClient(tracking_uri=(tmp_path / "mlruns").as_uri())


@pytest.fixture
def release(tmp_path):
    return write_controlled_release(tmp_path / "release")


@pytest.fixture
def data(release):
    return DataPaths(release.manifest_path, release.crop_report_path, release.crops_dir)


@pytest.fixture
def frozen(client, release, data, tmp_path):
    """Entrena un run corto y lo congela como candidato, como haría ML-08."""
    params = TrainingParams(
        optimizer="adam",
        batch_size=4,
        max_epochs=2,
        learning_rate=0.001,
        image_size=32,
        hidden_layers=[8],
        dropout=0.1,
        seed=3,
        patience=2,
        min_delta=0.0,
    )
    result = run_training(
        params,
        dataset_version=release.dataset_version,
        manifest_hash=release.manifest_hash,
        data=data,
        client=client,
        pretrained=False,
        git_commit=COMMIT,
    )
    candidate = CandidateSelection(
        matrix_id="ml07-test",
        split="validation",
        metric="best_val_loss",
        mode="min",
        tie_breakers=[TieBreaker(metric="entry", mode="min")],
        metric_value=min(epoch.val_loss for epoch in result.history),
        run_id=result.run_id,
        entry="r01",
        run_name="ml07-test-r01",
        checkpoint=result.checkpoint_uri,
        checkpoint_sha256=_checkpoint_sha256(client, result.run_id),
        dataset_version=release.dataset_version,
        manifest_hash=release.manifest_hash,
        training_git_commit=COMMIT,
        selection_git_commit=COMMIT,
        frozen_at=datetime.now(UTC),
        ranking=[
            RankedRun(
                entry="r01",
                run_id=result.run_id,
                best_epoch=result.best_epoch,
                best_val_loss=0.5,
                best_val_accuracy=0.5,
            )
        ],
    )
    path = tmp_path / "reports" / "candidates" / "candidate.json"
    freeze_candidate(candidate, path, evaluations_dir=tmp_path / "evaluations", client=client)
    return path


def _evaluate(client, data, frozen, tmp_path):
    return evaluate_frozen_candidate(
        client=client,
        data=data,
        candidate_path=frozen,
        evaluations_dir=tmp_path / "evaluations",
    )


def _test_crop_ids(release):
    manifest = json.loads(release.manifest_path.read_text(encoding="utf-8"))
    return sorted(r["crop_id"] for r in manifest["records"] if r["split"] == "test")


def test_evaluation_predicts_exactly_the_test_split(client, release, data, frozen, tmp_path):
    record = _evaluate(client, data, frozen, tmp_path)

    rows = list(csv.DictReader(record.predictions_path.open(encoding="utf-8")))
    assert sorted(row["crop_id"] for row in rows) == _test_crop_ids(release)
    assert record.total == len(rows) == 4
    assert record.evaluation.split == "test"


def test_evaluation_file_satisfies_the_shared_contract(client, release, data, frozen, tmp_path):
    record = _evaluate(client, data, frozen, tmp_path)

    stored = Evaluation.model_validate_json(record.evaluation_path.read_text(encoding="utf-8"))
    candidate = json.loads(frozen.read_text(encoding="utf-8"))
    assert stored == record.evaluation
    assert stored.run_id == candidate["run_id"]
    assert stored.checkpoint == candidate["checkpoint"]
    assert stored.manifest_hash == release.manifest_hash
    assert stored.class_names == ["dog", "cat"]
    assert sum(map(sum, stored.confusion_matrix)) == len(stored.predictions) == 4
    assert record.evaluation_path.parent == tmp_path / "evaluations" / "test"


def test_predictions_file_has_crop_id_labels_and_probabilities(
    client, release, data, frozen, tmp_path
):
    record = _evaluate(client, data, frozen, tmp_path)

    rows = list(csv.DictReader(record.predictions_path.open(encoding="utf-8")))
    assert list(rows[0]) == [
        "crop_id",
        "image_id",
        "annotation_id",
        "true_class",
        "predicted_class",
        "p_dog",
        "p_cat",
        "correct",
    ]
    by_annotation = {p.annotation_id: p for p in record.evaluation.predictions}
    crops = {crop.crop_id: crop for crop in release.report.crops}
    for row in rows:
        crop = crops[row["crop_id"]]
        assert (int(row["image_id"]), int(row["annotation_id"])) == (
            crop.image_id,
            crop.annotation_id,
        )
        assert row["true_class"] == crop.class_name
        prediction = by_annotation[crop.annotation_id]
        assert row["predicted_class"] == prediction.predicted_class
        assert float(row["p_dog"]) + float(row["p_cat"]) == pytest.approx(1.0)
        assert row["correct"] == str(row["true_class"] == row["predicted_class"])


def test_final_metrics_are_logged_to_the_candidate_run(client, data, frozen, tmp_path):
    record = _evaluate(client, data, frozen, tmp_path)

    run = client.get_run(record.evaluation.run_id)
    metrics, tags = run.data.metrics, run.data.tags
    assert metrics["test_accuracy"] == record.evaluation.metrics.accuracy_top1
    assert metrics["test_f1_macro"] == record.evaluation.metrics.f1_macro
    assert (metrics["test_correct"], metrics["test_total"]) == (record.correct, record.total)
    for entry in record.evaluation.per_class:
        assert metrics[f"test_precision_{entry.class_name}"] == entry.precision
        assert metrics[f"test_recall_{entry.class_name}"] == entry.recall
        assert metrics[f"test_support_{entry.class_name}"] == entry.support
    assert tags["test_evaluation_id"] == record.evaluation.evaluation_id
    assert tags["test_accuracy_meets_target"] == str(record.meets_target)
    assert tags["test_target_accuracy"] == "0.85"
    artifacts = {a.path for a in client.list_artifacts(run.info.run_id, "evaluation/test")}
    assert artifacts == {
        f"evaluation/test/{record.evaluation_path.name}",
        f"evaluation/test/{record.predictions_path.name}",
    }


def test_evaluation_never_trains(client, data, frozen, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("la evaluación no debe dar pasos de optimizador")

    monkeypatch.setattr(torch.optim.Optimizer, "step", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)

    record = _evaluate(client, data, frozen, tmp_path)

    assert record.total == 4


def test_evaluation_refuses_random_preprocessing(client, data, frozen, tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_module, "transform_for", lambda split, size: train_transform(size))

    with pytest.raises(ValueError, match="aleatori"):
        _evaluate(client, data, frozen, tmp_path)


def test_evaluation_requires_the_frozen_candidate(client, data, tmp_path):
    with pytest.raises(FileNotFoundError, match="candidato"):
        evaluate_frozen_candidate(
            client=client,
            data=data,
            candidate_path=tmp_path / "missing.json",
            evaluations_dir=tmp_path / "evaluations",
        )


def test_evaluation_refuses_a_checkpoint_other_than_the_frozen_one(client, data, frozen, tmp_path):
    candidate = json.loads(frozen.read_text(encoding="utf-8"))
    candidate["checkpoint_sha256"] = "0" * 64
    frozen.write_text(json.dumps(candidate), encoding="utf-8")

    with pytest.raises(ValueError, match="sha256"):
        _evaluate(client, data, frozen, tmp_path)


def test_evaluation_refuses_a_manifest_other_than_the_frozen_one(client, data, frozen, tmp_path):
    candidate = json.loads(frozen.read_text(encoding="utf-8"))
    candidate["manifest_hash"] = "sha256:" + "b" * 64
    frozen.write_text(json.dumps(candidate), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest_hash"):
        _evaluate(client, data, frozen, tmp_path)


def test_the_final_evaluation_happens_only_once(client, data, frozen, tmp_path):
    _evaluate(client, data, frozen, tmp_path)

    with pytest.raises(RuntimeError, match="auditor"):
        _evaluate(client, data, frozen, tmp_path)


def test_audit_recomputes_identical_predictions_without_writing(client, data, frozen, tmp_path):
    record = _evaluate(client, data, frozen, tmp_path)
    before = sorted((tmp_path / "evaluations").rglob("*"))
    metrics_before = client.get_run(record.evaluation.run_id).data.metrics

    differences = audit_evaluation(
        client=client,
        data=data,
        candidate_path=frozen,
        evaluations_dir=tmp_path / "evaluations",
    )

    assert differences == []
    assert sorted((tmp_path / "evaluations").rglob("*")) == before
    assert client.get_run(record.evaluation.run_id).data.metrics == metrics_before


def test_audit_detects_a_stored_prediction_that_was_altered(client, data, frozen, tmp_path):
    record = _evaluate(client, data, frozen, tmp_path)
    stored = json.loads(record.evaluation_path.read_text(encoding="utf-8"))
    first = stored["predictions"][0]
    winner = first["predicted_class"]
    loser = next(name for name in first["probabilities"] if name != winner)
    # Sigue siendo válida para el contrato (suma 1, mismo argmax), pero ya no es la real.
    first["probabilities"][winner] = min(1.0, first["probabilities"][winner] + 1e-4)
    first["probabilities"][loser] = 1.0 - first["probabilities"][winner]
    record.evaluation_path.write_text(json.dumps(stored), encoding="utf-8")

    differences = audit_evaluation(
        client=client,
        data=data,
        candidate_path=frozen,
        evaluations_dir=tmp_path / "evaluations",
    )

    assert differences == [f"annotation_id {first['annotation_id']}: distinto en probabilities"]


def _audit(client, data, frozen, tmp_path):
    return audit_evaluation(
        client=client,
        data=data,
        candidate_path=frozen,
        evaluations_dir=tmp_path / "evaluations",
    )


def _one_dog_and_one_cat(predictions):
    dog = next(p for p in predictions if p["true_class"] == "dog")
    cat = next(p for p in predictions if p["true_class"] == "cat")
    return dog, cat


def _misclassify_one_cat_as_dog(monkeypatch):
    """El modelo falla un cat (lo predice dog) igual al evaluar que al auditar."""
    predict = evaluation_module._predict

    def wrong_on_one_cat(*args):
        rows = predict(*args)
        cat = next(row for row in rows if row["true_class"] == "cat")
        cat["predicted_class"] = "dog"
        cat["probabilities"] = {
            "dog": cat["probabilities"]["cat"],
            "cat": cat["probabilities"]["dog"],
        }
        return rows

    monkeypatch.setattr(evaluation_module, "_predict", wrong_on_one_cat)


def test_audit_detects_true_labels_swapped_between_samples(
    client, data, frozen, tmp_path, monkeypatch
):
    # Caso de la revisión: dos muestras con la misma clase predicha y las etiquetas reales
    # intercambiadas. La matriz y las métricas no cambian; solo el registro por muestra.
    _misclassify_one_cat_as_dog(monkeypatch)
    record = _evaluate(client, data, frozen, tmp_path)
    stored = json.loads(record.evaluation_path.read_text(encoding="utf-8"))
    dog = next(p for p in stored["predictions"] if p["true_class"] == p["predicted_class"] == "dog")
    cat = next(p for p in stored["predictions"] if p["true_class"] == "cat" != p["predicted_class"])
    dog["true_class"], cat["true_class"] = cat["true_class"], dog["true_class"]
    record.evaluation_path.write_text(json.dumps(stored), encoding="utf-8")
    altered = Evaluation.model_validate(stored)
    assert altered.confusion_matrix == record.evaluation.confusion_matrix
    assert altered.metrics == record.evaluation.metrics

    differences = _audit(client, data, frozen, tmp_path)

    assert differences == [
        f"annotation_id {p['annotation_id']}: distinto en true_class"
        for p in sorted((dog, cat), key=lambda p: p["annotation_id"])
    ]


def test_audit_detects_whole_records_swapped_between_samples(client, data, frozen, tmp_path):
    record = _evaluate(client, data, frozen, tmp_path)
    stored = json.loads(record.evaluation_path.read_text(encoding="utf-8"))
    dog, cat = _one_dog_and_one_cat(stored["predictions"])
    fields = ("true_class", "predicted_class", "probabilities")
    for field in fields:
        dog[field], cat[field] = cat[field], dog[field]
    record.evaluation_path.write_text(json.dumps(stored), encoding="utf-8")
    assert Evaluation.model_validate(stored).confusion_matrix == record.evaluation.confusion_matrix

    differences = _audit(client, data, frozen, tmp_path)

    assert differences == [
        f"annotation_id {p['annotation_id']}: distinto en {', '.join(fields)}"
        for p in sorted((dog, cat), key=lambda p: p["annotation_id"])
    ]


def test_audit_detects_image_ids_swapped_between_samples(client, data, frozen, tmp_path):
    record = _evaluate(client, data, frozen, tmp_path)
    stored = json.loads(record.evaluation_path.read_text(encoding="utf-8"))
    first, second = stored["predictions"][:2]
    first["image_id"], second["image_id"] = second["image_id"], first["image_id"]
    record.evaluation_path.write_text(json.dumps(stored), encoding="utf-8")

    differences = _audit(client, data, frozen, tmp_path)

    assert differences == [
        f"annotation_id {p['annotation_id']}: distinto en image_id"
        for p in sorted((first, second), key=lambda p: p["annotation_id"])
    ]


def test_audit_compares_every_field_of_the_predictions_csv(client, data, frozen, tmp_path):
    record = _evaluate(client, data, frozen, tmp_path)
    with record.predictions_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    fieldnames = list(rows[0])
    dog, cat = _one_dog_and_one_cat(rows)
    # La matriz que se recalcula desde el CSV (verify) no cambia: se cruzan filas enteras
    # menos los identificadores.
    swapped = [
        name
        for name in fieldnames
        if name not in ("crop_id", "image_id", "annotation_id") and dog[name] != cat[name]
    ]
    assert {"true_class", "predicted_class", "p_dog", "p_cat"} <= set(swapped)
    for name in swapped:
        dog[name], cat[name] = cat[name], dog[name]
    with record.predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    differences = _audit(client, data, frozen, tmp_path)

    assert differences == [
        f"crop_id {row['crop_id']}: fila del CSV distinta en {', '.join(swapped)}"
        for row in sorted((dog, cat), key=lambda row: row["crop_id"])
    ]


def test_audit_detects_a_missing_or_extra_csv_row(client, data, frozen, tmp_path):
    record = _evaluate(client, data, frozen, tmp_path)
    lines = record.predictions_path.read_text(encoding="utf-8").splitlines(keepends=True)
    removed = lines.pop(1).split(",")[0]
    record.predictions_path.write_text("".join(lines), encoding="utf-8")

    assert _audit(client, data, frozen, tmp_path) == [
        f"crop_id {removed}: falta en el CSV de predicciones"
    ]


def test_audit_requires_the_predictions_csv(client, data, frozen, tmp_path):
    record = _evaluate(client, data, frozen, tmp_path)
    record.predictions_path.unlink()

    assert _audit(client, data, frozen, tmp_path) == [
        f"falta el CSV de predicciones {record.predictions_path.name}"
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("checkpoint", lambda run_id: f"runs:/{run_id}/checkpoints/otro.pt"),
        ("manifest_hash", lambda run_id: "sha256:" + "c" * 64),
        ("dataset_version", lambda run_id: "p3-otro"),
    ],
)
def test_audit_compares_the_header_of_the_evaluation(client, data, frozen, tmp_path, field, value):
    record = _evaluate(client, data, frozen, tmp_path)
    stored = json.loads(record.evaluation_path.read_text(encoding="utf-8"))
    stored[field] = value(stored["run_id"])
    record.evaluation_path.write_text(json.dumps(stored), encoding="utf-8")

    assert _audit(client, data, frozen, tmp_path) == [f"{field} distinto del recalculado"]


def test_candidate_stays_locked_and_frozen_before_the_test(client, data, frozen, tmp_path):
    record = _evaluate(client, data, frozen, tmp_path)
    candidate = CandidateSelection.model_validate_json(frozen.read_text(encoding="utf-8"))

    assert early_test_evaluations(candidate, tmp_path / "evaluations") == []
    other = candidate.model_copy(update={"run_id": "f" * 32, "entry": "otra"})
    with pytest.raises(CandidateLockedError, match="test"):
        freeze_candidate(other, frozen, evaluations_dir=tmp_path / "evaluations", replace=True)
    assert record.evaluation.created_at > candidate.frozen_at.strftime("%Y-%m-%dT%H:%M:%S")


# --- Agent Test: recalcular desde el archivo de predicciones y comparar con MLflow --------


def test_metrics_recomputed_from_the_predictions_file_match_mlflow(client, data, frozen, tmp_path):
    record = _evaluate(client, data, frozen, tmp_path)

    comparison = verify_against_mlflow(client, record.predictions_path, record.evaluation.run_id)

    assert set(comparison) >= {"test_accuracy", "test_f1_macro", "test_correct", "test_total"}
    assert all(row["equal"] for row in comparison.values()), comparison


def test_a_tampered_predictions_file_no_longer_matches_mlflow(client, data, frozen, tmp_path):
    record = _evaluate(client, data, frozen, tmp_path)
    rows = list(csv.DictReader(record.predictions_path.open(encoding="utf-8")))
    rows[0]["predicted_class"] = "cat" if rows[0]["predicted_class"] == "dog" else "dog"
    tampered = Path(tmp_path / "tampered.csv")
    with tampered.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    comparison = verify_against_mlflow(client, tampered, record.evaluation.run_id)

    assert comparison["test_correct"]["equal"] is False
    assert comparison["test_accuracy"]["equal"] is False


def test_audit_detects_the_correct_column_altered_alone(client, data, frozen, tmp_path):
    record = _evaluate(client, data, frozen, tmp_path)
    text = record.predictions_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    crop_id = lines[1].split(",")[0]
    lines[1] = lines[1].replace(",True", ",False")  # en el fixture todo crop acierta
    assert lines[1] != text.splitlines(keepends=True)[1]
    record.predictions_path.write_text("".join(lines), encoding="utf-8")

    assert _audit(client, data, frozen, tmp_path) == [
        f"crop_id {crop_id}: fila del CSV distinta en correct"
    ]


def test_audit_detects_a_repeated_csv_row(client, data, frozen, tmp_path):
    record = _evaluate(client, data, frozen, tmp_path)
    lines = record.predictions_path.read_text(encoding="utf-8").splitlines(keepends=True)
    record.predictions_path.write_text("".join([*lines, lines[1]]), encoding="utf-8")

    assert _audit(client, data, frozen, tmp_path) == ["crop_id repetido en el CSV de predicciones"]


def test_audit_reports_renamed_csv_columns(client, data, frozen, tmp_path):
    record = _evaluate(client, data, frozen, tmp_path)
    text = record.predictions_path.read_text(encoding="utf-8")
    record.predictions_path.write_text(text.replace("p_dog", "prob_dog", 1), encoding="utf-8")
    crop_ids = sorted(line.split(",")[0] for line in text.splitlines()[1:])

    differences = _audit(client, data, frozen, tmp_path)

    assert differences[0].startswith("columnas del CSV")
    assert differences[1:] == [
        f"crop_id {crop_id}: fila del CSV distinta en p_dog" for crop_id in crop_ids
    ]


# --- Recuperación: MLflow falla después de escribir la evaluación ------------------------


class _FailingMlflow:
    """Hace fallar un método del cliente de MLflow hasta que se repare."""

    def __init__(self, client, monkeypatch, method):
        self.broken = True
        original = getattr(client, method)

        def maybe_fail(*args, **kwargs):
            if self.broken:
                raise ConnectionError(f"MLflow no responde ({method})")
            return original(*args, **kwargs)

        monkeypatch.setattr(client, method, maybe_fail)


def _register(client, frozen, tmp_path):
    return register_evaluation(
        client=client, candidate_path=frozen, evaluations_dir=tmp_path / "evaluations"
    )


@pytest.mark.parametrize("method", ["log_batch", "set_tag", "log_artifact"])
def test_register_completes_mlflow_after_a_failure_without_inferring_again(
    client, data, frozen, tmp_path, monkeypatch, method
):
    mlflow = _FailingMlflow(client, monkeypatch, method)
    with pytest.raises(RuntimeError, match="register"):
        _evaluate(client, data, frozen, tmp_path)
    files = sorted((tmp_path / "evaluations" / "test").iterdir())
    assert [f.suffix for f in files] == [".json", ".csv"]
    contents = [f.read_bytes() for f in files]
    # La regla de evaluar una sola vez se mantiene; el error indica cómo recuperarse.
    with pytest.raises(RuntimeError, match="register"):
        _evaluate(client, data, frozen, tmp_path)

    mlflow.broken = False
    monkeypatch.setattr(evaluation_module, "_predict", _must_not_infer)
    record = _register(client, frozen, tmp_path)

    assert [f.read_bytes() for f in files] == contents
    comparison = verify_against_mlflow(client, record.predictions_path, record.evaluation.run_id)
    assert all(row["equal"] for row in comparison.values()), comparison
    tags = client.get_run(record.evaluation.run_id).data.tags
    assert tags["test_evaluation_id"] == record.evaluation.evaluation_id
    assert tags["test_evaluated_at"] == record.evaluation.created_at
    assert tags["test_accuracy_meets_target"] == str(record.meets_target)
    assert tags["test_target_accuracy"] == "0.85"
    artifacts = {a.path for a in client.list_artifacts(record.evaluation.run_id, "evaluation/test")}
    assert artifacts == {f"evaluation/test/{f.name}" for f in files}


def _must_not_infer(*args):
    raise AssertionError("register no debe volver a inferir el test")


def test_register_is_idempotent_after_a_complete_evaluation(
    client, data, frozen, tmp_path, monkeypatch
):
    first = _evaluate(client, data, frozen, tmp_path)
    before = client.get_run(first.evaluation.run_id).data
    monkeypatch.setattr(evaluation_module, "_predict", _must_not_infer)

    again = _register(client, frozen, tmp_path)

    after = client.get_run(first.evaluation.run_id).data
    assert after.metrics == before.metrics
    assert after.tags == before.tags
    assert (again.correct, again.total, again.meets_target) == (
        first.correct,
        first.total,
        first.meets_target,
    )


def test_register_refuses_a_predictions_csv_that_does_not_match_the_evaluation(
    client, data, frozen, tmp_path, monkeypatch
):
    mlflow = _FailingMlflow(client, monkeypatch, "log_batch")
    with pytest.raises(RuntimeError):
        _evaluate(client, data, frozen, tmp_path)
    mlflow.broken = False
    csv_path = next((tmp_path / "evaluations" / "test").glob("*.csv"))
    original = csv_path.read_bytes()
    csv_path.write_bytes(original.replace(b",True\r\n", b",False\r\n", 1))
    assert csv_path.read_bytes() != original

    with pytest.raises(ValueError, match="CSV"):
        _register(client, frozen, tmp_path)

    run_id = json.loads(frozen.read_text(encoding="utf-8"))["run_id"]
    assert "test_accuracy" not in client.get_run(run_id).data.metrics


def test_register_refuses_when_mlflow_has_another_test_evaluation(client, data, frozen, tmp_path):
    record = _evaluate(client, data, frozen, tmp_path)
    client.set_tag(record.evaluation.run_id, "test_evaluation_id", "test-otra-20260101T000000Z")

    with pytest.raises(RuntimeError, match="test-otra"):
        _register(client, frozen, tmp_path)


def test_register_requires_an_evaluation_on_disk(client, frozen, tmp_path):
    with pytest.raises(RuntimeError, match="evaluación de test"):
        _register(client, frozen, tmp_path)
