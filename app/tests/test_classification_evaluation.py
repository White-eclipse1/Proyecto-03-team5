"""ML-09: evaluación final del candidato congelado sobre el test, auditable y reproducible."""

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch
from mlflow.tracking import MlflowClient

import classification.dataset as dataset_module
from classification.evaluation import (
    TARGET_ACCURACY,
    audit_evaluation,
    compute_metrics,
    evaluate_frozen_candidate,
    meets_target,
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
    ("accuracy", "expected"),
    [
        (0.85, True),
        (17 / 20, True),
        (0.9, True),
        (0.8499999999, False),
        (0.84995, False),  # redondeado a 4 decimales sería 0.85
        (84 / 99, False),  # 0.8484…
    ],
)
def test_target_is_compared_without_rounding(accuracy, expected):
    assert TARGET_ACCURACY == 0.85
    assert meets_target(accuracy) is expected


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
