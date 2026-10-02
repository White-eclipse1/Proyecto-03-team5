"""ML-10: auditoría final de calidad del modelo y análisis de errores.

Todo se recalcula **solo desde el archivo de predicciones** (Agent Test) y se compara
con MLflow, la API y el portal. Cada discrepancia encontrada queda como test de
regresión.
"""

import csv
import json
from fractions import Fraction
from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

import classification.quality_audit as quality_audit
from classification.quality_audit import (
    TARGET_ACCURACY,
    PredictionsFileError,
    api_figures,
    api_prediction_differences,
    choose_examples,
    compare_figures,
    expected_portal_figures,
    flat_metrics,
    frozen_problems,
    frozen_split_problems,
    mlflow_figures,
    portal_figures,
    read_predictions,
    recompute,
)

FIELDS = (
    "crop_id",
    "image_id",
    "annotation_id",
    "true_class",
    "predicted_class",
    "p_dog",
    "p_cat",
    "correct",
)
# (image_id, annotation_id, real, predicha, p_dog): matriz [[3, 1], [1, 1]].
KNOWN = [
    (1, 1, "dog", "dog", 0.9),
    (2, 2, "dog", "dog", 0.8),
    (3, 3, "dog", "cat", 0.3),
    (4, 4, "cat", "cat", 0.2),
    (5, 5, "cat", "dog", 0.6),
    (6, 6, "dog", "dog", 0.7),
]
RUN_ID = "a" * 32


def _row(image_id, annotation_id, true, predicted, p_dog):
    return {
        "crop_id": f"img{image_id}-ann{annotation_id}",
        "image_id": str(image_id),
        "annotation_id": str(annotation_id),
        "true_class": true,
        "predicted_class": predicted,
        "p_dog": repr(p_dog),
        "p_cat": repr(1 - p_dog),
        "correct": str(true == predicted),
    }


def _write(path: Path, rows: list[dict], fields=FIELDS) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def known_csv(tmp_path):
    return _write(tmp_path / "known.predictions.csv", [_row(*entry) for entry in KNOWN])


@pytest.fixture
def known(known_csv):
    return recompute(read_predictions(known_csv))


# --- Agent Test: recalcular desde el archivo de predicciones ---------------------------


def test_metrics_from_known_predictions(known):
    assert known["class_names"] == ["dog", "cat"]
    assert known["confusion_matrix"] == [[3, 1], [1, 1]]
    assert (known["correct"], known["total"]) == (4, 6)
    assert known["accuracy"] == 4 / 6
    assert known["accuracy_fraction"] == "4/6"
    dog, cat = known["per_class"]
    assert (dog["precision"], dog["recall"], dog["f1"], dog["support"]) == (0.75, 0.75, 0.75, 4)
    assert (cat["precision"], cat["recall"], cat["f1"], cat["support"]) == (0.5, 0.5, 0.5, 2)
    assert known["f1_macro"] == 0.625


def test_matrix_adds_up_to_the_number_of_predictions(known):
    assert sum(map(sum, known["confusion_matrix"])) == known["total"] == len(KNOWN)


def test_a_class_never_predicted_has_zero_precision(tmp_path):
    rows = [_row(1, 1, "dog", "dog", 0.9), _row(2, 2, "cat", "dog", 0.8)]
    metrics = recompute(read_predictions(_write(tmp_path / "p.csv", rows)))

    cat = metrics["per_class"][1]
    assert (cat["precision"], cat["recall"], cat["f1"], cat["support"]) == (0.0, 0.0, 0.0, 1)


def test_recomputation_uses_only_the_predictions_file(tmp_path, known_csv):
    # Independiente de ML-09: no reutiliza su cálculo de métricas.
    source = Path(quality_audit.__file__).read_text(encoding="utf-8")
    assert "classification.evaluation" not in source
    assert "compute_metrics" not in source
    alone = tmp_path / "alone"
    alone.mkdir()
    copy = alone / "predictions.csv"
    copy.write_bytes(known_csv.read_bytes())

    assert recompute(read_predictions(copy))["confusion_matrix"] == [[3, 1], [1, 1]]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows.append(dict(rows[0])), "crop_id repetido"),
        (lambda rows: rows[0].update(crop_id="img99-ann1"), "crop_id"),
        (lambda rows: rows[0].update(true_class="person"), "person"),
        (lambda rows: rows[0].update(predicted_class="cat"), "argmax"),
        (lambda rows: rows[0].update(correct="False"), "correct"),
        (lambda rows: rows[0].update(p_cat="0.5"), "suman"),
        (lambda rows: rows[0].update(p_dog="1.2", p_cat="-0.2"), "entre 0 y 1"),
        (lambda rows: rows[0].update(image_id="x"), "image_id"),
    ],
)
def test_an_inconsistent_predictions_file_is_rejected(tmp_path, mutate, message):
    rows = [_row(*entry) for entry in KNOWN]
    mutate(rows)

    with pytest.raises(PredictionsFileError, match=message):
        read_predictions(_write(tmp_path / "bad.csv", rows))


def test_a_predictions_file_without_a_column_is_rejected(tmp_path):
    rows = [{k: v for k, v in _row(*entry).items() if k != "p_cat"} for entry in KNOWN]

    with pytest.raises(PredictionsFileError, match="p_cat"):
        read_predictions(_write(tmp_path / "bad.csv", rows, [f for f in FIELDS if f != "p_cat"]))


def test_an_empty_predictions_file_is_rejected(tmp_path):
    with pytest.raises(PredictionsFileError, match="vacío"):
        read_predictions(_write(tmp_path / "empty.csv", []))


# --- Meta, baseline y análisis de errores ---------------------------------------------


@pytest.mark.parametrize(
    ("correct", "total", "expected"),
    [
        (17, 20, True),
        (16, 20, False),
        (68, 71, True),
        (85 * 10**15 - 1, 10**17, False),  # en float valdría exactamente 0.85
    ],
)
def test_target_is_checked_with_integers(tmp_path, correct, total, expected):
    assert Fraction(85, 100) == TARGET_ACCURACY
    assert quality_audit.meets_target(correct, total) is expected


def test_majority_baseline_is_computed_on_the_same_test(known):
    assert known["baseline"] == {
        "classes": ["dog"],
        "correct": 4,
        "total": 6,
        "accuracy": 4 / 6,
    }
    assert known["accuracy_over_baseline"] == 0.0


def test_majority_baseline_reports_every_tied_class(tmp_path):
    rows = [_row(1, 1, "dog", "dog", 0.9), _row(2, 2, "cat", "cat", 0.1)]
    metrics = recompute(read_predictions(_write(tmp_path / "p.csv", rows)))

    assert metrics["baseline"]["classes"] == ["dog", "cat"]
    assert metrics["baseline"]["accuracy"] == 0.5


def test_most_confused_pair_and_class(known):
    # dog→cat y cat→dog empatan con 1 recorte; cat es la clase con menor recall.
    assert known["most_confused_pairs"] == [
        {"true_class": "dog", "predicted_class": "cat", "count": 1},
        {"true_class": "cat", "predicted_class": "dog", "count": 1},
    ]
    assert known["most_confused_classes"] == ["cat"]


def test_without_errors_nothing_is_confused(tmp_path):
    rows = [_row(1, 1, "dog", "dog", 0.9), _row(2, 2, "cat", "cat", 0.1)]
    metrics = recompute(read_predictions(_write(tmp_path / "p.csv", rows)))

    assert metrics["most_confused_pairs"] == []
    assert metrics["most_confused_classes"] == []


def test_accuracy_that_hides_a_low_recall_class_is_flagged(tmp_path):
    # 18/20 = 0.90 de accuracy, pero cat tiene recall 2/4 = 0.5.
    rows = [_row(i, i, "dog", "dog", 0.9) for i in range(1, 17)]
    rows += [_row(17, 17, "cat", "cat", 0.1), _row(18, 18, "cat", "cat", 0.2)]
    rows += [_row(19, 19, "cat", "dog", 0.7), _row(20, 20, "cat", "dog", 0.8)]
    metrics = recompute(read_predictions(_write(tmp_path / "p.csv", rows)))

    assert metrics["meets_target"] is True
    assert metrics["low_recall_classes"] == ["cat"]
    assert metrics["accuracy_hides_low_recall"] is True


def test_recall_exactly_at_the_target_is_not_low(tmp_path):
    rows = [_row(i, i, "dog", "dog", 0.9) for i in range(1, 18)]
    rows += [_row(i, i, "dog", "cat", 0.1) for i in range(18, 21)]
    rows += [_row(i, i, "cat", "cat", 0.1) for i in range(21, 41)]
    metrics = recompute(read_predictions(_write(tmp_path / "p.csv", rows)))

    assert metrics["per_class"][0]["recall"] == 17 / 20
    assert metrics["low_recall_classes"] == []
    assert metrics["accuracy_hides_low_recall"] is False


# --- Ejemplos reales del test ------------------------------------------------------------


def test_examples_include_every_error_and_hits_of_each_class(known_csv):
    examples = choose_examples(read_predictions(known_csv), hits_per_class=2)

    errors = [e["crop_id"] for e in examples["errors"]]
    # Primero el error más confiado: img3 (p_cat = 0.7) antes que img5 (p_dog = 0.6).
    assert errors == ["img3-ann3", "img5-ann5"]
    hits = {(e["true_class"], e["crop_id"]) for e in examples["hits"]}
    # Por clase: el acierto más confiado y el menos confiado.
    assert hits == {("dog", "img1-ann1"), ("dog", "img6-ann6"), ("cat", "img4-ann4")}
    for example in examples["errors"] + examples["hits"]:
        assert set(example) >= {"crop_id", "true_class", "predicted_class", "confidence"}


def test_examples_and_predictions_belong_to_the_frozen_test_split(known_csv):
    predictions = read_predictions(known_csv)
    records = [
        {"crop_id": f"img{i}-ann{a}", "source_image_id": i, "class": t, "split": "test"}
        for i, a, t, _, _ in KNOWN
    ]
    assert frozen_split_problems(predictions, records) == []

    records[0]["split"] = "train"
    records[1]["class"] = "cat"
    records.append({"crop_id": "img9-ann9", "source_image_id": 9, "class": "dog", "split": "test"})

    assert frozen_split_problems(predictions, records) == [
        "img1-ann1 no es del split test",
        "img2-ann2: clase real dog, en el manifiesto cat",
        "faltan predicciones de test: ['img9-ann9']",
    ]


# --- Comparación con MLflow, API y portal ------------------------------------------------


def test_flat_metrics_cover_every_reported_figure(known):
    flat = flat_metrics(known)

    assert flat["accuracy"] == 4 / 6
    assert flat["correct"] == 4.0
    assert flat["support_cat"] == 2.0
    assert flat["confusion_cat_dog"] == 1.0
    assert set(flat) >= {"f1_macro", "precision_dog", "recall_cat", "f1_dog", "total"}


def test_comparison_flags_a_different_or_missing_figure(known):
    expected = flat_metrics(known)
    reported = dict(expected, f1_macro=0.6)
    del reported["support_dog"]

    rows = compare_figures("MLflow", expected, reported)

    wrong = {row["metric"] for row in rows if not row["equal"]}
    assert wrong == {"f1_macro", "support_dog"}
    assert all(row["source"] == "MLflow" for row in rows)


def _mlflow_run(tmp_path, monkeypatch, known):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    client = MlflowClient(tracking_uri=(tmp_path / "mlruns").as_uri())
    experiment = client.create_experiment("ml10")
    run_id = client.create_run(experiment).info.run_id
    flat = flat_metrics(known)
    for name in ("accuracy", "f1_macro", "correct", "total"):
        client.log_metric(run_id, f"test_{name}", flat[name])
    for name in ("precision", "recall", "f1", "support"):
        for class_name in ("dog", "cat"):
            client.log_metric(run_id, f"test_{name}_{class_name}", flat[f"{name}_{class_name}"])
    client.set_tag(run_id, "test_evaluation_id", "test-r00-20261002T000000Z")
    client.set_tag(run_id, "test_accuracy_meets_target", "False")
    evaluation = tmp_path / "test-r00-20261002T000000Z.json"
    evaluation.write_text(json.dumps({"confusion_matrix": [[3, 1], [1, 1]]}), encoding="utf-8")
    client.log_artifact(run_id, str(evaluation), "evaluation/test")
    return client, run_id


def test_mlflow_figures_include_metrics_tag_and_logged_matrix(tmp_path, monkeypatch, known):
    client, run_id = _mlflow_run(tmp_path, monkeypatch, known)

    figures = mlflow_figures(client, run_id)

    assert figures["meets_target"] == "False"
    assert figures["confusion_dog_cat"] == 1.0
    expected = dict(flat_metrics(known), meets_target=str(known["meets_target"]))
    assert all(row["equal"] for row in compare_figures("MLflow", expected, figures))


def test_a_different_mlflow_metric_is_detected(tmp_path, monkeypatch, known):
    client, run_id = _mlflow_run(tmp_path, monkeypatch, known)
    client.log_metric(run_id, "test_recall_cat", 0.75)

    rows = compare_figures("MLflow", flat_metrics(known), mlflow_figures(client, run_id))

    assert [row["metric"] for row in rows if not row["equal"]] == ["recall_cat"]


def _overview(known_csv):
    predictions = read_predictions(known_csv)
    metrics = recompute(predictions)
    return {
        "state": "evaluated",
        "candidate": {"run_id": RUN_ID},
        "evaluation": {
            "run_id": RUN_ID,
            "class_names": ["dog", "cat"],
            "metrics": {"accuracy_top1": metrics["accuracy"], "f1_macro": metrics["f1_macro"]},
            "per_class": [
                {k: entry[k] for k in ("class_name", "precision", "recall", "f1", "support")}
                for entry in metrics["per_class"]
            ],
            "confusion_matrix": metrics["confusion_matrix"],
            "predictions": [
                {
                    "image_id": p.image_id,
                    "annotation_id": p.annotation_id,
                    "true_class": p.true_class,
                    "predicted_class": p.predicted_class,
                    "probabilities": p.probabilities,
                }
                for p in predictions
            ],
        },
    }


def test_api_figures_match_the_recomputation(known_csv, known):
    figures = api_figures(_overview(known_csv))

    rows = compare_figures("API", flat_metrics(known), figures)
    assert all(row["equal"] for row in rows), rows


def test_api_without_a_final_evaluation_has_no_figures(known_csv):
    overview = _overview(known_csv)
    overview.update(state="candidate_frozen", evaluation=None)

    assert api_figures(overview) == {}


def test_api_predictions_are_compared_sample_by_sample(known_csv):
    predictions = read_predictions(known_csv)
    overview = _overview(known_csv)
    assert api_prediction_differences(predictions, overview) == []

    # Etiquetas reales intercambiadas entre dos muestras: la matriz no cambia.
    api = overview["evaluation"]["predictions"]
    api[0]["true_class"], api[3]["true_class"] = api[3]["true_class"], api[0]["true_class"]
    api[0]["predicted_class"], api[3]["predicted_class"] = "cat", "dog"
    api[0]["probabilities"], api[3]["probabilities"] = (
        api[3]["probabilities"],
        api[0]["probabilities"],
    )
    api.pop()

    assert api_prediction_differences(predictions, overview) == [
        "annotation_id 1: distinto en true_class, predicted_class, probabilities",
        "annotation_id 4: distinto en true_class, predicted_class, probabilities",
        "annotation_id 6: falta en la API",
    ]


def _portal_dom(metrics: dict, **overrides) -> str:
    """El HTML que renderiza la pantalla Evaluation (APP-05), con sus textos."""
    f = {**expected_portal_figures(metrics), **overrides}
    dog, cat = metrics["per_class"]
    matrix = metrics["confusion_matrix"]
    return f"""
    <main>
      <p>Accuracy top-1 (test)</p><p>{f["accuracy"]}</p>
      <div><p>{f["correct"]} de {f["total"]} recortes de test</p>
      <p>Meta 0.85: {f["meets_target"]}</p></div>
      <p>F1 macro (test)</p><p>{f["f1_macro"]}</p>
      <p>Baseline de clase mayoritaria</p><p>{f["baseline_accuracy"]}</p>
      <div>Predecir siempre «{f["baseline_class"]}» en el mismo test</div>
      <table><thead><tr><th>Clase</th><th>Precision</th><th>Recall</th><th>F1</th>
      <th>Support</th></tr></thead><tbody>
      <tr><td>dog</td><td>{f["precision_dog"]}</td><td>{f["recall_dog"]}</td>
      <td>{f["f1_dog"]}</td><td>{dog["support"]}</td></tr>
      <tr><td>cat</td><td>{f["precision_cat"]}</td><td>{f["recall_cat"]}</td>
      <td>{f["f1_cat"]}</td><td>{cat["support"]}</td></tr></tbody></table>
      <table><thead><tr><th>Real \\ Predicha</th><th>dog</th><th>cat</th>
      <th>Total real</th></tr></thead><tbody>
      <tr><th>dog</th><td>{matrix[0][0]}</td><td>{f["confusion_dog_cat"]}</td><td>4</td></tr>
      <tr><th>cat</th><td>{f["confusion_cat_dog"]}</td><td>{matrix[1][1]}</td><td>2</td></tr>
      <tr><th>Total predicho</th><td>4</td><td>2</td><td>6</td></tr></tbody></table>
      <p>La matriz suma 6 = 6 predicciones de test.</p>
      <h2>Interpretación de errores</h2>
      <ul><li>Confusión más frecuente (real → predicha): dog → cat: 1 recortes.</li>
      <li>El recall de dog es 0.7500, por debajo de 0.85.</li>
      <li>El recall de cat es 0.5000, por debajo de 0.85.</li></ul>
      <button>Incorrectos ({f["errors_shown"]})</button>
      <button>Correctos ({f["hits_shown"]})</button>
      <a>Descargar predicciones (CSV)</a>
      <figure><span>Real: dog</span><span>Predicha: cat</span><span>p = 0.700</span>
      <span>img3-ann3</span></figure>
      <figure><span>Real: cat</span><span>Predicha: dog</span>
      <span>p = {f["error_confidences"].split(", ")[-1]}</span>
      <span>{f["error_examples"].split(", ")[-1]}</span></figure>
    </main>
    """


def test_portal_figures_are_read_from_the_rendered_page(known):
    figures = portal_figures(_portal_dom(known))

    assert figures["accuracy"] == "0.6667"
    assert figures["confusion_dog_dog"] == "3"
    assert figures["support_cat"] == "2"
    assert figures["matrix_total"] == "6"
    assert figures["low_recall_classes"] == "dog, cat"
    assert figures["error_examples"] == "img3-ann3, img5-ann5"
    assert figures["error_confidences"] == "0.700, 0.600"
    rows = compare_figures("Portal", expected_portal_figures(known), figures)
    assert all(row["equal"] for row in rows), [row for row in rows if not row["equal"]]


def test_a_different_figure_on_the_portal_is_detected(known):
    dom = _portal_dom(
        known,
        recall_cat="0.5500",
        errors_shown="1",
        error_examples="img4-ann4",
        error_confidences="0.700, 0.900",
    )
    figures = portal_figures(dom)

    rows = compare_figures("Portal", expected_portal_figures(known), figures)

    assert {row["metric"] for row in rows if not row["equal"]} == {
        "recall_cat",
        "errors_shown",
        "error_examples",
        "error_confidences",
    }


def test_portal_text_that_cannot_be_read_is_reported_as_missing(known):
    rows = compare_figures("Portal", expected_portal_figures(known), portal_figures("<main/>"))

    assert rows and not any(row["equal"] for row in rows)


# --- El candidato no cambió después de consultar el test --------------------------------


def _frozen_state():
    candidate = {
        "run_id": RUN_ID,
        "checkpoint": f"runs:/{RUN_ID}/checkpoints/best.pt",
        "dataset_version": "v0.1.1",
        "manifest_hash": "sha256:" + "1" * 64,
        "frozen_at": "2026-10-02T00:37:26.037627Z",
    }
    evaluation = {
        "evaluation_id": "test-r03-20261002T012633Z",
        "run_id": RUN_ID,
        "checkpoint": candidate["checkpoint"],
        "dataset_version": "v0.1.1",
        "manifest_hash": candidate["manifest_hash"],
        "created_at": "2026-10-02T01:26:33.097538Z",
    }
    tags = {
        RUN_ID: {
            "candidate": "true",
            "candidate_frozen_at": candidate["frozen_at"],
            "test_evaluation_id": evaluation["evaluation_id"],
        },
        "b" * 32: {},
    }
    return candidate, evaluation, tags


def test_frozen_candidate_unchanged_after_the_test_has_no_problems():
    assert frozen_problems(*_frozen_state()) == []


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c, e, t: e.update(created_at="2026-10-02T00:00:00.000000Z"), "antes de congelar"),
        (lambda c, e, t: e.update(run_id="b" * 32), "run_id"),
        (lambda c, e, t: e.update(checkpoint=f"runs:/{RUN_ID}/x.pt"), "checkpoint"),
        (lambda c, e, t: e.update(manifest_hash="sha256:" + "2" * 64), "manifest_hash"),
        (lambda c, e, t: t["b" * 32].update(candidate="true"), "otro run"),
        (lambda c, e, t: t[RUN_ID].update(candidate="false"), "candidate=true"),
        (lambda c, e, t: t[RUN_ID].update(candidate_replaced_by="b" * 32), "reemplaz"),
        (lambda c, e, t: t[RUN_ID].update(candidate_frozen_at="2026-10-03T00:00:00Z"), "frozen"),
        (lambda c, e, t: t[RUN_ID].update(test_evaluation_id="test-otra"), "test_evaluation_id"),
    ],
)
def test_a_candidate_changed_after_the_test_is_detected(mutate, message):
    candidate, evaluation, tags = _frozen_state()
    mutate(candidate, evaluation, tags)

    problems = frozen_problems(candidate, evaluation, tags)

    assert len(problems) == 1 and message in problems[0], problems


# --- Auditoría completa (run_audit) ------------------------------------------------------


@pytest.fixture
def audit_inputs(tmp_path, monkeypatch, known_csv, known):
    client, run_id = _mlflow_run(tmp_path, monkeypatch, known)
    evaluation_id = "test-r00-20261002T000000Z"
    candidate = {
        "run_id": run_id,
        "checkpoint": f"runs:/{run_id}/checkpoints/best.pt",
        "dataset_version": "v0.1.1",
        "manifest_hash": "sha256:" + "1" * 64,
        "frozen_at": "2026-10-02T00:00:00.000000Z",
    }
    client.set_tag(run_id, "candidate", "true")
    client.set_tag(run_id, "candidate_frozen_at", candidate["frozen_at"])
    test_dir = tmp_path / "evaluations" / "test"
    test_dir.mkdir(parents=True)
    predictions = test_dir / f"{evaluation_id}.predictions.csv"
    predictions.write_bytes(known_csv.read_bytes())
    header = {k: candidate[k] for k in ("run_id", "checkpoint", "dataset_version", "manifest_hash")}
    (test_dir / f"{evaluation_id}.json").write_text(
        json.dumps(
            {**header, "evaluation_id": evaluation_id, "created_at": "2026-10-02T01:00:00Z"}
        ),
        encoding="utf-8",
    )
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    records = [
        {"crop_id": f"img{i}-ann{a}", "source_image_id": i, "class": t, "split": "test"}
        for i, a, t, _, _ in KNOWN
    ]
    manifest.write_text(json.dumps({"records": records}), encoding="utf-8")
    crops_dir = tmp_path / "crops"
    crops = []
    for i, a, t, _, _ in KNOWN:
        (crops_dir / t).mkdir(parents=True, exist_ok=True)
        (crops_dir / t / f"img{i}-ann{a}.png").write_bytes(f"png {i}".encode())
        crops.append({"crop_id": f"img{i}-ann{a}", "crop_path": f"{t}/img{i}-ann{a}.png"})
    crop_report = tmp_path / "crops.json"
    crop_report.write_text(json.dumps({"crops": crops}), encoding="utf-8")
    overview = _overview(known_csv)
    overview["evaluation"]["run_id"] = run_id

    served = {f"/crops/{c['crop_id']}": (crops_dir / c["crop_path"]).read_bytes() for c in crops}

    def fake_get(url):
        path = url.removeprefix("http://api")
        if path == "/evaluation":
            return 200, "application/json", json.dumps(overview).encode()
        if path in served:
            return 200, "image/png", served[path]
        return 404, "application/json", b"{}"

    monkeypatch.setattr(quality_audit, "_http_get", fake_get)
    return {
        "overview": overview,
        "client": client,
        "predictions_path": predictions,
        "candidate_path": candidate_path,
        "manifest_path": manifest,
        "crop_report_path": crop_report,
        "crops_dir": crops_dir,
        "api_url": "http://api",
        "portal_html": _portal_dom(known),
        "served": served,
    }


def test_a_consistent_evaluation_is_verified_by_every_source(audit_inputs):
    inputs = {k: v for k, v in audit_inputs.items() if k not in ("served", "overview")}

    report = quality_audit.run_audit(**inputs)

    assert report["verified"] is True, report["problems"]
    assert report["problems"] == []
    assert {row["source"] for row in report["comparisons"]} == {"MLflow", "API", "Portal"}
    assert all(row["equal"] for row in report["comparisons"])
    assert report["metrics"]["accuracy_fraction"] == "4/6"
    assert {c["crop_id"] for c in report["crops"]} == {
        e["crop_id"] for e in report["examples"]["errors"] + report["examples"]["hits"]
    }
    assert all(c["served"] and c["same_file"] for c in report["crops"])


def test_an_unreachable_api_or_portal_is_reported_as_not_verified(audit_inputs, monkeypatch):
    inputs = {k: v for k, v in audit_inputs.items() if k not in ("served", "overview")}

    def down(url):
        raise OSError("Connection refused")

    monkeypatch.setattr(quality_audit, "_http_get", down)
    report = quality_audit.run_audit(**{**inputs, "portal_html": None})

    assert report["verified"] is False
    assert any(p.startswith("API no verificada") for p in report["problems"])
    assert any(p.startswith("Portal no verificado") for p in report["problems"])
    assert {row["source"] for row in report["comparisons"]} == {"MLflow"}


def test_a_crop_served_with_other_content_is_detected(audit_inputs):
    inputs = {k: v for k, v in audit_inputs.items() if k not in ("served", "overview")}
    audit_inputs["served"]["/crops/img3-ann3"] = b"otra imagen"

    report = quality_audit.run_audit(**inputs)

    assert report["verified"] is False
    assert "img3-ann3: la API sirve otro recorte" in report["problems"]


def test_a_discrepancy_in_any_source_fails_the_audit(audit_inputs):
    inputs = {k: v for k, v in audit_inputs.items() if k not in ("served", "overview")}
    inputs["client"].log_metric(
        json.loads(inputs["candidate_path"].read_text())["run_id"], "test_f1_macro", 0.1
    )

    report = quality_audit.run_audit(**inputs)

    assert report["verified"] is False
    assert "MLflow: f1_macro distinto" in report["problems"]


def test_the_audit_writes_nothing_and_leaves_the_candidate_unchanged(audit_inputs):
    inputs = {k: v for k, v in audit_inputs.items() if k not in ("served", "overview")}
    test_dir = inputs["predictions_path"].parent
    listing = sorted(test_dir.iterdir())
    files = [inputs["candidate_path"], *listing]
    before = {path: path.read_bytes() for path in files}
    run_id = json.loads(inputs["candidate_path"].read_text())["run_id"]
    run_before = inputs["client"].get_run(run_id).data

    quality_audit.run_audit(**inputs)

    assert {path: path.read_bytes() for path in files} == before
    assert sorted(test_dir.iterdir()) == listing
    run_after = inputs["client"].get_run(run_id).data
    assert (run_after.metrics, run_after.tags) == (run_before.metrics, run_before.tags)


def test_examples_sheet_shows_every_chosen_test_crop(tmp_path, known_csv):
    from PIL import Image

    examples = choose_examples(read_predictions(known_csv))
    crops_dir = tmp_path / "crops"
    crops = []
    for example in examples["errors"] + examples["hits"]:
        path = crops_dir / f"{example['crop_id']}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (40, 30), "gray").save(path)
        crops.append({"crop_id": example["crop_id"], "crop_path": path.name})
    crop_report = tmp_path / "crops.json"
    crop_report.write_text(json.dumps({"crops": crops}), encoding="utf-8")
    out = tmp_path / "sheet.jpg"

    quality_audit.write_examples_sheet(examples, crop_report, crops_dir, out, tile=100)

    with Image.open(out) as sheet:
        assert sheet.format == "JPEG"
        assert sheet.size == (4 * 100, 2 * (100 + quality_audit.CAPTION_HEIGHT))


def test_precision_and_baseline_use_columns_and_rows_of_an_asymmetric_matrix(tmp_path):
    # Matriz [[2, 1], [0, 1]]: columnas y filas suman distinto.
    rows = [
        _row(1, 1, "dog", "dog", 0.9),
        _row(2, 2, "dog", "dog", 0.8),
        _row(3, 3, "dog", "cat", 0.4),
        _row(4, 4, "cat", "cat", 0.1),
    ]
    metrics = recompute(read_predictions(_write(tmp_path / "p.csv", rows)))

    dog, cat = metrics["per_class"]
    assert (dog["precision"], dog["recall"]) == (1.0, 2 / 3)
    assert (cat["precision"], cat["recall"]) == (0.5, 1.0)
    assert metrics["baseline"] == {"classes": ["dog"], "correct": 3, "total": 4, "accuracy": 0.75}


def test_errors_are_ordered_by_confidence_not_by_file_order(tmp_path):
    rows = [
        _row(1, 1, "cat", "dog", 0.55),
        _row(2, 2, "cat", "dog", 0.95),
        _row(3, 3, "cat", "dog", 0.75),
    ]
    examples = choose_examples(read_predictions(_write(tmp_path / "p.csv", rows)))

    assert [e["crop_id"] for e in examples["errors"]] == ["img2-ann2", "img3-ann3", "img1-ann1"]


def test_portal_rounding_matches_javascript_to_fixed():
    # 1/32 = 0.03125 es exacto en binario: toFixed(4) desempata hacia arriba (0.0313).
    assert quality_audit._fixed(1 / 32, 4) == "0.0313"
    assert quality_audit._fixed(0.9577464788732394, 4) == "0.9577"
    assert quality_audit._fixed(0.7691, 3) == "0.769"


def test_api_predictions_that_differ_fail_the_audit(audit_inputs):
    inputs = {k: v for k, v in audit_inputs.items() if k not in ("served", "overview")}
    api = audit_inputs["overview"]["evaluation"]["predictions"]
    api[0]["image_id"], api[1]["image_id"] = api[1]["image_id"], api[0]["image_id"]

    report = quality_audit.run_audit(**inputs)

    assert report["verified"] is False
    assert "API: annotation_id 1: distinto en image_id" in report["problems"]
