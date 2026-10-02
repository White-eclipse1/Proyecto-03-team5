"""APP-05: API de la pantalla Evaluation (`training/evaluation_view.py`, `ml-api`).

TDD Requirement del issue #22: estados `candidate_not_frozen`, `candidate_frozen` y
`evaluated`. Lee lo que dejan ML-08 (`reports/candidates/ml08_candidate.json`) y
ML-09 (`reports/evaluations/test/<id>.json` + `<id>.predictions.csv`).

Agent Test: con una evaluación de test en disco pero sin candidato congelado, la
respuesta no revela ningún resultado del test.
"""

import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from starlette.testclient import TestClient

from presentation.ml_contracts import ErrorResponse, EvaluationOverview
from training.queue import TrainingJobQueue
from training.server import create_app

EXAMPLES = Path(__file__).resolve().parents[1] / "presentation" / "examples" / "ml"
ROOT = Path(__file__).resolve().parents[2]
TEST_EVALUATION = next(
    e
    for e in json.loads((EXAMPLES / "evaluations.json").read_text(encoding="utf-8"))["evaluations"]
    if e["split"] == "test"
)
VALIDATION_EVALUATION = next(
    e
    for e in json.loads((EXAMPLES / "evaluations.json").read_text(encoding="utf-8"))["evaluations"]
    if e["split"] == "validation"
)
FROZEN_AT = "2026-01-14T14:00:00.000000Z"  # antes de created_at de la evaluación (14:10)
CSV = "crop_id,image_id,annotation_id,true_class,predicted_class,p_dog,p_cat,correct\n"


def candidate_document(**changes) -> dict:
    """Forma de `CandidateSelection` (ML-08), alineada con la evaluación de ejemplo."""
    document = json.loads((ROOT / "reports" / "candidates" / "ml08_candidate.json").read_text())
    document.update(
        run_id=TEST_EVALUATION["run_id"],
        run_name="mlp-512-256-adam",
        checkpoint=TEST_EVALUATION["checkpoint"],
        dataset_version=TEST_EVALUATION["dataset_version"],
        manifest_hash=TEST_EVALUATION["manifest_hash"],
        frozen_at=FROZEN_AT,
    )
    document.update(changes)
    return document


def write_candidate(reports: Path, **changes) -> None:
    path = reports / "candidates" / "ml08_candidate.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(candidate_document(**changes)), encoding="utf-8")


def write_evaluation(reports: Path, document: dict | None = None, *, csv: bool = True) -> str:
    document = document or TEST_EVALUATION
    folder = reports / "evaluations" / "test"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{document['evaluation_id']}.json").write_text(json.dumps(document), "utf-8")
    if csv:
        (folder / f"{document['evaluation_id']}.predictions.csv").write_text(
            CSV, "utf-8", newline=""
        )
    return document["evaluation_id"]


def client_for(tmp_path: Path, reports: Path, crops_dir: Path | None = None) -> TestClient:
    queue = TrainingJobQueue(create_engine(f"sqlite:///{tmp_path / 'jobs.db'}"))
    queue.create_tables()
    return TestClient(create_app(queue=queue, reports_dir=reports, crops_dir=crops_dir))


@pytest.fixture
def reports(tmp_path) -> Path:
    path = tmp_path / "reports"
    path.mkdir()
    return path


def overview(client: TestClient) -> EvaluationOverview:
    response = client.get("/evaluation")
    assert response.status_code == 200, response.text
    return EvaluationOverview.model_validate(response.json())


def assert_no_test_numbers(response_text: str) -> None:
    for marker in ("accuracy_top1", "f1_macro", "confusion_matrix", "predictions"):
        assert marker not in response_text


# --- Estados (TDD del issue #22) ----------------------------------------------------


def test_candidate_not_frozen(tmp_path, reports):
    result = overview(client_for(tmp_path, reports))
    assert (result.state, result.candidate, result.evaluation) == (
        "candidate_not_frozen",
        None,
        None,
    )


def test_agent_test_test_results_exist_but_nothing_is_revealed_before_the_freeze(tmp_path, reports):
    write_evaluation(reports)
    client = client_for(tmp_path, reports)

    response = client.get("/evaluation")

    result = EvaluationOverview.model_validate(response.json())
    assert result.state == "candidate_not_frozen" and result.evaluation is None
    assert "sin candidato congelado" in result.problem
    assert_no_test_numbers(response.text)


def test_candidate_frozen_without_evaluation(tmp_path, reports):
    write_candidate(reports)

    result = overview(client_for(tmp_path, reports))

    assert result.state == "candidate_frozen" and result.evaluation is None
    assert result.problem is None
    candidate = result.candidate
    assert candidate.run_id == TEST_EVALUATION["run_id"]
    assert (candidate.selection_metric, candidate.frozen_at) == ("best_val_loss", FROZEN_AT)


def test_evaluated_shows_the_test_evaluation_of_the_candidate(tmp_path, reports):
    write_candidate(reports)
    write_evaluation(reports)

    result = overview(client_for(tmp_path, reports))

    assert result.state == "evaluated"
    assert result.evaluation.model_dump() == TEST_EVALUATION


# --- Lo que no cuadra no se muestra -------------------------------------------------


@pytest.mark.parametrize(
    ("candidate_changes", "fragment"),
    [
        (
            {
                "run_id": "f" * 32,
                "checkpoint": f"runs:/{'f' * 32}/checkpoints/best.pt",
            },
            "candidato",
        ),
        ({"manifest_hash": "md5:" + "0" * 32}, "candidato"),
        ({"frozen_at": "2026-01-14T15:00:00Z"}, "antes de congelar"),
    ],
    ids=["other-run", "other-manifest", "evaluated-before-freeze"],
)
def test_evaluation_that_does_not_match_the_candidate_is_withheld(
    tmp_path, reports, candidate_changes, fragment
):
    write_candidate(reports, **candidate_changes)
    write_evaluation(reports)
    client = client_for(tmp_path, reports)

    response = client.get("/evaluation")

    result = EvaluationOverview.model_validate(response.json())
    assert result.state == "candidate_frozen" and result.evaluation is None
    assert fragment in result.problem
    assert_no_test_numbers(response.text)


def test_validation_evaluation_in_the_test_folder_is_never_shown_as_test(tmp_path, reports):
    write_candidate(reports)
    write_evaluation(reports, VALIDATION_EVALUATION)

    result = overview(client_for(tmp_path, reports))

    assert result.state == "candidate_frozen" and result.evaluation is None
    assert "validation" in result.problem


def test_more_than_one_test_evaluation_is_withheld(tmp_path, reports):
    write_candidate(reports)
    write_evaluation(reports)
    write_evaluation(reports, {**TEST_EVALUATION, "evaluation_id": "eval-0003"})

    result = overview(client_for(tmp_path, reports))

    assert result.state == "candidate_frozen" and result.evaluation is None
    assert "una sola vez" in result.problem


@pytest.mark.parametrize("content", ["{no es json", '{"run_id": "x"}'], ids=["json", "contract"])
def test_unreadable_candidate_counts_as_not_frozen(tmp_path, reports, content):
    path = reports / "candidates" / "ml08_candidate.json"
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding="utf-8")
    write_evaluation(reports)
    client = client_for(tmp_path, reports)

    response = client.get("/evaluation")

    result = EvaluationOverview.model_validate(response.json())
    assert result.state == "candidate_not_frozen"
    assert "ml08_candidate.json" in result.problem
    assert_no_test_numbers(response.text)


def test_unreadable_test_evaluation_is_withheld(tmp_path, reports):
    write_candidate(reports)
    folder = reports / "evaluations" / "test"
    folder.mkdir(parents=True)
    (folder / "eval-roto.json").write_text('{"split": "test"}', encoding="utf-8")

    result = overview(client_for(tmp_path, reports))

    assert result.state == "candidate_frozen" and result.evaluation is None
    assert "eval-roto.json" in result.problem


# --- Predicciones por recorte (exportar) -------------------------------------------


def test_predictions_csv_is_downloadable_only_when_evaluated(tmp_path, reports):
    write_candidate(reports)
    client = client_for(tmp_path, reports)
    error = client.get("/evaluation/predictions.csv")
    assert error.status_code == 404
    assert ErrorResponse.model_validate(error.json()).error.code == "predictions_not_available"

    write_evaluation(reports)
    response = client.get("/evaluation/predictions.csv")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]
    assert response.text == CSV


def test_predictions_csv_is_not_served_before_the_freeze(tmp_path, reports):
    write_evaluation(reports)
    response = client_for(tmp_path, reports).get("/evaluation/predictions.csv")
    assert response.status_code == 404


# --- Recortes para los ejemplos ------------------------------------------------------


@pytest.fixture
def crops(tmp_path, reports) -> Path:
    crops_dir = tmp_path / "crops"
    (crops_dir / "dog").mkdir(parents=True)
    (crops_dir / "dog" / "img1-ann1.png").write_bytes(b"\x89PNG fake crop")
    (reports / "crops.json").write_text(
        json.dumps({"crops": [{"crop_id": "img1-ann1", "crop_path": "dog/img1-ann1.png"}]}),
        encoding="utf-8",
    )
    return crops_dir


def test_crop_image_is_served_by_crop_id(tmp_path, reports, crops):
    response = client_for(tmp_path, reports, crops).get("/crops/img1-ann1")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == b"\x89PNG fake crop"


@pytest.mark.parametrize(
    ("crop_id", "status"),
    [("img9-ann9", 404), ("..%2F..%2Fsecret", 400), ("img1-ann1.png", 400)],
    ids=["unknown", "traversal", "not-an-id"],
)
def test_crop_lookup_is_by_id_only(tmp_path, reports, crops, crop_id, status):
    assert client_for(tmp_path, reports, crops).get(f"/crops/{crop_id}").status_code == status


def test_crops_not_configured(tmp_path, reports):
    response = client_for(tmp_path, reports).get("/crops/img1-ann1")
    assert response.status_code == 503
    assert ErrorResponse.model_validate(response.json()).error.code == "crops_not_configured"
