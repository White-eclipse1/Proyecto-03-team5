"""APP-03: API de jobs de entrenamiento (`training/server.py`), detrás de /api/ml/.

Pruebas de integración del TDD Requirement del issue #15: creación de jobs, estado,
falla y persistencia al refrescar. El worker de OPS-04 se simula con los mismos
métodos de la cola que usará (`claim_next`, `start`, `report_progress`, `log`,
`fail`).
"""

import json
import math
import shutil
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from starlette.testclient import TestClient

from presentation.ml_contracts import (
    ContractError,
    ErrorResponse,
    TrainingJob,
    TrainingJobsResponse,
    TrainingLogsResponse,
)
from training.queue import TrainingJobQueue
from training.server import create_app

EXAMPLES = Path(__file__).resolve().parents[1] / "presentation" / "examples" / "ml"
# QualityReport real de P2 (v0.1.1): los fixtures solo cambian versión y estado.
REAL_QUALITY = (
    Path(__file__).resolve().parents[2] / "reports" / "releases" / "v0.1.1" / "quality.json"
)
RELEASE = "demo-v1.0.0"
RUN_ID = "0a1b2c3d4e5f60718293a4b5c6d7e8f9"


def example(name: str) -> dict:
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))


def write_release(reports: Path, *, status="warning", provenance=True, manifest=True) -> None:
    release_dir = reports / "releases" / RELEASE
    release_dir.mkdir(parents=True)
    (reports / "versions.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "releases": [
                    {
                        "dataset_version": RELEASE,
                        "quality_file": f"releases/{RELEASE}/quality.json",
                        "splits_file": f"releases/{RELEASE}/splits.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    quality = json.loads(REAL_QUALITY.read_text(encoding="utf-8"))
    quality.update(dataset_version=RELEASE, status=status)
    (release_dir / "quality.json").write_text(json.dumps(quality), encoding="utf-8")
    if provenance:
        shutil.copy(EXAMPLES / "provenance.json", release_dir / "provenance.json")
    if manifest:
        shutil.copy(EXAMPLES / "manifest.json", release_dir / "manifest.json")


@pytest.fixture
def reports(tmp_path) -> Path:
    write_release(tmp_path / "reports")
    return tmp_path / "reports"


@pytest.fixture
def database_url(tmp_path) -> str:
    return f"sqlite:///{tmp_path / 'jobs.db'}"


def make_queue(database_url: str) -> TrainingJobQueue:
    queue = TrainingJobQueue(create_engine(database_url))
    queue.create_tables()
    return queue


def client_for(database_url: str, reports: Path) -> TestClient:
    return TestClient(create_app(queue=make_queue(database_url), reports_dir=reports))


@pytest.fixture
def client(database_url, reports) -> TestClient:
    return client_for(database_url, reports)


def post_job(client: TestClient, **param_changes):
    body = example("training_request.json")
    body["params"].update(param_changes)
    return client.post("/training/jobs", json=body)


def assert_error(response, status_code: int, code: str) -> ErrorResponse:
    assert response.status_code == status_code, response.text
    error = ErrorResponse.model_validate(response.json())
    assert error.error.code == code
    return error


# --- Creación ----------------------------------------------------------------------


def test_post_queues_the_job_without_training_inside_the_request(client):
    response = post_job(client)

    assert response.status_code == 202, response.text
    job = TrainingJob.model_validate(response.json())
    assert job.status == "queued"
    assert job.started_at is None and job.run_id is None
    assert response.headers["location"] == f"/api/ml/training/jobs/{job.job_id}"

    listed = TrainingJobsResponse.model_validate(client.get("/training/jobs").json())
    assert [j.job_id for j in listed.jobs] == [job.job_id]


def test_params_reach_the_job_exactly_as_sent(client):
    job = TrainingJob.model_validate(post_job(client, batch_size=64, dropout=0.35).json())
    assert (job.params.batch_size, job.params.dropout) == (64, 0.35)


def test_invalid_params_are_rejected_before_any_job_exists(client):
    error = assert_error(post_job(client, batch_size=0), 422, "invalid_request")
    assert "params.batch_size" in error.error.message
    assert client.get("/training/jobs").json()["jobs"] == []


def test_malformed_json_is_rejected(client):
    response = client.post(
        "/training/jobs", content="{no es json", headers={"Content-Type": "application/json"}
    )
    assert_error(response, 400, "invalid_json")


def test_unknown_release_is_rejected_with_422_not_404(client):
    """El frontend lee un 404 en el POST como 'servicio no conectado'."""
    body = example("training_request.json")
    body["dataset_version"] = "v9.9.9"
    assert_error(client.post("/training/jobs", json=body), 422, "release_not_found")
    assert client.get("/training/jobs").json()["jobs"] == []


@pytest.mark.parametrize(
    ("release_files", "fragment"),
    [
        ({"status": "failed"}, "Quality Gate"),
        ({"provenance": False}, "provenance"),
        ({"manifest": False}, "manifest"),
    ],
    ids=["quality-gate-failed", "no-provenance", "no-manifest"],
)
def test_untrainable_release_blocks_the_job(database_url, tmp_path, release_files, fragment):
    reports = tmp_path / "blocked-reports"
    write_release(reports, **release_files)
    client = client_for(database_url, reports)

    error = assert_error(post_job(client), 409, "training_blocked")

    assert fragment in error.error.message
    assert client.get("/training/jobs").json()["jobs"] == []


@pytest.mark.parametrize(
    "status",
    ["FAILED", "unknown", "", "passed ", None],
    ids=["uppercase", "unknown", "empty", "trailing-space", "null"],
)
def test_quality_gate_with_an_invalid_status_blocks_the_job(database_url, tmp_path, status):
    """Solo `passed` y `warning` aprueban la compuerta; cualquier otro estado bloquea."""
    reports = tmp_path / "invalid-status"
    write_release(reports, status=status)
    client = client_for(database_url, reports)

    error = assert_error(post_job(client), 409, "training_blocked")

    assert "Quality Gate" in error.error.message
    assert client.get("/training/jobs").json()["jobs"] == []


def test_quality_report_of_another_version_blocks_the_job(database_url, tmp_path):
    reports = tmp_path / "other-version"
    write_release(reports, status="passed")
    quality_path = reports / "releases" / RELEASE / "quality.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["dataset_version"] = "v0.1.0"
    quality_path.write_text(json.dumps(quality), encoding="utf-8")
    client = client_for(database_url, reports)

    error = assert_error(post_job(client), 409, "training_blocked")

    assert "v0.1.0" in error.error.message
    assert client.get("/training/jobs").json()["jobs"] == []


@pytest.mark.parametrize(
    "content",
    [None, "{no es json", '{"status": "passed"}'],
    ids=["missing", "not-json", "not-a-quality-report"],
)
def test_unreadable_quality_report_blocks_the_job(database_url, tmp_path, content):
    reports = tmp_path / "unreadable"
    write_release(reports, status="passed")
    quality_path = reports / "releases" / RELEASE / "quality.json"
    if content is None:
        quality_path.unlink()
    else:
        quality_path.write_text(content, encoding="utf-8")
    client = client_for(database_url, reports)

    error = assert_error(post_job(client), 409, "training_blocked")

    assert "quality.json" in error.error.message
    assert client.get("/training/jobs").json()["jobs"] == []


def test_corrupt_manifest_counts_as_missing(database_url, tmp_path):
    reports = tmp_path / "corrupt-reports"
    write_release(reports)
    (reports / "releases" / RELEASE / "manifest.json").write_text("{}", encoding="utf-8")

    assert_error(post_job(client_for(database_url, reports)), 409, "training_blocked")


def test_manifest_hash_must_match_the_release(client):
    body = example("training_request.json")
    body["manifest_hash"] = "md5:" + "0" * 32
    error = assert_error(client.post("/training/jobs", json=body), 409, "training_blocked")
    assert "manifest_hash" in error.error.message


# --- Estado, falla y persistencia ---------------------------------------------------


def test_job_is_still_visible_after_reloading_the_service(database_url, reports):
    job_id = post_job(client_for(database_url, reports)).json()["job_id"]

    reloaded = client_for(database_url, reports)

    assert reloaded.get(f"/training/jobs/{job_id}").json()["status"] == "queued"


def test_non_finite_progress_is_served_as_json_null(client, database_url):
    job_id = post_job(client).json()["job_id"]
    worker = make_queue(database_url)
    worker.claim_next("worker-1")
    worker.start(job_id, experiment_id="1", run_id=RUN_ID)
    worker.report_progress(job_id, epoch=1, metrics={"val_loss": math.nan, "val_accuracy": 0.6})

    def reject_non_json(value):
        raise ValueError(f"no es JSON estándar: {value}")

    for path in ("/training/jobs", f"/training/jobs/{job_id}"):
        response = client.get(path)
        assert response.status_code == 200
        json.loads(response.text, parse_constant=reject_non_json)
    job = client.get(f"/training/jobs/{job_id}").json()
    assert job["progress"]["metrics"] == {"val_loss": None, "val_accuracy": 0.6}


def test_running_job_exposes_progress_and_logs(client, database_url):
    job_id = post_job(client).json()["job_id"]
    worker = make_queue(database_url)
    worker.claim_next("worker-1")
    worker.start(job_id, experiment_id="1", run_id=RUN_ID)
    worker.log(job_id, "Run creado.")
    worker.report_progress(job_id, epoch=3, metrics={"val_accuracy": 0.7})
    worker.log(job_id, "Época 3/50.")

    job = TrainingJob.model_validate(client.get(f"/training/jobs/{job_id}").json())
    assert job.status == "running" and job.run_id == RUN_ID
    assert (job.progress.epoch, job.progress.metrics) == (3, {"val_accuracy": 0.7})

    logs = TrainingLogsResponse.model_validate(client.get(f"/training/jobs/{job_id}/logs").json())
    assert [e.message for e in logs.entries] == ["Run creado.", "Época 3/50."]
    newer = client.get(f"/training/jobs/{job_id}/logs", params={"after": 1}).json()
    assert [e["seq"] for e in newer["entries"]] == [2]


def test_failed_job_keeps_its_error_after_reload(database_url, reports):
    job_id = post_job(client_for(database_url, reports)).json()["job_id"]
    worker = make_queue(database_url)
    worker.claim_next("worker-1")
    worker.start(job_id, experiment_id="1", run_id=RUN_ID)
    worker.fail(job_id, ContractError(code="out_of_memory", message="Sin memoria.", retryable=True))

    job = client_for(database_url, reports).get(f"/training/jobs/{job_id}").json()

    assert job["status"] == "failed"
    assert job["error"] == {"code": "out_of_memory", "message": "Sin memoria.", "retryable": True}


def test_unknown_job_and_its_logs_are_404(client):
    assert_error(client.get("/training/jobs/job-missing"), 404, "job_not_found")
    assert_error(client.get("/training/jobs/job-missing/logs"), 404, "job_not_found")


def test_bad_logs_cursor_is_rejected(client):
    job_id = post_job(client).json()["job_id"]
    assert_error(client.get(f"/training/jobs/{job_id}/logs?after=-1"), 400, "invalid_request")


def test_health_reports_database_access(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
