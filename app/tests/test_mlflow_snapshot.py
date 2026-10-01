"""ML-07: snapshot de MLflow (metadatos + artefactos) para restaurar los mismos run IDs."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from tracking.snapshot import (
    ARTIFACTS_DIR,
    DUMP_FILE,
    MANIFEST_FILE,
    dump_database,
    export_snapshot,
    restore_artifacts,
    restore_database,
    verify_snapshot,
)

MATRIX = "ml07-test"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    return MlflowClient(tracking_uri=(tmp_path / "mlruns").as_uri())


def _run(client, tmp_path, name, *, matrix=MATRIX, status="FINISHED", files=None):
    experiment = client.get_experiment_by_name("dogcat-classifier")
    experiment_id = (
        experiment.experiment_id if experiment else client.create_experiment("dogcat-classifier")
    )
    run = client.create_run(
        experiment_id, run_name=name, tags={"experiment_matrix": matrix, "matrix_entry": name}
    )
    for relpath, content in (files or {"checkpoints/best.pt": name.encode() * 50}).items():
        local = tmp_path / "src" / name / relpath
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(content)
        client.log_artifact(run.info.run_id, str(local), str(Path(relpath).parent))
    client.set_terminated(run.info.run_id, status)
    return run.info.run_id


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _artifacts_root(client, run_id) -> Path:
    return Path(client.get_run(run_id).info.artifact_uri.removeprefix("file://"))


# --- Export ----------------------------------------------------------------------------


def test_export_downloads_only_finished_runs_of_the_matrix(client, tmp_path):
    keep = _run(
        client,
        tmp_path,
        "r01",
        files={"checkpoints/best.pt": b"pesos", "curves/history.json": b"{}"},
    )
    _run(client, tmp_path, "r02", status="FAILED")
    _run(client, tmp_path, "r03", matrix="otra")
    deleted = _run(client, tmp_path, "r04")
    client.delete_run(deleted)
    dest = tmp_path / "snapshot"

    manifest = export_snapshot(client, MATRIX, dest)

    assert list(manifest["runs"]) == [keep]
    assert manifest["matrix_id"] == MATRIX
    assert manifest["runs"][keep]["files"] == {
        "checkpoints/best.pt": {"size": 5, "sha256": _sha(b"pesos")},
        "curves/history.json": {"size": 2, "sha256": _sha(b"{}")},
    }
    assert (dest / ARTIFACTS_DIR / keep / "checkpoints" / "best.pt").read_bytes() == b"pesos"
    assert json.loads((dest / MANIFEST_FILE).read_text(encoding="utf-8")) == manifest


def test_export_refuses_a_matrix_without_runs(client, tmp_path):
    with pytest.raises(ValueError, match=MATRIX):
        export_snapshot(client, MATRIX, tmp_path / "snapshot")


def test_export_replaces_a_previous_snapshot(client, tmp_path):
    dest = tmp_path / "snapshot"
    (dest / ARTIFACTS_DIR / "viejo").mkdir(parents=True)
    run_id = _run(client, tmp_path, "r01")

    export_snapshot(client, MATRIX, dest)

    assert sorted(path.name for path in (dest / ARTIFACTS_DIR).iterdir()) == [run_id]


# --- Restore y verificación ---------------------------------------------------------------


def test_restore_uploads_missing_artifacts_to_the_same_run_ids(client, tmp_path):
    run_id = _run(
        client,
        tmp_path,
        "r01",
        files={"checkpoints/best.pt": b"pesos", "curves/training_curves.png": b"png"},
    )
    dest = tmp_path / "snapshot"
    export_snapshot(client, MATRIX, dest)
    # Clon limpio: la base ya trae los runs, pero el almacén de artefactos está vacío.
    shutil.rmtree(_artifacts_root(client, run_id))
    assert verify_snapshot(client, dest)[run_id] != []

    restore_artifacts(client, dest)

    assert verify_snapshot(client, dest) == {run_id: []}
    local = client.download_artifacts(run_id, "checkpoints/best.pt", str(tmp_path / "dl"))
    assert Path(local).read_bytes() == b"pesos"


def test_verify_reports_missing_runs_and_changed_files(client, tmp_path):
    run_id = _run(client, tmp_path, "r01", files={"checkpoints/best.pt": b"pesos"})
    dest = tmp_path / "snapshot"
    export_snapshot(client, MATRIX, dest)
    manifest = json.loads((dest / MANIFEST_FILE).read_text(encoding="utf-8"))
    manifest["runs"]["f" * 32] = {"run_name": "fantasma", "files": {}}
    (dest / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")
    (_artifacts_root(client, run_id) / "checkpoints" / "best.pt").write_bytes(b"otro")

    problems = verify_snapshot(client, dest)

    assert problems["f" * 32] == ["run no existe en MLflow"]
    assert problems[run_id] == ["checkpoints/best.pt: tamaño 4 != 5"]


def test_verify_can_check_sha256_too(client, tmp_path):
    run_id = _run(client, tmp_path, "r01", files={"checkpoints/best.pt": b"pesos"})
    dest = tmp_path / "snapshot"
    export_snapshot(client, MATRIX, dest)
    (_artifacts_root(client, run_id) / "checkpoints" / "best.pt").write_bytes(b"pexos")

    assert verify_snapshot(client, dest)[run_id] == []
    assert verify_snapshot(client, dest, deep=True)[run_id] == [
        "checkpoints/best.pt: sha256 distinto"
    ]


# --- Base de datos de MLflow en MariaDB (Compose) -----------------------------------------


class FakeRunner:
    def __init__(self, stdout=b""):
        self.calls = []
        self.stdout = stdout

    def __call__(self, args, **kwargs):
        self.calls.append((args, kwargs))

        class Done:
            returncode = 0
            stdout = self.stdout

        return Done()


def test_dump_database_uses_mariadb_dump_inside_the_container(tmp_path):
    runner = FakeRunner(stdout=b"-- dump\nCREATE DATABASE `mlflow`;\n")

    path = dump_database(tmp_path, compose=["docker", "compose", "-p", "x"], runner=runner)

    [(args, _kwargs)] = runner.calls
    assert args[:6] == ["docker", "compose", "-p", "x", "exec", "-T"]
    assert args[6] == "mariadb"
    assert "mariadb-dump" in args[-1] and "--databases mlflow" in args[-1]
    # La contraseña se lee dentro del contenedor; nunca aparece en la línea de comandos.
    assert '"$MARIADB_ROOT_PASSWORD"' in args[-1]
    assert path == tmp_path / DUMP_FILE
    assert path.read_bytes() == b"-- dump\nCREATE DATABASE `mlflow`;\n"


def test_dump_database_refuses_an_empty_dump(tmp_path):
    with pytest.raises(RuntimeError, match="dump"):
        dump_database(tmp_path, compose=["docker", "compose"], runner=FakeRunner(stdout=b""))


def test_restore_database_loads_the_dump_and_restarts_mlflow(tmp_path):
    (tmp_path / DUMP_FILE).write_bytes(b"-- dump")
    runner = FakeRunner()

    restore_database(tmp_path, compose=["docker", "compose"], runner=runner)

    load, restart = runner.calls
    assert load[0][:5] == ["docker", "compose", "exec", "-T", "mariadb"]
    assert 'mariadb -uroot -p"$MARIADB_ROOT_PASSWORD"' in load[0][-1]
    assert load[1]["input"] == b"-- dump"
    assert restart[0] == ["docker", "compose", "restart", "mlflow"]
