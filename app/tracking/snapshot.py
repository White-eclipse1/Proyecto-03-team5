"""ML-07 — snapshot de MLflow para llevar las corridas de la matriz a otro clon.

El MLflow de OPS-03 vive en los volúmenes Docker de cada máquina (MariaDB +
MinIO): un clon limpio arranca vacío. El snapshot conserva los **mismos run IDs**
que lista `reports/experiments/ml07_runs.json`:

- `mlflow.sql`: volcado de la base `mlflow` de MariaDB (experimentos, runs,
  parámetros, métricas, tags), hecho con `mariadb-dump` dentro del contenedor.
  La contraseña se lee de su entorno y nunca pasa por la línea de comandos.
- `artifacts/<run_id>/...`: artefactos de los runs `FINISHED` de la matriz
  (checkpoints, curvas), descargados por la API de MLflow.
- `snapshot.json`: tamaño y sha256 de cada artefacto.

`data/mlflow-snapshot` se versiona con DVC (remote `prod`). Para restaurarlo en
un clon (desde la raíz, con `docker compose up -d --wait mlflow`):

    dvc pull -r prod data/mlflow-snapshot.dvc
    cd app && uv run python -m tracking.snapshot restore

`restore` carga el volcado en MariaDB, reinicia MLflow, sube los artefactos a los
mismos run IDs por la API (sin credenciales de MinIO) y verifica tamaños y sha256.
"""

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_DIR = REPO_ROOT / "data" / "mlflow-snapshot"
DUMP_FILE = "mlflow.sql"
MANIFEST_FILE = "snapshot.json"
ARTIFACTS_DIR = "artifacts"
MATRIX_TAG = "experiment_matrix"
DEFAULT_COMPOSE = ["docker", "compose"]
DUMP_SCRIPT = (
    'mariadb-dump -uroot -p"$MARIADB_ROOT_PASSWORD" --single-transaction '
    "--skip-dump-date --databases mlflow"
)
LOAD_SCRIPT = 'mariadb -uroot -p"$MARIADB_ROOT_PASSWORD"'


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_files(client: MlflowClient, run_id: str, path: str = "") -> dict[str, int]:
    files: dict[str, int] = {}
    for info in client.list_artifacts(run_id, path or None):
        if info.is_dir:
            files.update(_artifact_files(client, run_id, info.path))
        else:
            files[info.path] = info.file_size
    return files


def _matrix_runs(client: MlflowClient, matrix_id: str) -> list:
    experiments = [experiment.experiment_id for experiment in client.search_experiments()]
    runs = client.search_runs(
        experiments, filter_string=f"tags.{MATRIX_TAG} = '{matrix_id}'", max_results=1000
    )
    finished = [run for run in runs if run.info.status == "FINISHED"]
    return sorted(finished, key=lambda run: run.info.run_name or run.info.run_id)


def export_snapshot(client: MlflowClient, matrix_id: str, dest: Path) -> dict:
    """Descarga los artefactos de los runs FINISHED de la matriz y escribe el manifiesto."""
    runs = _matrix_runs(client, matrix_id)
    if not runs:
        raise ValueError(f"No hay runs FINISHED de la matriz {matrix_id} en MLflow")
    artifacts_root = dest / ARTIFACTS_DIR
    shutil.rmtree(artifacts_root, ignore_errors=True)
    manifest: dict = {
        "matrix_id": matrix_id,
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mlflow_version": mlflow.__version__,
        "runs": {},
    }
    for run in runs:
        run_id = run.info.run_id
        run_dir = artifacts_root / run_id
        run_dir.mkdir(parents=True)
        for info in client.list_artifacts(run_id):
            client.download_artifacts(run_id, info.path, str(run_dir))
        manifest["runs"][run_id] = {
            "run_name": run.info.run_name,
            "files": {
                relpath: {"size": size, "sha256": _sha256(run_dir / relpath)}
                for relpath, size in sorted(_artifact_files(client, run_id).items())
            },
        }
    dest.mkdir(parents=True, exist_ok=True)
    (dest / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _manifest(src: Path) -> dict:
    return json.loads((src / MANIFEST_FILE).read_text(encoding="utf-8"))


def restore_artifacts(client: MlflowClient, src: Path) -> None:
    """Sube cada artefacto del snapshot al mismo run_id y la misma ruta."""
    for run_id, entry in _manifest(src)["runs"].items():
        for relpath in entry["files"]:
            # Rutas de artefacto de MLflow: siempre con "/", también en Windows.
            parent = str(PurePosixPath(relpath).parent)
            client.log_artifact(
                run_id,
                str(src / ARTIFACTS_DIR / run_id / relpath),
                None if parent == "." else parent,
            )


def verify_snapshot(client: MlflowClient, src: Path, *, deep: bool = False) -> dict[str, list]:
    """Problemas por run_id (lista vacía = el run y sus artefactos coinciden)."""
    problems: dict[str, list] = {}
    for run_id, entry in _manifest(src)["runs"].items():
        try:
            client.get_run(run_id)
        except MlflowException:
            problems[run_id] = ["run no existe en MLflow"]
            continue
        actual = _artifact_files(client, run_id)
        found = []
        for relpath, expected in entry["files"].items():
            if relpath not in actual:
                found.append(f"{relpath}: falta")
            elif actual[relpath] != expected["size"]:
                found.append(f"{relpath}: tamaño {actual[relpath]} != {expected['size']}")
            elif deep:
                with tempfile.TemporaryDirectory() as tmp:
                    local = Path(client.download_artifacts(run_id, relpath, tmp))
                    if _sha256(local) != expected["sha256"]:
                        found.append(f"{relpath}: sha256 distinto")
        problems[run_id] = found
    return problems


def dump_database(dest: Path, *, compose=DEFAULT_COMPOSE, runner=subprocess.run) -> Path:
    result = runner(
        [*compose, "exec", "-T", "mariadb", "sh", "-c", DUMP_SCRIPT],
        capture_output=True,
        check=True,
        cwd=REPO_ROOT,
    )
    if not result.stdout:
        raise RuntimeError("mariadb-dump devolvió un dump vacío de la base mlflow")
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / DUMP_FILE
    path.write_bytes(result.stdout)
    return path


def restore_database(src: Path, *, compose=DEFAULT_COMPOSE, runner=subprocess.run) -> None:
    """Carga el volcado (reemplaza las tablas de la base mlflow) y reinicia MLflow."""
    runner(
        [*compose, "exec", "-T", "mariadb", "sh", "-c", LOAD_SCRIPT],
        input=(src / DUMP_FILE).read_bytes(),
        check=True,
        cwd=REPO_ROOT,
    )
    runner([*compose, "restart", "mlflow"], check=True, cwd=REPO_ROOT)


def _wait_for_mlflow(timeout_s: float = 120) -> None:
    from tracking.client import TrackingServerUnavailableError, check_server

    deadline = time.monotonic() + timeout_s
    while True:
        try:
            check_server()
            return
        except TrackingServerUnavailableError:
            if time.monotonic() > deadline:
                raise
            time.sleep(2)


def main(argv: list[str] | None = None) -> int:
    from tracking.client import tracking_client

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["create", "restore", "verify"])
    parser.add_argument("--matrix-id", default="ml07-v1")
    parser.add_argument("--dir", type=Path, default=SNAPSHOT_DIR)
    parser.add_argument(
        "--compose",
        default="docker compose",
        help="comando de Compose del stack (por ejemplo, con -p o -f)",
    )
    parser.add_argument("--deep", action="store_true", help="verificar también sha256")
    args = parser.parse_args(argv)
    compose = shlex.split(args.compose)
    client = tracking_client()

    if args.command == "create":
        manifest = export_snapshot(client, args.matrix_id, args.dir)
        dump_database(args.dir, compose=compose)
        print(f"Snapshot de {len(manifest['runs'])} runs en {args.dir}")
        return 0
    if args.command == "restore":
        restore_database(args.dir, compose=compose)
        _wait_for_mlflow()
        restore_artifacts(client, args.dir)
    problems = verify_snapshot(client, args.dir, deep=args.deep or args.command == "restore")
    for run_id, found in problems.items():
        print(f"{run_id}: {'OK' if not found else '; '.join(found)}")
    return 0 if not any(problems.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
