"""ML-07 — matriz de experimentos, ejecución reanudable y reporte desde MLflow.

La matriz es un YAML versionado (`ml07_matrix.yaml`): `base` con todos los
`TrainingParams` y `runs`, cada uno con un `name` y solo lo que cambia respecto de
la base. `load_matrix` exige que los siete parámetros de la rúbrica (`VARIED_PARAMS`)
tomen al menos dos valores y que no haya configuraciones duplicadas.

`run_matrix` entrena cada entrada con `run_training` (ML-04/05/06) sobre el mismo
manifiesto y etiqueta el run con `experiment_matrix` y `matrix_entry`. Una entrada
que ya tiene un run `FINISHED` de la misma matriz no se repite, así una ejecución
interrumpida se reanuda sin inflar el conteo. Sin `git_commit` explícito, se niega a
correr con cambios sin commit: el commit registrado debe ser el código que entrenó.

`matrix_report` consulta solo la API de MLflow y comprueba los criterios de ML-07
(rúbrica 3.1 y 3.2) sobre los runs `FINISHED` de la matriz.

    uv run python -m classification.experiments run --matrix classification/ml07_matrix.yaml
    uv run python -m classification.experiments report --matrix classification/ml07_matrix.yaml \\
        --out ../reports/experiments/ml07_runs.json
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml
from mlflow.tracking import MlflowClient
from pydantic import ValidationError

from classification.manifest import load_manifest
from classification.training import (
    CHECKPOINT_ARTIFACT,
    EXPERIMENT_NAME,
    METRIC_NAMES,
    REPO_ROOT,
    DataPaths,
    run_training,
)
from presentation.ml_contracts import TrainingParams

MATRIX_TAG = "experiment_matrix"
ENTRY_TAG = "matrix_entry"
VARIED_PARAMS = (
    "optimizer",
    "batch_size",
    "max_epochs",
    "learning_rate",
    "image_size",
    "hidden_layers",
    "dropout",
)
PROVENANCE_TAGS = ("git_commit", "dataset_version", "dvc_images_hash", "dvc_annotations_hash")


@dataclass(frozen=True)
class MatrixEntry:
    name: str
    params: TrainingParams


@dataclass(frozen=True)
class ExperimentMatrix:
    matrix_id: str
    dataset_version: str
    entries: list[MatrixEntry]


def _config_key(params: TrainingParams) -> tuple:
    return tuple(json.dumps(value) for value in params.model_dump().values())


def load_matrix(path: Path) -> ExperimentMatrix:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries: list[MatrixEntry] = []
    for run in doc["runs"]:
        changes = {key: value for key, value in run.items() if key != "name"}
        try:
            params = TrainingParams(**{**doc["base"], **changes})
        except ValidationError as exc:
            raise ValueError(f"Entrada {run['name']} inválida: {exc}") from exc
        entries.append(MatrixEntry(name=run["name"], params=params))

    names = [entry.name for entry in entries]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"Nombres repetidos en la matriz: {repeated}")
    keys = [_config_key(entry.params) for entry in entries]
    duplicated = sorted(
        entry.name for entry, key in zip(entries, keys, strict=True) if keys.count(key) > 1
    )
    if duplicated:
        raise ValueError(f"Configuraciones duplicadas en la matriz: {duplicated}")
    for param in VARIED_PARAMS:
        values = {json.dumps(getattr(entry.params, param)) for entry in entries}
        if len(values) < 2:
            raise ValueError(f"{param} debe tomar al menos dos valores en la matriz")
    return ExperimentMatrix(
        matrix_id=doc["matrix_id"], dataset_version=doc["dataset_version"], entries=entries
    )


def _dirty_files() -> list[str]:
    output = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [line[3:] for line in output.splitlines() if line.strip()]


def _matrix_runs(client: MlflowClient, matrix_id: str) -> list:
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return []
    return client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.{MATRIX_TAG} = '{matrix_id}'",
        max_results=1000,
    )


def run_matrix(
    matrix: ExperimentMatrix,
    *,
    manifest_hash: str,
    data: DataPaths,
    client: MlflowClient,
    only: list[str] | None = None,
    pretrained: bool = True,
    git_commit: str | None = None,
) -> list[tuple[str, str]]:
    """Entrena las entradas pendientes; devuelve `(entrada, run_id)` en el orden de la matriz."""
    if git_commit is None:
        dirty = _dirty_files()
        if dirty:
            raise RuntimeError(
                f"Hay cambios sin commit ({dirty[:5]}): haz commit antes de correr la matriz "
                "para que el git_commit registrado sea el código que entrena"
            )
    finished = {
        run.data.tags.get(ENTRY_TAG): run.info.run_id
        for run in _matrix_runs(client, matrix.matrix_id)
        if run.info.status == "FINISHED"
    }
    results = []
    for entry in matrix.entries:
        if only is not None and entry.name not in only:
            continue
        if entry.name in finished:
            print(f"[{entry.name}] ya terminó: {finished[entry.name]}")
            results.append((entry.name, finished[entry.name]))
            continue
        print(f"[{entry.name}] entrenando: {entry.params.model_dump()}")
        result = run_training(
            entry.params,
            dataset_version=matrix.dataset_version,
            manifest_hash=manifest_hash,
            data=data,
            client=client,
            run_name=f"{matrix.matrix_id}-{entry.name}",
            pretrained=pretrained,
            git_commit=git_commit,
            tags={MATRIX_TAG: matrix.matrix_id, ENTRY_TAG: entry.name},
        )
        print(
            f"[{entry.name}] {result.run_id}: best_epoch={result.best_epoch} "
            f"best_val_loss={min(e.val_loss for e in result.history):.4f}"
        )
        results.append((entry.name, result.run_id))
    return results


def _run_summary(client: MlflowClient, run) -> dict:
    run_id = run.info.run_id
    metrics = run.data.metrics
    best_epoch = int(metrics.get("best_epoch", 0))
    val_accuracy = {m.step: m.value for m in client.get_metric_history(run_id, "val_accuracy")}
    epoch_metrics = {name: len(client.get_metric_history(run_id, name)) for name in METRIC_NAMES}
    checkpoints = [a.path for a in client.list_artifacts(run_id, "checkpoints")]
    return {
        "run_id": run_id,
        "entry": run.data.tags.get(ENTRY_TAG),
        "run_name": run.info.run_name,
        "params": {name: run.data.params.get(name) for name in (*VARIED_PARAMS, "seed")},
        "best_epoch": best_epoch,
        "best_val_loss": metrics.get("best_val_loss"),
        "best_val_accuracy": val_accuracy.get(best_epoch),
        "epochs_completed": int(metrics.get("epochs_completed", 0)),
        "stopped_epoch": int(metrics["stopped_epoch"]) if "stopped_epoch" in metrics else None,
        "epoch_metrics": epoch_metrics,
        "checkpoint": (
            f"runs:/{run_id}/{CHECKPOINT_ARTIFACT}" if CHECKPOINT_ARTIFACT in checkpoints else None
        ),
        "git_commit": run.data.tags.get("git_commit"),
        "dataset_version": run.data.tags.get("dataset_version"),
        "dvc_images_hash": run.data.tags.get("dvc_images_hash"),
        "dvc_annotations_hash": run.data.tags.get("dvc_annotations_hash"),
        "manifest_hash": run.data.tags.get("manifest_hash"),
        "classes": run.data.tags.get("classes"),
        "train_samples": run.data.params.get("train_samples"),
    }


def matrix_report(client: MlflowClient, matrix: ExperimentMatrix, *, min_runs: int = 10) -> dict:
    """Runs válidos de la matriz y criterios de ML-07, leídos solo de la API de MLflow."""
    valid, excluded = [], []
    for run in _matrix_runs(client, matrix.matrix_id):
        entry = run.data.tags.get(ENTRY_TAG)
        if run.info.status != "FINISHED":
            excluded.append(
                {"run_id": run.info.run_id, "entry": entry, "reason": f"status {run.info.status}"}
            )
            continue
        valid.append(_run_summary(client, run))
    valid.sort(key=lambda run: run["entry"] or "")
    excluded.sort(key=lambda run: run["entry"] or "")

    def distinct(field: str) -> set:
        return {run[field] for run in valid}

    varied = {param: sorted({run["params"][param] for run in valid}) for param in VARIED_PARAMS}
    configs = [tuple(sorted(run["params"].items())) for run in valid]
    checks = {
        "at_least_min_runs": len(valid) >= min_runs,
        "seven_params_vary": all(len(values) >= 2 for values in varied.values()),
        "no_duplicate_configs": len(configs) == len(set(configs)),
        "same_manifest_hash": len(distinct("manifest_hash")) == 1,
        "same_dataset_version": distinct("dataset_version") == {matrix.dataset_version},
        "same_classes_dog_cat": distinct("classes") == {"dog,cat"},
        "all_record_seed": all(run["params"]["seed"] for run in valid),
        "all_record_git_commit": all(run["git_commit"] for run in valid),
        "all_record_dvc_release": all(
            run["dvc_images_hash"] and run["dvc_annotations_hash"] for run in valid
        ),
        "all_have_epoch_metrics": all(
            min(run["epoch_metrics"].values()) == run["epochs_completed"] > 0 for run in valid
        ),
        "all_have_checkpoint": all(run["checkpoint"] for run in valid),
        "all_trained_on_real_split": all(
            run["train_samples"] and int(run["train_samples"]) > 0 for run in valid
        ),
    }
    return {
        "matrix_id": matrix.matrix_id,
        "experiment": EXPERIMENT_NAME,
        "dataset_version": matrix.dataset_version,
        "manifest_hash": next(iter(distinct("manifest_hash")), None),
        "min_runs": min_runs,
        "checks": checks,
        "varied_params": varied,
        "valid_runs": valid,
        "excluded_runs": excluded,
    }


def main(argv: list[str] | None = None) -> int:
    from tracking.client import tracking_client

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["run", "report"])
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    matrix = load_matrix(args.matrix)
    client = tracking_client()
    if args.command == "run":
        data = DataPaths.for_release(matrix.dataset_version)
        run_matrix(
            matrix,
            manifest_hash=load_manifest(data.manifest).manifest_hash,
            data=data,
            client=client,
            only=args.only,
        )
        return 0
    report = matrix_report(client, matrix)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(json.dumps(report["checks"], indent=2))
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
