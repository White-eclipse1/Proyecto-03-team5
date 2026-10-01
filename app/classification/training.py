"""ML-04 — loop de entrenamiento por minibatches con instrumentación MLflow.

`run_training` es lo que ejecuta el worker de OPS-04 por cada job de la cola
(APP-03, `training/queue.py`):

1. Carga train y validation del manifiesto P3 con el Dataset de ML-02 y exige que
   sea el release y el `manifest_hash` al que está fijado el job. Si no, falla
   **antes** de crear el run.
2. Construye el modelo de ML-03 con `image_size`, `hidden_layers` y `dropout` de
   `TrainingParams`, y el optimizador configurado (`adam`, `adamw` o `sgd`) con
   su `learning_rate` sobre los parámetros entrenables.
3. Crea un run en el experimento `EXPERIMENT_NAME` con los parámetros efectivos y
   las etiquetas de procedencia: commit de Git, release DVC de P2 y sus hashes,
   `manifest_hash`, clases y `class_map`.
4. Por época: un `optimizer.step()` por minibatch del DataLoader de train
   (`batch_size` de la configuración), evaluación sin gradiente en validation, y
   `train_loss`, `train_accuracy`, `val_loss`, `val_accuracy` en MLflow con
   `step=epoch`. Con `batch_size=1`, todas las BatchNorm usan sus estadísticas
   guardadas (`batchnorm_statistics=frozen`): las de una sola imagen son ruido, y
   con `image_size=32` (layer4 en 1x1) BatchNorm no puede normalizar un único
   valor. Con batches mayores, si el último de train tendría una sola muestra se
   descarta (`train_drop_last`) por la misma razón.
5. Sube el checkpoint final a `checkpoints/last.pt` y cierra el run como
   `FINISHED`. Cualquier excepción después de crear el run lo deja `FAILED` (o
   `KILLED` si se interrumpe), guarda el error en la etiqueta `error` y se
   propaga para que el worker marque el job como `failed`.

Los `hooks` conectan el loop con la cola sin que este módulo dependa de ella:
`on_run_started` -> `queue.start`, `on_epoch_end` -> `queue.report_progress`,
`log` -> `queue.log`. Early stopping y mejor checkpoint son ML-06.

Reproducibilidad (ML-05): cada run registra la semilla del split (la del
manifiesto de OPS-02) y las del DataLoader, la augmentation y la inicialización
de pesos (`TrainingParams.seed`, cada una en su propio generador), las versiones
de librerías y plataforma (`environment_tags`), y el orden en que entró cada crop
de train por época (`SAMPLE_ORDER_ARTIFACT`, etiqueta `train_order_sha256`).
El entrenamiento nunca carga el split de test y se niega a correr si validation
tuviera transforms aleatorios. Operaciones no deterministas conocidas:
`classification/README.md`, sección ML-05.
"""

import json
import os
import platform
import re
import subprocess
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol

import mlflow
import numpy
import PIL
import torch
import torchvision
from mlflow.entities import Metric, Param
from mlflow.tracking import MlflowClient
from torch import nn
from torch.utils.data import DataLoader

from classification.dataset import build_dataloader, load_split
from classification.manifest import load_manifest
from classification.model import (
    ARCHITECTURE,
    CLASS_MAP,
    WEIGHTS_ORIGIN,
    ModelConfig,
    build_model,
    save_checkpoint,
)
from classification.transforms import random_transform_names
from crops.classes import CLASS_NAMES
from presentation.ml_contracts import TrainingParams

EXPERIMENT_NAME = "dogcat-classifier"
METRIC_NAMES = ("train_loss", "train_accuracy", "val_loss", "val_accuracy")
CHECKPOINT_ARTIFACT = "checkpoints/last.pt"
SAMPLE_ORDER_ARTIFACT = "reproducibility/sample_order.json"
REPO_ROOT = Path(__file__).resolve().parents[2]
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class DataPaths:
    manifest: Path
    crop_report: Path
    crops_dir: Path

    @classmethod
    def for_release(cls, dataset_version: str, repo_root: Path = REPO_ROOT) -> "DataPaths":
        """Rutas del repo: manifiesto del release, `reports/crops.json` y `data/crops`."""
        return cls(
            manifest=repo_root / "reports" / "releases" / dataset_version / "manifest.json",
            crop_report=repo_root / "reports" / "crops.json",
            crops_dir=repo_root / "data" / "crops",
        )


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float

    def as_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in METRIC_NAMES}


@dataclass(frozen=True)
class TrainingResult:
    experiment_id: str
    run_id: str
    checkpoint_uri: str
    history: list[EpochMetrics]
    optimizer_steps: int
    sample_order: dict[int, list[str]]


class TrainingHooks(Protocol):
    def on_run_started(self, experiment_id: str, run_id: str) -> None: ...

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None: ...

    def log(self, message: str, level: str = "info") -> None: ...


class _NoHooks:
    def on_run_started(self, experiment_id: str, run_id: str) -> None:
        pass

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        pass

    def log(self, message: str, level: str = "info") -> None:
        pass


class JobQueue(Protocol):
    """Lo que el loop usa de `training.queue.TrainingJobQueue` (APP-03)."""

    def start(self, job_id: str, *, experiment_id: str, run_id: str) -> object: ...

    def report_progress(self, job_id: str, *, epoch: int, metrics: dict[str, float]) -> object: ...

    def log(self, job_id: str, message: str, *, level: str = "info") -> object: ...


class JobQueueHooks:
    """Hooks de `run_training` para un job de la cola: start, progreso por época y logs.

    `succeed`/`fail` quedan en el worker, que conoce el resultado (`checkpoint_uri`) o
    la excepción que propaga `run_training`.
    """

    def __init__(self, queue: JobQueue, job_id: str) -> None:
        self._queue = queue
        self._job_id = job_id

    def on_run_started(self, experiment_id: str, run_id: str) -> None:
        self._queue.start(self._job_id, experiment_id=experiment_id, run_id=run_id)

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        self._queue.report_progress(self._job_id, epoch=epoch, metrics=metrics)

    def log(self, message: str, level: str = "info") -> None:
        self._queue.log(self._job_id, message, level=level)


def environment_tags() -> dict[str, str]:
    """Versiones de librerías y plataforma con las que corre el entrenamiento."""
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "numpy_version": numpy.__version__,
        "pillow_version": PIL.__version__,
        "mlflow_version": mlflow.__version__,
        "platform": platform.platform(),
        "torch_num_threads": str(torch.get_num_threads()),
        "torch_deterministic_algorithms": str(torch.are_deterministic_algorithms_enabled()),
        "cuda_available": str(torch.cuda.is_available()),
    }


def resolve_git_commit() -> str:
    """Commit del código que entrena: `GIT_COMMIT` (p. ej. en Docker) o `git rev-parse HEAD`."""
    commit = os.environ.get("GIT_COMMIT")
    if commit is None:
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError(
                "No se pudo leer el commit con git; define GIT_COMMIT con el SHA de 40 caracteres"
            ) from exc
    if not COMMIT_PATTERN.fullmatch(commit):
        raise ValueError(f"GIT_COMMIT debe ser un SHA de 40 caracteres hexadecimales: {commit!r}")
    return commit


def build_optimizer(
    name: str, parameters: list[nn.Parameter], learning_rate: float
) -> torch.optim.Optimizer:
    if name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate)
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9)
    raise ValueError(f"optimizer no soportado: {name!r}")


def _train_epoch(
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module
) -> tuple[float, float, int, list[str]]:
    """Una época por minibatches: un `optimizer.step()` por batch.

    Devuelve loss, accuracy, pasos y el orden en que entraron los crops.
    """
    model.train()
    total_loss, correct, seen, steps = 0.0, 0, 0, 0
    order: list[str] = []
    for batch in loader:
        images, labels = batch["image"], batch["label"]
        order.extend(batch["crop_id"])
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        steps += 1
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += len(labels)
    return total_loss / seen, correct / seen, steps, order


def _evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            images, labels = batch["image"], batch["label"]
            logits = model(images)
            total_loss += criterion(logits, labels).item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += len(labels)
    return total_loss / seen, correct / seen


def _experiment_id(client: MlflowClient, name: str) -> str:
    experiment = client.get_experiment_by_name(name)
    return experiment.experiment_id if experiment else client.create_experiment(name)


def _mlflow_params(
    params: TrainingParams,
    config: ModelConfig,
    train_samples: int,
    validation_samples: int,
    drop_last: bool,
    batchnorm_statistics: str,
    split_seed: int,
) -> dict[str, str]:
    """Parámetros efectivos como texto, igual que `ExperimentRun.params` del portal."""
    values = params.model_dump()
    values["hidden_layers"] = ",".join(str(width) for width in params.hidden_layers)
    effective = {name: str(value) for name, value in values.items()}
    effective.update(
        architecture=ARCHITECTURE,
        pretrained=str(config.pretrained),
        trainable=config.trainable,
        train_samples=str(train_samples),
        validation_samples=str(validation_samples),
        train_drop_last=str(drop_last),
        batchnorm_statistics=batchnorm_statistics,
        seed_split=str(split_seed),
        seed_dataloader=str(params.seed),
        seed_augmentation=str(params.seed),
        seed_weight_init=str(params.seed),
    )
    return effective


def run_training(
    params: TrainingParams,
    *,
    dataset_version: str,
    manifest_hash: str,
    data: DataPaths,
    client: MlflowClient,
    hooks: TrainingHooks | None = None,
    run_name: str | None = None,
    pretrained: bool = True,
    git_commit: str | None = None,
    experiment_name: str = EXPERIMENT_NAME,
) -> TrainingResult:
    hooks = hooks or _NoHooks()
    train_set = load_split(
        data.manifest,
        data.crop_report,
        crops_dir=data.crops_dir,
        split="train",
        params=params,
        augmentation_seed=params.seed,
    )
    validation_set = load_split(
        data.manifest, data.crop_report, crops_dir=data.crops_dir, split="validation", params=params
    )
    random_steps = random_transform_names(validation_set.transform)
    if random_steps:
        raise ValueError(f"validation no debe tener transforms aleatorios: {random_steps}")
    if train_set.dataset_version != dataset_version:
        raise ValueError(
            f"dataset_version del job ({dataset_version}) distinto del manifiesto "
            f"({train_set.dataset_version})"
        )
    if train_set.manifest_hash != manifest_hash:
        raise ValueError(
            f"manifest_hash del job ({manifest_hash}) distinto del manifiesto "
            f"({train_set.manifest_hash})"
        )
    config = ModelConfig.from_params(params, pretrained=pretrained)
    drop_last = params.batch_size > 1 and len(train_set) % params.batch_size == 1
    commit = git_commit or resolve_git_commit()
    manifest = load_manifest(data.manifest)
    provenance = manifest.provenance

    # Semilla de inicialización de pesos (cabeza nueva) y de la secuencia de dropout.
    torch.manual_seed(params.seed)
    model = build_model(config)
    if params.batch_size == 1:
        model.freeze_batchnorm_statistics()
    optimizer = build_optimizer(
        params.optimizer,
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        params.learning_rate,
    )
    criterion = nn.CrossEntropyLoss()
    train_loader = build_dataloader(
        train_set, batch_size=params.batch_size, seed=params.seed, drop_last=drop_last
    )
    validation_loader = build_dataloader(
        validation_set, batch_size=params.batch_size, seed=params.seed
    )

    experiment_id = _experiment_id(client, experiment_name)
    run = client.create_run(
        experiment_id,
        run_name=run_name,
        tags={
            "mlflow.source.git.commit": commit,
            "git_commit": commit,
            "dataset_version": dataset_version,
            "manifest_hash": manifest_hash,
            "manifest_version": manifest.manifest_version,
            "dvc_images_hash": provenance.images_dvc_hash,
            "dvc_annotations_hash": provenance.annotations_dvc_hash,
            "quality_report": provenance.quality_report,
            "crops_sha256": provenance.crops_sha256,
            "classes": ",".join(CLASS_NAMES),
            "class_map": json.dumps(CLASS_MAP),
            "weights_origin": json.dumps(WEIGHTS_ORIGIN if pretrained else None),
            "trainable_summary": json.dumps(model.trainable_summary()),
            **environment_tags(),
        },
    )
    run_id = run.info.run_id
    history: list[EpochMetrics] = []
    sample_order: dict[int, list[str]] = {}
    steps = 0
    try:
        client.log_batch(
            run_id,
            params=[
                Param(key, value)
                for key, value in _mlflow_params(
                    params,
                    config,
                    len(train_set),
                    len(validation_set),
                    drop_last,
                    model.batchnorm_statistics,
                    manifest.seed,
                ).items()
            ],
        )
        hooks.on_run_started(experiment_id, run_id)
        hooks.log(
            f"Run {run_id}: {len(train_set)} crops de train y {len(validation_set)} de "
            f"validation, {params.max_epochs} épocas, batch_size={params.batch_size}"
        )
        for epoch in range(1, params.max_epochs + 1):
            train_set.set_epoch(epoch)
            train_loss, train_accuracy, epoch_steps, order = _train_epoch(
                model, train_loader, optimizer, criterion
            )
            steps += epoch_steps
            sample_order[epoch] = order
            val_loss, val_accuracy = _evaluate(model, validation_loader, criterion)
            metrics = EpochMetrics(epoch, train_loss, train_accuracy, val_loss, val_accuracy)
            timestamp = int(time.time() * 1000)
            client.log_batch(
                run_id,
                metrics=[
                    Metric(name, value, timestamp, epoch)
                    for name, value in metrics.as_dict().items()
                ],
            )
            history.append(metrics)
            hooks.on_epoch_end(epoch, metrics.as_dict())
            hooks.log(
                f"Época {epoch}/{params.max_epochs}: train_loss={train_loss:.4f} "
                f"train_acc={train_accuracy:.4f} val_loss={val_loss:.4f} "
                f"val_acc={val_accuracy:.4f}"
            )
        order_json = json.dumps({str(epoch): ids for epoch, ids in sample_order.items()})
        client.set_tag(run_id, "train_order_sha256", sha256(order_json.encode()).hexdigest())
        with TemporaryDirectory() as tmp:
            order_path = Path(tmp) / Path(SAMPLE_ORDER_ARTIFACT).name
            order_path.write_text(order_json, encoding="utf-8")
            client.log_artifact(run_id, str(order_path), str(Path(SAMPLE_ORDER_ARTIFACT).parent))
            checkpoint = save_checkpoint(
                model,
                Path(tmp) / Path(CHECKPOINT_ARTIFACT).name,
                metadata={
                    "run_id": run_id,
                    "dataset_version": dataset_version,
                    "manifest_hash": manifest_hash,
                    "git_commit": commit,
                    "epochs": params.max_epochs,
                },
            )
            client.log_artifact(run_id, str(checkpoint), str(Path(CHECKPOINT_ARTIFACT).parent))
        client.set_terminated(run_id, "FINISHED")
    except BaseException as exc:
        status = "KILLED" if isinstance(exc, KeyboardInterrupt) else "FAILED"
        client.set_tag(run_id, "error", f"{type(exc).__name__}: {exc}"[:5000])
        client.set_terminated(run_id, status)
        hooks.log(f"Run {run_id} terminó en {status}: {exc}", "error")
        raise
    hooks.log(f"Run {run_id} FINISHED; checkpoint en {CHECKPOINT_ARTIFACT}")
    return TrainingResult(
        experiment_id=experiment_id,
        run_id=run_id,
        checkpoint_uri=f"runs:/{run_id}/{CHECKPOINT_ARTIFACT}",
        history=history,
        optimizer_steps=steps,
        sample_order=sample_order,
    )


__all__ = [
    "CHECKPOINT_ARTIFACT",
    "EXPERIMENT_NAME",
    "METRIC_NAMES",
    "SAMPLE_ORDER_ARTIFACT",
    "DataPaths",
    "EpochMetrics",
    "JobQueue",
    "JobQueueHooks",
    "TrainingHooks",
    "TrainingResult",
    "build_optimizer",
    "environment_tags",
    "resolve_git_commit",
    "run_training",
]
