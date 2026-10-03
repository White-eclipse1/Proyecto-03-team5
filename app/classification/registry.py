"""OPS-06 — semantic model package and persistent model registry.

Flujo real (`publish`): candidato congelado de ML-08 + evaluación final de ML-09 →
descarga `checkpoints/best.pt` del run de MLflow y verifica su sha256 contra el
congelado → arma el paquete (checkpoint, arquitectura, class map, preprocesamiento,
dependencias reales y model card) en `data/models/<modelo>/<versión>/` → lo registra en
`reports/models/registry.json` con el sha256 de cada archivo del paquete.

El paquete pesa ~45 MB (checkpoint): se versiona con DVC (`data/models.dvc`). Una
versión registrada no se reescribe; las anteriores siguen resolubles.

    uv run python -m classification.registry publish 1.0.0
    uv run python -m classification.registry resolve 1.0.0   # checkpoint verificado
    uv run python -m classification.registry verify 1.0.0    # Agent Test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import sys
import tempfile
from importlib.metadata import version as installed_version
from pathlib import Path
from typing import Annotated, Literal

import torch
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

from classification.model import load_checkpoint
from classification.selection import (
    CANDIDATE_PATH,
    EVALUATIONS_DIR,
    _evaluations,
    require_frozen_candidate,
)
from classification.training import CHECKPOINT_ARTIFACT

SEMVER_PATTERN = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "reports" / "models" / "registry.json"
PACKAGES_ROOT = REPO_ROOT / "data" / "models"
PACKAGE_CHECKPOINT = "checkpoint/best.pt"
# `trainable` de classification.model: qué se ajustó sobre los pesos preentrenados.
TRAINED_LAYERS = {
    "head": "solo la cabeza; el backbone queda congelado",
    "layer4": "layer4 (último bloque residual) y la cabeza; el resto del backbone congelado",
    "all": "toda la red",
}
# Archivos del paquete con sha256 en el registro (package.json los lista a ellos).
PACKAGE_FILES = (PACKAGE_CHECKPOINT, "model-card.md", "dependencies.json")


class ModelVersion(RootModel[str]):
    """Versión semántica estricta MAJOR.MINOR.PATCH."""

    @model_validator(mode="after")
    def valid_semver(self) -> "ModelVersion":
        if not SEMVER_PATTERN.fullmatch(self.root):
            raise ValueError(f"model_version debe usar SemVer MAJOR.MINOR.PATCH: {self.root!r}")
        return self

    def __str__(self) -> str:
        return self.root


class Architecture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    image_size: int
    hidden_layers: list[int]
    dropout: float
    pretrained: bool
    trainable: str


class Preprocessing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resize: str
    image_size: int
    mean: list[float]
    std: list[float]


class ModelCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    purpose: str
    dataset_release: str
    manifest_hash: str
    mlflow_run_id: str
    test_metrics: dict[str, float]
    limitations: Annotated[list[str], Field(min_length=1)]
    pretrained_weights: str | None = None


class ModelPackage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"

    model_version: ModelVersion
    model_name: str

    run_id: str
    checkpoint: str
    checkpoint_sha256: str

    dataset_version: str
    manifest_hash: str

    architecture: Architecture
    class_map: dict[str, int]
    preprocessing: Preprocessing
    dependencies: dict[str, str]
    metrics: dict[str, float]
    weights_origin: dict | None
    model_card: ModelCard
    # Solo en paquetes materializados por `publish`: dónde están y el sha256 de cada archivo.
    package_path: str | None = None
    files: dict[str, str] | None = None

    @model_validator(mode="after")
    def traceability_is_consistent(self) -> "ModelPackage":
        expected_checkpoint = f"runs:/{self.run_id}/checkpoints/best.pt"

        if self.checkpoint != expected_checkpoint:
            raise ValueError(f"checkpoint no corresponde al run_id: {self.checkpoint!r}")

        if not re.fullmatch(r"[0-9a-f]{64}", self.checkpoint_sha256):
            raise ValueError("checkpoint_sha256 debe tener 64 caracteres hexadecimales")

        if self.model_card.mlflow_run_id != self.run_id:
            raise ValueError("model_card.mlflow_run_id debe coincidir con run_id")

        if self.model_card.dataset_release != self.dataset_version:
            raise ValueError("model_card.dataset_release debe coincidir con dataset_version")

        if self.model_card.manifest_hash != self.manifest_hash:
            raise ValueError("model_card.manifest_hash debe coincidir con manifest_hash")

        if self.preprocessing.image_size != self.architecture.image_size:
            raise ValueError("preprocessing.image_size debe coincidir con architecture.image_size")

        return self


class ModelRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    models: list[ModelPackage] = Field(default_factory=list)


def _load_registry(path: Path) -> ModelRegistry:
    if not path.is_file():
        return ModelRegistry()

    return ModelRegistry.model_validate_json(path.read_text(encoding="utf-8"))


def _write_registry(registry: ModelRegistry, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        registry.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )


def register_model(package: ModelPackage, path: Path) -> ModelPackage:
    registry = _load_registry(path)

    for existing in registry.models:
        if existing.model_version == package.model_version:
            if existing == package:
                return existing

            raise ValueError(
                f"La model_version {package.model_version} ya existe con metadata distinta"
            )

    registry.models.append(package)
    _write_registry(registry, path)

    return package


def resolve_model(version: str, path: Path) -> ModelPackage:
    requested = ModelVersion(root=version)
    registry = _load_registry(path)

    for package in registry.models:
        if package.model_version == requested:
            return package

    raise KeyError(f"No existe model_version {version}")


def build_model_package(
    model_version: str,
    *,
    checkpoint_path: Path,
    candidate_path: Path,
    evaluation_path: Path,
    dependencies: dict[str, str],
) -> ModelPackage:
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))

    if candidate["run_id"] != evaluation["run_id"]:
        raise ValueError("candidate y evaluation tienen distinto run_id")

    if candidate["checkpoint"] != evaluation["checkpoint"]:
        raise ValueError("candidate y evaluation tienen distinto checkpoint")

    if candidate["dataset_version"] != evaluation["dataset_version"]:
        raise ValueError("candidate y evaluation tienen distinto dataset_version")

    if candidate["manifest_hash"] != evaluation["manifest_hash"]:
        raise ValueError("candidate y evaluation tienen distinto manifest_hash")

    if evaluation.get("split") != "test":
        raise ValueError("La evaluación usada para empaquetar debe ser del split test")

    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    if checkpoint_sha256 != candidate["checkpoint_sha256"]:
        raise ValueError("El sha256 del checkpoint local no coincide con el candidato congelado")

    fields = _checkpoint_fields(checkpoint_path)
    architecture = fields["architecture"]
    preprocessing = fields["preprocessing"]
    class_map = fields["class_map"]
    weights_origin = fields["weights_origin"]

    metrics = {
        "accuracy_top1": evaluation["metrics"]["accuracy_top1"],
        "f1_macro": evaluation["metrics"]["f1_macro"],
    }
    test_metrics = {**metrics, **_test_details(evaluation)}

    return ModelPackage(
        model_version=model_version,
        model_name="dog-cat-resnet18",
        run_id=candidate["run_id"],
        checkpoint=candidate["checkpoint"],
        checkpoint_sha256=checkpoint_sha256,
        dataset_version=candidate["dataset_version"],
        manifest_hash=candidate["manifest_hash"],
        architecture=architecture,
        class_map=class_map,
        preprocessing=preprocessing,
        dependencies=dependencies,
        metrics=metrics,
        weights_origin=weights_origin,
        model_card=ModelCard(
            purpose=(
                "Clasificar recortes de cajas COCO (un objeto por recorte) en dog o cat "
                "dentro del portal de anotación del Proyecto 3."
            ),
            dataset_release=candidate["dataset_version"],
            manifest_hash=candidate["manifest_hash"],
            mlflow_run_id=candidate["run_id"],
            test_metrics=test_metrics,
            limitations=_limitations(evaluation, candidate["dataset_version"]),
            pretrained_weights=(
                weights_origin.get("weights") if isinstance(weights_origin, dict) else None
            ),
        ),
    )


def _checkpoint_fields(checkpoint_path: Path) -> dict:
    """Arquitectura, class map, preprocesamiento y origen de pesos del checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return {
        "architecture": {"name": checkpoint["architecture"], **checkpoint["config"]},
        "preprocessing": checkpoint["preprocessing"],
        "class_map": checkpoint["class_map"],
        "weights_origin": checkpoint.get("weights_origin"),
    }


def _test_details(evaluation: dict) -> dict[str, float]:
    """Aciertos, total y métricas por clase del test, si la evaluación las trae."""
    matrix = evaluation.get("confusion_matrix")
    if not matrix:
        return {}
    details = {
        "correct": float(sum(matrix[i][i] for i in range(len(matrix)))),
        "total": float(sum(map(sum, matrix))),
    }
    for entry in evaluation.get("per_class", []):
        for field in ("precision", "recall", "f1", "support"):
            details[f"{field}_{entry['class_name']}"] = float(entry[field])
    return details


def _limitations(evaluation: dict, dataset_version: str) -> list[str]:
    limitations = [
        "Solo distingue dog y cat: cualquier otro objeto se asigna a una de las dos clases.",
        "Clasifica el recorte de una caja; no imágenes completas con varios objetos.",
    ]
    matrix, names = evaluation.get("confusion_matrix"), evaluation.get("class_names")
    if not matrix or not names:
        return [
            *limitations,
            f"Evaluado solo sobre el release {dataset_version}; fuera de ese dominio puede variar.",
        ]
    total = sum(map(sum, matrix))
    limitations.append(
        f"Evaluado en el test congelado del release {dataset_version} ({total} recortes); "
        "con pocas muestras por clase las métricas tienen incertidumbre alta."
    )
    recalls = [(matrix[i][i] / sum(matrix[i]), i) for i in range(len(names)) if sum(matrix[i])]
    recall, weakest = min(recalls)
    confusions = [
        f"{names[t]}→{names[p]}: {matrix[t][p]}"
        for t in range(len(names))
        for p in range(len(names))
        if t != p and matrix[t][p]
    ]
    limitations.append(
        f"Clase más débil en test: {names[weakest]}, recall {recall:.4f} "
        f"({matrix[weakest][weakest]}/{sum(matrix[weakest])}). "
        f"Errores de test: {', '.join(confusions) or 'ninguno'}."
    )
    return limitations


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_card_markdown(package: ModelPackage) -> str:
    card = package.model_card
    origin = package.weights_origin or {}
    trainable = package.architecture.trainable
    pretrained = (
        f"{card.pretrained_weights} ({origin.get('dataset', '?')}, {origin.get('library', '?')}); "
        f"entrenado: {TRAINED_LAYERS.get(trainable, trainable)}."
        if card.pretrained_weights
        else "Ninguno: pesos inicializados al azar y entrenados desde cero."
    )
    return "\n".join(
        [
            f"# Model Card — {package.model_name} {package.model_version}",
            "",
            "## Propósito",
            "",
            card.purpose,
            "",
            "## Datos y trazabilidad",
            "",
            f"- Model version: {package.model_version} (independiente del dataset)",
            f"- Release P2: {package.dataset_version}",
            f"- Manifest hash (split 70/20/10): {package.manifest_hash}",
            f"- MLflow run_id: {package.run_id}",
            f"- Checkpoint: {package.checkpoint}",
            f"- Checkpoint SHA256: {package.checkpoint_sha256}",
            f"- Clases: {package.class_map}",
            "",
            "## Arquitectura y preprocesamiento",
            "",
            f"- {package.architecture.model_dump()}",
            f"- {package.preprocessing.model_dump()}",
            "",
            "## Métricas de test",
            "",
            *(
                [f"- {name}: {value!r}" for name, value in card.test_metrics.items()]
                or ["Sin evaluación en test (ver limitaciones)."]
            ),
            "",
            "## Limitaciones",
            "",
            *[f"- {limitation}" for limitation in card.limitations],
            "",
            "## Pesos preentrenados",
            "",
            pretrained,
            "",
        ]
    )


def materialize_model_package(
    package: ModelPackage,
    *,
    checkpoint_path: Path,
    output_dir: Path,
) -> Path:
    # Integridad: solo se empaqueta el checkpoint cuyo sha256 es el registrado.
    actual = _sha256_file(checkpoint_path)
    if actual != package.checkpoint_sha256:
        raise ValueError(
            f"El checkpoint {checkpoint_path.name} tiene sha256 {actual}, no el del paquete "
            f"{package.checkpoint_sha256}: no se materializa"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = output_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_target = checkpoint_dir / "best.pt"
    shutil.copy2(checkpoint_path, checkpoint_target)
    if _sha256_file(checkpoint_target) != package.checkpoint_sha256:
        checkpoint_target.unlink()
        raise ValueError(f"La copia de {checkpoint_target} no conserva el sha256 registrado")

    package_json = output_dir / "package.json"
    package_json.write_text(
        package.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    dependencies_json = output_dir / "dependencies.json"
    dependencies_json.write_text(
        json.dumps(package.dependencies, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    model_card = output_dir / "model-card.md"
    model_card.write_text(_model_card_markdown(package), encoding="utf-8")

    return output_dir


# --- Flujo real: MLflow → paquete verificable → registro ---------------------------------


def runtime_dependencies() -> dict[str, str]:
    """Versiones instaladas de verdad y el sha256 del lockfile del repo que las fija."""
    dependencies = {"python": platform.python_version()}
    for name in ("torch", "torchvision", "pillow", "numpy", "pydantic"):
        dependencies[name] = installed_version(name)
    lockfile = REPO_ROOT / "app" / "uv.lock"
    dependencies["lockfile"] = "app/uv.lock"
    dependencies["uv.lock sha256"] = _sha256_file(lockfile)
    return dependencies


def _registered(path: Path, version: ModelVersion) -> ModelPackage | None:
    return next((p for p in _load_registry(path).models if p.model_version == version), None)


def publish_model_version(
    model_version: str,
    *,
    client: MlflowClient,
    candidate_path: Path = CANDIDATE_PATH,
    evaluations_dir: Path = EVALUATIONS_DIR,
    registry_path: Path = REGISTRY_PATH,
    packages_root: Path = PACKAGES_ROOT,
    repo_root: Path = REPO_ROOT,
) -> ModelPackage:
    """Empaqueta y registra el checkpoint del run congelado, descargado de MLflow."""
    version = ModelVersion(root=model_version)
    candidate = require_frozen_candidate(candidate_path)
    evaluations = [
        e for e in _evaluations(evaluations_dir, "test") if e["run_id"] == candidate.run_id
    ]
    if len(evaluations) != 1:
        raise RuntimeError(
            "Se necesita la evaluación de test final del candidato congelado (ML-09): "
            f"hay {len(evaluations)}"
        )
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(client.download_artifacts(candidate.run_id, CHECKPOINT_ARTIFACT, tmp))
        # Verifica el sha256 del checkpoint de MLflow contra el congelado (ValueError si no).
        package = build_model_package(
            str(version),
            checkpoint_path=checkpoint,
            candidate_path=candidate_path,
            evaluation_path=Path(evaluations[0]["path"]),
            dependencies=runtime_dependencies(),
        )
        return _materialize_and_register(
            package,
            checkpoint,
            staging=Path(tmp) / "package",
            registry_path=registry_path,
            packages_root=packages_root,
            repo_root=repo_root,
        )


def _materialize_and_register(
    package: ModelPackage,
    checkpoint: Path,
    *,
    staging: Path,
    registry_path: Path,
    packages_root: Path,
    repo_root: Path,
) -> ModelPackage:
    """Materializa el paquete verificado y lo registra; una versión no se reescribe."""
    version = package.model_version
    materialize_model_package(package, checkpoint_path=checkpoint, output_dir=staging)
    package_dir = packages_root / package.model_name / str(version)
    final = ModelPackage.model_validate(
        {
            **package.model_dump(),
            "package_path": package_dir.relative_to(repo_root).as_posix(),
            "files": {rel: _sha256_file(staging / rel) for rel in PACKAGE_FILES},
        }
    )
    (staging / "package.json").write_text(final.model_dump_json(indent=2) + "\n", encoding="utf-8")

    existing = _registered(registry_path, version)
    if existing is not None:
        if existing == final:
            resolve_checkpoint(str(version), registry_path=registry_path, repo_root=repo_root)
            return existing
        raise ValueError(
            f"La model_version {version} ya está registrada con otro contenido: "
            "publica una versión nueva"
        )
    if package_dir.exists():
        raise FileExistsError(f"{package_dir} existe pero no está en el registro")
    package_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(staging, package_dir)
    register_model(final, registry_path)
    return final


def _semver(version: ModelVersion) -> tuple[int, ...]:
    return tuple(int(part) for part in str(version).split("."))


def publish_previous_version(
    model_version: str,
    *,
    client: MlflowClient,
    run_id: str,
    candidate_path: Path = CANDIDATE_PATH,
    registry_path: Path = REGISTRY_PATH,
    packages_root: Path = PACKAGES_ROOT,
    repo_root: Path = REPO_ROOT,
) -> ModelPackage:
    """Versión anterior: otro run terminado de la matriz, nunca evaluado en test.

    La versión vigente es el candidato congelado (ML-08), evaluado una vez en test
    (ML-09). Una versión anterior empaqueta otro run de la misma matriz, release y
    manifest para poder volver a ella y comparar artefactos; no trae métricas de test
    porque el test no se usa para evaluar ni comparar otros modelos.
    """
    version = ModelVersion(root=model_version)
    candidate = require_frozen_candidate(candidate_path)
    if run_id == candidate.run_id:
        raise ValueError("Ese run es el candidato congelado: publícalo con publish_model_version")
    current = [p for p in _load_registry(registry_path).models if p.run_id == candidate.run_id]
    if not current:
        raise ValueError("Publica primero la versión vigente (el candidato congelado)")
    if _semver(version) >= min(_semver(p.model_version) for p in current):
        raise ValueError(
            f"{version} no es anterior a la versión vigente "
            f"{min((p.model_version for p in current), key=_semver)}"
        )
    run = client.get_run(run_id)
    if run.info.status != "FINISHED":
        raise ValueError(f"El run {run_id} está {run.info.status}, no FINISHED")
    tags = run.data.tags
    expected = {
        "dataset_version": candidate.dataset_version,
        "manifest_hash": candidate.manifest_hash,
        "experiment_matrix": candidate.matrix_id,
    }
    for tag, value in expected.items():
        if tags.get(tag) != value:
            raise ValueError(f"El run {run_id} tiene {tag}={tags.get(tag)}, no {value}")
    tested = sorted(name for name in run.data.metrics if name.startswith("test_"))
    if tested:
        raise ValueError(f"El run {run_id} tiene métricas de test {tested}: no se publica")

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(client.download_artifacts(run_id, CHECKPOINT_ARTIFACT, tmp))
        fields = _checkpoint_fields(checkpoint)
        metrics = run.data.metrics
        validation = (
            f"best_val_loss {metrics.get('best_val_loss', float('nan')):.4f} en validation"
            f" (época {int(metrics.get('best_epoch', 0))})"
        )
        entry = tags.get("matrix_entry", run.info.run_name or run_id)
        package = ModelPackage(
            model_version=str(version),
            model_name="dog-cat-resnet18",
            run_id=run_id,
            checkpoint=f"runs:/{run_id}/{CHECKPOINT_ARTIFACT}",
            checkpoint_sha256=_sha256_file(checkpoint),
            dataset_version=candidate.dataset_version,
            manifest_hash=candidate.manifest_hash,
            architecture=fields["architecture"],
            class_map=fields["class_map"],
            preprocessing=fields["preprocessing"],
            dependencies=runtime_dependencies(),
            metrics={},
            weights_origin=fields["weights_origin"],
            model_card=ModelCard(
                purpose=(
                    "Versión anterior del clasificador dog/cat: otro run de la misma matriz "
                    f"de experimentos ({entry}), para poder volver a ella y comparar artefactos."
                ),
                dataset_release=candidate.dataset_version,
                manifest_hash=candidate.manifest_hash,
                mlflow_run_id=run_id,
                test_metrics={},
                limitations=[
                    f"{entry} no es el candidato congelado de ML-08 "
                    f"({candidate.entry}, elegido por {candidate.metric}); {validation}.",
                    "no se evaluó en test: el test se usa una sola vez, para el candidato, "
                    "y no para evaluar ni comparar otros modelos.",
                    "Solo distingue dog y cat: cualquier otro objeto se asigna a una de las dos.",
                    "Clasifica el recorte de una caja; no imágenes completas con varios objetos.",
                ],
                pretrained_weights=(
                    fields["weights_origin"].get("weights")
                    if isinstance(fields["weights_origin"], dict)
                    else None
                ),
            ),
        )
        return _materialize_and_register(
            package,
            checkpoint,
            staging=Path(tmp) / "package",
            registry_path=registry_path,
            packages_root=packages_root,
            repo_root=repo_root,
        )


def resolve_checkpoint(
    version: str, *, registry_path: Path = REGISTRY_PATH, repo_root: Path = REPO_ROOT
) -> Path:
    """model_version → checkpoint del paquete, verificado contra el registro."""
    package = resolve_model(version, registry_path)
    if not package.package_path or not package.files:
        raise ValueError(f"La model_version {version} solo tiene metadata, sin paquete")
    package_dir = repo_root / package.package_path
    for relpath, sha256 in package.files.items():
        path = package_dir / relpath
        if not path.is_file():
            raise FileNotFoundError(f"Falta {path} (¿dvc pull data/models.dvc?)")
        if _sha256_file(path) != sha256:
            raise ValueError(f"{relpath} de {version}: sha256 distinto del registrado")
    checkpoint = package_dir / PACKAGE_CHECKPOINT
    if package.files[PACKAGE_CHECKPOINT] != package.checkpoint_sha256:
        raise ValueError(f"El registro de {version} no es coherente: checkpoint_sha256 distinto")
    return checkpoint


def verify_model_version(
    version: str,
    *,
    client: MlflowClient,
    registry_path: Path = REGISTRY_PATH,
    repo_root: Path = REPO_ROOT,
    manifest_path: Path | None = None,
) -> list[str]:
    """Agent Test: model_version → MLflow run → checkpoint → release → manifest hash."""
    package = resolve_model(version, registry_path)
    problems = []
    try:
        checkpoint = resolve_checkpoint(version, registry_path=registry_path, repo_root=repo_root)
    except (ValueError, FileNotFoundError) as error:
        checkpoint = None
        problems.append(f"paquete: {error}")
    else:
        stored = (checkpoint.parents[1] / "package.json").read_text(encoding="utf-8")
        if ModelPackage.model_validate_json(stored) != package:
            problems.append("paquete: package.json distinto de la entrada del registro")
        model = load_checkpoint(checkpoint)
        architecture = package.architecture.model_dump(exclude={"name"})
        if dict(model.class_map) != package.class_map or (
            model.config.model_dump() != architecture
        ):
            problems.append("paquete: class map o arquitectura del checkpoint distintos")

    try:
        run = client.get_run(package.run_id)
    except MlflowException:
        problems.append(f"MLflow: no existe el run {package.run_id}")
    else:
        for field in ("dataset_version", "manifest_hash"):
            logged = run.data.tags.get(field)
            if logged != getattr(package, field):
                problems.append(
                    f"MLflow: {field} del run ({logged}) distinto del registrado "
                    f"({getattr(package, field)})"
                )
        with tempfile.TemporaryDirectory() as tmp:
            remote = Path(client.download_artifacts(package.run_id, CHECKPOINT_ARTIFACT, tmp))
            remote_sha = _sha256_file(remote)
        if remote_sha != package.checkpoint_sha256:
            problems.append(
                f"MLflow: el checkpoint del run tiene sha256 {remote_sha}, no "
                f"{package.checkpoint_sha256}"
            )

    manifest_file = manifest_path or (
        repo_root / "reports" / "releases" / package.dataset_version / "manifest.json"
    )
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("dataset_version") != package.dataset_version:
        problems.append(f"release: el manifiesto es de {manifest.get('dataset_version')}")
    if manifest.get("manifest_hash") != package.manifest_hash:
        problems.append(
            f"release: manifest_hash del manifiesto ({manifest.get('manifest_hash')}) distinto "
            f"del registrado ({package.manifest_hash})"
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    from tracking.client import tracking_client

    parser = argparse.ArgumentParser(description="OPS-06: paquete y registro de modelos")
    parser.add_argument("command", choices=["publish", "publish-previous", "resolve", "verify"])
    parser.add_argument("version")
    parser.add_argument("--run-id", help="publish-previous: run de la matriz a empaquetar")
    args = parser.parse_args(argv)

    if args.command == "resolve":
        print(resolve_checkpoint(args.version))
        return 0
    client = tracking_client()
    if args.command in ("publish", "publish-previous"):
        if args.command == "publish":
            package = publish_model_version(args.version, client=client)
        else:
            if not args.run_id:
                parser.error("publish-previous necesita --run-id")
            package = publish_previous_version(args.version, client=client, run_id=args.run_id)
        print(f"{package.model_name} {package.model_version} → {package.package_path}")
        for relpath, sha256 in (package.files or {}).items():
            print(f"  {relpath}: {sha256}")
        return 0
    problems = verify_model_version(args.version, client=client)
    package = resolve_model(args.version, REGISTRY_PATH)
    print(
        f"{package.model_version} → run {package.run_id} → checkpoint "
        f"{package.checkpoint_sha256[:12]}… → release {package.dataset_version} → "
        f"{package.manifest_hash}"
    )
    print("Cadena verificada" if not problems else "\n".join(problems))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
