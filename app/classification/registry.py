"""OPS-06 — semantic model package and persistent model registry."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Annotated, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

SEMVER_PATTERN = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


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

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    architecture = {
        "name": checkpoint["architecture"],
        **checkpoint["config"],
    }

    preprocessing = checkpoint["preprocessing"]
    class_map = checkpoint["class_map"]
    weights_origin = checkpoint.get("weights_origin")

    metrics = {
        "accuracy_top1": evaluation["metrics"]["accuracy_top1"],
        "f1_macro": evaluation["metrics"]["f1_macro"],
    }

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
            purpose="Clasificar crops dog/cat con el candidato congelado.",
            dataset_release=candidate["dataset_version"],
            manifest_hash=candidate["manifest_hash"],
            mlflow_run_id=candidate["run_id"],
            test_metrics=metrics,
            limitations=[
                "Clasificador binario dog/cat.",
                "La calidad depende de que las imágenes sean similares al dominio evaluado.",
            ],
            pretrained_weights=(
                weights_origin.get("weights") if isinstance(weights_origin, dict) else None
            ),
        ),
    )


def materialize_model_package(
    package: ModelPackage,
    *,
    checkpoint_path: Path,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = output_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_target = checkpoint_dir / "best.pt"
    shutil.copy2(checkpoint_path, checkpoint_target)

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
    model_card.write_text(
        "\n".join(
            [
                f"# Model Card — {package.model_name}",
                "",
                f"- Model version: {package.model_version}",
                f"- Dataset release: {package.dataset_version}",
                f"- Manifest hash: {package.manifest_hash}",
                f"- MLflow run_id: {package.run_id}",
                f"- Checkpoint: {package.checkpoint}",
                f"- Checkpoint SHA256: {package.checkpoint_sha256}",
                "",
                "## Purpose",
                "",
                package.model_card.purpose,
                "",
                "## Test metrics",
                "",
                *[f"- {name}: {value}" for name, value in package.model_card.test_metrics.items()],
                "",
                "## Limitations",
                "",
                *[f"- {limitation}" for limitation in package.model_card.limitations],
                "",
                "## Pretrained weights",
                "",
                package.model_card.pretrained_weights or "None",
                "",
            ]
        ),
        encoding="utf-8",
    )

    return output_dir
