"""OPS-06: semantic model package and persistent registry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from classification.registry import (
    ModelPackage,
    ModelRegistry,
    ModelVersion,
    build_model_package,
    materialize_model_package,
    register_model,
    resolve_model,
)


def package_payload() -> dict:
    return {
        "schema_version": "1.0",
        "model_version": "1.0.0",
        "model_name": "dog-cat-resnet18",
        "run_id": "bb448230424146349a969253d30db43b",
        "checkpoint": ("runs:/bb448230424146349a969253d30db43b/checkpoints/best.pt"),
        "checkpoint_sha256": ("84d6c88b4bd84c3e6f0229dd23f7f188ff85622bbb39e5c3988a97598fa90d52"),
        "dataset_version": "v0.1.1",
        "manifest_hash": (
            "sha256:178b28bddb1bef5c05975fc7150710658627e31af08a1a12d94b98f9e99cb2b2"
        ),
        "architecture": {
            "name": "resnet18",
            "image_size": 128,
            "hidden_layers": [],
            "dropout": 0.2,
            "pretrained": True,
            "trainable": "layer4",
        },
        "class_map": {
            "dog": 0,
            "cat": 1,
        },
        "preprocessing": {
            "resize": "square",
            "image_size": 128,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "dependencies": {
            "python": "3.12",
            "torch": "2.x",
            "torchvision": "0.x",
        },
        "metrics": {
            "accuracy_top1": 0.9577464788732394,
            "f1_macro": 0.9548441806232775,
        },
        "weights_origin": {
            "dataset": "ImageNet-1K",
            "library": "torchvision",
        },
        "model_card": {
            "purpose": "Clasificar crops dog/cat.",
            "dataset_release": "v0.1.1",
            "manifest_hash": (
                "sha256:178b28bddb1bef5c05975fc7150710658627e31af08a1a12d94b98f9e99cb2b2"
            ),
            "mlflow_run_id": "bb448230424146349a969253d30db43b",
            "test_metrics": {
                "accuracy_top1": 0.9577464788732394,
                "f1_macro": 0.9548441806232775,
            },
            "limitations": [
                "Clasificador binario dog/cat.",
                "Depende de crops similares al dominio de entrenamiento.",
            ],
            "pretrained_weights": "ResNet18 ImageNet-1K",
        },
    }


def test_model_version_requires_semver():
    ModelVersion(root="1.0.0")

    with pytest.raises(ValidationError):
        ModelVersion(root="v1.0.0")

    with pytest.raises(ValidationError):
        ModelVersion(root="1.0")

    with pytest.raises(ValidationError):
        ModelVersion(root="v0.1.1")


def test_model_version_is_independent_from_dataset_version():
    package = ModelPackage.model_validate(package_payload())

    assert str(package.model_version) == "1.0.0"
    assert package.dataset_version == "v0.1.1"
    assert str(package.model_version) != package.dataset_version


def test_package_contains_required_reproducibility_metadata():
    package = ModelPackage.model_validate(package_payload())

    assert package.run_id == "bb448230424146349a969253d30db43b"
    assert package.checkpoint.endswith("/checkpoints/best.pt")
    assert len(package.checkpoint_sha256) == 64
    assert package.architecture.name == "resnet18"
    assert package.class_map == {"dog": 0, "cat": 1}
    assert package.preprocessing.image_size == 128
    assert package.dependencies
    assert package.model_card.mlflow_run_id == package.run_id
    assert package.model_card.dataset_release == package.dataset_version
    assert package.model_card.manifest_hash == package.manifest_hash


def test_registry_persists_and_resolves_versions(tmp_path):
    registry_path = tmp_path / "registry.json"

    first = ModelPackage.model_validate(package_payload())
    register_model(first, registry_path)

    loaded = ModelRegistry.model_validate_json(registry_path.read_text(encoding="utf-8"))

    assert len(loaded.models) == 1

    resolved = resolve_model("1.0.0", registry_path)

    assert resolved.run_id == first.run_id
    assert resolved.checkpoint == first.checkpoint
    assert resolved.checkpoint_sha256 == first.checkpoint_sha256


def test_registering_new_version_keeps_previous_version(tmp_path):
    registry_path = tmp_path / "registry.json"

    first = ModelPackage.model_validate(package_payload())
    register_model(first, registry_path)

    second_payload = package_payload()
    second_payload["model_version"] = "1.1.0"
    second_payload["run_id"] = "11111111111111111111111111111111"
    second_payload["checkpoint"] = "runs:/11111111111111111111111111111111/checkpoints/best.pt"
    second_payload["model_card"]["mlflow_run_id"] = "11111111111111111111111111111111"

    second = ModelPackage.model_validate(second_payload)
    register_model(second, registry_path)

    old = resolve_model("1.0.0", registry_path)
    new = resolve_model("1.1.0", registry_path)

    assert old.run_id == first.run_id
    assert new.run_id == second.run_id


def test_registering_same_version_with_different_model_is_rejected(tmp_path):
    registry_path = tmp_path / "registry.json"

    first = ModelPackage.model_validate(package_payload())
    register_model(first, registry_path)

    changed_payload = package_payload()
    changed_payload["run_id"] = "22222222222222222222222222222222"
    changed_payload["checkpoint"] = "runs:/22222222222222222222222222222222/checkpoints/best.pt"
    changed_payload["model_card"]["mlflow_run_id"] = "22222222222222222222222222222222"

    changed = ModelPackage.model_validate(changed_payload)

    with pytest.raises(ValueError, match=r"1\.0\.0"):
        register_model(changed, registry_path)


def test_resolve_unknown_version_fails(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "models": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(KeyError, match=r"9\.9\.9"):
        resolve_model("9.9.9", registry_path)


def test_build_model_package_from_frozen_candidate_and_evaluation(tmp_path):
    checkpoint = tmp_path / "best.pt"

    payload = {
        "format_version": 1,
        "architecture": "resnet18",
        "config": {
            "image_size": 128,
            "hidden_layers": [],
            "dropout": 0.2,
            "pretrained": True,
            "trainable": "layer4",
        },
        "class_map": {
            "dog": 0,
            "cat": 1,
        },
        "preprocessing": {
            "image_size": 128,
            "resize": "square",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "weights_origin": {
            "weights": "ResNet18_Weights.IMAGENET1K_V1",
            "dataset": "ImageNet-1K",
            "library": "torchvision",
        },
        "metadata": {
            "run_id": "abc123",
            "dataset_version": "v0.1.1",
            "manifest_hash": "sha256:manifest",
        },
        "state_dict": {},
    }
    torch.save(payload, checkpoint)

    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(
        json.dumps(
            {
                "run_id": "abc123",
                "checkpoint": "runs:/abc123/checkpoints/best.pt",
                "checkpoint_sha256": checkpoint_sha256,
                "dataset_version": "v0.1.1",
                "manifest_hash": "sha256:manifest",
            }
        ),
        encoding="utf-8",
    )

    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(
        json.dumps(
            {
                "run_id": "abc123",
                "checkpoint": "runs:/abc123/checkpoints/best.pt",
                "dataset_version": "v0.1.1",
                "manifest_hash": "sha256:manifest",
                "split": "test",
                "metrics": {
                    "accuracy_top1": 0.95,
                    "f1_macro": 0.94,
                },
            }
        ),
        encoding="utf-8",
    )

    package = build_model_package(
        "1.0.0",
        checkpoint_path=checkpoint,
        candidate_path=candidate_path,
        evaluation_path=evaluation_path,
        dependencies={
            "python": "3.12.12",
            "torch": "2.7.1",
            "torchvision": "0.22.1",
        },
    )

    assert str(package.model_version) == "1.0.0"
    assert package.run_id == "abc123"
    assert package.dataset_version == "v0.1.1"
    assert package.manifest_hash == "sha256:manifest"
    assert package.checkpoint_sha256 == checkpoint_sha256

    assert package.architecture.name == "resnet18"
    assert package.architecture.image_size == 128

    assert package.class_map == {"dog": 0, "cat": 1}

    assert package.preprocessing.image_size == 128
    assert package.preprocessing.resize == "square"

    assert package.metrics["accuracy_top1"] == 0.95
    assert package.metrics["f1_macro"] == 0.94

    assert package.weights_origin["dataset"] == "ImageNet-1K"

    assert package.model_card.dataset_release == "v0.1.1"
    assert package.model_card.manifest_hash == "sha256:manifest"
    assert package.model_card.mlflow_run_id == "abc123"
    assert package.model_card.test_metrics["accuracy_top1"] == 0.95


def test_materialize_model_package_writes_all_required_files(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"real-checkpoint")

    payload = package_payload()
    payload["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    package = ModelPackage.model_validate(payload)

    package_dir = tmp_path / "models" / "1.0.0"

    result = materialize_model_package(
        package,
        checkpoint_path=checkpoint,
        output_dir=package_dir,
    )

    assert result == package_dir

    assert (package_dir / "checkpoint" / "best.pt").read_bytes() == b"real-checkpoint"
    assert (package_dir / "package.json").is_file()
    assert (package_dir / "model-card.md").is_file()
    assert (package_dir / "dependencies.json").is_file()

    stored = json.loads((package_dir / "package.json").read_text(encoding="utf-8"))

    assert stored["model_version"] == "1.0.0"
    assert stored["run_id"] == package.run_id
    assert stored["dataset_version"] == "v0.1.1"
    assert stored["manifest_hash"] == package.manifest_hash

    card = (package_dir / "model-card.md").read_text(encoding="utf-8")

    assert "dog-cat-resnet18" in card
    assert "v0.1.1" in card
    assert package.run_id in card
    assert "0.9577464788732394" in card


def test_real_candidate_and_evaluation_are_traceable():
    repo_root = Path(__file__).resolve().parents[2]

    candidate_path = repo_root / "reports" / "candidates" / "ml08_candidate.json"

    evaluation_path = (
        repo_root / "reports" / "evaluations" / "test" / "test-r03-sgd-20261002T012633Z.json"
    )

    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))

    assert candidate["run_id"] == evaluation["run_id"]

    assert candidate["checkpoint"] == evaluation["checkpoint"]

    assert candidate["dataset_version"] == evaluation["dataset_version"]

    assert candidate["manifest_hash"] == evaluation["manifest_hash"]

    assert candidate["run_id"] == "bb448230424146349a969253d30db43b"

    assert candidate["checkpoint"] == ("runs:/bb448230424146349a969253d30db43b/checkpoints/best.pt")

    assert candidate["dataset_version"] == "v0.1.1"

    assert evaluation["metrics"]["accuracy_top1"] == pytest.approx(0.9577464788732394)

    assert evaluation["metrics"]["f1_macro"] == pytest.approx(0.9548441806232775)
