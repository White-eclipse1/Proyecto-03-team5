"""OPS-06: semantic model package and persistent registry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from mlflow.tracking import MlflowClient
from pydantic import ValidationError

from classification.model import load_checkpoint
from classification.registry import (
    ModelPackage,
    ModelRegistry,
    ModelVersion,
    build_model_package,
    materialize_model_package,
    publish_model_version,
    register_model,
    resolve_checkpoint,
    resolve_model,
    verify_model_version,
)
from classification.training import DataPaths
from tests._classification_fixtures import write_controlled_release
from tests._frozen_candidate import evaluate_test_once, freeze_short_run


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


# --- Revisión del PR: integridad del checkpoint y flujo real desde MLflow ---------------


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_materialize_rejects_a_checkpoint_whose_sha256_does_not_match(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"un archivo que no es el checkpoint registrado")
    package = ModelPackage.model_validate(package_payload())
    output_dir = tmp_path / "models" / "1.0.0"

    with pytest.raises(ValueError, match="sha256"):
        materialize_model_package(package, checkpoint_path=checkpoint, output_dir=output_dir)

    assert not output_dir.exists()


def _publish(client, frozen, tmp_path, version="1.0.0"):
    return publish_model_version(
        version,
        client=client,
        candidate_path=frozen,
        evaluations_dir=tmp_path / "evaluations",
        registry_path=tmp_path / "reports" / "models" / "registry.json",
        packages_root=tmp_path / "data" / "models",
        repo_root=tmp_path,
    )


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    return MlflowClient(tracking_uri=(tmp_path / "mlruns").as_uri())


@pytest.fixture
def release(tmp_path):
    return write_controlled_release(tmp_path / "release")


@pytest.fixture
def data(release):
    return DataPaths(release.manifest_path, release.crop_report_path, release.crops_dir)


@pytest.fixture
def frozen(client, release, data, tmp_path):
    return freeze_short_run(client, release, data, tmp_path)


@pytest.fixture
def published(client, data, frozen, tmp_path):
    evaluate_test_once(client, data, frozen, tmp_path)
    return _publish(client, frozen, tmp_path)


def test_publish_downloads_the_run_checkpoint_and_registers_a_verifiable_package(
    client, frozen, published, tmp_path
):
    candidate = json.loads(frozen.read_text(encoding="utf-8"))
    package_dir = tmp_path / published.package_path

    assert published.run_id == candidate["run_id"]
    assert published.checkpoint == candidate["checkpoint"]
    assert published.checkpoint_sha256 == candidate["checkpoint_sha256"]
    assert published.package_path == "data/models/dog-cat-resnet18/1.0.0"
    assert set(published.files) == {"checkpoint/best.pt", "model-card.md", "dependencies.json"}
    for relpath, sha256 in published.files.items():
        assert _sha(package_dir / relpath) == sha256
    # El checkpoint del paquete es exactamente el artefacto del run de MLflow.
    mlflow_copy = Path(
        client.download_artifacts(published.run_id, "checkpoints/best.pt", str(tmp_path / "dl"))
    )
    assert (package_dir / "checkpoint" / "best.pt").read_bytes() == mlflow_copy.read_bytes()
    stored = ModelPackage.model_validate_json((package_dir / "package.json").read_text("utf-8"))
    assert stored == published
    assert resolve_model("1.0.0", tmp_path / "reports" / "models" / "registry.json") == published
    model = load_checkpoint(package_dir / "checkpoint" / "best.pt")
    assert dict(model.class_map) == published.class_map


def test_the_model_card_documents_purpose_data_run_metrics_and_limitations(published, tmp_path):
    card = (tmp_path / published.package_path / "model-card.md").read_text(encoding="utf-8")

    for text in (
        published.model_card.purpose,
        f"Release P2: {published.dataset_version}",
        published.manifest_hash,
        published.run_id,
        "accuracy_top1",
        "recall_cat",
        "## Limitaciones",
        "## Pesos preentrenados",
    ):
        assert text in card, text
    assert (
        published.model_card.test_metrics["correct"] <= published.model_card.test_metrics["total"]
    )


def test_dependencies_are_the_real_installed_versions(published):
    import torch as installed_torch
    import torchvision

    assert published.dependencies["torch"] == installed_torch.__version__
    assert published.dependencies["torchvision"] == torchvision.__version__
    assert len(published.dependencies["uv.lock sha256"]) == 64


def test_publish_refuses_a_run_checkpoint_that_is_not_the_frozen_one(
    client, data, frozen, tmp_path
):
    evaluate_test_once(client, data, frozen, tmp_path)
    run_id = json.loads(frozen.read_text(encoding="utf-8"))["run_id"]
    other = tmp_path / "otro" / "best.pt"
    other.parent.mkdir()
    other.write_bytes(b"checkpoint reemplazado en MLflow")
    client.log_artifact(run_id, str(other), "checkpoints")

    with pytest.raises(ValueError, match="sha256"):
        _publish(client, frozen, tmp_path)

    assert not (tmp_path / "reports" / "models" / "registry.json").exists()
    assert not (tmp_path / "data" / "models").exists()


def test_publish_requires_the_final_test_evaluation(client, frozen, tmp_path):
    with pytest.raises(RuntimeError, match="evaluación de test"):
        _publish(client, frozen, tmp_path)


def test_a_new_version_keeps_the_previous_one_resolvable(client, frozen, published, tmp_path):
    registry_path = tmp_path / "reports" / "models" / "registry.json"
    previous = {
        relpath: _sha(tmp_path / published.package_path / relpath) for relpath in published.files
    }

    newer = _publish(client, frozen, tmp_path, version="1.1.0")

    assert [
        str(p.model_version)
        for p in ModelRegistry.model_validate_json(registry_path.read_text("utf-8")).models
    ] == ["1.0.0", "1.1.0"]
    assert resolve_model("1.0.0", registry_path) == published
    assert resolve_checkpoint("1.0.0", registry_path=registry_path, repo_root=tmp_path).is_file()
    assert resolve_checkpoint("1.1.0", registry_path=registry_path, repo_root=tmp_path).is_file()
    assert newer.package_path == "data/models/dog-cat-resnet18/1.1.0"
    assert {
        relpath: _sha(tmp_path / published.package_path / relpath) for relpath in published.files
    } == previous


def test_republishing_the_same_version_changes_nothing(client, frozen, published, tmp_path):
    registry_path = tmp_path / "reports" / "models" / "registry.json"
    before = registry_path.read_bytes()

    again = _publish(client, frozen, tmp_path)

    assert again == published
    assert registry_path.read_bytes() == before


def test_resolve_checkpoint_rejects_a_tampered_package(published, tmp_path):
    registry_path = tmp_path / "reports" / "models" / "registry.json"
    checkpoint = resolve_checkpoint("1.0.0", registry_path=registry_path, repo_root=tmp_path)
    assert _sha(checkpoint) == published.checkpoint_sha256

    checkpoint.write_bytes(b"alterado")

    with pytest.raises(ValueError, match="sha256"):
        resolve_checkpoint("1.0.0", registry_path=registry_path, repo_root=tmp_path)


def _verify(client, release, tmp_path):
    return verify_model_version(
        "1.0.0",
        client=client,
        registry_path=tmp_path / "reports" / "models" / "registry.json",
        repo_root=tmp_path,
        manifest_path=release.manifest_path,
    )


def test_agent_test_resolves_the_whole_chain(client, release, published, tmp_path):
    # model_version → MLflow run → checkpoint → dataset release → manifest hash.
    assert _verify(client, release, tmp_path) == []


def test_agent_test_detects_a_manifest_of_another_release(client, release, published, tmp_path):
    manifest = json.loads(release.manifest_path.read_text(encoding="utf-8"))
    manifest["manifest_hash"] = "sha256:" + "f" * 64
    release.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    problems = _verify(client, release, tmp_path)

    assert len(problems) == 1 and "manifest_hash" in problems[0], problems


def test_agent_test_detects_a_run_of_another_dataset(client, release, published, tmp_path):
    client.set_tag(published.run_id, "dataset_version", "v9.9.9")

    problems = _verify(client, release, tmp_path)

    assert len(problems) == 1 and "dataset_version" in problems[0], problems


def test_agent_test_detects_a_changed_checkpoint_in_mlflow(client, release, published, tmp_path):
    other = tmp_path / "otro" / "best.pt"
    other.parent.mkdir()
    other.write_bytes(b"otro checkpoint")
    client.log_artifact(published.run_id, str(other), "checkpoints")

    problems = _verify(client, release, tmp_path)

    assert len(problems) == 1 and "MLflow" in problems[0], problems


def test_committed_registry_points_to_a_verifiable_package():
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "reports" / "models" / "registry.json").read_text(encoding="utf-8")
    assert "Ã" not in text  # sin texto con la codificación rota
    registry = ModelRegistry.model_validate_json(text)
    candidate = json.loads(
        (repo_root / "reports" / "candidates" / "ml08_candidate.json").read_text(encoding="utf-8")
    )

    entry = next(p for p in registry.models if str(p.model_version) == "1.0.0")

    assert entry.run_id == candidate["run_id"]
    assert entry.checkpoint_sha256 == candidate["checkpoint_sha256"]
    assert entry.files["checkpoint/best.pt"] == candidate["checkpoint_sha256"]
    assert entry.package_path == "data/models/dog-cat-resnet18/1.0.0"
    # El paquete (45 MB) se versiona con DVC; el .dvc está en git.
    assert (repo_root / "data" / "models.dvc").is_file()
