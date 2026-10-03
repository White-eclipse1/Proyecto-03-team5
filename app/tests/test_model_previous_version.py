"""OPS-10 (rúbricas 5.3 y 6.4): una versión anterior real del modelo.

El candidato congelado (ML-08) es la versión vigente (`1.0.0`). Una versión anterior
empaqueta **otro** run terminado de la misma matriz, con el mismo release y manifest,
que nunca se evaluó en test: su tarjeta lo dice y no trae métricas de test (el test se
usa una sola vez, para el candidato). Así cambiar de versión cambia el artefacto real.
"""

import hashlib
import json
from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from classification.registry import (
    ModelRegistry,
    publish_model_version,
    publish_previous_version,
    resolve_checkpoint,
    resolve_model,
    verify_model_version,
)
from classification.training import CHECKPOINT_ARTIFACT, DataPaths, run_training
from presentation.ml_contracts import TrainingParams
from tests._classification_fixtures import write_controlled_release
from tests._frozen_candidate import COMMIT, evaluate_test_once, freeze_short_run
from training.models_view import list_models


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    return MlflowClient(tracking_uri=(tmp_path / "mlruns").as_uri())


@pytest.fixture
def release(tmp_path):
    return write_controlled_release(tmp_path / "release")


@pytest.fixture
def setup(client, release, tmp_path):
    data = DataPaths(release.manifest_path, release.crop_report_path, release.crops_dir)
    frozen = freeze_short_run(client, release, data, tmp_path)
    evaluate_test_once(client, data, frozen, tmp_path)
    paths = {
        "candidate_path": frozen,
        "registry_path": tmp_path / "reports" / "models" / "registry.json",
        "packages_root": tmp_path / "data" / "models",
        "repo_root": tmp_path,
    }
    current = publish_model_version(
        "1.0.0",
        client=client,
        evaluations_dir=tmp_path / "evaluations",
        **paths,
    )
    other = run_training(
        TrainingParams(
            optimizer="sgd",
            batch_size=4,
            max_epochs=2,
            learning_rate=0.01,
            image_size=64,
            hidden_layers=[16],
            dropout=0.2,
            seed=5,
            patience=2,
            min_delta=0.0,
        ),
        dataset_version=release.dataset_version,
        manifest_hash=release.manifest_hash,
        data=data,
        client=client,
        pretrained=False,
        git_commit=COMMIT,
        tags={"experiment_matrix": "ml07-test", "matrix_entry": "r02"},
    )
    return {"paths": paths, "current": current, "other": other, "release": release}


def _previous(client, setup, run_id=None, version="0.9.0"):
    return publish_previous_version(
        version,
        client=client,
        run_id=run_id or setup["other"].run_id,
        **setup["paths"],
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_previous_version_packages_another_finished_run_of_the_same_matrix(client, setup):
    package = _previous(client, setup)

    other = setup["other"]
    remote = Path(
        client.download_artifacts(
            other.run_id, CHECKPOINT_ARTIFACT, str(setup["paths"]["repo_root"] / "dl")
        )
    )
    assert package.run_id == other.run_id
    assert package.checkpoint_sha256 == _sha(remote)
    assert package.checkpoint_sha256 != setup["current"].checkpoint_sha256
    assert package.architecture.image_size == 64
    assert (package.dataset_version, package.manifest_hash) == (
        setup["current"].dataset_version,
        setup["current"].manifest_hash,
    )
    assert package.package_path == "data/models/dog-cat-resnet18/0.9.0"


def test_the_previous_version_has_no_test_metrics_and_its_card_says_why(client, setup):
    package = _previous(client, setup)

    assert package.metrics == {}
    assert package.model_card.test_metrics == {}
    limitations = " ".join(package.model_card.limitations)
    assert "no es el candidato congelado" in limitations
    assert "no se evaluó en test" in limitations
    assert "best_val_loss" in limitations
    card = (setup["paths"]["repo_root"] / package.package_path / "model-card.md").read_text(
        encoding="utf-8"
    )
    assert "Sin evaluación en test" in card


def test_changing_the_version_changes_the_real_artifact(client, setup):
    _previous(client, setup)
    registry_path = setup["paths"]["registry_path"]
    root = setup["paths"]["repo_root"]

    current = resolve_checkpoint("1.0.0", registry_path=registry_path, repo_root=root)
    previous = resolve_checkpoint("0.9.0", registry_path=registry_path, repo_root=root)

    assert current != previous
    assert _sha(current) != _sha(previous)
    assert resolve_model("0.9.0", registry_path).architecture.image_size == 64


def test_the_current_version_stays_first_and_untouched(client, setup):
    root = setup["paths"]["repo_root"]
    current_dir = root / setup["current"].package_path
    before = {path: path.read_bytes() for path in current_dir.rglob("*") if path.is_file()}

    _previous(client, setup)

    registry = ModelRegistry.model_validate_json(
        setup["paths"]["registry_path"].read_text(encoding="utf-8")
    )
    assert [str(p.model_version) for p in registry.models] == ["1.0.0", "0.9.0"]
    assert resolve_model("1.0.0", setup["paths"]["registry_path"]) == setup["current"]
    assert {path: path.read_bytes() for path in current_dir.rglob("*") if path.is_file()} == before


def test_the_agent_test_chain_holds_for_the_previous_version(client, setup):
    _previous(client, setup)

    problems = verify_model_version(
        "0.9.0",
        client=client,
        registry_path=setup["paths"]["registry_path"],
        repo_root=setup["paths"]["repo_root"],
        manifest_path=setup["release"].manifest_path,
    )

    assert problems == []


def test_models_shows_the_previous_version_without_test_metrics(client, setup):
    _previous(client, setup)
    root = setup["paths"]["repo_root"]

    models = list_models(root / "reports", root).models

    assert [m.model_version for m in models] == ["1.0.0", "0.9.0"]
    assert models[0].test_metrics is not None
    assert models[1].test_metrics is None
    assert models[1].servable is True


def test_the_frozen_candidate_is_not_a_previous_version(client, setup):
    candidate = json.loads(setup["paths"]["candidate_path"].read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="candidato"):
        _previous(client, setup, run_id=candidate["run_id"])


def test_a_run_with_test_metrics_is_refused(client, setup):
    client.log_metric(setup["other"].run_id, "test_accuracy", 0.9)

    with pytest.raises(ValueError, match="test"):
        _previous(client, setup)


@pytest.mark.parametrize(
    ("tag", "value"),
    [
        ("manifest_hash", "sha256:" + "f" * 64),
        ("dataset_version", "v9.9.9"),
        ("experiment_matrix", "otra-matriz"),
    ],
)
def test_a_run_of_other_data_or_matrix_is_refused(client, setup, tag, value):
    client.set_tag(setup["other"].run_id, tag, value)

    with pytest.raises(ValueError, match=tag):
        _previous(client, setup)


def test_an_unfinished_run_is_refused(client, setup):
    experiment = client.get_run(setup["other"].run_id).info.experiment_id
    running = client.create_run(experiment)
    for tag in ("dataset_version", "manifest_hash", "experiment_matrix"):
        client.set_tag(
            running.info.run_id, tag, client.get_run(setup["other"].run_id).data.tags[tag]
        )

    with pytest.raises(ValueError, match="FINISHED"):
        _previous(client, setup, run_id=running.info.run_id)


def test_a_previous_version_must_be_lower_than_the_current_one(client, setup):
    with pytest.raises(ValueError, match="anterior"):
        _previous(client, setup, version="1.1.0")
