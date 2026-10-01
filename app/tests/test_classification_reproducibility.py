"""ML-05: semillas registradas, orden de muestras reproducible y augmentation solo en train."""

import json
import platform
from pathlib import Path

import mlflow
import numpy
import PIL
import pytest
import torch
import torchvision
from mlflow.tracking import MlflowClient

import classification.dataset as dataset_module
import classification.training as training_module
from classification.dataset import load_split
from classification.training import (
    EXPERIMENT_NAME,
    SAMPLE_ORDER_ARTIFACT,
    DataPaths,
    environment_tags,
    run_training,
)
from classification.transforms import random_transform_names, train_transform
from presentation.ml_contracts import TrainingParams
from tests._classification_fixtures import write_controlled_release

COMMIT = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    return MlflowClient(tracking_uri=(tmp_path / "mlruns").as_uri())


@pytest.fixture
def release(tmp_path):
    return write_controlled_release(tmp_path / "release")


def _params(**overrides):
    values = {
        "optimizer": "adam",
        "batch_size": 3,
        "max_epochs": 2,
        "learning_rate": 0.001,
        "image_size": 32,
        "hidden_layers": [16],
        "dropout": 0.1,
        "seed": 7,
        "patience": 1,
        "min_delta": 0.0,
    }
    values.update(overrides)
    return TrainingParams(**values)


def _data(release):
    return DataPaths(release.manifest_path, release.crop_report_path, release.crops_dir)


def _split(release, split, **kwargs):
    return load_split(
        release.manifest_path,
        release.crop_report_path,
        crops_dir=release.crops_dir,
        split=split,
        params=_params(image_size=32),
        **kwargs,
    )


def _train(client, release, params=None, **kwargs):
    return run_training(
        params or _params(),
        dataset_version=release.dataset_version,
        manifest_hash=release.manifest_hash,
        data=_data(release),
        client=client,
        pretrained=False,
        git_commit=COMMIT,
        **kwargs,
    )


# --- Transforms: aleatorios solo en train ------------------------------------------------


def test_random_transform_names_finds_augmentation_only_in_train():
    from classification.transforms import transform_for

    assert random_transform_names(train_transform(32)) == [
        "RandomResizedCrop",
        "RandomHorizontalFlip",
        "ColorJitter",
    ]
    for split in ("validation", "test", "inference"):
        assert random_transform_names(transform_for(split, 32)) == []


# --- Semilla de augmentation por muestra ------------------------------------------------


def _read(dataset, index, *, global_seed):
    torch.manual_seed(global_seed)
    return dataset[index]["image"]


def test_seeded_train_augmentation_ignores_the_global_rng(release):
    train = _split(release, "train", augmentation_seed=5)

    reads = [_read(train, 0, global_seed=seed) for seed in range(5)]

    assert all(torch.equal(image, reads[0]) for image in reads[1:])


def test_seeded_augmentation_still_varies_by_epoch_and_sample(release):
    train = _split(release, "train", augmentation_seed=5)
    first = _read(train, 0, global_seed=0)

    train.set_epoch(2)

    assert not torch.equal(_read(train, 0, global_seed=0), first)
    train.set_epoch(1)
    assert torch.equal(_read(train, 0, global_seed=0), first)
    assert not torch.equal(_read(train, 1, global_seed=0), _read(train, 0, global_seed=0))


def test_augmentation_seed_reproduces_across_dataset_instances(release):
    one = _split(release, "train", augmentation_seed=5)
    two = _split(release, "train", augmentation_seed=5)
    other = _split(release, "train", augmentation_seed=6)

    assert torch.equal(_read(one, 3, global_seed=1), _read(two, 3, global_seed=2))
    assert not torch.equal(_read(one, 3, global_seed=1), _read(other, 3, global_seed=1))


def test_seeded_augmentation_does_not_consume_the_global_rng(release):
    train = _split(release, "train", augmentation_seed=5)
    torch.manual_seed(11)
    expected = torch.rand(4)

    torch.manual_seed(11)
    train[0]

    assert torch.equal(torch.rand(4), expected)


@pytest.mark.parametrize("split", ["validation", "test"])
def test_validation_and_test_ignore_the_augmentation_seed(release, split):
    plain = _split(release, split)
    seeded = _split(release, split, augmentation_seed=5)
    seeded.set_epoch(3)

    assert random_transform_names(seeded.transform) == []
    assert torch.equal(_read(seeded, 0, global_seed=1), _read(plain, 0, global_seed=2))


# --- Semillas y versiones en MLflow --------------------------------------------------------


def test_run_records_every_seed(client, release):
    result = _train(client, release, _params(seed=13))

    params = client.get_run(result.run_id).data.params
    assert params["seed_split"] == "42"  # semilla con la que OPS-02 generó el manifiesto
    assert params["seed_dataloader"] == "13"
    assert params["seed_augmentation"] == "13"
    assert params["seed_weight_init"] == "13"
    assert params["seed"] == "13"


def test_environment_tags_record_library_and_platform_versions():
    tags = environment_tags()

    assert tags["python_version"] == platform.python_version()
    assert tags["torch_version"] == torch.__version__
    assert tags["torchvision_version"] == torchvision.__version__
    assert tags["numpy_version"] == numpy.__version__
    assert tags["pillow_version"] == PIL.__version__
    assert tags["mlflow_version"] == mlflow.__version__
    assert tags["platform"] == platform.platform()
    assert tags["torch_num_threads"] == str(torch.get_num_threads())
    assert tags["cuda_available"] == str(torch.cuda.is_available())
    assert "torch_deterministic_algorithms" in tags


def test_run_records_the_environment(client, release):
    result = _train(client, release)

    tags = client.get_run(result.run_id).data.tags
    for key, value in environment_tags().items():
        assert tags[key] == value


# --- Orden de muestras (Agent Test) ---------------------------------------------------------


def test_two_short_runs_with_the_same_seed_see_the_same_sample_order(client, release):
    first = _train(client, release, _params(seed=21))
    again = _train(client, release, _params(seed=21))

    assert first.sample_order == again.sample_order
    assert first.history == again.history
    assert first.sample_order[1] != first.sample_order[2]  # se baraja en cada época


def test_another_seed_changes_the_sample_order(client, release):
    first = _train(client, release, _params(seed=21))
    other = _train(client, release, _params(seed=22))

    assert first.sample_order != other.sample_order


def test_sample_order_covers_every_train_crop_each_epoch(client, release):
    result = _train(client, release, _params(batch_size=4))
    train_ids = {crop.crop_id for crop in _split(release, "train")._samples}

    assert set(result.sample_order) == {1, 2}
    for order in result.sample_order.values():
        assert sorted(order) == sorted(train_ids)


def test_sample_order_is_logged_as_an_artifact(client, release, tmp_path):
    result = _train(client, release)

    path = client.download_artifacts(result.run_id, SAMPLE_ORDER_ARTIFACT, str(tmp_path))
    logged = json.loads(Path(path).read_text(encoding="utf-8"))
    assert logged == {str(epoch): order for epoch, order in result.sample_order.items()}
    tag = client.get_run(result.run_id).data.tags["train_order_sha256"]
    assert len(tag) == 64


# --- Protección de validation/test en el entrenamiento --------------------------------------


def test_training_never_loads_the_test_split(client, release, monkeypatch):
    loaded = []
    real_load = training_module.load_split

    def spy(*args, split, **kwargs):
        loaded.append(split)
        return real_load(*args, split=split, **kwargs)

    monkeypatch.setattr(training_module, "load_split", spy)

    _train(client, release)

    assert sorted(loaded) == ["train", "validation"]


def test_training_refuses_a_validation_split_with_random_transforms(client, release, monkeypatch):
    monkeypatch.setattr(dataset_module, "transform_for", lambda split, size: train_transform(size))

    with pytest.raises(ValueError, match="validation"):
        _train(client, release)

    assert client.get_experiment_by_name(EXPERIMENT_NAME) is None
