"""ML-03: clasificador dog/cat ResNet18 (ImageNet) con cabeza configurable."""

import pytest
import torch
from torch import nn

import classification.model as model_module
from classification.model import (
    CLASS_MAP,
    WEIGHTS_ORIGIN,
    ModelConfig,
    build_model,
    load_checkpoint,
    predict_proba,
    save_checkpoint,
)
from classification.transforms import IMAGENET_MEAN, IMAGENET_STD
from presentation.ml_contracts import TrainingParams


def _config(**overrides):
    # pretrained=False: misma arquitectura sin descargar pesos (los tests no usan red).
    values = {"image_size": 64, "hidden_layers": [32], "dropout": 0.25, "pretrained": False}
    values.update(overrides)
    return ModelConfig(**values)


def _batch(size=4, image_size=64, seed=0):
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(size, 3, image_size, image_size, generator=generator)
    labels = torch.tensor([0, 1] * (size // 2))
    return images, labels


def _linear_dims(head):
    return [
        (layer.in_features, layer.out_features) for layer in head if isinstance(layer, nn.Linear)
    ]


def _snapshot(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _train_steps(model, steps=5, image_size=64):
    images, labels = _batch(image_size=image_size)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.1
    )
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        nn.functional.cross_entropy(model(images), labels).backward()
        optimizer.step()


# --- Output y class map ----------------------------------------------------------


@pytest.mark.parametrize("image_size", [32, 64, 128])
@pytest.mark.parametrize("hidden_layers", [[], [32], [128, 64]])
def test_output_has_exactly_two_classes(image_size, hidden_layers):
    model = build_model(_config(image_size=image_size, hidden_layers=hidden_layers))
    images, _ = _batch(image_size=image_size)

    assert model(images).shape == (4, 2)


def test_class_map_is_dog_0_and_cat_1():
    model = build_model(_config())

    assert CLASS_MAP == {"dog": 0, "cat": 1}
    assert model.class_map == {"dog": 0, "cat": 1}
    assert _linear_dims(model.head)[-1][1] == len(CLASS_MAP)


def test_predict_proba_returns_a_distribution_per_crop():
    model = build_model(_config())
    images, _ = _batch()

    probabilities = predict_proba(model, images)

    assert probabilities.shape == (4, 2)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(4))


def test_forward_rejects_images_of_another_size():
    model = build_model(_config(image_size=64))
    images, _ = _batch(image_size=96)

    with pytest.raises(ValueError, match="image_size"):
        model(images)


# --- Configuración ---------------------------------------------------------------


def test_hidden_layers_define_the_head():
    model = build_model(_config(hidden_layers=[256, 64]))

    assert _linear_dims(model.head) == [(512, 256), (256, 64), (64, 2)]


def test_head_without_hidden_layers_is_a_single_linear():
    model = build_model(_config(hidden_layers=[]))

    assert _linear_dims(model.head) == [(512, 2)]


def test_dropout_is_configurable():
    model = build_model(_config(hidden_layers=[64, 32], dropout=0.4))

    dropouts = [layer.p for layer in model.head if isinstance(layer, nn.Dropout)]
    assert dropouts == [0.4, 0.4]


def test_config_comes_from_training_params():
    params = TrainingParams(
        optimizer="adamw",
        batch_size=16,
        max_epochs=10,
        learning_rate=0.0003,
        image_size=160,
        hidden_layers=[128],
        dropout=0.3,
        seed=1,
        patience=3,
        min_delta=0.001,
    )

    config = ModelConfig.from_params(params)

    assert (config.image_size, config.hidden_layers, config.dropout) == (160, [128], 0.3)
    assert config.pretrained is True
    assert config.trainable == "layer4"


@pytest.mark.parametrize(
    "overrides",
    [
        {"image_size": 48},
        {"image_size": 0},
        {"image_size": 2048},
        {"dropout": 1.0},
        {"dropout": -0.1},
        {"hidden_layers": [0]},
        {"hidden_layers": [5000]},
        {"hidden_layers": [8, 8, 8, 8, 8, 8]},
        {"trainable": "backbone"},
    ],
)
def test_invalid_configuration_is_rejected(overrides):
    with pytest.raises(ValueError):
        _config(**overrides)


# --- Pesos preentrenados y capas entrenables ---------------------------------------


def test_pretrained_uses_torchvision_imagenet1k_v1(monkeypatch):
    seen = {}
    real_resnet18 = model_module.resnet18

    def spy(*, weights):
        seen["weights"] = weights
        return real_resnet18(weights=None)

    monkeypatch.setattr(model_module, "resnet18", spy)

    build_model(_config(pretrained=True))

    assert str(seen["weights"]) == "ResNet18_Weights.IMAGENET1K_V1"
    assert WEIGHTS_ORIGIN["weights"] == "ResNet18_Weights.IMAGENET1K_V1"
    assert WEIGHTS_ORIGIN["url"].startswith("https://download.pytorch.org/models/resnet18-")


@pytest.mark.parametrize(
    ("trainable", "trainable_blocks"),
    [
        ("head", {"head"}),
        ("layer4", {"layer4", "head"}),
        ("all", {"conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "head"}),
    ],
)
def test_trainable_policy_freezes_the_rest(trainable, trainable_blocks):
    model = build_model(_config(trainable=trainable))
    summary = model.trainable_summary()

    assert set(summary["trainable_blocks"]) == trainable_blocks
    for name, parameter in model.named_parameters():
        block = "head" if name.startswith("head.") else name.split(".")[1]
        assert parameter.requires_grad is (block in trainable_blocks), name
    assert summary["trainable_params"] + summary["frozen_params"] == sum(
        parameter.numel() for parameter in model.parameters()
    )


# --- Agent Test: una corrida corta cambia los pesos ----------------------------------


def test_optimizer_steps_change_trainable_weights_and_keep_frozen_ones():
    model = build_model(_config(trainable="layer4"))
    before = _snapshot(model)

    _train_steps(model)

    after = model.state_dict()
    changed = {name for name in before if not torch.equal(before[name], after[name])}
    assert any(name.startswith("head.") for name in changed)
    assert any(name.startswith("backbone.layer4.") for name in changed)
    # Ni los pesos ni las estadísticas de BatchNorm de los bloques congelados cambian.
    assert not [name for name in changed if name.startswith(("backbone.layer1.", "backbone.bn1."))]


@pytest.mark.parametrize("trainable", ["layer4", "all"])
def test_frozen_batchnorm_statistics_allow_single_sample_batches(trainable):
    model = build_model(_config(image_size=32, trainable=trainable))
    model.freeze_batchnorm_statistics()
    before = _snapshot(model)
    images, _ = _batch(size=1, image_size=32)
    labels = torch.tensor([1])
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)

    model.train()
    optimizer.zero_grad()
    nn.functional.cross_entropy(model(images), labels).backward()  # a 32 px, layer4 es 1x1
    optimizer.step()

    batchnorms = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    assert batchnorms and not any(m.training for m in batchnorms)
    after = model.state_dict()
    running = [k for k in before if k.endswith(("running_mean", "running_var"))]
    assert all(torch.equal(before[k], after[k]) for k in running)
    # Los pesos entrenables (incluidos gamma/beta de BatchNorm en layer4) sí cambian.
    assert not torch.equal(
        before["backbone.layer4.1.bn2.weight"], after["backbone.layer4.1.bn2.weight"]
    )
    assert model.batchnorm_statistics == "frozen"


def test_batchnorm_uses_batch_statistics_by_default():
    model = build_model(_config()).train()

    assert model.batchnorm_statistics == "batch"
    assert model.backbone.layer4[1].bn2.training


def test_inference_before_and_after_training_differs():
    model = build_model(_config())
    images, _ = _batch(seed=1)
    before = predict_proba(model, images)

    _train_steps(model)

    assert not torch.allclose(before, predict_proba(model, images))


# --- Checkpoint ------------------------------------------------------------------------


def test_checkpoint_round_trip_restores_identical_predictions(tmp_path):
    model = build_model(_config(hidden_layers=[64], dropout=0.3))
    _train_steps(model)
    images, _ = _batch(seed=2)
    expected = predict_proba(model, images)

    path = save_checkpoint(model, tmp_path / "model.pt")
    restored = load_checkpoint(path)

    assert restored.config == model.config
    assert restored.class_map == {"dog": 0, "cat": 1}
    assert not restored.training
    assert torch.equal(predict_proba(restored, images), expected)


def _offline_resnet18(monkeypatch):
    """`pretrained=True` sin descargar: misma arquitectura con pesos aleatorios."""
    real_resnet18 = model_module.resnet18
    monkeypatch.setattr(model_module, "resnet18", lambda *, weights: real_resnet18(weights=None))


def test_checkpoint_records_class_map_preprocessing_and_weights_origin(tmp_path, monkeypatch):
    _offline_resnet18(monkeypatch)
    model = build_model(_config(image_size=96, pretrained=True))

    payload = torch.load(save_checkpoint(model, tmp_path / "model.pt"), weights_only=True)

    assert payload["class_map"] == {"dog": 0, "cat": 1}
    assert payload["config"]["image_size"] == 96
    assert payload["preprocessing"] == {
        "image_size": 96,
        "resize": "square",
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
    }
    assert payload["weights_origin"] == WEIGHTS_ORIGIN
    assert payload["architecture"] == "resnet18"


def test_model_trained_from_scratch_declares_no_weights_origin(tmp_path):
    model = build_model(_config(pretrained=False))

    payload = torch.load(save_checkpoint(model, tmp_path / "model.pt"), weights_only=True)

    assert payload["weights_origin"] is None


def test_load_checkpoint_never_downloads_weights(tmp_path, monkeypatch):
    path = save_checkpoint(build_model(_config(pretrained=False)), tmp_path / "model.pt")
    payload = torch.load(path, weights_only=True)
    payload["config"]["pretrained"] = True
    torch.save(payload, path)
    requested = []
    real_resnet18 = model_module.resnet18

    def spy(*, weights):
        requested.append(weights)
        return real_resnet18(weights=None)

    monkeypatch.setattr(model_module, "resnet18", spy)

    restored = load_checkpoint(path)

    # Los pesos salen del checkpoint: ni ImageNet ni caché de torch hub.
    assert requested == [None]
    assert restored.config.pretrained is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("class_map", {"cat": 0, "dog": 1}, "class_map"),
        ("architecture", "resnet50", "architecture"),
    ],
)
def test_load_checkpoint_rejects_incompatible_payload(tmp_path, field, value, message):
    path = save_checkpoint(build_model(_config()), tmp_path / "model.pt")
    payload = torch.load(path, weights_only=True)
    payload[field] = value
    torch.save(payload, path)

    with pytest.raises(ValueError, match=message):
        load_checkpoint(path)
