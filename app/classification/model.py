"""ML-03 — clasificador dog/cat: ResNet18 preentrenada en ImageNet + cabeza propia.

Arquitectura (ver `classification/README.md`):

- Backbone: `torchvision.models.resnet18` con los pesos
  `ResNet18_Weights.IMAGENET1K_V1` (`WEIGHTS_ORIGIN`) y su `fc` original
  reemplazada por la identidad: entrega un vector de 512 características.
- Cabeza: por cada entero de `hidden_layers`, `Linear -> ReLU -> Dropout(dropout)`,
  y al final `Linear(…, 2)`: un logit por clase de `CLASS_MAP` (dog=0, cat=1).
- `trainable` decide qué se entrena: `head` (solo la cabeza), `layer4` (último
  bloque residual + cabeza, por defecto) o `all`. Los bloques congelados no
  reciben gradiente y sus BatchNorm quedan en modo eval también durante el
  entrenamiento, para que sus estadísticas de ImageNet no cambien.
- `freeze_batchnorm_statistics()` deja **todas** las BatchNorm en modo eval al
  entrenar: usan sus estadísticas guardadas en vez de las del batch, y sus pesos
  (gamma/beta) siguen entrenándose donde sean entrenables. Es lo que permite
  `batch_size=1`, donde las estadísticas de un solo crop son ruido (o, con
  `image_size=32`, un único valor por canal que BatchNorm no puede normalizar).

El checkpoint guarda pesos, configuración, `class_map`, preprocesamiento y
origen de los pesos iniciales; `load_checkpoint` reconstruye la red sin
descargar nada y la deja en modo eval.
"""

from pathlib import Path
from typing import Annotated, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from classification.transforms import IMAGENET_MEAN, IMAGENET_STD
from crops.classes import CLASS_NAMES, CLASS_TO_INDEX
from presentation.ml_contracts import TrainingParams

ARCHITECTURE = "resnet18"
CHECKPOINT_FORMAT = 1
CLASS_MAP: dict[str, int] = dict(CLASS_TO_INDEX)
BACKBONE_FEATURES = 512
BACKBONE_BLOCKS = ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4")
TRAINABLE_BLOCKS = {
    "head": ("head",),
    "layer4": ("layer4", "head"),
    "all": (*BACKBONE_BLOCKS, "head"),
}
PRETRAINED_WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
WEIGHTS_ORIGIN = {
    "weights": str(PRETRAINED_WEIGHTS),
    "url": PRETRAINED_WEIGHTS.url,
    "dataset": "ImageNet-1K",
    "library": "torchvision",
}

Trainable = Literal["head", "layer4", "all"]


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    image_size: int
    hidden_layers: Annotated[list[Annotated[int, Field(ge=1, le=4096)]], Field(max_length=5)]
    dropout: float = Field(ge=0, lt=1)
    pretrained: bool = True
    trainable: Trainable = "layer4"

    @field_validator("image_size")
    @classmethod
    def square_multiple_of_32(cls, value: int) -> int:
        # Mismo criterio que `TrainingParams.image_size` y `classification.transforms`.
        if not 32 <= value <= 1024 or value % 32:
            raise ValueError(f"image_size debe ser múltiplo de 32 entre 32 y 1024: {value}")
        return value

    @classmethod
    def from_params(
        cls, params: TrainingParams, *, pretrained: bool = True, trainable: Trainable = "layer4"
    ) -> "ModelConfig":
        return cls(
            image_size=params.image_size,
            hidden_layers=list(params.hidden_layers),
            dropout=params.dropout,
            pretrained=pretrained,
            trainable=trainable,
        )


def _head(hidden_layers: list[int], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = BACKBONE_FEATURES
    for hidden in hidden_layers:
        layers += [nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout)]
        width = hidden
    if not hidden_layers:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(width, len(CLASS_NAMES)))
    return nn.Sequential(*layers)


class DogCatResNet18(nn.Module):
    def __init__(self, config: ModelConfig, backbone: nn.Module) -> None:
        super().__init__()
        backbone.fc = nn.Identity()
        self.config = config
        self.class_map = dict(CLASS_MAP)
        self.backbone = backbone
        self.head = _head(config.hidden_layers, config.dropout)
        trainable = TRAINABLE_BLOCKS[config.trainable]
        self._frozen_blocks = tuple(block for block in BACKBONE_BLOCKS if block not in trainable)
        for block in self._frozen_blocks:
            getattr(self.backbone, block).requires_grad_(False)
        self._freeze_all_batchnorm = False

    @property
    def batchnorm_statistics(self) -> str:
        """`frozen` si todas las BatchNorm usan sus estadísticas guardadas; si no, `batch`."""
        return "frozen" if self._freeze_all_batchnorm else "batch"

    def freeze_batchnorm_statistics(self) -> "DogCatResNet18":
        self._freeze_all_batchnorm = True
        return self.train(self.training)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        expected = (3, self.config.image_size, self.config.image_size)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected:
            raise ValueError(
                f"Se esperaba un batch N x {expected} (image_size={self.config.image_size}); "
                f"llegó {tuple(images.shape)}"
            )
        return self.head(self.backbone(images))

    def train(self, mode: bool = True) -> "DogCatResNet18":
        super().train(mode)
        for block in self._frozen_blocks:
            getattr(self.backbone, block).eval()
        if self._freeze_all_batchnorm:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def trainable_summary(self) -> dict:
        trainable = list(TRAINABLE_BLOCKS[self.config.trainable])
        counts = {True: 0, False: 0}
        for parameter in self.parameters():
            counts[parameter.requires_grad] += parameter.numel()
        return {
            "trainable_blocks": trainable,
            "frozen_blocks": list(self._frozen_blocks),
            "trainable_params": counts[True],
            "frozen_params": counts[False],
        }


def build_model(config: ModelConfig, *, download_weights: bool = True) -> DogCatResNet18:
    """Red nueva; con `pretrained`, parte de los pesos ImageNet de torchvision."""
    weights = PRETRAINED_WEIGHTS if config.pretrained and download_weights else None
    return DogCatResNet18(config, resnet18(weights=weights))


def predict_proba(model: DogCatResNet18, images: torch.Tensor) -> torch.Tensor:
    """Probabilidades softmax en modo eval, sin gradiente; restaura el modo previo."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            return torch.softmax(model(images), dim=1)
    finally:
        model.train(was_training)


def save_checkpoint(model: DogCatResNet18, path: Path, *, metadata: dict | None = None) -> Path:
    payload = {
        "format_version": CHECKPOINT_FORMAT,
        "architecture": ARCHITECTURE,
        "config": model.config.model_dump(),
        "class_map": dict(model.class_map),
        "preprocessing": {
            "image_size": model.config.image_size,
            "resize": "square",
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
        },
        "weights_origin": dict(WEIGHTS_ORIGIN) if model.config.pretrained else None,
        "metadata": metadata or {},
        "state_dict": model.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def load_checkpoint(path: Path) -> DogCatResNet18:
    """Reconstruye el modelo guardado (sin descargar pesos) y lo deja en modo eval."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("architecture") != ARCHITECTURE:
        raise ValueError(f"architecture no soportada: {payload.get('architecture')!r}")
    if payload.get("class_map") != CLASS_MAP:
        raise ValueError(f"class_map {payload.get('class_map')} distinto de {CLASS_MAP}")
    config = ModelConfig.model_validate(payload["config"])
    model = build_model(config, download_weights=False)
    model.load_state_dict(payload["state_dict"])
    return model.eval()
