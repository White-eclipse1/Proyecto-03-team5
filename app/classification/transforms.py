"""ML-02 — preprocesamiento del clasificador: aumentación solo en train.

`eval_transform` es el único preprocesamiento de validation, test e inference:
redimensiona a `image_size x image_size`, pasa a tensor float32 en [0, 1] y
normaliza con la media/desviación de ImageNet (compatible con pesos
preentrenados de torchvision). No tiene pasos aleatorios.

`train_transform` agrega aumentación aleatoria (recorte-escala, espejo
horizontal y color) y termina con el mismo tamaño, tipo y normalización, así el
modelo ve en train el mismo formato de tensor que en evaluación.
"""

import torch
from PIL import Image
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EVAL_SPLITS = frozenset({"validation", "test", "inference"})


def _check_image_size(image_size: int) -> None:
    # Mismo criterio que `TrainingParams.image_size` (presentation/ml_contracts.py).
    if not 32 <= image_size <= 1024 or image_size % 32:
        raise ValueError(f"image_size debe ser múltiplo de 32 entre 32 y 1024: {image_size}")


def _to_normalized_tensor() -> list:
    return [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]


def eval_transform(image_size: int) -> v2.Compose:
    _check_image_size(image_size)
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((image_size, image_size), antialias=True),
            *_to_normalized_tensor(),
        ]
    )


def train_transform(image_size: int) -> v2.Compose:
    _check_image_size(image_size)
    return v2.Compose(
        [
            v2.ToImage(),
            v2.RandomResizedCrop((image_size, image_size), scale=(0.8, 1.0), antialias=True),
            v2.RandomHorizontalFlip(),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            *_to_normalized_tensor(),
        ]
    )


def random_transform_names(transform: v2.Compose) -> list[str]:
    """Pasos aleatorios de un transform (`Random*`, `ColorJitter`): vacío fuera de train."""
    names = [type(step).__name__ for step in transform.transforms]
    return [name for name in names if name.startswith("Random") or name == "ColorJitter"]


def transform_for(split: str, image_size: int) -> v2.Compose:
    if split == "train":
        return train_transform(image_size)
    if split in EVAL_SPLITS:
        return eval_transform(image_size)
    raise ValueError(f"split desconocido: {split!r}")


def preprocess_image(image: Image.Image, image_size: int) -> torch.Tensor:
    """Preprocesamiento de inferencia: el mismo de validation/test, sobre RGB."""
    return eval_transform(image_size)(image.convert("RGB"))
