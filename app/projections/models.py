"""Validated, versioned parameters for the RGB projection stage."""

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, StrictBool, StringConstraints

MAX_SAFE_INTEGER = 9007199254740991


def mathematical_integer(value):
    """Accept JSON integral numbers, but never strings or booleans."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


SafeInteger = Annotated[
    int, Field(ge=0, le=MAX_SAFE_INTEGER), BeforeValidator(mathematical_integer)
]


class ProjectionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class FeatureConfig(ProjectionModel):
    representation: Literal["rgb_pixels"]
    width: Literal[16]
    height: Literal[16]
    channels: Literal[3]
    resize_filter: Literal["LANCZOS"]
    exif_orientation: Literal["transpose"]
    normalization: Literal["divide_by_255"]


class PCAConfig(ProjectionModel):
    n_components: Literal[2]
    svd_solver: Literal["full"]
    whiten: StrictBool


class TSNEConfig(ProjectionModel):
    n_components: Literal[2]
    random_state: SafeInteger = Field(ge=0, le=2**32 - 1)
    init: Literal["pca"]
    learning_rate: Literal["auto"]
    max_iter: SafeInteger = Field(ge=250)
    perplexity: float = Field(gt=0)
    method: Literal["barnes_hut"]


class ProjectionsConfig(ProjectionModel):
    dataset_version: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")]
    features: FeatureConfig
    pca: PCAConfig
    tsne: TSNEConfig


def load_projections_config(path: Path | None = None) -> ProjectionsConfig:
    path = path if path is not None else Path(__file__).with_name("projections.yaml")
    return ProjectionsConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def relative_image_name(value: str) -> str:
    """Relative POSIX data filename, never an absolute path or parent traversal."""
    if (
        not value
        or any(char in value for char in ("\\", ":", "\x00"))
        or any(part in ("", ".", "..") for part in value.split("/"))
    ):
        raise ValueError("file_name must be a relative POSIX image path without traversal")
    return value
