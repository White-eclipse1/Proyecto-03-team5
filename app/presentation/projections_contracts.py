"""Independent v1.0 contract for real dataset projections."""

from typing import Annotated, Literal, Self

from projections.models import (
    FeatureConfig,
    PCAConfig,
    SafeInteger,
    TSNEConfig,
    relative_image_name,
)
from pydantic import Field, StringConstraints, field_validator, model_validator

from presentation.contracts import ContractModel, Identifier


class ProjectionCategory(ContractModel):
    id: SafeInteger
    name: str = Field(min_length=1)


class ProjectionPoint(ContractModel):
    image_id: SafeInteger
    file_name: str
    category_ids: list[SafeInteger]
    x: float
    y: float

    _relative_name = field_validator("file_name")(relative_image_name)

    @field_validator("category_ids")
    @classmethod
    def unique_labels(cls, value: list[int]) -> list[int]:
        if len(value) != len(set(value)):
            raise ValueError("category_ids must be unique per point")
        return value


class PCAProjection(ContractModel):
    parameters: PCAConfig
    explained_variance_ratio: list[Annotated[float, Field(ge=0, le=1)]] = Field(
        min_length=2, max_length=2
    )
    points: list[ProjectionPoint]

    @field_validator("explained_variance_ratio")
    @classmethod
    def variance_sum(cls, value: list[float]) -> list[float]:
        if sum(value) > 1 + 1e-6:
            raise ValueError("explained variance cannot exceed 1")
        return value


class TSNEProjection(ContractModel):
    parameters: TSNEConfig
    points: list[ProjectionPoint]


class ProjectionsReport(ContractModel):
    schema_version: Literal["1.0"]
    dataset_version: Identifier
    dataset_fingerprint: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    total_images: SafeInteger = Field(gt=0)
    categories: list[ProjectionCategory] = Field(min_length=1)
    features: FeatureConfig
    pca: PCAProjection
    tsne: TSNEProjection

    @model_validator(mode="after")
    def consistent_points(self) -> Self:
        category_ids = {category.id for category in self.categories}
        if len(category_ids) != len(self.categories):
            raise ValueError("Category IDs must be unique")
        metadata = []
        for method in (self.pca, self.tsne):
            points = method.points
            if len(points) != self.total_images:
                raise ValueError("Point count must equal total_images")
            by_id = {point.image_id: point for point in points}
            if len(by_id) != len(points):
                raise ValueError("Image IDs must be unique within each projection")
            if any(not set(point.category_ids) <= category_ids for point in points):
                raise ValueError("Point references an unknown category")
            metadata.append(
                {
                    point.image_id: (point.file_name, tuple(sorted(point.category_ids)))
                    for point in points
                }
            )
        if metadata[0] != metadata[1]:
            raise ValueError("PCA/t-SNE must cover identical images, filenames and labels")
        if self.tsne.parameters.perplexity >= self.total_images:
            raise ValueError("Perplexity must be smaller than total_images")
        return self
