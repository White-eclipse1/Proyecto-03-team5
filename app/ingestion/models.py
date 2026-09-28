"""Tier 1 — modelos Pydantic v2 del COCO crudo producido por el Proyecto 1.

`CocoDataset.model_validate(...)` es la única puerta de entrada del pipeline:
un dato mal formado (bbox corto, category_id inexistente, image_id huérfano)
se rechaza aquí con un error de Pydantic que nombra el campo roto, antes de
llegar a `analyzers`/`policies` como un dict crudo y reventar con un
`KeyError`/`IndexError`.
"""

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CocoModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


Id = Annotated[int, Field(ge=0)]


class Category(CocoModel):
    id: Id
    name: Annotated[str, Field(min_length=1)]


class Image(CocoModel):
    id: Id
    file_name: Annotated[str, Field(min_length=1)]
    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]


class Annotation(CocoModel):
    id: Id
    image_id: Id
    category_id: Id
    bbox: list[float]
    area: Annotated[float, Field(ge=0)]
    iscrowd: Annotated[int, Field(ge=0, le=1)]
    segmentation: list = Field(default_factory=list)

    @field_validator("bbox")
    @classmethod
    def bbox_is_x_y_width_height(cls, value: list[float]) -> list[float]:
        if len(value) != 4:
            raise ValueError(
                f"bbox debe tener exactamente 4 valores [x, y, width, height]; "
                f"se recibieron {len(value)}"
            )
        _x, _y, width, height = value
        if width <= 0 or height <= 0:
            raise ValueError("bbox debe tener width y height positivos")
        return value


class CocoDataset(CocoModel):
    images: list[Image] = Field(min_length=1)
    annotations: list[Annotation]
    categories: list[Category] = Field(min_length=1)

    @model_validator(mode="after")
    def ids_are_unique(self) -> Self:
        for label, items in (
            ("images", self.images),
            ("annotations", self.annotations),
            ("categories", self.categories),
        ):
            ids = [item.id for item in items]
            if len(ids) != len(set(ids)):
                duplicates = sorted({item_id for item_id in ids if ids.count(item_id) > 1})
                raise ValueError(f"{label} tiene ids repetidos: {duplicates}")
        return self

    @model_validator(mode="after")
    def annotations_reference_declared_ids(self) -> Self:
        image_ids = {image.id for image in self.images}
        category_ids = {category.id for category in self.categories}

        for index, annotation in enumerate(self.annotations):
            if annotation.image_id not in image_ids:
                raise ValueError(
                    f"annotations[{index}].image_id={annotation.image_id} no corresponde "
                    "a ninguna imagen declarada en 'images' (image_id huérfano)"
                )
            if annotation.category_id not in category_ids:
                raise ValueError(
                    f"annotations[{index}].category_id={annotation.category_id} no corresponde "
                    "a ninguna categoría declarada en 'categories'"
                )
        return self
