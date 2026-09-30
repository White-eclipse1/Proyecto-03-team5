"""ML-02 — lectura del manifiesto P3 70/20/10 que genera OPS-02.

Cada registro es un crop de ML-01 asignado a una partición: `crop_id`,
`source_image_id`, `duplicate_group`, `class` y `split`. Aquí solo se valida la
forma de cada registro; que el crop exista y coincida con `reports/crops.json`
lo comprueba `classification.dataset`. Se ignoran campos adicionales (hash,
versión, conteos) para no acoplar el Dataset a la cabecera exacta de OPS-02.
"""

from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from crops.models import ClassName, Identifier

Split = Literal["train", "validation", "test"]


class ManifestRecord(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True, frozen=True)

    crop_id: Annotated[str, StringConstraints(min_length=1)]
    source_image_id: Annotated[int, Field(ge=0)]
    duplicate_group: str | int
    class_name: ClassName = Field(alias="class")
    split: Split


class CropManifest(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    dataset_version: Identifier
    records: Annotated[list[ManifestRecord], Field(min_length=1)]

    @model_validator(mode="after")
    def unique_crop_ids(self) -> Self:
        crop_ids = [record.crop_id for record in self.records]
        if len(crop_ids) != len(set(crop_ids)):
            raise ValueError("crop_id repetido en el manifiesto")
        return self


def load_manifest(path: Path) -> CropManifest:
    return CropManifest.model_validate_json(path.read_text(encoding="utf-8"))
