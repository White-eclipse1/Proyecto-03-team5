"""ML-01 — contrato JSON v1.0 del reporte de crops (`reports/crops.json`).

`crops` es el manifiesto de muestras aceptadas; `rejections` registra cada
anotación excluida con sus motivos. Una misma anotación no puede aparecer
en ambas listas.
"""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from crops.classes import CLASS_NAMES

ClassName = Literal["dog", "cat"]
Count = Annotated[int, Field(ge=0)]
Id = Annotated[int, Field(ge=0)]
Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")]

RejectionReason = Literal[
    "malformed_bbox",
    "non_positive_width",
    "non_positive_height",
    "negative_x",
    "negative_y",
    "exceeds_image_width",
    "exceeds_image_height",
    "out_of_bounds",
    "unknown_image_id",
    "unknown_category_id",
    "unsupported_category",
    "missing_image",
    "unreadable_image",
    "image_size_mismatch",
]


class CropModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class CropRecord(CropModel):
    crop_id: Identifier
    image_id: Id
    annotation_id: Id
    category_id: Id
    class_name: ClassName
    bbox: Annotated[list[float], Field(min_length=4, max_length=4)]
    crop_box: Annotated[list[int], Field(min_length=4, max_length=4)]
    source_file_name: Annotated[str, Field(min_length=1)]
    crop_path: Annotated[str, Field(min_length=1)]
    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]
    sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class CropRejection(CropModel):
    # `annotation_id`/`image_id`/`category_id` vienen crudos del COCO, así que
    # se aceptan tal cual para poder registrar incluso la anotación malformada.
    annotation_id: int
    image_id: int | None
    category_id: int | None
    class_name: ClassName | None
    bbox: list
    reasons: Annotated[list[RejectionReason], Field(min_length=1)]


class CropProvenance(CropModel):
    release_version: Identifier
    images_dvc_hash: Annotated[str, Field(min_length=1)]
    annotations_dvc_hash: Annotated[str, Field(min_length=1)]
    quality_report: Annotated[str, Field(min_length=1)]


class CropSummary(CropModel):
    total_annotations: Count
    accepted: Count
    rejected: Count
    accepted_per_class: dict[ClassName, Count]
    accepted_images_per_class: dict[ClassName, Count]
    rejected_per_reason: dict[RejectionReason, Count]


class CropReport(CropModel):
    schema_version: Literal["1.0"]
    dataset_version: Identifier
    provenance: CropProvenance
    classes: list[ClassName]
    summary: CropSummary
    crops: list[CropRecord]
    rejections: list[CropRejection]

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.provenance.release_version != self.dataset_version:
            raise ValueError("provenance.release_version debe coincidir con dataset_version")
        if tuple(self.classes) != CLASS_NAMES:
            raise ValueError(f"classes debe ser {list(CLASS_NAMES)}")
        crop_ids = [crop.crop_id for crop in self.crops]
        if len(crop_ids) != len(set(crop_ids)):
            raise ValueError("crop_id repetido en el manifiesto")
        accepted = {crop.annotation_id for crop in self.crops}
        rejected = {rejection.annotation_id for rejection in self.rejections}
        if len(accepted) != len(self.crops) or accepted & rejected:
            raise ValueError("Cada annotation_id debe aparecer una sola vez entre crops/rejections")
        if (
            self.summary.accepted != len(self.crops)
            or self.summary.rejected != len(self.rejections)
            or self.summary.total_annotations != len(self.crops) + len(self.rejections)
        ):
            raise ValueError("summary no coincide con crops/rejections")

        expected_images_per_class = {
            class_name: len({crop.image_id for crop in self.crops if crop.class_name == class_name})
            for class_name in CLASS_NAMES
        }
        if self.summary.accepted_images_per_class != expected_images_per_class:
            raise ValueError("accepted_images_per_class no coincide con los image_id aceptados")
        return self
