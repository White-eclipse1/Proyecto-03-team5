from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, StringConstraints, model_validator
from typing_extensions import Annotated

Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")]
ManifestHash = Annotated[
    str,
    StringConstraints(pattern=r"^(md5:[0-9a-f]{32}|sha256:[0-9a-f]{64})$"),
]
SplitName = Literal["train", "validation", "test"]
ClassName = Literal["dog", "cat"]


class ManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class ManifestRecord(ManifestModel):
    crop_id: Identifier
    source_image_id: int = Field(ge=0)
    duplicate_group: Identifier
    class_name: ClassName = Field(
        validation_alias=AliasChoices("class_name", "class"),
        serialization_alias="class",
    )
    split: SplitName


class ManifestSplit(ManifestModel):
    image_count: int = Field(ge=0)
    ratio: float = Field(ge=0, le=1)


class ManifestSplits(ManifestModel):
    train: ManifestSplit
    validation: ManifestSplit
    test: ManifestSplit


class P3Manifest(ManifestModel):
    schema_version: Literal["1.0"]
    manifest_version: Identifier
    dataset_version: Identifier
    source_release: Identifier | None = None
    seed: int = Field(ge=0)

    @model_validator(mode="after")
    def default_source_release(self):
        if self.source_release is None:
            self.source_release = self.dataset_version
        return self

    manifest_hash: ManifestHash
    total_images: int = Field(gt=0)
    splits: ManifestSplits
    records: list[ManifestRecord]
    counts: dict
