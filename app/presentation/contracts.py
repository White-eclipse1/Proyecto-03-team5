"""Frozen JSON v1.0 contracts; validation only, without pipeline execution or I/O."""

from math import isclose
from re import fullmatch
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    ValidationInfo,
    field_validator,
    model_validator,
)

from analyzers.base import AnalyzerResult

Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")]
Count = Annotated[int, Field(ge=0)]
Ratio = Annotated[float, Field(ge=0, le=1)]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class QualityCheck(AnalyzerResult):
    """Analyzer output plus the policy action supplied by the upstream gate."""

    model_config = ContractModel.model_config

    check_name: Identifier
    metric_value: float
    details: dict[str, JsonValue] = Field(default_factory=dict)
    action: Literal["warn", "fail"]


class QualityReport(ContractModel):
    schema_version: Literal["1.0"]
    dataset_version: Identifier
    status: Literal["passed", "warning", "failed"]
    checks: list[QualityCheck] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_check_names(self) -> Self:
        names = [check.check_name for check in self.checks]
        if len(names) != len(set(names)):
            raise ValueError("check_name must be unique within a quality report")
        return self


class SplitSummary(ContractModel):
    image_count: Count
    ratio: Ratio


class DatasetSplits(ContractModel):
    train: SplitSummary
    validation: SplitSummary
    test: SplitSummary


class SplitsReport(ContractModel):
    schema_version: Literal["1.0"]
    dataset_version: Identifier
    total_images: int = Field(gt=0)
    splits: DatasetSplits
    class_distribution: dict[str, dict[str, Count]] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    leakage: dict[str, JsonValue] = Field(default_factory=dict, exclude_if=lambda value: not value)

    @model_validator(mode="after")
    def consistent_split_summary(self) -> Self:
        splits = (self.splits.train, self.splits.validation, self.splits.test)
        if sum(split.image_count for split in splits) != self.total_images:
            raise ValueError("split counts must sum to total_images")
        for split in splits:
            if not isclose(
                split.ratio, split.image_count / self.total_images, rel_tol=0, abs_tol=1e-6
            ):
                raise ValueError("each ratio must match image_count / total_images within 1e-6")
        return self


class DatasetRelease(ContractModel):
    dataset_version: Identifier
    quality_file: str
    splits_file: str

    @field_validator("quality_file", "splits_file")
    @classmethod
    def relative_report_reference(cls, value: str, info: ValidationInfo) -> str:
        expected = "quality.json" if info.field_name == "quality_file" else "splits.json"
        parts = value.split("/")
        if (
            parts[-1] != expected
            or any(part in ("", ".", "..") for part in parts)
            or any(fullmatch(r"[A-Za-z0-9._-]+", part) is None for part in parts)
        ):
            raise ValueError(f"reference must be a relative POSIX path ending in {expected}")
        return value


class VersionsReport(ContractModel):
    schema_version: Literal["1.0"]
    releases: list[DatasetRelease]

    @model_validator(mode="after")
    def unique_dataset_versions(self) -> Self:
        versions = [release.dataset_version for release in self.releases]
        if len(versions) != len(set(versions)):
            raise ValueError("dataset_version must be unique within the release catalog")
        return self
