"""APP-01: JSON v1.0 contracts for the model screens; validation only, without I/O.

Mirrored by `frontend/src/ml/schemas.ts`. Both sides validate the shared corpus in
`presentation/examples/ml/` (valid examples plus `invalid_cases.json`), so an
incompatible change on one side breaks the other side's tests.

IDs are the real identifiers of each system, never display labels:
- dataset_version: dataset release (same format as the P2 reports).
- manifest_hash: hash of the exact file manifest used (`md5:` DVC or `sha256:`).
- experiment_id / run_id: MLflow tracking IDs (numeric / 32 lowercase hex).
- checkpoint: MLflow artifact URI `runs:/<run_id>/<path>` of the same run.
- model_version: MLflow Model Registry version (positive integer as string).
"""

from collections import Counter
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")]
ManifestHash = Annotated[
    str, StringConstraints(pattern=r"^(md5:[0-9a-f]{32}|sha256:[0-9a-f]{64})$")
]
RunId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
ExperimentId = Annotated[str, StringConstraints(pattern=r"^[0-9]+$")]
Checkpoint = Annotated[str, StringConstraints(pattern=r"^runs:/[0-9a-f]{32}/[^\s]+$")]
ModelVersion = Annotated[str, StringConstraints(pattern=r"^[1-9][0-9]*$")]
ErrorCode = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*$")]
Timestamp = Annotated[
    str, StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z$")
]
Label = Annotated[str, StringConstraints(min_length=1)]
Count = Annotated[int, Field(ge=0)]
Ratio = Annotated[float, Field(ge=0, le=1)]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


def checkpoint_run_id(checkpoint: str) -> str:
    return checkpoint.removeprefix("runs:/").split("/", 1)[0]


def require_checkpoint_of_run(checkpoint: str | None, run_id: str | None) -> None:
    if checkpoint is not None and checkpoint_run_id(checkpoint) != run_id:
        raise ValueError("checkpoint must belong to the same run_id")


def require_unique(values: list, message: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(message)


class ContractError(ContractModel):
    """Error state shared by failed jobs and HTTP error bodies."""

    code: ErrorCode
    message: Label
    retryable: bool


class ErrorResponse(ContractModel):
    schema_version: Literal["1.0"]
    error: ContractError


TrainingStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


class TrainingJob(ContractModel):
    job_id: Identifier
    status: TrainingStatus
    dataset_version: Identifier
    manifest_hash: ManifestHash
    experiment_id: ExperimentId
    run_id: RunId | None
    checkpoint: Checkpoint | None
    params: dict[str, str | int | float | bool]
    created_at: Timestamp
    started_at: Timestamp | None
    finished_at: Timestamp | None
    error: ContractError | None

    @model_validator(mode="after")
    def status_matches_lifecycle(self) -> Self:
        if self.status == "queued" and (self.run_id is not None or self.started_at is not None):
            raise ValueError("a queued job has no run_id or started_at yet")
        if self.status in ("running", "succeeded") and (
            self.run_id is None or self.started_at is None
        ):
            raise ValueError("running and succeeded jobs require run_id and started_at")
        if (self.finished_at is not None) != (self.status in ("succeeded", "failed", "cancelled")):
            raise ValueError("finished_at is set exactly when the job is terminal")
        if (self.checkpoint is not None) != (self.status == "succeeded"):
            raise ValueError("checkpoint is set exactly when the job succeeded")
        if (self.error is not None) != (self.status == "failed"):
            raise ValueError("error is set exactly when the job failed")
        require_checkpoint_of_run(self.checkpoint, self.run_id)
        return self


class TrainingJobsResponse(ContractModel):
    schema_version: Literal["1.0"]
    jobs: list[TrainingJob]

    @model_validator(mode="after")
    def unique_jobs(self) -> Self:
        require_unique([job.job_id for job in self.jobs], "job_id must be unique")
        return self


RunStatus = Literal["SCHEDULED", "RUNNING", "FINISHED", "FAILED", "KILLED"]


class ExperimentRun(ContractModel):
    """MLflow run, with MLflow's own status names; params are strings, as in MLflow."""

    run_id: RunId
    experiment_id: ExperimentId
    run_name: Label
    status: RunStatus
    start_time: Timestamp
    end_time: Timestamp | None
    dataset_version: Identifier
    manifest_hash: ManifestHash
    params: dict[str, str]
    metrics: dict[str, float]

    @model_validator(mode="after")
    def end_time_matches_status(self) -> Self:
        if (self.end_time is None) != (self.status in ("SCHEDULED", "RUNNING")):
            raise ValueError("end_time is null exactly while the run is active")
        return self


class RunsResponse(ContractModel):
    schema_version: Literal["1.0"]
    runs: list[ExperimentRun]

    @model_validator(mode="after")
    def unique_runs(self) -> Self:
        require_unique([run.run_id for run in self.runs], "run_id must be unique")
        return self


class EvaluationMetrics(ContractModel):
    map50: Ratio
    map50_95: Ratio
    precision: Ratio
    recall: Ratio


class ClassEvaluation(ContractModel):
    category_name: Label
    ap50: Ratio
    support: Count


class Evaluation(ContractModel):
    evaluation_id: Identifier
    run_id: RunId
    checkpoint: Checkpoint
    model_name: Identifier | None
    model_version: ModelVersion | None
    dataset_version: Identifier
    manifest_hash: ManifestHash
    split: Literal["validation", "test"]
    metrics: EvaluationMetrics
    per_class: list[ClassEvaluation]
    created_at: Timestamp

    @model_validator(mode="after")
    def consistent_references(self) -> Self:
        if (self.model_name is None) != (self.model_version is None):
            raise ValueError("model_name and model_version are both set or both null")
        require_checkpoint_of_run(self.checkpoint, self.run_id)
        require_unique(
            [entry.category_name for entry in self.per_class], "category_name must be unique"
        )
        return self


class EvaluationsResponse(ContractModel):
    schema_version: Literal["1.0"]
    evaluations: list[Evaluation]

    @model_validator(mode="after")
    def unique_evaluations(self) -> Self:
        require_unique(
            [evaluation.evaluation_id for evaluation in self.evaluations],
            "evaluation_id must be unique",
        )
        return self


class RegisteredModelVersion(ContractModel):
    """MLflow Model Registry version, with MLflow's own status names."""

    model_name: Identifier
    model_version: ModelVersion
    status: Literal["PENDING_REGISTRATION", "READY", "FAILED_REGISTRATION"]
    aliases: list[Identifier]
    run_id: RunId
    checkpoint: Checkpoint
    dataset_version: Identifier
    manifest_hash: ManifestHash
    created_at: Timestamp

    @model_validator(mode="after")
    def checkpoint_of_run(self) -> Self:
        require_checkpoint_of_run(self.checkpoint, self.run_id)
        require_unique(self.aliases, "aliases must be unique")
        return self


class ModelsResponse(ContractModel):
    schema_version: Literal["1.0"]
    models: list[RegisteredModelVersion]

    @model_validator(mode="after")
    def unique_versions_and_aliases(self) -> Self:
        require_unique(
            [(model.model_name, model.model_version) for model in self.models],
            "(model_name, model_version) must be unique",
        )
        aliases = Counter(
            (model.model_name, alias) for model in self.models for alias in model.aliases
        )
        if any(count > 1 for count in aliases.values()):
            raise ValueError("an alias points to a single version per model")
        return self


class ImageSize(ContractModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class Prediction(ContractModel):
    category_name: Label
    score: Ratio
    bbox: Annotated[list[Annotated[float, Field(ge=0)]], Field(min_length=4, max_length=4)]
    """COCO format: [x, y, width, height] in pixels."""

    @model_validator(mode="after")
    def positive_box_size(self) -> Self:
        if self.bbox[2] <= 0 or self.bbox[3] <= 0:
            raise ValueError("bbox width and height must be positive")
        return self


class InferenceResponse(ContractModel):
    schema_version: Literal["1.0"]
    request_id: Identifier
    model_name: Identifier
    model_version: ModelVersion
    run_id: RunId
    dataset_version: Identifier
    image: ImageSize
    predictions: list[Prediction]
    latency_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def boxes_inside_image(self) -> Self:
        for prediction in self.predictions:
            x, y, width, height = prediction.bbox
            if x + width > self.image.width or y + height > self.image.height:
                raise ValueError("bbox must fit inside the image")
        return self


CONTRACTS: dict[str, type[ContractModel]] = {
    "training_jobs": TrainingJobsResponse,
    "runs": RunsResponse,
    "evaluations": EvaluationsResponse,
    "models": ModelsResponse,
    "inference": InferenceResponse,
    "error": ErrorResponse,
}
