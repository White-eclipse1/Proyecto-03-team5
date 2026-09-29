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
- image_id / annotation_id: COCO IDs of a crop. The model classifies crops, one
  per COCO annotation (ML-01), so a crop is identified by its annotation.

The model is a multiclass crop classifier: evaluation reports accuracy_top1,
f1_macro, per-class metrics, the confusion matrix and the per-crop predictions,
and every metric must agree with the matrix; inference returns one class with its
probability distribution, never bounding boxes.
"""

from collections import Counter
from math import isclose
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
CocoId = Annotated[int, Field(ge=0)]

METRIC_TOLERANCE = 1e-3
"""Reported metrics and probabilities may be rounded to 3 decimals."""


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


def require_close(reported: float, expected: float, name: str) -> None:
    if not isclose(reported, expected, abs_tol=METRIC_TOLERANCE):
        raise ValueError(f"{name} must equal {expected:.4f} (recomputed)")


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def f1_score(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def require_distribution(
    probabilities: dict[str, float], predicted_class: str, class_names: list[str] | None
) -> None:
    """Softmax output: one probability per class, summing ~1, argmax = predicted_class."""
    if class_names is not None and set(probabilities) != set(class_names):
        raise ValueError("probabilities must have exactly one entry per class")
    if len(probabilities) < 2:
        raise ValueError("probabilities needs at least two classes")
    require_close(sum(probabilities.values()), 1.0, "sum of probabilities")
    if probabilities.get(predicted_class) != max(probabilities.values()):
        raise ValueError("predicted_class must be the class with the highest probability")


class ContractError(ContractModel):
    """Error state shared by failed jobs and HTTP error bodies."""

    code: ErrorCode
    message: Label
    retryable: bool


class ErrorResponse(ContractModel):
    schema_version: Literal["1.0"]
    error: ContractError


class TrainingParams(ContractModel):
    """The 7 required hyperparameters of a training job; all required, no defaults."""

    optimizer: Literal["adam", "adamw", "sgd"]
    batch_size: int = Field(ge=1, le=256)
    max_epochs: int = Field(ge=1, le=500)
    learning_rate: float = Field(gt=0, le=1)
    image_size: int = Field(ge=32, le=1024, multiple_of=32)
    hidden_layers: list[Annotated[int, Field(ge=1, le=4096)]] = Field(max_length=5)
    dropout: float = Field(ge=0, lt=1)


TrainingStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


class TrainingJob(ContractModel):
    job_id: Identifier
    status: TrainingStatus
    dataset_version: Identifier
    manifest_hash: ManifestHash
    experiment_id: ExperimentId
    run_id: RunId | None
    checkpoint: Checkpoint | None
    params: TrainingParams
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
    accuracy_top1: Ratio
    f1_macro: Ratio


class ClassMetrics(ContractModel):
    class_name: Label
    precision: Ratio
    recall: Ratio
    f1: Ratio
    support: Count


class CropPrediction(ContractModel):
    """Prediction for one crop of the split, so the metrics can be recomputed."""

    image_id: CocoId
    annotation_id: CocoId
    true_class: Label
    predicted_class: Label
    probabilities: dict[Label, Ratio]


class Evaluation(ContractModel):
    """Classification metrics of a checkpoint on one split.

    `class_names` fixes the order of `per_class` and of both axes of
    `confusion_matrix` (rows: true class, columns: predicted class). The matrix is
    the source of truth: supports, per-class metrics, accuracy and f1_macro are
    recomputed from it, and `predictions` must add up to it crop by crop.
    """

    evaluation_id: Identifier
    run_id: RunId
    checkpoint: Checkpoint
    model_name: Identifier | None
    model_version: ModelVersion | None
    dataset_version: Identifier
    manifest_hash: ManifestHash
    split: Literal["validation", "test"]
    class_names: list[Label] = Field(min_length=2)
    metrics: EvaluationMetrics
    per_class: list[ClassMetrics]
    confusion_matrix: list[list[Count]]
    predictions: list[CropPrediction]
    created_at: Timestamp

    @model_validator(mode="after")
    def consistent_references(self) -> Self:
        if (self.model_name is None) != (self.model_version is None):
            raise ValueError("model_name and model_version are both set or both null")
        require_checkpoint_of_run(self.checkpoint, self.run_id)
        require_unique(self.class_names, "class_names must be unique")
        if [entry.class_name for entry in self.per_class] != self.class_names:
            raise ValueError("per_class must follow class_names, one entry per class")
        return self

    @model_validator(mode="after")
    def metrics_match_confusion_matrix(self) -> Self:
        matrix = self.confusion_matrix
        size = len(self.class_names)
        if len(matrix) != size or any(len(row) != size for row in matrix):
            raise ValueError("confusion_matrix must be square with one row per class")
        total = sum(map(sum, matrix))
        if total == 0:
            raise ValueError("confusion_matrix cannot be empty")
        f1_scores = []
        for index, entry in enumerate(self.per_class):
            hits = matrix[index][index]
            actual = sum(matrix[index])
            predicted = sum(row[index] for row in matrix)
            precision, recall = ratio(hits, predicted), ratio(hits, actual)
            f1_scores.append(f1_score(precision, recall))
            if entry.support != actual:
                raise ValueError("support must equal the row sum of its class")
            require_close(entry.precision, precision, f"{entry.class_name} precision")
            require_close(entry.recall, recall, f"{entry.class_name} recall")
            require_close(entry.f1, f1_scores[-1], f"{entry.class_name} f1")
        trace = sum(matrix[index][index] for index in range(size))
        require_close(self.metrics.accuracy_top1, trace / total, "accuracy_top1")
        require_close(self.metrics.f1_macro, sum(f1_scores) / size, "f1_macro")
        return self

    @model_validator(mode="after")
    def predictions_match_confusion_matrix(self) -> Self:
        require_unique(
            [prediction.annotation_id for prediction in self.predictions],
            "annotation_id must be unique",
        )
        position = {name: index for index, name in enumerate(self.class_names)}
        counts = Counter()
        for prediction in self.predictions:
            if prediction.true_class not in position or prediction.predicted_class not in position:
                raise ValueError("true_class and predicted_class must be in class_names")
            require_distribution(
                prediction.probabilities, prediction.predicted_class, self.class_names
            )
            counts[position[prediction.true_class], position[prediction.predicted_class]] += 1
        for row, cells in enumerate(self.confusion_matrix):
            for column, count in enumerate(cells):
                if counts[row, column] != count:
                    raise ValueError("predictions must add up to confusion_matrix")
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


class CropSelection(ContractModel):
    """A crop chosen in the UI: one COCO annotation of a dataset release."""

    dataset_version: Identifier
    image_id: CocoId
    annotation_id: CocoId


class InferenceRequest(ContractModel):
    """Body of `POST /api/ml/inference`: classify one existing crop."""

    schema_version: Literal["1.0"]
    model_name: Identifier
    model_version: ModelVersion
    crop: CropSelection


class InferenceResponse(ContractModel):
    """One class for the crop plus the full distribution; `dataset_version` is the
    release the model was trained on, `crop.dataset_version` the crop's release."""

    schema_version: Literal["1.0"]
    request_id: Identifier
    model_name: Identifier
    model_version: ModelVersion
    run_id: RunId
    dataset_version: Identifier
    crop: CropSelection
    predicted_class: Label
    probabilities: dict[Label, Ratio]
    latency_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def probabilities_are_a_distribution(self) -> Self:
        require_distribution(self.probabilities, self.predicted_class, None)
        return self


CONTRACTS: dict[str, type[ContractModel]] = {
    "training_jobs": TrainingJobsResponse,
    "runs": RunsResponse,
    "evaluations": EvaluationsResponse,
    "models": ModelsResponse,
    "inference_request": InferenceRequest,
    "inference": InferenceResponse,
    "error": ErrorResponse,
}
