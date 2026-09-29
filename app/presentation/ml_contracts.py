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

APP-02 adds the training request (`TrainingJobRequest`, sharing `TrainingParams`
with the job), the reproducibility and early-stopping params, the Quality Gate
rule (`training_blocked_reason`), and the per-release files
`releases/<version>/provenance.json` and `releases/<version>/manifest.json`.
"""

from collections import Counter
from math import isclose
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationInfo,
    field_validator,
    model_validator,
)

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
    """The 7 required hyperparameters (APP-01) plus seed and early stopping (APP-02).

    Every field is required, no defaults.
    """

    optimizer: Literal["adam", "adamw", "sgd"]
    batch_size: int = Field(ge=1, le=256)
    max_epochs: int = Field(ge=1, le=500)
    learning_rate: float = Field(gt=0, le=1)
    image_size: int = Field(ge=32, le=1024, multiple_of=32)
    hidden_layers: list[Annotated[int, Field(ge=1, le=4096)]] = Field(max_length=5)
    dropout: float = Field(ge=0, lt=1)
    seed: int = Field(ge=0, le=2**31 - 1)
    patience: int = Field(ge=1)
    min_delta: float = Field(ge=0, le=1)

    @field_validator("patience")
    @classmethod
    def patience_within_max_epochs(cls, patience: int, info: ValidationInfo) -> int:
        max_epochs = info.data.get("max_epochs")
        if max_epochs is not None and patience > max_epochs:
            raise ValueError("patience cannot exceed max_epochs")
        return patience


class TrainingJobRequest(ContractModel):
    """Body of `POST /api/ml/training/jobs`, validated before any job exists.

    `manifest_hash` pins the job to the exact 70/20/10 manifest of the release;
    `training_request_rejection` checks it against `releases/<v>/manifest.json`.
    """

    schema_version: Literal["1.0"]
    dataset_version: Identifier
    manifest_hash: ManifestHash
    params: TrainingParams


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


class DvcOutput(ContractModel):
    """One `outs` entry of a `.dvc` file: the content hash DVC computed."""

    path: Label
    md5: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}(\.dir)?$")]
    nfiles: Count | None


class ReleaseProvenance(ContractModel):
    """`releases/<version>/provenance.json`, written by `cut_release`."""

    schema_version: Literal["1.0"]
    dataset_version: Identifier
    dvc_outputs: list[DvcOutput] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_paths(self) -> Self:
        require_unique([output.path for output in self.dvc_outputs], "path must be unique")
        return self


class ManifestSplit(ContractModel):
    image_count: Count
    ratio: Ratio


class ManifestSplits(ContractModel):
    train: ManifestSplit
    validation: ManifestSplit
    test: ManifestSplit


SPLIT_TARGETS = {"train": 0.7, "validation": 0.2, "test": 0.1}


class TrainingManifest(ContractModel):
    """`releases/<version>/manifest.json`: the exact split used to train (70/20/10).

    Each split must hold its target share up to integer rounding: its image_count
    is less than one image away from `target * total_images`.
    """

    schema_version: Literal["1.0"]
    dataset_version: Identifier
    manifest_hash: ManifestHash
    total_images: int = Field(gt=0)
    splits: ManifestSplits

    @model_validator(mode="after")
    def counts_and_ratios_agree(self) -> Self:
        splits = (self.splits.train, self.splits.validation, self.splits.test)
        if sum(split.image_count for split in splits) != self.total_images:
            raise ValueError("split counts must sum to total_images")
        for split in splits:
            if not isclose(split.ratio, split.image_count / self.total_images, abs_tol=1e-6):
                raise ValueError("ratio must equal image_count / total_images")
        return self

    @model_validator(mode="after")
    def split_is_70_20_10(self) -> Self:
        for name, target in SPLIT_TARGETS.items():
            count = getattr(self.splits, name).image_count
            if abs(count - target * self.total_images) >= 1:
                raise ValueError(f"{name} must be {target:.0%} of total_images")
        return self


def training_blocked_reason(
    quality_status: str,
    provenance: ReleaseProvenance | None,
    manifest: TrainingManifest | None,
) -> str | None:
    """Whether a release can be trained reproducibly; `None` means it can.

    A `failed` Quality Gate blocks (warn does not), and so does a missing or
    invalid DVC provenance or 70/20/10 manifest: pass `None` for either when the
    file is missing or does not validate.
    """
    if quality_status == "failed":
        return "El release no pasó el Quality Gate (failed); no se puede entrenar con él."
    if provenance is None:
        return (
            "El release no tiene un provenance.json de DVC válido; "
            "no se puede entrenar de forma reproducible."
        )
    if manifest is None:
        return (
            "El release no tiene un manifest.json 70/20/10 válido; "
            "no se puede entrenar de forma reproducible."
        )
    if provenance.dataset_version != manifest.dataset_version:
        return "La procedencia y el manifiesto corresponden a releases distintos."
    return None


def training_request_rejection(
    request: TrainingJobRequest,
    quality_status: str,
    provenance: ReleaseProvenance | None,
    manifest: TrainingManifest | None,
) -> str | None:
    """What `POST /api/ml/training/jobs` checks, after the contract, before creating a job."""
    reason = training_blocked_reason(quality_status, provenance, manifest)
    if reason is not None:
        return reason
    if manifest.dataset_version != request.dataset_version:
        return "El manifiesto no corresponde al release solicitado."
    if manifest.manifest_hash != request.manifest_hash:
        return "El manifest_hash no coincide con el manifiesto del release."
    return None


CONTRACTS: dict[str, type[ContractModel]] = {
    "training_jobs": TrainingJobsResponse,
    "runs": RunsResponse,
    "evaluations": EvaluationsResponse,
    "models": ModelsResponse,
    "inference_request": InferenceRequest,
    "inference": InferenceResponse,
    "error": ErrorResponse,
    "training_request": TrainingJobRequest,
    "provenance": ReleaseProvenance,
    "manifest": TrainingManifest,
}
