"""APP-05 — lo que la pantalla Evaluation puede mostrar, sin revelar el test antes de tiempo.

Lee los archivos que dejan ML-08 y ML-09 en `reports/`:

- `candidates/ml08_candidate.json`: el candidato elegido solo con validation y
  congelado (`classification.selection.CandidateSelection`).
- `evaluations/test/<id>.json` (contrato `Evaluation`) y `<id>.predictions.csv`: la
  evaluación final del test (`classification.evaluation`).

Los resultados del test solo salen si hay un candidato congelado válido y una sola
evaluación de test que sea suya (mismo run, checkpoint, release y manifest) y
posterior al congelamiento. Si algo no cuadra se ocultan y `problem` dice por qué:
mostrar un test que no corresponde, o consultado antes de congelar, es lo que la
rúbrica penaliza.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from presentation.ml_contracts import Evaluation, EvaluationOverview, FrozenCandidate

CANDIDATE_FILE = Path("candidates") / "ml08_candidate.json"
TEST_EVALUATIONS_DIR = Path("evaluations") / "test"
PREDICTIONS_SUFFIX = ".predictions.csv"


class _CandidateFile(BaseModel):
    """Los campos de `CandidateSelection` (ML-08) que necesita la pantalla."""

    model_config = ConfigDict(extra="ignore")

    run_id: str
    run_name: str
    checkpoint: str
    checkpoint_sha256: str
    dataset_version: str
    manifest_hash: str
    metric: str
    metric_value: float
    frozen_at: datetime


@dataclass(frozen=True)
class EvaluationView:
    overview: EvaluationOverview
    predictions_csv: Path | None  # solo en `evaluated`


def _first_error(exc: ValidationError) -> str:
    error = exc.errors()[0]
    field = ".".join(str(part) for part in error["loc"]) or "documento"
    return f"{field}: {error['msg']}"


def _timestamp(moment: datetime) -> str:
    moment = moment if moment.tzinfo else moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _instant(timestamp: str) -> datetime:
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


def _load_candidate(reports_dir: Path) -> tuple[FrozenCandidate | None, str | None]:
    path = reports_dir / CANDIDATE_FILE
    if not path.is_file():
        return None, None
    try:
        raw = _CandidateFile.model_validate_json(path.read_text(encoding="utf-8"))
        candidate = FrozenCandidate(
            run_id=raw.run_id,
            run_name=raw.run_name,
            checkpoint=raw.checkpoint,
            checkpoint_sha256=raw.checkpoint_sha256,
            dataset_version=raw.dataset_version,
            manifest_hash=raw.manifest_hash,
            selection_metric=raw.metric,
            selection_value=raw.metric_value,
            frozen_at=_timestamp(raw.frozen_at),
        )
    except (OSError, ValidationError) as exc:
        detail = _first_error(exc) if isinstance(exc, ValidationError) else str(exc)
        return None, f"El {CANDIDATE_FILE.as_posix()} no es un candidato válido ({detail})."
    return candidate, None


def _load_test_evaluations(reports_dir: Path) -> tuple[list[tuple[Evaluation, Path]], str | None]:
    folder = reports_dir / TEST_EVALUATIONS_DIR
    if not folder.is_dir():
        return [], None
    evaluations = []
    for path in sorted(folder.glob("*.json")):
        try:
            evaluation = Evaluation.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            detail = _first_error(exc) if isinstance(exc, ValidationError) else "no es JSON"
            return [], f"La evaluación {path.name} no cumple el contrato Evaluation ({detail})."
        evaluations.append((evaluation, path))
    return evaluations, None


def _mismatch(candidate: FrozenCandidate, evaluation: Evaluation) -> str | None:
    if evaluation.split != "test":
        return (
            f"La evaluación {evaluation.evaluation_id} es de {evaluation.split}: no se "
            "presenta como evaluación final de test."
        )
    if (
        evaluation.run_id != candidate.run_id
        or evaluation.checkpoint != candidate.checkpoint
        or evaluation.dataset_version != candidate.dataset_version
        or evaluation.manifest_hash != candidate.manifest_hash
    ):
        return (
            f"La evaluación {evaluation.evaluation_id} no es del candidato congelado "
            f"(run {candidate.run_id}, manifest {candidate.manifest_hash})."
        )
    if _instant(evaluation.created_at) < _instant(candidate.frozen_at):
        return (
            f"La evaluación {evaluation.evaluation_id} se hizo antes de congelar el "
            "candidato; no se muestra."
        )
    return None


def load_evaluation_view(reports_dir: Path) -> EvaluationView:
    candidate, candidate_problem = _load_candidate(reports_dir)
    evaluations, evaluations_problem = _load_test_evaluations(reports_dir)

    def view(state, *, evaluation=None, problem=None, csv=None) -> EvaluationView:
        overview = EvaluationOverview(
            schema_version="1.0",
            state=state,
            candidate=candidate,
            evaluation=evaluation,
            problem=problem,
        )
        return EvaluationView(overview=overview, predictions_csv=csv)

    if candidate is None:
        problem = candidate_problem
        if problem is None and (evaluations or evaluations_problem):
            problem = (
                "Hay una evaluación de test sin candidato congelado; no se muestra "
                "hasta que ML-08 congele el candidato."
            )
        return view("candidate_not_frozen", problem=problem)
    if evaluations_problem is not None:
        return view("candidate_frozen", problem=evaluations_problem)
    if not evaluations:
        return view("candidate_frozen")
    if len(evaluations) > 1:
        return view(
            "candidate_frozen",
            problem=(
                f"Hay {len(evaluations)} evaluaciones de test; el test se evalúa una sola vez."
            ),
        )
    evaluation, path = evaluations[0]
    problem = _mismatch(candidate, evaluation)
    if problem is not None:
        return view("candidate_frozen", problem=problem)
    csv = path.with_name(path.stem + PREDICTIONS_SUFFIX)
    return view("evaluated", evaluation=evaluation, csv=csv if csv.is_file() else None)


def crop_path(reports_dir: Path, crops_dir: Path, crop_id: str) -> Path | None:
    """`crops_dir/<crop_path>` del crop según `reports/crops.json` (ML-01), si existe."""
    try:
        report = json.loads((reports_dir / "crops.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    entry = next((c for c in report.get("crops", []) if c.get("crop_id") == crop_id), None)
    if entry is None:
        return None
    root = crops_dir.resolve()
    path = (root / entry["crop_path"]).resolve()
    if root not in path.parents or not path.is_file():
        return None
    return path
