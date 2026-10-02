"""ML-08 — selección del candidato solo con validation y congelamiento antes del test.

1. La política (`selection_policy.yaml`) predeclara la métrica: menor
   `best_val_loss` entre los runs `FINISHED` de la matriz de ML-07, con
   desempates fijos. `SelectionPolicy` rechaza cualquier métrica que no sea de
   validation.
2. `select_candidate` lee solo esas métricas desde MLflow. Se niega a elegir si
   algún run de la matriz ya tiene una métrica de test.
3. `freeze_candidate` escribe `reports/candidates/ml08_candidate.json` (run_id,
   checkpoint y su sha256, manifest_hash, commits, timestamp y ranking) y marca el
   run en MLflow (`candidate=true`).
   - Congelar otra vez el mismo run no cambia nada.
   - Cambiar de candidato exige `replace=True`; el run anterior queda
     `candidate=false` con `candidate_replaced_by` y `candidate_replaced_at`, así
     siempre hay un solo run con `candidate=true`.
   - Si ya existe **cualquier** evaluación de test en `reports/evaluations/`, ni
     se congela por primera vez ni se cambia.

Para ML-09: `require_frozen_candidate()` antes de evaluar test, y
`early_test_evaluations()` para demostrar que toda evaluación de test es
posterior al congelamiento (Agent Test).

    uv run python -m classification.selection select    # ranking, sin congelar
    uv run python -m classification.selection freeze    # elige y congela
    uv run python -m classification.selection check     # congelado antes de todo test
"""

import argparse
import hashlib
import json
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

import yaml
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, ConfigDict, model_validator

from classification.experiments import _matrix_runs, _run_summary
from classification.training import CHECKPOINT_ARTIFACT, REPO_ROOT, resolve_git_commit

POLICY_PATH = Path(__file__).with_name("selection_policy.yaml")
CANDIDATE_PATH = REPO_ROOT / "reports" / "candidates" / "ml08_candidate.json"
EVALUATIONS_DIR = REPO_ROOT / "reports" / "evaluations"
VALIDATION_METRICS = ("best_val_loss", "best_val_accuracy")
TIE_BREAK_FIELDS = (*VALIDATION_METRICS, "entry")


class CandidateLockedError(RuntimeError):
    """El candidato no puede congelarse o cambiarse (ya hay uno, o ya se consultó test)."""


class TieBreaker(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    metric: str
    mode: Literal["min", "max"]


class SelectionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    matrix_id: str
    split: Literal["validation"]
    metric: str
    mode: Literal["min", "max"]
    tie_breakers: list[TieBreaker]

    @model_validator(mode="after")
    def validation_metrics_only(self) -> Self:
        if self.metric not in VALIDATION_METRICS:
            raise ValueError(f"La métrica de selección debe ser de validation: {self.metric}")
        for tie in self.tie_breakers:
            if tie.metric not in TIE_BREAK_FIELDS:
                raise ValueError(f"Desempate no permitido (solo validation o entry): {tie.metric}")
            if tie.metric == "entry" and tie.mode != "min":
                raise ValueError("El desempate por entry solo admite mode=min")
        return self


class RankedRun(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    entry: str
    run_id: str
    best_epoch: int
    best_val_loss: float
    best_val_accuracy: float


class CandidateSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    matrix_id: str
    split: Literal["validation"]
    metric: str
    mode: Literal["min", "max"]
    tie_breakers: list[TieBreaker]
    metric_value: float
    run_id: str
    entry: str
    run_name: str
    checkpoint: str
    checkpoint_sha256: str
    dataset_version: str
    manifest_hash: str
    training_git_commit: str
    selection_git_commit: str
    frozen_at: datetime
    ranking: list[RankedRun]


def load_policy(path: Path = POLICY_PATH) -> SelectionPolicy:
    return SelectionPolicy.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def _sort_key(policy: SelectionPolicy):
    rules = [(policy.metric, policy.mode)] + [(t.metric, t.mode) for t in policy.tie_breakers]

    def key(summary: dict) -> tuple:
        parts = []
        for metric, mode in rules:
            value = summary[metric]
            parts.append(value if metric == "entry" or mode == "min" else -value)
        return tuple(parts)

    return key


def _checkpoint_sha256(client: MlflowClient, run_id: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(client.download_artifacts(run_id, CHECKPOINT_ARTIFACT, tmp))
        return hashlib.sha256(local.read_bytes()).hexdigest()


def select_candidate(
    client: MlflowClient, policy: SelectionPolicy, *, selection_git_commit: str | None = None
) -> CandidateSelection:
    runs = _matrix_runs(client, policy.matrix_id)
    with_test = sorted(
        run.data.tags.get("matrix_entry") or run.info.run_id
        for run in runs
        if any("test" in name for name in run.data.metrics)
    )
    if with_test:
        raise ValueError(
            f"Runs de {policy.matrix_id} con métricas de test ({with_test}): el candidato "
            "debe elegirse antes de consultar el test"
        )
    summaries = [_run_summary(client, run) for run in runs if run.info.status == "FINISHED"]
    if not summaries:
        raise ValueError(f"No hay runs FINISHED de la matriz {policy.matrix_id}")
    summaries.sort(key=_sort_key(policy))
    best = summaries[0]
    return CandidateSelection(
        matrix_id=policy.matrix_id,
        split=policy.split,
        metric=policy.metric,
        mode=policy.mode,
        tie_breakers=policy.tie_breakers,
        metric_value=best[policy.metric],
        run_id=best["run_id"],
        entry=best["entry"],
        run_name=best["run_name"],
        checkpoint=f"runs:/{best['run_id']}/{CHECKPOINT_ARTIFACT}",
        checkpoint_sha256=_checkpoint_sha256(client, best["run_id"]),
        dataset_version=best["dataset_version"],
        manifest_hash=best["manifest_hash"],
        training_git_commit=best["git_commit"],
        selection_git_commit=selection_git_commit or resolve_git_commit(),
        frozen_at=datetime.now(UTC),
        ranking=[
            RankedRun(
                entry=s["entry"],
                run_id=s["run_id"],
                best_epoch=s["best_epoch"],
                best_val_loss=s["best_val_loss"],
                best_val_accuracy=s["best_val_accuracy"],
            )
            for s in summaries
        ],
    )


def _evaluations(directory: Path, split: str) -> list[dict]:
    if not directory.is_dir():
        return []
    found = []
    for path in sorted(directory.rglob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(document, dict) and document.get("split") == split:
            found.append({**document, "path": str(path)})
    return found


def load_frozen_candidate(path: Path = CANDIDATE_PATH) -> CandidateSelection:
    return CandidateSelection.model_validate_json(path.read_text(encoding="utf-8"))


def require_frozen_candidate(path: Path = CANDIDATE_PATH) -> CandidateSelection:
    """Para ML-09: el test solo se evalúa sobre un candidato ya congelado."""
    if not path.is_file():
        raise FileNotFoundError(f"No hay candidato congelado en {path}: corre ML-08 primero")
    return load_frozen_candidate(path)


def freeze_candidate(
    candidate: CandidateSelection,
    path: Path = CANDIDATE_PATH,
    *,
    evaluations_dir: Path = EVALUATIONS_DIR,
    client: MlflowClient | None = None,
    replace: bool = False,
) -> CandidateSelection:
    test_evaluations = _evaluations(evaluations_dir, "test")
    previous: CandidateSelection | None = None
    if path.is_file():
        existing = load_frozen_candidate(path)
        if existing.run_id == candidate.run_id:
            return existing
        if test_evaluations:
            raise CandidateLockedError(
                f"Ya existe una evaluación de test ({test_evaluations[0]['path']}): el "
                f"candidato {existing.run_id} no se puede cambiar"
            )
        if not replace:
            raise CandidateLockedError(
                f"Ya hay un candidato congelado ({existing.entry}, {existing.run_id}); "
                "usa replace=True para cambiarlo antes de evaluar test"
            )
        previous = existing
    elif test_evaluations:
        raise CandidateLockedError(
            f"Ya existe una evaluación de test ({test_evaluations[0]['path']}): el candidato "
            "debe congelarse antes de consultar el test"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(candidate.model_dump_json(indent=2) + "\n", encoding="utf-8")
    if client is not None:
        frozen_at = candidate.frozen_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        client.set_tag(candidate.run_id, "candidate", "true")
        client.set_tag(candidate.run_id, "candidate_frozen_at", frozen_at)
        if previous is not None:
            # Un solo run con candidate=true; el anterior conserva la traza del reemplazo.
            client.set_tag(previous.run_id, "candidate", "false")
            client.delete_tag(previous.run_id, "candidate_frozen_at")
            client.set_tag(previous.run_id, "candidate_replaced_by", candidate.run_id)
            client.set_tag(previous.run_id, "candidate_replaced_at", frozen_at)
    return candidate


def early_test_evaluations(candidate: CandidateSelection, evaluations_dir: Path) -> list[dict]:
    """Evaluaciones de test anteriores al congelamiento (debe ser una lista vacía)."""
    early = []
    for evaluation in _evaluations(evaluations_dir, "test"):
        created = datetime.fromisoformat(evaluation["created_at"].replace("Z", "+00:00"))
        if created < candidate.frozen_at:
            early.append(evaluation)
    return early


def main(argv: list[str] | None = None) -> int:
    from tracking.client import tracking_client

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["select", "freeze", "show", "check"])
    parser.add_argument("--policy", type=Path, default=POLICY_PATH)
    parser.add_argument("--candidate", type=Path, default=CANDIDATE_PATH)
    parser.add_argument("--evaluations", type=Path, default=EVALUATIONS_DIR)
    args = parser.parse_args(argv)

    if args.command in ("show", "check"):
        candidate = require_frozen_candidate(args.candidate)
        if args.command == "show":
            print(candidate.model_dump_json(indent=2))
            return 0
        early = early_test_evaluations(candidate, args.evaluations)
        tests = _evaluations(args.evaluations, "test")
        print(
            f"Candidato {candidate.entry} ({candidate.run_id}) congelado en "
            f"{candidate.frozen_at.isoformat()}; evaluaciones de test: {len(tests)}, "
            f"anteriores al congelamiento: {len(early)}"
        )
        return 0 if not early else 1

    client = tracking_client()
    candidate = select_candidate(client, load_policy(args.policy))
    for position, row in enumerate(candidate.ranking, start=1):
        print(
            f"{position:2d}. {row.entry:34s} {row.run_id} best_val_loss={row.best_val_loss:.4f} "
            f"best_val_accuracy={row.best_val_accuracy:.4f} (época {row.best_epoch})"
        )
    if args.command == "freeze":
        candidate = freeze_candidate(
            candidate, args.candidate, evaluations_dir=args.evaluations, client=client
        )
        print(f"Candidato congelado: {candidate.entry} {candidate.run_id} en {args.candidate}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
