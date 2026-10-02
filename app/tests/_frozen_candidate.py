"""Candidato congelado real (run corto en MLflow + ML-08) y su evaluación de test (ML-09).

Para pruebas que necesitan la cadena completa sin el dataset real: entrena con el
release controlado de `_classification_fixtures`, congela el run como haría ML-08 y
evalúa el test una vez como ML-09.
"""

from datetime import UTC, datetime
from pathlib import Path

from mlflow.tracking import MlflowClient

from classification.evaluation import evaluate_frozen_candidate
from classification.selection import (
    CandidateSelection,
    RankedRun,
    TieBreaker,
    _checkpoint_sha256,
    freeze_candidate,
)
from classification.training import DataPaths, run_training
from presentation.ml_contracts import TrainingParams

COMMIT = "0123456789abcdef0123456789abcdef01234567"


def freeze_short_run(client: MlflowClient, release, data: DataPaths, tmp_path: Path) -> Path:
    """Entrena un run corto, lo congela como candidato y devuelve la ruta del candidato."""
    params = TrainingParams(
        optimizer="adam",
        batch_size=4,
        max_epochs=2,
        learning_rate=0.001,
        image_size=32,
        hidden_layers=[8],
        dropout=0.1,
        seed=3,
        patience=2,
        min_delta=0.0,
    )
    result = run_training(
        params,
        dataset_version=release.dataset_version,
        manifest_hash=release.manifest_hash,
        data=data,
        client=client,
        pretrained=False,
        git_commit=COMMIT,
    )
    candidate = CandidateSelection(
        matrix_id="ml07-test",
        split="validation",
        metric="best_val_loss",
        mode="min",
        tie_breakers=[TieBreaker(metric="entry", mode="min")],
        metric_value=min(epoch.val_loss for epoch in result.history),
        run_id=result.run_id,
        entry="r01",
        run_name="ml07-test-r01",
        checkpoint=result.checkpoint_uri,
        checkpoint_sha256=_checkpoint_sha256(client, result.run_id),
        dataset_version=release.dataset_version,
        manifest_hash=release.manifest_hash,
        training_git_commit=COMMIT,
        selection_git_commit=COMMIT,
        frozen_at=datetime.now(UTC),
        ranking=[
            RankedRun(
                entry="r01",
                run_id=result.run_id,
                best_epoch=result.best_epoch,
                best_val_loss=0.5,
                best_val_accuracy=0.5,
            )
        ],
    )
    path = tmp_path / "reports" / "candidates" / "candidate.json"
    freeze_candidate(candidate, path, evaluations_dir=tmp_path / "evaluations", client=client)
    return path


def evaluate_test_once(client: MlflowClient, data: DataPaths, candidate: Path, tmp_path: Path):
    return evaluate_frozen_candidate(
        client=client,
        data=data,
        candidate_path=candidate,
        evaluations_dir=tmp_path / "evaluations",
    )
