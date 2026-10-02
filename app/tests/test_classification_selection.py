"""ML-08: selección del candidato solo con validation y congelamiento antes del test."""

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from mlflow.tracking import MlflowClient

from classification.selection import (
    CandidateLockedError,
    SelectionPolicy,
    early_test_evaluations,
    freeze_candidate,
    load_frozen_candidate,
    load_policy,
    require_frozen_candidate,
    select_candidate,
)

MATRIX = "ml07-test"
MANIFEST = "sha256:" + "a" * 64
COMMIT = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    return MlflowClient(tracking_uri=(tmp_path / "mlruns").as_uri())


def _trained_run(
    client,
    tmp_path,
    entry,
    *,
    val_loss,
    val_accuracy,
    matrix=MATRIX,
    status="FINISHED",
    extra_metrics=None,
):
    """Run con lo que registra `run_training` (ML-04/05/06), sin entrenar de verdad."""
    experiment = client.get_experiment_by_name("dogcat-classifier")
    experiment_id = (
        experiment.experiment_id if experiment else client.create_experiment("dogcat-classifier")
    )
    run = client.create_run(
        experiment_id,
        run_name=f"{matrix}-{entry}",
        tags={
            "experiment_matrix": matrix,
            "matrix_entry": entry,
            "dataset_version": "v0.1.1",
            "manifest_hash": MANIFEST,
            "git_commit": COMMIT,
            "mlflow.source.git.commit": COMMIT,
            "classes": "dog,cat",
            "dvc_images_hash": "0" * 32 + ".dir",
            "dvc_annotations_hash": "1" * 32 + ".dir",
        },
    )
    run_id = run.info.run_id
    for name, value in {
        "optimizer": "adam",
        "batch_size": "32",
        "max_epochs": "3",
        "learning_rate": "0.001",
        "image_size": "64",
        "hidden_layers": "16",
        "dropout": "0.1",
        "seed": "42",
        "train_samples": "8",
    }.items():
        client.log_param(run_id, name, value)
    for epoch, (loss, accuracy) in enumerate(
        [(val_loss + 0.2, val_accuracy - 0.1), (val_loss, val_accuracy)], start=1
    ):
        for name, value in {
            "train_loss": loss,
            "train_accuracy": accuracy,
            "val_loss": loss,
            "val_accuracy": accuracy,
        }.items():
            client.log_metric(run_id, name, value, step=epoch)
    client.log_metric(run_id, "best_epoch", 2)
    client.log_metric(run_id, "best_val_loss", val_loss)
    client.log_metric(run_id, "epochs_completed", 2)
    for name, value in (extra_metrics or {}).items():
        client.log_metric(run_id, name, value)
    checkpoint = tmp_path / "ckpt" / entry / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(f"pesos-{entry}".encode())
    client.log_artifact(run_id, str(checkpoint), "checkpoints")
    client.set_terminated(run_id, status)
    return run_id


def _policy(**overrides):
    values = {
        "matrix_id": MATRIX,
        "split": "validation",
        "metric": "best_val_loss",
        "mode": "min",
        "tie_breakers": [
            {"metric": "best_val_accuracy", "mode": "max"},
            {"metric": "entry", "mode": "min"},
        ],
    }
    values.update(overrides)
    return SelectionPolicy(**values)


def _select(client, **kwargs):
    kwargs.setdefault("selection_git_commit", COMMIT)
    return select_candidate(client, _policy(), **kwargs)


# --- Política predeclarada -----------------------------------------------------------


def test_committed_policy_predeclares_a_validation_metric():
    policy = load_policy()

    assert policy.matrix_id == "ml07-v1"
    assert policy.split == "validation"
    assert (policy.metric, policy.mode) == ("best_val_loss", "min")
    assert [tb.metric for tb in policy.tie_breakers] == ["best_val_accuracy", "entry"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"metric": "test_accuracy"},
        {"metric": "best_test_loss"},
        {"split": "test"},
        {"tie_breakers": [{"metric": "test_f1_macro", "mode": "max"}]},
        {"mode": "median"},
        {"metric": "train_loss"},
    ],
)
def test_policy_rejects_anything_but_validation_metrics(overrides):
    with pytest.raises(ValueError):
        _policy(**overrides)


def test_load_policy_reads_yaml(tmp_path):
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(_policy().model_dump()), encoding="utf-8")

    assert load_policy(path) == _policy()


# --- Selección -------------------------------------------------------------------------


def test_selects_the_lowest_validation_loss(client, tmp_path):
    _trained_run(client, tmp_path, "r01", val_loss=0.30, val_accuracy=0.90)
    best = _trained_run(client, tmp_path, "r02", val_loss=0.10, val_accuracy=0.85)
    _trained_run(client, tmp_path, "r03", val_loss=0.20, val_accuracy=0.99)

    candidate = _select(client)

    assert candidate.run_id == best
    assert candidate.entry == "r02"
    assert candidate.metric_value == 0.10
    assert [row.entry for row in candidate.ranking] == ["r02", "r03", "r01"]


def test_ties_are_broken_by_validation_accuracy_then_entry(client, tmp_path):
    _trained_run(client, tmp_path, "r03", val_loss=0.10, val_accuracy=0.95)
    _trained_run(client, tmp_path, "r02", val_loss=0.10, val_accuracy=0.97)
    _trained_run(client, tmp_path, "r01", val_loss=0.10, val_accuracy=0.95)

    candidate = _select(client)

    assert [row.entry for row in candidate.ranking] == ["r02", "r01", "r03"]
    assert candidate.entry == "r02"


def test_only_finished_runs_of_the_matrix_compete(client, tmp_path):
    keep = _trained_run(client, tmp_path, "r01", val_loss=0.30, val_accuracy=0.90)
    _trained_run(client, tmp_path, "r02", val_loss=0.01, val_accuracy=0.99, status="FAILED")
    _trained_run(client, tmp_path, "r03", val_loss=0.01, val_accuracy=0.99, matrix="otra")

    candidate = _select(client)

    assert candidate.run_id == keep
    assert [row.entry for row in candidate.ranking] == ["r01"]


def test_selection_refuses_runs_that_already_have_test_metrics(client, tmp_path):
    _trained_run(client, tmp_path, "r01", val_loss=0.30, val_accuracy=0.90)
    _trained_run(
        client, tmp_path, "r02", val_loss=0.10, val_accuracy=0.85, extra_metrics={"test_acc": 0.9}
    )

    with pytest.raises(ValueError, match="test"):
        _select(client)


def test_selection_refuses_an_empty_matrix(client):
    with pytest.raises(ValueError, match=MATRIX):
        _select(client)


def test_candidate_records_run_checkpoint_manifest_commit_and_timestamp(client, tmp_path):
    run_id = _trained_run(client, tmp_path, "r01", val_loss=0.10, val_accuracy=0.95)
    before = datetime.now(UTC)

    candidate = _select(client)

    assert candidate.run_id == run_id
    assert candidate.checkpoint == f"runs:/{run_id}/checkpoints/best.pt"
    assert candidate.checkpoint_sha256 == hashlib.sha256(b"pesos-r01").hexdigest()
    assert candidate.manifest_hash == MANIFEST
    assert candidate.dataset_version == "v0.1.1"
    assert candidate.training_git_commit == COMMIT
    assert candidate.selection_git_commit == COMMIT
    assert candidate.split == "validation"
    assert before <= candidate.frozen_at <= datetime.now(UTC)


# --- Congelamiento ------------------------------------------------------------------------


def _evaluation(directory: Path, split: str, created_at: datetime, run_id="f" * 32):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"eval-{split}-{created_at.timestamp():.0f}.json"
    path.write_text(
        json.dumps(
            {
                "evaluation_id": f"eval-{split}",
                "run_id": run_id,
                "split": split,
                "created_at": created_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        ),
        encoding="utf-8",
    )
    return path


def test_freeze_persists_the_candidate(client, tmp_path):
    _trained_run(client, tmp_path, "r01", val_loss=0.10, val_accuracy=0.95)
    candidate = _select(client)
    path = tmp_path / "reports" / "candidate.json"

    freeze_candidate(candidate, path, evaluations_dir=tmp_path / "evaluations")

    assert load_frozen_candidate(path) == candidate


def test_freezing_the_same_run_again_keeps_the_original_timestamp(client, tmp_path):
    _trained_run(client, tmp_path, "r01", val_loss=0.10, val_accuracy=0.95)
    path = tmp_path / "candidate.json"
    first = freeze_candidate(_select(client), path, evaluations_dir=tmp_path / "ev")

    again = freeze_candidate(_select(client), path, evaluations_dir=tmp_path / "ev")

    assert again.frozen_at == first.frozen_at
    assert load_frozen_candidate(path).frozen_at == first.frozen_at


def test_a_different_candidate_needs_an_explicit_replace(client, tmp_path):
    _trained_run(client, tmp_path, "r01", val_loss=0.10, val_accuracy=0.95)
    path = tmp_path / "candidate.json"
    freeze_candidate(_select(client), path, evaluations_dir=tmp_path / "ev")
    other = _trained_run(client, tmp_path, "r02", val_loss=0.05, val_accuracy=0.96)

    with pytest.raises(CandidateLockedError, match="congelado"):
        freeze_candidate(_select(client), path, evaluations_dir=tmp_path / "ev")

    replaced = freeze_candidate(
        _select(client), path, evaluations_dir=tmp_path / "ev", replace=True
    )
    assert replaced.run_id == other


def test_candidate_cannot_change_once_a_test_evaluation_exists(client, tmp_path):
    # Requisito TDD del issue: después de evaluar test, no se puede cambiar el candidato.
    first = _trained_run(client, tmp_path, "r01", val_loss=0.10, val_accuracy=0.95)
    path = tmp_path / "candidate.json"
    evaluations = tmp_path / "evaluations"
    freeze_candidate(_select(client), path, evaluations_dir=evaluations)
    _evaluation(evaluations, "test", datetime.now(UTC) + timedelta(seconds=1), run_id=first)
    _trained_run(client, tmp_path, "r02", val_loss=0.05, val_accuracy=0.96)

    with pytest.raises(CandidateLockedError, match="test"):
        freeze_candidate(_select(client), path, evaluations_dir=evaluations, replace=True)

    assert load_frozen_candidate(path).run_id == first


def test_a_validation_evaluation_does_not_lock_the_candidate(client, tmp_path):
    _trained_run(client, tmp_path, "r01", val_loss=0.10, val_accuracy=0.95)
    path = tmp_path / "candidate.json"
    evaluations = tmp_path / "evaluations"
    freeze_candidate(_select(client), path, evaluations_dir=evaluations)
    _evaluation(evaluations, "validation", datetime.now(UTC))
    other = _trained_run(client, tmp_path, "r02", val_loss=0.05, val_accuracy=0.96)

    replaced = freeze_candidate(_select(client), path, evaluations_dir=evaluations, replace=True)

    assert replaced.run_id == other


def test_freezing_after_a_test_evaluation_is_refused(client, tmp_path):
    # El test debe seguir oculto hasta el freeze: si ya hay una evaluación de test, no se congela.
    _trained_run(client, tmp_path, "r01", val_loss=0.10, val_accuracy=0.95)
    evaluations = tmp_path / "evaluations"
    _evaluation(evaluations, "test", datetime.now(UTC))

    with pytest.raises(CandidateLockedError, match="test"):
        freeze_candidate(_select(client), tmp_path / "c.json", evaluations_dir=evaluations)


def test_freeze_tags_the_selected_run_in_mlflow(client, tmp_path):
    run_id = _trained_run(client, tmp_path, "r01", val_loss=0.10, val_accuracy=0.95)

    candidate = freeze_candidate(
        _select(client), tmp_path / "c.json", evaluations_dir=tmp_path / "ev", client=client
    )

    tags = client.get_run(run_id).data.tags
    assert tags["candidate"] == "true"
    assert tags["candidate_frozen_at"] == candidate.frozen_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _candidate_runs(client):
    experiment = client.get_experiment_by_name("dogcat-classifier").experiment_id
    return [
        run.info.run_id
        for run in client.search_runs([experiment], filter_string="tags.candidate = 'true'")
    ]


def test_replacing_the_candidate_unmarks_the_previous_run(client, tmp_path):
    # Revisión del PR #44: al reemplazar, el run anterior conservaba candidate=true.
    first = _trained_run(client, tmp_path, "r01", val_loss=0.10, val_accuracy=0.95)
    path = tmp_path / "candidate.json"
    freeze_candidate(_select(client), path, evaluations_dir=tmp_path / "ev", client=client)
    second = _trained_run(client, tmp_path, "r02", val_loss=0.05, val_accuracy=0.96)

    replaced = freeze_candidate(
        _select(client), path, evaluations_dir=tmp_path / "ev", client=client, replace=True
    )

    assert _candidate_runs(client) == [second]
    old = client.get_run(first).data.tags
    assert old["candidate"] == "false"
    assert "candidate_frozen_at" not in old
    assert old["candidate_replaced_by"] == second
    assert old["candidate_replaced_at"] == replaced.frozen_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    assert client.get_run(second).data.tags["candidate"] == "true"


def test_refreezing_the_same_run_keeps_a_single_candidate(client, tmp_path):
    run_id = _trained_run(client, tmp_path, "r01", val_loss=0.10, val_accuracy=0.95)
    path = tmp_path / "candidate.json"
    freeze_candidate(_select(client), path, evaluations_dir=tmp_path / "ev", client=client)

    freeze_candidate(_select(client), path, evaluations_dir=tmp_path / "ev", client=client)

    assert _candidate_runs(client) == [run_id]
    assert "candidate_replaced_by" not in client.get_run(run_id).data.tags


def test_a_refused_replacement_leaves_the_tags_untouched(client, tmp_path):
    first = _trained_run(client, tmp_path, "r01", val_loss=0.10, val_accuracy=0.95)
    path = tmp_path / "candidate.json"
    freeze_candidate(_select(client), path, evaluations_dir=tmp_path / "ev", client=client)
    _trained_run(client, tmp_path, "r02", val_loss=0.05, val_accuracy=0.96)

    with pytest.raises(CandidateLockedError):
        freeze_candidate(_select(client), path, evaluations_dir=tmp_path / "ev", client=client)

    assert _candidate_runs(client) == [first]


# --- Para ML-09: candidato obligatorio y orden temporal (Agent Test) ----------------------


def test_require_frozen_candidate_fails_without_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="candidato"):
        require_frozen_candidate(tmp_path / "missing.json")


def test_test_evaluations_must_come_after_the_freeze(client, tmp_path):
    run_id = _trained_run(client, tmp_path, "r01", val_loss=0.10, val_accuracy=0.95)
    candidate = freeze_candidate(
        _select(client), tmp_path / "c.json", evaluations_dir=tmp_path / "e"
    )
    evaluations = tmp_path / "evaluations"
    _evaluation(evaluations, "test", candidate.frozen_at - timedelta(minutes=5), run_id=run_id)
    _evaluation(evaluations, "test", candidate.frozen_at + timedelta(minutes=5), run_id=run_id)
    _evaluation(evaluations, "validation", candidate.frozen_at - timedelta(minutes=9))

    early = early_test_evaluations(candidate, evaluations)

    assert [item["split"] for item in early] == ["test"]
    assert len(early) == 1
