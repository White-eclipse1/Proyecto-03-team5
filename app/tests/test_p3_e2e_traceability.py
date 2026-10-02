"""OPS-09 — cadena P3 trazable de extremo a extremo.

Release P2 aprobado
→ DVC hashes
→ manifest P3
→ training corto
→ MLflow run
→ checkpoint
→ evaluación congelada
→ model version
→ publicación S3/MinIO
→ inferencia en proceso limpio.
"""

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

import boto3
import pytest
from releases.service import P2ReleaseService

from classification.evaluation import evaluate_frozen_candidate
from classification.publication import publish_to_s3, verify_publication
from classification.registry import publish_model_version, verify_model_version
from classification.selection import (
    CandidateSelection,
    RankedRun,
    TieBreaker,
    _checkpoint_sha256,
    freeze_candidate,
)
from classification.training import DataPaths, run_training
from presentation.ml_contracts import TrainingParams
from tracking.client import tracking_client

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_VERSION = "v0.1.1"
MODEL_VERSION = "9.0.0"

pytestmark = pytest.mark.skipif(
    os.environ.get("OPS09_E2E") != "1",
    reason="requiere datos PROD materializados, MLflow y MinIO",
)


def _cleanup_bucket(s3, bucket: str) -> None:
    versions = s3.list_object_versions(Bucket=bucket)

    for item in [
        *versions.get("Versions", []),
        *versions.get("DeleteMarkers", []),
    ]:
        s3.delete_object(
            Bucket=bucket,
            Key=item["Key"],
            VersionId=item["VersionId"],
        )

    for item in s3.list_objects_v2(Bucket=bucket).get("Contents", []):
        s3.delete_object(Bucket=bucket, Key=item["Key"])

    s3.delete_bucket(Bucket=bucket)


def test_p3_end_to_end_traceability(tmp_path):
    # 1. Release P2 real y aprobado.
    releases = P2ReleaseService.from_catalog(REPO_ROOT / "app" / "releases" / "p2_releases.json")
    release = releases.verify_release(DATASET_VERSION, REPO_ROOT)

    data = DataPaths.for_release(DATASET_VERSION, REPO_ROOT)
    manifest = json.loads(data.manifest.read_text(encoding="utf-8"))

    assert manifest["dataset_version"] == DATASET_VERSION
    assert manifest["provenance"]["release_version"] == DATASET_VERSION
    assert manifest["provenance"]["images_dvc_hash"] == release["images_dvc_hash"]
    assert manifest["provenance"]["annotations_dvc_hash"] == release["annotations_dvc_hash"]

    manifest_hash = manifest["manifest_hash"]
    git_commit = os.environ["GIT_COMMIT"]

    # 2. Training corto real sobre el release P2.
    client = tracking_client()

    params = TrainingParams(
        optimizer="adam",
        batch_size=128,
        max_epochs=1,
        learning_rate=0.001,
        image_size=32,
        hidden_layers=[],
        dropout=0.0,
        seed=42,
        patience=1,
        min_delta=0.0,
    )

    training = run_training(
        params,
        dataset_version=DATASET_VERSION,
        manifest_hash=manifest_hash,
        data=data,
        client=client,
        pretrained=False,
        git_commit=git_commit,
        run_name="ops09-e2e",
        tags={"ops09_e2e": "true"},
    )

    run = client.get_run(training.run_id)

    assert run.info.status == "FINISHED"
    assert run.data.tags["dataset_version"] == DATASET_VERSION
    assert run.data.tags["manifest_hash"] == manifest_hash

    checkpoint_sha256 = _checkpoint_sha256(client, training.run_id)

    assert training.checkpoint_uri == (f"runs:/{training.run_id}/checkpoints/best.pt")
    assert len(checkpoint_sha256) == 64

    # 3. Congelar el mismo run antes de consultar test.
    candidate = CandidateSelection(
        matrix_id="ops09-e2e",
        split="validation",
        metric="best_val_loss",
        mode="min",
        tie_breakers=[TieBreaker(metric="entry", mode="min")],
        metric_value=float(run.data.metrics["best_val_loss"]),
        run_id=training.run_id,
        entry="ops09-r01",
        run_name="ops09-e2e",
        checkpoint=training.checkpoint_uri,
        checkpoint_sha256=checkpoint_sha256,
        dataset_version=DATASET_VERSION,
        manifest_hash=manifest_hash,
        training_git_commit=git_commit,
        selection_git_commit=git_commit,
        frozen_at=datetime.now(UTC),
        ranking=[
            RankedRun(
                entry="ops09-r01",
                run_id=training.run_id,
                best_epoch=training.best_epoch,
                best_val_loss=float(run.data.metrics["best_val_loss"]),
                best_val_accuracy=float(run.data.metrics["val_accuracy"]),
            )
        ],
    )

    candidate_path = tmp_path / "candidate.json"
    evaluations_dir = tmp_path / "evaluations"

    freeze_candidate(
        candidate,
        candidate_path,
        evaluations_dir=evaluations_dir,
        client=client,
    )

    # 4. Evaluación controlada del test congelado.
    evaluation = evaluate_frozen_candidate(
        client=client,
        data=data,
        candidate_path=candidate_path,
        evaluations_dir=evaluations_dir,
    )

    assert evaluation.evaluation.run_id == training.run_id
    assert evaluation.evaluation.dataset_version == DATASET_VERSION
    assert evaluation.evaluation.manifest_hash == manifest_hash

    # 5. Registrar una model version reproducible.
    registry_path = tmp_path / "registry.json"
    packages_root = tmp_path / "models"

    package = publish_model_version(
        MODEL_VERSION,
        client=client,
        candidate_path=candidate_path,
        evaluations_dir=evaluations_dir,
        registry_path=registry_path,
        packages_root=packages_root,
        repo_root=tmp_path,
    )

    assert package.run_id == training.run_id
    assert package.dataset_version == DATASET_VERSION
    assert package.manifest_hash == manifest_hash
    assert package.checkpoint_sha256 == checkpoint_sha256

    assert (
        verify_model_version(
            MODEL_VERSION,
            client=client,
            registry_path=registry_path,
            repo_root=tmp_path,
            manifest_path=data.manifest,
        )
        == []
    )

    # 6. Publicación en el S3 de integración (MinIO).
    endpoint = os.environ["OPS09_S3_ENDPOINT"]
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name="us-east-1",
    )

    bucket = f"ops09-{uuid.uuid4().hex[:10]}"
    s3.create_bucket(Bucket=bucket)
    s3.put_bucket_versioning(
        Bucket=bucket,
        VersioningConfiguration={"Status": "Enabled"},
    )

    publications_path = tmp_path / "s3_publications.json"

    try:
        publication = publish_to_s3(
            MODEL_VERSION,
            s3=s3,
            bucket=bucket,
            prefix="models",
            registry_path=registry_path,
            publications_path=publications_path,
            repo_root=tmp_path,
        )

        image = next((REPO_ROOT / "data" / "crops").rglob("*.png"))

        verification = verify_publication(
            MODEL_VERSION,
            s3=s3,
            publications_path=publications_path,
            workdir=tmp_path / "clean-download",
            image=image,
        )

        assert publication["run_id"] == training.run_id
        assert publication["checkpoint_sha256"] == checkpoint_sha256

        assert verification["problems"] == []
        assert verification["inference"] is not None
        assert verification["inference"]["checkpoint_sha256"] == checkpoint_sha256

        probabilities = verification["inference"]["probabilities"]

        assert set(probabilities) == {"dog", "cat"}
        assert abs(sum(probabilities.values()) - 1.0) < 1e-5

        # 7. Evidencia completa de la cadena OPS-09.
        trace = {
            "dataset_version": DATASET_VERSION,
            "images_dvc_hash": release["images_dvc_hash"],
            "annotations_dvc_hash": release["annotations_dvc_hash"],
            "manifest_hash": manifest_hash,
            "run_id": training.run_id,
            "checkpoint": training.checkpoint_uri,
            "checkpoint_sha256": checkpoint_sha256,
            "evaluation_id": evaluation.evaluation.evaluation_id,
            "model_version": MODEL_VERSION,
            "s3_bucket": bucket,
            "s3_checkpoint_key": publication["files"]["checkpoint/best.pt"]["key"],
            "prediction": verification["inference"]["predicted_class"],
            "probabilities": probabilities,
        }

        trace_path = Path(
            os.environ.get(
                "OPS09_TRACE_OUTPUT",
                str(tmp_path / "ops09-traceability.json"),
            )
        )
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(
            json.dumps(trace, indent=2) + "\n",
            encoding="utf-8",
        )

        # Agent Test: si el manifiesto deja de apuntar al hash del modelo,
        # la verificación de trazabilidad debe detectarlo.
        tampered = dict(manifest)
        tampered["manifest_hash"] = "sha256:" + "0" * 64
        tampered_manifest = tmp_path / "tampered-manifest.json"
        tampered_manifest.write_text(
            json.dumps(tampered),
            encoding="utf-8",
        )

        problems = verify_model_version(
            MODEL_VERSION,
            client=client,
            registry_path=registry_path,
            repo_root=tmp_path,
            manifest_path=tampered_manifest,
        )

        assert any("manifest_hash" in problem for problem in problems)

    finally:
        _cleanup_bucket(s3, bucket)
