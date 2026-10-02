"""OPS-07: publicación del paquete del modelo en S3 y recuperación desde un entorno limpio.

Las pruebas usan un S3 en memoria con la semántica de boto3 que importa aquí
(versiones, `VersionId`, `ETag` y `ChecksumSHA256`). La prueba de integración contra
un S3 real (MinIO) corre si se define `S3_INTEGRATION_ENDPOINT`.
"""

import base64
import hashlib
import io
import json
import os
import re
import subprocess
import uuid
from pathlib import Path

import pytest
from botocore.exceptions import ClientError
from mlflow.tracking import MlflowClient

import classification.publication as publication_module
from classification.publication import (
    PUBLISHED_FILES,
    infer_in_clean_process,
    publish_to_s3,
    verify_publication,
)
from classification.registry import publish_model_version
from classification.training import DataPaths
from tests._classification_fixtures import write_controlled_release
from tests._frozen_candidate import evaluate_test_once, freeze_short_run

BUCKET = "models-test"
REPO_ROOT = Path(__file__).resolve().parents[2]


class FakeS3:
    """Lo mínimo de un cliente boto3 S3: put/head/get con versiones y checksums."""

    def __init__(self, *, versioning: bool = True):
        self.versioning = versioning
        self.objects: dict[tuple[str, str], list[dict]] = {}
        self.corrupt_downloads = False
        self.wrong_checksum = False

    def put_object(self, *, Bucket, Key, Body, Metadata=None, ChecksumAlgorithm=None, **_):
        body = Body.read() if hasattr(Body, "read") else Body
        version = {
            "VersionId": uuid.uuid4().hex if self.versioning else None,
            "Body": body,
            "Metadata": dict(Metadata or {}),
            # El ETag de S3 no es un SHA-256 (aquí, como en S3 sin multipart: md5).
            "ETag": f'"{hashlib.md5(body).hexdigest()}"',
            "ChecksumSHA256": (
                base64.b64encode(hashlib.sha256(body).digest()).decode()
                if ChecksumAlgorithm == "SHA256"
                else None
            ),
        }
        history = self.objects.setdefault((Bucket, Key), [])
        if not self.versioning:
            history.clear()
        history.append(version)
        reply = {"ETag": version["ETag"]}
        if version["VersionId"]:
            reply["VersionId"] = version["VersionId"]
        return reply

    def _version(self, bucket, key, version_id, operation):
        history = self.objects.get((bucket, key))
        if not history:
            raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, operation)
        if version_id is None:
            return history[-1]
        for version in history:
            if version["VersionId"] == version_id:
                return version
        raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, operation)

    def head_object(self, *, Bucket, Key, VersionId=None, ChecksumMode=None):
        version = self._version(Bucket, Key, VersionId, "HeadObject")
        reply = {
            "ContentLength": len(version["Body"]),
            "ETag": version["ETag"],
            "Metadata": version["Metadata"],
        }
        if version["VersionId"]:
            reply["VersionId"] = version["VersionId"]
        if ChecksumMode == "ENABLED" and version["ChecksumSHA256"]:
            checksum = version["ChecksumSHA256"]
            if self.wrong_checksum:
                checksum = base64.b64encode(b"\0" * 32).decode()
            reply["ChecksumSHA256"] = checksum
        return reply

    def get_object(self, *, Bucket, Key, VersionId=None):
        version = self._version(Bucket, Key, VersionId, "GetObject")
        body = b"corrupto" if self.corrupt_downloads else version["Body"]
        reply = {"Body": io.BytesIO(body), "ContentLength": len(body)}
        if version["VersionId"]:
            reply["VersionId"] = version["VersionId"]
        return reply


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    return MlflowClient(tracking_uri=(tmp_path / "mlruns").as_uri())


@pytest.fixture
def package(client, tmp_path):
    """Un paquete real de OPS-06: run corto congelado, evaluado y empaquetado."""
    release = write_controlled_release(tmp_path / "release")
    data = DataPaths(release.manifest_path, release.crop_report_path, release.crops_dir)
    frozen = freeze_short_run(client, release, data, tmp_path)
    evaluate_test_once(client, data, frozen, tmp_path)
    for version in ("1.0.0", "1.1.0"):
        publish_model_version(
            version,
            client=client,
            candidate_path=frozen,
            evaluations_dir=tmp_path / "evaluations",
            registry_path=tmp_path / "reports" / "models" / "registry.json",
            packages_root=tmp_path / "data" / "models",
            repo_root=tmp_path,
        )
    image = next(release.crops_dir.rglob("*.png"))
    return {"repo_root": tmp_path, "image": image}


def _publish(s3, package, version="1.0.0"):
    root = package["repo_root"]
    return publish_to_s3(
        version,
        s3=s3,
        bucket=BUCKET,
        prefix="models",
        registry_path=root / "reports" / "models" / "registry.json",
        publications_path=root / "reports" / "models" / "s3_publications.json",
        repo_root=root,
    )


def _verify(s3, package, version="1.0.0"):
    root = package["repo_root"]
    return verify_publication(
        version,
        s3=s3,
        publications_path=root / "reports" / "models" / "s3_publications.json",
        workdir=root / "clean",
        image=package["image"],
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- Publicación -------------------------------------------------------------------------


def test_publish_uploads_checkpoint_and_card_and_records_bucket_key_version_and_sha(package):
    s3 = FakeS3()

    record = _publish(s3, package)

    root = package["repo_root"]
    package_dir = root / "data" / "models" / "dog-cat-resnet18" / "1.0.0"
    assert record["bucket"] == BUCKET
    assert record["model_version"] == "1.0.0"
    assert set(record["files"]) == set(PUBLISHED_FILES)
    for relpath, entry in record["files"].items():
        assert entry["key"] == f"models/dog-cat-resnet18/1.0.0/{relpath}"
        assert entry["sha256"] == _sha(package_dir / relpath)  # SHA-256 local real
        assert entry["version_id"] == s3.objects[(BUCKET, entry["key"])][-1]["VersionId"]
        assert entry["size"] == (package_dir / relpath).stat().st_size
        assert entry["download_sha256"] == entry["sha256"]
        stored = s3.objects[(BUCKET, entry["key"])][-1]
        assert stored["Body"] == (package_dir / relpath).read_bytes()
        assert stored["Metadata"]["sha256"] == entry["sha256"]
    stored = json.loads((root / "reports" / "models" / "s3_publications.json").read_text("utf-8"))
    assert stored["publications"] == [record]


def test_the_etag_is_recorded_but_never_taken_as_sha256(package):
    record = _publish(FakeS3(), package)

    checkpoint = record["files"]["checkpoint/best.pt"]
    assert checkpoint["etag"] != checkpoint["sha256"]
    assert len(checkpoint["sha256"]) == 64
    source = Path(publication_module.__file__).read_text(encoding="utf-8")
    assert not re.search(r"sha256\w*\s*=\s*\S*etag", source, re.IGNORECASE)


def test_a_wrong_s3_sha256_checksum_fails_the_publication(package):
    s3 = FakeS3()
    s3.wrong_checksum = True

    with pytest.raises(ValueError, match="ChecksumSHA256"):
        _publish(s3, package)

    assert not (package["repo_root"] / "reports" / "models" / "s3_publications.json").exists()


def test_a_download_that_does_not_match_fails_the_publication(package):
    s3 = FakeS3()
    s3.corrupt_downloads = True

    with pytest.raises(ValueError, match="descarga"):
        _publish(s3, package)


def test_a_tampered_local_package_is_not_published(package):
    checkpoint = (
        package["repo_root"] / "data" / "models" / "dog-cat-resnet18" / "1.0.0" / "checkpoint"
    ) / "best.pt"
    checkpoint.write_bytes(b"alterado")
    s3 = FakeS3()

    with pytest.raises(ValueError, match="sha256"):
        _publish(s3, package)

    assert s3.objects == {}


def test_without_bucket_versioning_the_version_id_is_recorded_as_absent(package):
    record = _publish(FakeS3(versioning=False), package)

    assert all(entry["version_id"] is None for entry in record["files"].values())


def test_republishing_the_same_version_uploads_nothing(package):
    s3 = FakeS3()
    first = _publish(s3, package)

    again = _publish(s3, package)

    assert again == first
    assert all(len(history) == 1 for history in s3.objects.values())


def test_a_different_object_already_at_the_key_is_never_overwritten(package):
    s3 = FakeS3()
    key = "models/dog-cat-resnet18/1.0.0/checkpoint/best.pt"
    s3.put_object(Bucket=BUCKET, Key=key, Body=b"otro modelo")

    with pytest.raises(ValueError, match="ya existe"):
        _publish(s3, package)

    assert len(s3.objects[(BUCKET, key)]) == 1


# --- Agent Test: head-object, get-object, SHA-256, carga e inferencia --------------------


def test_agent_test_recovers_and_runs_the_model_in_a_clean_process(package):
    s3 = FakeS3()
    _publish(s3, package)

    report = _verify(s3, package)

    assert report["problems"] == []
    for relpath in PUBLISHED_FILES:
        check = report["files"][relpath]
        assert check["head_object"] is True
        assert check["download_sha256"] == check["recorded_sha256"]
    assert (package["repo_root"] / "clean" / "model-card.md").is_file()
    probabilities = report["inference"]["probabilities"]
    assert set(probabilities) == {"dog", "cat"}
    assert abs(sum(probabilities.values()) - 1) < 1e-5
    assert report["inference"]["predicted_class"] == max(probabilities, key=probabilities.get)
    assert report["inference"]["pid"] != os.getpid()


def test_agent_test_detects_an_object_changed_in_s3(package):
    s3 = FakeS3(versioning=False)
    record = _publish(s3, package)
    key = record["files"]["model-card.md"]["key"]
    s3.put_object(Bucket=BUCKET, Key=key, Body=b"otra tarjeta")

    report = _verify(s3, package)

    assert report["problems"] == ["model-card.md: el SHA-256 descargado no es el registrado"]


def test_a_previous_model_version_remains_recoverable(package):
    s3 = FakeS3()
    _publish(s3, package, "1.0.0")
    _publish(s3, package, "1.1.0")
    # Alguien sube algo encima de la clave de 1.0.0: el VersionId registrado lo protege.
    key = "models/dog-cat-resnet18/1.0.0/checkpoint/best.pt"
    s3.put_object(Bucket=BUCKET, Key=key, Body=b"sobrescrito")

    report = _verify(s3, package, "1.0.0")

    assert report["problems"] == []
    assert report["files"]["checkpoint/best.pt"]["latest_differs"] is True
    assert _verify(s3, package, "1.1.0")["problems"] == []


def test_clean_process_inference_uses_only_the_downloaded_checkpoint(package, tmp_path):
    checkpoint = (
        package["repo_root"] / "data" / "models" / "dog-cat-resnet18" / "1.0.0" / "checkpoint"
    ) / "best.pt"
    copy = tmp_path / "solo" / "best.pt"
    copy.parent.mkdir()
    copy.write_bytes(checkpoint.read_bytes())

    result = infer_in_clean_process(copy, package["image"])

    assert set(result["probabilities"]) == {"dog", "cat"}
    assert result["checkpoint_sha256"] == _sha(copy)


# --- Credenciales -------------------------------------------------------------------------


def test_no_aws_access_keys_are_hardcoded_in_tracked_files():
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    pattern = re.compile(r"(?<![A-Z0-9])(AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])")
    secret = re.compile(r"aws_secret_access_key\s*[=:]\s*['\"]?[A-Za-z0-9/+]{40}")
    offenders = []
    for name in tracked:
        path = REPO_ROOT / name
        if path.suffix in {".png", ".jpg", ".pt", ".lock"} or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if pattern.search(text) or secret.search(text):
            offenders.append(name)
    assert offenders == []


def test_publication_uses_the_default_credential_chain_only():
    source = Path(publication_module.__file__).read_text(encoding="utf-8")

    assert "aws_access_key_id" not in source
    assert "aws_secret_access_key" not in source


# --- Integración con un S3 real (MinIO) ---------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("S3_INTEGRATION_ENDPOINT"),
    reason="define S3_INTEGRATION_ENDPOINT (y credenciales en el entorno) para correrla",
)
def test_publish_and_recover_against_a_real_s3_endpoint(package):
    import boto3

    s3 = boto3.client("s3", endpoint_url=os.environ["S3_INTEGRATION_ENDPOINT"])
    bucket = f"ops07-it-{uuid.uuid4().hex[:8]}"
    s3.create_bucket(Bucket=bucket)
    s3.put_bucket_versioning(Bucket=bucket, VersioningConfiguration={"Status": "Enabled"})
    try:
        _publish_and_verify_against(s3, bucket, package)
    finally:
        # Solo el bucket temporal que creó esta prueba en el S3 de integración.
        versions = s3.list_object_versions(Bucket=bucket).get("Versions", [])
        for version in versions:
            s3.delete_object(Bucket=bucket, Key=version["Key"], VersionId=version["VersionId"])
        s3.delete_bucket(Bucket=bucket)


def _publish_and_verify_against(s3, bucket, package):
    root = package["repo_root"]
    record = publish_to_s3(
        "1.0.0",
        s3=s3,
        bucket=bucket,
        prefix="models",
        registry_path=root / "reports" / "models" / "registry.json",
        publications_path=root / "reports" / "models" / "s3_publications.json",
        repo_root=root,
    )
    report = verify_publication(
        "1.0.0",
        s3=s3,
        publications_path=root / "reports" / "models" / "s3_publications.json",
        workdir=root / "clean",
        image=package["image"],
    )

    assert report["problems"] == []
    assert record["files"]["checkpoint/best.pt"]["version_id"]
