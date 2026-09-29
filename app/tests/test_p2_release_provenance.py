import pytest
from releases.service import (
    P2ReleaseNotApprovedError,
    P2ReleaseNotFoundError,
    P2ReleaseService,
    P2ReleaseUnverifiableError,
)


def test_select_valid_approved_release():
    releases = {
        "v1.0.0": {
            "release_version": "v1.0.0",
            "dvc_hash": "abc123",
            "quality_status": "warning",
            "quality_report": "reports/v1.0.0/quality_report.json",
            "coco_path": "data/validated/coco.json",
            "images_path": "data/images",
        }
    }

    service = P2ReleaseService(releases)

    result = service.select_release("v1.0.0")

    assert result["release_version"] == "v1.0.0"
    assert result["dvc_hash"] == "abc123"
    assert result["quality_status"] == "warning"


def test_select_nonexistent_release_is_rejected():
    service = P2ReleaseService({})

    with pytest.raises(P2ReleaseNotFoundError):
        service.select_release("does-not-exist")


def test_select_failed_gate_release_is_rejected():
    releases = {
        "v-bad": {
            "release_version": "v-bad",
            "dvc_hash": "deadbeef",
            "quality_status": "failed",
            "quality_report": "reports/v-bad/quality_report.json",
            "coco_path": "data/validated/coco.json",
            "images_path": "data/images",
        }
    }

    service = P2ReleaseService(releases)

    with pytest.raises(P2ReleaseNotApprovedError):
        service.select_release("v-bad")


def test_release_without_dvc_provenance_is_rejected():
    releases = {
        "v-unverifiable": {
            "release_version": "v-unverifiable",
            "quality_status": "warning",
            "quality_report": "reports/v-unverifiable/quality_report.json",
            "coco_path": "data/validated/coco.json",
            "images_path": "data/images",
        }
    }

    service = P2ReleaseService(releases)

    with pytest.raises(P2ReleaseUnverifiableError):
        service.select_release("v-unverifiable")


def test_service_loads_releases_from_catalog(tmp_path):
    catalog = tmp_path / "p2_releases.json"
    catalog.write_text(
        """
        {
          "releases": [
            {
              "release_version": "v1.0.0",
              "dvc_hash": "457cc9fcf36cd8699b21f292ebbe49cc",
              "images_dvc_hash": "8a9a091030b6d1b1bf67f30771f78d5f.dir",
              "quality_status": "warning",
              "quality_report": "metadata/snapshots/v1.0.0/summary.json",
              "coco_path": "data/validated/coco.json",
              "images_path": "data/images"
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    service = P2ReleaseService.from_catalog(catalog)
    result = service.select_release("v1.0.0")

    assert result["release_version"] == "v1.0.0"
    assert result["dvc_hash"] == "457cc9fcf36cd8699b21f292ebbe49cc"
    assert result["images_dvc_hash"] == ("8a9a091030b6d1b1bf67f30771f78d5f.dir")


def test_real_p2_catalog_preserves_verified_provenance():
    service = P2ReleaseService.from_catalog("releases/p2_releases.json")

    result = service.select_release("v0.1.1")

    assert result["release_version"] == "v0.1.1"
    assert result["quality_status"] == "warning"
    assert result["dvc_hash"] == ("951150dd4fb053f4665089fcb37a1c87.dir")
    assert result["images_dvc_hash"] == ("951150dd4fb053f4665089fcb37a1c87.dir")
    assert result["annotations_dvc_hash"] == ("c7cb86ae7ece94ef7b853620e464a4d7.dir")
    assert result["images_dvc_file"] == "data/raw/images.dvc"
    assert result["annotations_dvc_file"] == "data/raw/annotations.dvc"
    assert result["quality_report"] == ("reports/releases/v0.1.1/quality.json")
    assert result["image_count"] == 600


def test_switching_release_changes_provenance():
    releases = {
        "v-test-1": {
            "release_version": "v-test-1",
            "dvc_hash": "coco-hash-1",
            "images_dvc_hash": "images-hash-1.dir",
            "quality_status": "warning",
            "quality_report": "reports/v-test-1/quality.json",
            "coco_path": "data/v-test-1/coco.json",
            "images_path": "data/v-test-1/images",
        },
        "v-test-2": {
            "release_version": "v-test-2",
            "dvc_hash": "coco-hash-2",
            "images_dvc_hash": "images-hash-2.dir",
            "quality_status": "warning",
            "quality_report": "reports/v-test-2/quality.json",
            "coco_path": "data/v-test-2/coco.json",
            "images_path": "data/v-test-2/images",
        },
    }

    service = P2ReleaseService(releases)

    first = service.select_release("v-test-1")
    second = service.select_release("v-test-2")

    assert first["release_version"] != second["release_version"]
    assert first["dvc_hash"] != second["dvc_hash"]
    assert first["images_dvc_hash"] != second["images_dvc_hash"]


def test_verify_release_rejects_mismatched_dvc_hash(tmp_path):
    images_dvc = tmp_path / "images.dvc"
    annotations_dvc = tmp_path / "annotations.dvc"
    quality_report = tmp_path / "quality.json"

    images_dvc.write_text(
        """
outs:
- md5: actual-images-hash.dir
  path: images
""",
        encoding="utf-8",
    )

    annotations_dvc.write_text(
        """
outs:
- md5: actual-annotations-hash.dir
  path: annotations
""",
        encoding="utf-8",
    )

    quality_report.write_text(
        """
{
  "dataset_version": "v-test",
  "status": "warning"
}
""",
        encoding="utf-8",
    )

    releases = {
        "v-test": {
            "release_version": "v-test",
            "quality_status": "warning",
            "quality_report": "quality.json",
            "images_dvc_file": "images.dvc",
            "images_dvc_hash": "WRONG-HASH.dir",
            "annotations_dvc_file": "annotations.dvc",
            "annotations_dvc_hash": "actual-annotations-hash.dir",
            "dvc_hash": "WRONG-HASH.dir",
            "coco_path": "annotations",
            "images_path": "images",
        }
    }

    service = P2ReleaseService(releases)

    with pytest.raises(P2ReleaseUnverifiableError):
        service.verify_release("v-test", tmp_path)


def test_real_p2_release_verifies_against_repository():
    service = P2ReleaseService.from_catalog("releases/p2_releases.json")

    result = service.verify_release("v0.1.1", "..")

    assert result["release_version"] == "v0.1.1"
    assert result["quality_status"] == "warning"
    assert result["image_count"] == 600


def test_unregistered_release_is_rejected(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    registry = reports_dir / "versions.json"
    registry.write_text(
        """
{
  "releases": [
    {
      "dataset_version": "v-official",
      "quality_file": "releases/v-official/quality.json",
      "splits_file": "releases/v-official/splits.json"
    }
  ]
}
""",
        encoding="utf-8",
    )

    catalog = tmp_path / "p2_releases.json"
    catalog.write_text(
        """
{
  "release_registry": "reports/versions.json",
  "releases": [
    {
      "release_version": "v-invented",
      "dvc_hash": "abc123",
      "quality_status": "warning",
      "quality_report": "reports/releases/v-invented/quality.json",
      "coco_path": "data/raw/annotations",
      "images_path": "data/raw/images"
    }
  ]
}
""",
        encoding="utf-8",
    )

    with pytest.raises(P2ReleaseUnverifiableError):
        P2ReleaseService.from_catalog(catalog)


def test_invalid_quality_status_is_rejected():
    releases = {
        "v-pending": {
            "release_version": "v-pending",
            "dvc_hash": "abc123",
            "quality_status": "pending",
            "quality_report": "reports/v-pending/quality.json",
            "coco_path": "data/raw/annotations",
            "images_path": "data/raw/images",
        }
    }

    service = P2ReleaseService(releases)

    with pytest.raises(P2ReleaseNotApprovedError):
        service.select_release("v-pending")


def test_quality_report_must_match_official_registry(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    registry = reports_dir / "versions.json"
    registry.write_text(
        """
{
  "releases": [
    {
      "dataset_version": "v1.0.0",
      "quality_file": "releases/v1.0.0/quality.json",
      "splits_file": "releases/v1.0.0/splits.json"
    }
  ]
}
""",
        encoding="utf-8",
    )

    catalog = tmp_path / "p2_releases.json"
    catalog.write_text(
        """
{
  "release_registry": "reports/versions.json",
  "releases": [
    {
      "release_version": "v1.0.0",
      "dvc_hash": "abc123",
      "quality_status": "warning",
      "quality_report": "reports/releases/v1.0.0/OTHER.json",
      "coco_path": "data/raw/annotations",
      "images_path": "data/raw/images"
    }
  ]
}
""",
        encoding="utf-8",
    )

    with pytest.raises(P2ReleaseUnverifiableError):
        P2ReleaseService.from_catalog(catalog)
