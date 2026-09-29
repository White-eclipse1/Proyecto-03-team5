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
    assert result["images_dvc_hash"] == (
        "8a9a091030b6d1b1bf67f30771f78d5f.dir"
    )


def test_real_p2_catalog_preserves_verified_provenance():
    service = P2ReleaseService.from_catalog("releases/p2_releases.json")

    result = service.select_release("v1.0.0")

    assert result["release_version"] == "v1.0.0"
    assert result["p2_commit"] == (
        "7ce2c84689214088507dc50a2a6852a9672c6730"
    )
    assert result["dvc_hash"] == "457cc9fcf36cd8699b21f292ebbe49cc"
    assert result["images_dvc_hash"] == (
        "8a9a091030b6d1b1bf67f30771f78d5f.dir"
    )
    assert result["quality_report_dvc_hash"] == (
        "24d0df32896503d7622f1f268f3ac97a"
    )
    assert result["coco_path"] == "data/validated/coco.json"
    assert result["images_path"] == "data/images"


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
