import dvc_crops_stage
import pytest
from releases.service import P2ReleaseUnverifiableError


def test_unverifiable_p2_release_stops_crops_stage(monkeypatch, tmp_path):
    """Un release P2 no verificable debe detener ML-01 antes de generar crops."""

    class RejectingReleaseService:
        @classmethod
        def from_catalog(cls, catalog_path):
            return cls()

        def verify_release(self, release_version, repo_root):
            raise P2ReleaseUnverifiableError("invalid DVC provenance")

    extract_called = False

    def fake_extract(*args, **kwargs):
        nonlocal extract_called
        extract_called = True
        return object()

    monkeypatch.setattr(
        dvc_crops_stage,
        "P2ReleaseService",
        RejectingReleaseService,
        raising=False,
    )
    monkeypatch.setattr(
        dvc_crops_stage,
        "P2_RELEASE_VERSION",
        "v-invalid",
        raising=False,
    )
    monkeypatch.setattr(
        dvc_crops_stage,
        "P2_RELEASE_CATALOG",
        tmp_path / "p2_releases.json",
        raising=False,
    )
    monkeypatch.setattr(dvc_crops_stage, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(dvc_crops_stage, "CROPS_DIR", tmp_path / "crops")
    monkeypatch.setattr(dvc_crops_stage, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(
        dvc_crops_stage,
        "assert_quality_gate_passed",
        lambda: None,
    )
    monkeypatch.setattr(
        dvc_crops_stage,
        "merge_raw_batches",
        lambda _: {"images": [], "annotations": [], "categories": []},
    )
    monkeypatch.setattr(dvc_crops_stage, "extract_crops", fake_extract)
    monkeypatch.setattr(
        dvc_crops_stage,
        "write_crop_report",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(P2ReleaseUnverifiableError):
        dvc_crops_stage.write_crops()

    assert extract_called is False


def test_verified_p2_release_drives_crops_inputs(monkeypatch, tmp_path):
    """ML-01 debe consumir rutas y versión del release P2 verificado."""

    verified_release = {
        "release_version": "v-test",
        "coco_path": "verified/annotations",
        "images_path": "verified/images",
        "images_dvc_hash": "images-hash",
        "annotations_dvc_hash": "annotations-hash",
        "quality_report": "reports/releases/v-test/quality.json",
    }

    class ApprovedReleaseService:
        @classmethod
        def from_catalog(cls, catalog_path):
            return cls()

        def verify_release(self, release_version, repo_root):
            assert release_version == "v-test"
            assert repo_root == tmp_path
            return verified_release

    seen = {}

    def fake_merge(path):
        seen["annotations_dir"] = path
        return {"images": [], "annotations": [], "categories": []}

    def fake_extract(
        coco,
        *,
        images_dir,
        output_dir,
        dataset_version,
        provenance=None,
    ):
        seen["images_dir"] = images_dir
        seen["output_dir"] = output_dir
        seen["dataset_version"] = dataset_version
        seen["provenance"] = provenance
        return "crop-report"

    def fake_write_report(report, path):
        seen["report"] = report
        seen["report_path"] = path

    monkeypatch.setattr(
        dvc_crops_stage,
        "P2ReleaseService",
        ApprovedReleaseService,
    )
    monkeypatch.setattr(dvc_crops_stage, "P2_RELEASE_VERSION", "v-test")
    monkeypatch.setattr(
        dvc_crops_stage,
        "P2_RELEASE_CATALOG",
        tmp_path / "catalog.json",
    )
    monkeypatch.setattr(dvc_crops_stage, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(dvc_crops_stage, "CROPS_DIR", tmp_path / "crops")
    monkeypatch.setattr(dvc_crops_stage, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(
        dvc_crops_stage,
        "assert_quality_gate_passed",
        lambda: None,
    )
    monkeypatch.setattr(dvc_crops_stage, "merge_raw_batches", fake_merge)
    monkeypatch.setattr(dvc_crops_stage, "extract_crops", fake_extract)
    monkeypatch.setattr(
        dvc_crops_stage,
        "assert_min_images_per_class",
        lambda report, minimum: seen.update(minimum_checked=(report, minimum)),
        raising=False,
    )
    monkeypatch.setattr(
        dvc_crops_stage,
        "write_crop_report",
        fake_write_report,
    )

    dvc_crops_stage.write_crops()

    assert seen["annotations_dir"] == tmp_path / "verified" / "annotations"
    assert seen["images_dir"] == tmp_path / "verified" / "images"
    assert seen["output_dir"] == tmp_path / "crops"
    assert seen["dataset_version"] == "v-test"
    assert seen["provenance"] == {
        "release_version": "v-test",
        "images_dvc_hash": "images-hash",
        "annotations_dvc_hash": "annotations-hash",
        "quality_report": "reports/releases/v-test/quality.json",
    }
    assert seen["minimum_checked"] == ("crop-report", 300)
    assert seen["report"] == "crop-report"
    assert seen["report_path"] == tmp_path / "reports" / "crops.json"


def test_crops_dvc_stage_tracks_p2_release_provenance():
    """El stage ML debe depender explícitamente de la procedencia de OPS-01."""
    import yaml

    dvc = yaml.safe_load((dvc_crops_stage.REPO_ROOT / "dvc.yaml").read_text(encoding="utf-8"))
    deps = set(dvc["stages"]["crops"]["deps"])

    assert "../data/raw/annotations" in deps
    assert "../data/raw/images" in deps
    assert "../reports/versions.json" in deps
    assert "../reports/releases/v0.1.1/quality.json" in deps
    assert "releases/p2_releases.json" in deps
    assert "releases/service.py" in deps


def test_crops_stage_tracks_versioned_release_selection():
    """Cambiar el release seleccionado debe invalidar el stage DVC."""
    import yaml

    dvc = yaml.safe_load((dvc_crops_stage.REPO_ROOT / "dvc.yaml").read_text(encoding="utf-8"))
    crops = dvc["stages"]["crops"]

    assert "params" in crops
    assert {
        "releases/selection.yaml": [
            "release_version",
            "catalog",
        ]
    } in crops["params"]
