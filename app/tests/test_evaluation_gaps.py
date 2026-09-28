import json
from pathlib import Path

import dvc_gate_stage
import dvc_split_stage
import pytest
import yaml

from analyzers.spatial_bias import SpatialBiasConfig, analyze_spatial_bias
from policies.models import load_quality_policy
from presentation.release import cut_release, diff_releases
from presentation.splits import build_splits_report
from splits.models import SplitsConfig
from splits.stratified import split_dataset

ROOT = Path(__file__).resolve().parents[2]


def _quality_report(path: Path, status: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset_version": "test",
                "status": status,
                "checks": [
                    {
                        "check_name": "fixture",
                        "passed": status != "failed",
                        "metric_value": 0.0,
                        "details": {},
                        "action": "fail",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _policy_path(tmp_path: Path) -> Path:
    path = tmp_path / "quality.yaml"
    path.write_text(
        """
min_images_per_class:
  threshold: 1
  action: fail
max_imbalance_ratio:
  threshold: 999
  action: warn
max_small_object_ratio:
  threshold: 0.99
  width_px: 1
  height_px: 1
  action: warn
degenerate_boxes:
  threshold: 0
  action: fail
cross_split_leakage:
  threshold: 0
  action: fail
duplicate_similarity_threshold:
  threshold: 0.999
  action: warn
min_spatial_dispersion:
  threshold: 0.0
  action: warn
""",
        encoding="utf-8",
    )
    return path


def test_compose_does_not_version_operational_database_credentials():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "MARIADB_ROOT_PASSWORD: password" not in compose
    assert "root:password@" not in compose
    assert "-ppassword" not in compose
    assert "${MARIADB_ROOT_PASSWORD" in compose


def test_split_stage_rejects_failed_quality_gate_before_loading_dataset(tmp_path, monkeypatch):
    report_path = tmp_path / "reports" / "quality.json"
    marker_path = tmp_path / "reports" / ".quality_gate.passed"
    _quality_report(report_path, "failed")
    marker_path.write_text("passed\n", encoding="utf-8")

    monkeypatch.setattr(dvc_gate_stage, "REPORT_PATH", report_path)
    monkeypatch.setattr(dvc_gate_stage, "PASS_MARKER", marker_path)
    monkeypatch.setattr(
        dvc_split_stage,
        "load_quality_policy",
        lambda: pytest.fail("the split must validate the gate first"),
    )

    with pytest.raises(RuntimeError, match="failed"):
        dvc_split_stage.write_split_report()


def test_dvc_declares_gate_sources_and_split_guard_source():
    stages = yaml.safe_load((ROOT / "dvc.yaml").read_text(encoding="utf-8"))["stages"]

    assert "dvc_gate_stage.py" in stages["quality_gate"]["deps"]
    assert "presentation/contracts.py" in stages["quality_gate"]["deps"]
    assert "dvc_gate_stage.py" in stages["split"]["deps"]


def test_spatial_bias_publishes_median_and_percentiles():
    coco = {
        "images": [{"id": 1, "width": 100, "height": 100, "file_name": "a.jpg"}],
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": [
            {
                "id": index,
                "image_id": 1,
                "category_id": 1,
                "bbox": [x, 0, 10, 10],
            }
            for index, x in enumerate((5, 15, 75, 85), start=1)
        ],
    }

    result = analyze_spatial_bias(coco, SpatialBiasConfig(min_std_dev=0.0))

    assert result.details["median_center_x"] == pytest.approx(0.5)
    assert result.details["center_x_percentiles"] == pytest.approx(
        {"p25": 0.175, "p50": 0.5, "p75": 0.825, "p90": 0.87, "p95": 0.885}
    )


def test_release_diff_reports_annotation_delta_threshold_crossing_and_metric_delta(tmp_path):
    from tests._dataset_fixtures import write_coco_dataset

    reports_dir = tmp_path / "reports"
    policy = load_quality_policy(_policy_path(tmp_path))
    dataset_a = write_coco_dataset(tmp_path / "dataset-a", cats=3, dogs=3)
    cut_release("v0.1.0", dataset_dir=dataset_a, reports_dir=reports_dir, policy=policy)

    dataset_b = write_coco_dataset(tmp_path / "dataset-b", cats=6, dogs=3)
    cut_release("v0.2.0", dataset_dir=dataset_b, reports_dir=reports_dir, policy=policy)

    diff = diff_releases("v0.1.0", "v0.2.0", reports_dir=reports_dir)

    assert diff["annotations"] == {"from": 6, "to": 9, "delta": 3}
    assert diff["images_per_category"]["cat"] == {"from": 3, "to": 6}
    assert diff["image_deltas"]["cat"] == 3
    assert diff["classes_crossing_minimum"] == {
        "entered": [],
        "left": [],
    }
    assert diff["checks"]["max_small_object_ratio"]["metric_delta"] == pytest.approx(0.0)


def test_splits_report_exposes_class_distribution_and_leakage_summary():
    coco = {
        "images": [
            {"id": image_id, "width": 10, "height": 10, "file_name": f"{image_id}.jpg"}
            for image_id in range(6)
        ],
        "categories": [
            {"id": 1, "name": "cat"},
            {"id": 2, "name": "dog"},
        ],
        "annotations": [
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [0, 0, 5, 5],
                "area": 25,
                "iscrowd": 0,
            }
            for annotation_id, (image_id, category_id) in enumerate(
                ((0, 1), (1, 1), (2, 1), (3, 2), (4, 2), (5, 2)),
                start=1,
            )
        ],
    }
    result = split_dataset(
        coco,
        SplitsConfig(train=0.5, val=0.25, test=0.25, seed=7),
        image_contents={image_id: str(image_id).encode() for image_id in range(6)},
        duplicate_pairs=[],
    )

    report = build_splits_report(result, dataset_version="test")

    assert report.class_distribution["train"] == {"1": 1, "2": 2}
    assert report.leakage == {
        "status": "passed",
        "checked_groups": 6,
        "duplicate_groups": 0,
        "cross_split_groups": 0,
        "coverage": 6,
    }
