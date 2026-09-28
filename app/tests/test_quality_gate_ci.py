import json

import dvc_gate_stage

from policies.models import load_quality_policy
from presentation.gate import build_quality_report
from tests._dataset_fixtures import write_coco_dataset

QUALITY_YAML = """
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
"""


def test_representative_coco_fixture_passes_quality_gate(tmp_path, monkeypatch):
    dataset_dir = write_coco_dataset(tmp_path / "dataset", cats=6, dogs=6)
    policy_path = tmp_path / "quality.yaml"
    policy_path.write_text(QUALITY_YAML, encoding="utf-8")

    report = build_quality_report(
        dataset_dir=dataset_dir,
        policy=load_quality_policy(policy_path),
        dataset_version="ci-fixture",
    )

    assert report.status != "failed"
    assert all(check.passed for check in report.checks if check.action == "fail")

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    report_path = reports_dir / "quality.json"
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    marker_path = reports_dir / ".quality_gate.passed"
    monkeypatch.setattr(dvc_gate_stage, "REPORT_PATH", report_path)
    monkeypatch.setattr(dvc_gate_stage, "PASS_MARKER", marker_path)

    assert dvc_gate_stage.enforce_quality_gate() == 0
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] != "failed"
    assert marker_path.read_text(encoding="utf-8") == "passed\n"
