import json

import pytest

from policies.models import load_quality_policy
from presentation.contracts import QualityCheck, QualityReport
from presentation.gate import build_quality_report, main, run
from storage.settings import Settings
from tests._dataset_fixtures import write_coco_dataset

QUALITY_YAML = """
min_images_per_class:
  threshold: {min_images}
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


def _policy_path(tmp_path, min_images=2):
    path = tmp_path / "quality.yaml"
    path.write_text(QUALITY_YAML.format(min_images=min_images), encoding="utf-8")
    return path


def test_report_has_seven_checks(tmp_path):
    dataset_dir = write_coco_dataset(tmp_path / "dataset")
    policy_path = _policy_path(tmp_path)
    policy = load_quality_policy(policy_path)

    report = build_quality_report(
        dataset_dir=dataset_dir,
        policy=policy,
        dataset_version="test-1",
    )

    names = {check.check_name for check in report.checks}
    assert names == {
        "min_images_per_class",
        "max_imbalance_ratio",
        "max_small_object_ratio",
        "degenerate_boxes",
        "duplicate_similarity_threshold",
        "spatial_bias",
        "cross_split_leakage",
    }
    assert "cross_split_leakage" in names


def test_status_passed_when_everything_meets_threshold(tmp_path):
    dataset_dir = write_coco_dataset(tmp_path / "dataset", cats=2, dogs=2)
    policy_path = _policy_path(tmp_path, min_images=2)
    policy = load_quality_policy(policy_path)

    report = build_quality_report(dataset_dir=dataset_dir, policy=policy, dataset_version="test-1")

    assert report.status == "passed"
    assert all(check.passed for check in report.checks)


def test_status_failed_when_a_fail_action_check_does_not_pass(tmp_path):
    dataset_dir = write_coco_dataset(tmp_path / "dataset", cats=2, dogs=2)
    # Umbral imposible: ninguna clase real tiene 99999 imágenes.
    policy_path = _policy_path(tmp_path, min_images=99999)
    policy = load_quality_policy(policy_path)

    report = build_quality_report(dataset_dir=dataset_dir, policy=policy, dataset_version="test-1")

    assert report.status == "failed"
    min_images_check = next(c for c in report.checks if c.check_name == "min_images_per_class")
    assert not min_images_check.passed


def test_warn_action_check_failing_sets_warning_status_and_does_not_block(tmp_path):
    """P2-31: un check `action: warn` que no pasa no debe comportarse como un
    `fail` — status debe ser "warning" (no "failed"), y main() debe devolver
    0 (no 1), aunque el warn quede registrado y siga siendo visible."""
    dataset_dir = write_coco_dataset(tmp_path / "dataset", cats=1, dogs=2)
    policy_path = tmp_path / "quality.yaml"
    policy_path.write_text(
        """
min_images_per_class:
  threshold: 1
  action: fail
max_imbalance_ratio:
  threshold: 1.0
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
    policy = load_quality_policy(policy_path)

    report = build_quality_report(dataset_dir=dataset_dir, policy=policy, dataset_version="test-1")

    # 1 gato / 2 perros con umbral de imbalance 1.0: ratio 2/1=2 > 1.0 falla,
    # pero es `action: warn`; ningún check `fail` falla (min_images pasa con
    # umbral 1, degenerate_boxes pasa porque las cajas son válidas).
    imbalance_check = next(c for c in report.checks if c.check_name == "max_imbalance_ratio")
    assert not imbalance_check.passed
    assert imbalance_check.action == "warn"

    fail_checks = [c for c in report.checks if c.action == "fail"]
    assert all(c.passed for c in fail_checks), "ningún check fail debería estar fallando aquí"

    # Criterio 2 y 4: el warn queda registrado y sigue diferenciado de un fail.
    assert imbalance_check in report.checks
    assert report.status != "failed"

    # Criterio 3: el estado final refleja la advertencia, no un passed limpio.
    assert report.status == "warning"


def test_warn_only_report_does_not_block_main_exit_code(monkeypatch):
    """P2-31 criterio 1: un warn no detiene la ejecución. Se mockea run()
    (ya probado por separado) para no repetir I/O real ni tocar
    policies/quality.yaml — solo se verifica la rama de main() para
    status="warning", igual que test_main_returns_nonzero_exit_code_when_failed
    verifica la rama para status="failed"."""
    warn_only_report = QualityReport(
        schema_version="1.0",
        dataset_version="test-1",
        status="warning",
        checks=[
            QualityCheck(
                check_name="max_imbalance_ratio",
                passed=False,
                metric_value=2.0,
                details={},
                action="warn",
            )
        ],
    )
    monkeypatch.setattr("presentation.gate.run", lambda: warn_only_report)

    assert main() == 0


def test_min_images_per_class_check_reports_classes_below_minimum(tmp_path):
    dataset_dir = write_coco_dataset(tmp_path / "dataset", cats=1, dogs=2)
    policy_path = _policy_path(tmp_path, min_images=2)
    policy = load_quality_policy(policy_path)

    report = build_quality_report(dataset_dir=dataset_dir, policy=policy, dataset_version="test-1")

    min_images_check = next(c for c in report.checks if c.check_name == "min_images_per_class")
    below = min_images_check.details["classes_below_minimum"]
    assert [c["category_name"] for c in below] == ["cat"]


def test_run_writes_quality_json(tmp_path, monkeypatch):
    dataset_dir = write_coco_dataset(tmp_path / "dataset", cats=2, dogs=2)
    reports_dir = tmp_path / "reports"
    _policy_path(tmp_path, min_images=2)  # queda en tmp_path, no se usa: run() usa el real

    monkeypatch.setenv("DATABASE_URL", "mysql+pymysql://u:p@localhost/db")
    monkeypatch.setenv("MINIO_ENDPOINT", "localhost")
    monkeypatch.setenv("MINIO_PORT", "9000")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "ak")
    monkeypatch.setenv("MINIO_SECRET_KEY", "sk")
    monkeypatch.setenv("MINIO_BUCKET", "bucket")
    monkeypatch.setenv("DATASET_DIR", str(dataset_dir))
    monkeypatch.setenv("REPORTS_DIR", str(reports_dir))

    report = run(Settings())

    written = json.loads((reports_dir / "quality.json").read_text(encoding="utf-8"))
    assert QualityReport.model_validate(written) == report
    assert all("criterion" in check["details"] for check in written["checks"])
    assert written["schema_version"] == "1.0"
    assert written["status"] == report.status
    assert len(written["checks"]) == 7


def test_main_returns_nonzero_exit_code_when_failed(tmp_path, monkeypatch):
    dataset_dir = write_coco_dataset(tmp_path / "dataset", cats=2, dogs=2)
    monkeypatch.setenv("DATABASE_URL", "mysql+pymysql://u:p@localhost/db")
    monkeypatch.setenv("MINIO_ENDPOINT", "localhost")
    monkeypatch.setenv("MINIO_PORT", "9000")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "ak")
    monkeypatch.setenv("MINIO_SECRET_KEY", "sk")
    monkeypatch.setenv("MINIO_BUCKET", "bucket")
    monkeypatch.setenv("DATASET_DIR", str(dataset_dir))
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    # El quality.yaml real (min_images_per_class=300) siempre falla contra
    # un dataset sintético de cuatro imágenes: exit code 1 garantizado.

    assert main() == 1


@pytest.mark.parametrize("bad_field", ["min_images_per_class", "max_imbalance_ratio"])
def test_broken_policy_is_rejected_by_pydantic_not_a_key_error(tmp_path, bad_field):
    policy_path = tmp_path / "quality.yaml"
    document = QUALITY_YAML.format(min_images=2).replace(
        f"{bad_field}:\n", f"{bad_field}:\n  extra_unexpected_key: 1\n", 1
    )
    policy_path.write_text(document, encoding="utf-8")

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        load_quality_policy(policy_path)


def test_seven_criteria_use_the_same_custom_policy(tmp_path):
    dataset_dir = write_coco_dataset(tmp_path / "dataset", cats=2, dogs=4)
    policy = load_quality_policy(_policy_path(tmp_path, min_images=2))
    policy.max_imbalance_ratio.threshold = 2.0
    policy.max_small_object_ratio.threshold = 1.0
    policy.max_small_object_ratio.width_px = 41.0
    policy.max_small_object_ratio.height_px = 41.0
    policy.degenerate_boxes.threshold = 3.0
    policy.duplicate_similarity_threshold.threshold = 1.0
    policy.min_spatial_dispersion.threshold = 0.0
    # No policy_path: all configurations must use this in-memory policy,
    # not reload either the temporary YAML or the repository defaults.
    before = policy.model_dump()
    report = build_quality_report(dataset_dir=dataset_dir, policy=policy, dataset_version="test-1")
    expected = {
        "min_images_per_class": (2.0, 2.0, ">=", True, "fail"),
        "max_imbalance_ratio": (2.0, 2.0, "<=", True, "warn"),
        "max_small_object_ratio": (1.0, 1.0, "<=", True, "warn"),
        "degenerate_boxes": (0.0, 3.0, "<=", True, "fail"),
        "duplicate_similarity_threshold": (0.0, 0, "==", True, "warn"),
        "cross_split_leakage": (0.0, 0, "<=", True, "fail"),
        "spatial_bias": (0.0, 0.0, ">=", True, "warn"),
    }
    assert {check.check_name for check in report.checks} == expected.keys()
    for check in report.checks:
        value, threshold, operator, passed, action = expected[check.check_name]
        assert check.metric_value == value
        assert check.passed is passed
        assert check.action == action
        assert check.details["criterion"]["metric"] == "metric_value"
        assert check.details["criterion"]["threshold"] == threshold
        assert check.details["criterion"]["operator"] == operator
    checks = {check.check_name: check for check in report.checks}
    assert checks["max_small_object_ratio"].details["small_box_detection"] == {
        "width_px": 41.0,
        "height_px": 41.0,
        "operator": "<",
        "combination": "and",
    }
    duplicate = checks["duplicate_similarity_threshold"]
    assert duplicate.details["similarity_threshold"] == 1.0
    assert duplicate.details["similarity_operator"] == ">="
    assert duplicate.details["similarity_formula"] == "1 - hamming_distance / hash_bits"
    assert len(duplicate.details["image_pairs"]) == duplicate.metric_value
    assert report.status == "passed"
    assert policy.model_dump() == before
    assert QualityReport.model_validate_json(report.model_dump_json()) == report
    assert report.schema_version == "1.0"


def test_p230_phash_detection_is_not_a_pair_count_limit(tmp_path):
    dataset_dir = write_coco_dataset(tmp_path / "dataset", cats=2, dogs=2)
    images = dataset_dir / "images"
    (images / "dog.0.jpg").write_bytes((images / "cat.0.jpg").read_bytes())
    policy = load_quality_policy(_policy_path(tmp_path, min_images=1))
    policy.duplicate_similarity_threshold.threshold = 1.0
    report = build_quality_report(dataset_dir=dataset_dir, policy=policy, dataset_version="test-1")
    check = next(c for c in report.checks if c.check_name == "duplicate_similarity_threshold")
    assert check.metric_value == 1.0
    assert check.details["similarity_threshold"] == 1.0
    assert check.details["image_pairs"][0]["similarity"] == 1.0
    assert check.details["criterion"] == {
        "metric": "metric_value",
        "threshold": 0,
        "operator": "==",
    }
    assert not check.passed  # 1 pair must fail, even though 1 <= similarity threshold.
    assert check.action == "warn"
    assert report.status == "warning"


def test_p230_empty_category_cannot_pass_with_zero_ratio(tmp_path):
    dataset_dir = write_coco_dataset(tmp_path / "dataset", cats=0, dogs=3)
    policy = load_quality_policy(_policy_path(tmp_path, min_images=1))
    report = build_quality_report(dataset_dir=dataset_dir, policy=policy, dataset_version="test-1")
    check = next(c for c in report.checks if c.check_name == "max_imbalance_ratio")
    assert check.metric_value == 0.0
    assert check.details["ratio_defined"] is False
    assert check.details["criterion"]["requires_ratio_defined"] is True
    assert not check.passed
    assert check.action == "warn"
    assert report.status == "failed"  # Minimum images is an independent blocking check.
    assert QualityReport.model_validate_json(report.model_dump_json()) == report


@pytest.mark.parametrize(
    "action,threshold,expected",
    [("fail", 0, "failed"), ("warn", 0, "warning"), ("fail", 1, "warning")],
)
def test_real_leakage_from_contaminated_split_changes_status(
    tmp_path, monkeypatch, action, threshold, expected
):
    from dataclasses import replace

    from presentation import gate

    dataset = write_coco_dataset(tmp_path / "data", cats=2, dogs=2)
    (dataset / "images/dog.0.jpg").write_bytes((dataset / "images/cat.0.jpg").read_bytes())
    policy = load_quality_policy(_policy_path(tmp_path, min_images=1))
    policy.cross_split_leakage.action = action
    policy.cross_split_leakage.threshold = threshold
    original = gate.split_dataset

    def faulty_split(*args, **kwargs):
        result = original(*args, **kwargs)
        # Fault injection: same real pHash pair (1,3), valid coverage, different splits.
        return replace(result, assignments={"train": (1, 2), "val": (3,), "test": (4,)})

    monkeypatch.setattr(gate, "split_dataset", faulty_split)
    report = build_quality_report(dataset_dir=dataset, policy=policy, dataset_version="test-1")
    check = next(c for c in report.checks if c.check_name == "cross_split_leakage")
    assert check.metric_value == 1
    assert check.passed is (threshold >= 1)
    assert check.action == action
    assert check.details["criterion"] == {
        "metric": "metric_value",
        "threshold": threshold,
        "operator": "<=",
    }
    assert report.status == expected
    assert QualityReport.model_validate_json(report.model_dump_json()) == report


def test_zero_similarity_cannot_fabricate_three_splits(tmp_path):
    dataset = write_coco_dataset(tmp_path / "data", cats=3, dogs=3)
    policy = load_quality_policy(_policy_path(tmp_path))
    policy.duplicate_similarity_threshold.threshold = 0.0
    with pytest.raises(ValueError, match="three independent image groups"):
        build_quality_report(dataset_dir=dataset, policy=policy, dataset_version="test-1")


@pytest.mark.parametrize("action,status", [("warn", "warning"), ("fail", "failed")])
def test_spatial_metric_from_coco_controls_global_status(tmp_path, action, status):
    dataset = write_coco_dataset(tmp_path / "data", cats=2, dogs=2)
    policy = load_quality_policy(_policy_path(tmp_path))
    policy.min_spatial_dispersion.threshold = 0.15
    policy.min_spatial_dispersion.action = action
    report = build_quality_report(dataset_dir=dataset, policy=policy, dataset_version="test-1")
    spatial = next(c for c in report.checks if c.check_name == "spatial_bias")
    assert spatial.details["total_boxes"] == 4
    assert spatial.details["mean_center_x"] == 20 / 64
    assert spatial.metric_value == 0
    assert not spatial.passed
    assert report.status == status
    policy.min_spatial_dispersion.threshold = 0.0
    report = build_quality_report(dataset_dir=dataset, policy=policy, dataset_version="test-1")
    assert report.status == "passed"  # Equality at zero passes.
