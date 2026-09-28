import json
import re
import unittest
from pathlib import Path

from pydantic import ValidationError

from analyzers.base import AnalyzerResult
from presentation.contracts import QualityCheck, QualityReport, SplitsReport, VersionsReport

EXAMPLES = Path(__file__).resolve().parents[1] / "presentation" / "examples"
MODELS = {
    "quality.json": QualityReport,
    "splits.json": SplitsReport,
    "versions.json": VersionsReport,
}


def reject_non_json_number(value):
    raise ValueError(f"Not a JSON number: {value}")


def load_example(filename):
    return json.loads(
        (EXAMPLES / filename).read_text(encoding="utf-8"),
        parse_constant=reject_non_json_number,
    )


class JsonContractsTests(unittest.TestCase):
    def test_quality_example(self):
        QualityReport.model_validate(load_example("quality.json"))

    def test_splits_example(self):
        SplitsReport.model_validate(load_example("splits.json"))

    def test_versions_example(self):
        VersionsReport.model_validate(load_example("versions.json"))

    def test_examples_round_trip_without_changing_values(self):
        for filename, model in MODELS.items():
            with self.subTest(filename=filename):
                document = load_example(filename)
                report = model.model_validate(document)
                self.assertEqual(json.loads(report.model_dump_json()), document)

    def test_catalog_references_existing_matching_reports(self):
        catalog = VersionsReport.model_validate(load_example("versions.json"))
        for release in catalog.releases:
            quality = QualityReport.model_validate(load_example(release.quality_file))
            splits = SplitsReport.model_validate(load_example(release.splits_file))
            self.assertEqual(release.dataset_version, quality.dataset_version)
            self.assertEqual(release.dataset_version, splits.dataset_version)
            self.assertEqual(catalog.schema_version, quality.schema_version)
            self.assertEqual(catalog.schema_version, splits.schema_version)

    def test_schema_version_is_required_and_frozen(self):
        for filename, model in MODELS.items():
            for value in (None, "2.0", 1):
                with self.subTest(filename=filename, value=value):
                    document = load_example(filename)
                    if value is None:
                        document.pop("schema_version")
                    else:
                        document["schema_version"] = value
                    with self.assertRaises(ValidationError):
                        model.model_validate(document)

    def test_unknown_fields_are_rejected(self):
        for filename, model in MODELS.items():
            with self.subTest(filename=filename):
                document = load_example(filename)
                document["unexpected"] = True
                with self.assertRaises(ValidationError):
                    model.model_validate(document)
        document = load_example("quality.json")
        document["checks"][0]["unexpected"] = True
        with self.assertRaises(ValidationError):
            QualityReport.model_validate(document)

    def test_quality_status_and_check_values_are_typed(self):
        for key, value in (("passed", "true"), ("action", "ignore"), ("metric_value", "0.4")):
            with self.subTest(key=key):
                document = load_example("quality.json")
                document["checks"][0][key] = value
                with self.assertRaises(ValidationError):
                    QualityReport.model_validate(document)
        document = load_example("quality.json")
        document["status"] = "unknown"
        with self.assertRaises(ValidationError):
            QualityReport.model_validate(document)

    def test_check_details_are_json_and_metrics_finite(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                document = load_example("quality.json")
                document["checks"][0]["metric_value"] = value
                with self.assertRaises(ValidationError):
                    QualityReport.model_validate(document)
        document = load_example("quality.json")
        document["checks"][0]["details"] = {"unsupported": object()}
        with self.assertRaises(ValidationError):
            QualityReport.model_validate(document)

    def test_empty_or_duplicate_checks_are_rejected(self):
        document = load_example("quality.json")
        document["checks"].append(document["checks"][0].copy())
        with self.assertRaises(ValidationError):
            QualityReport.model_validate(document)
        document["checks"] = []
        with self.assertRaises(ValidationError):
            QualityReport.model_validate(document)

    def test_existing_analyzer_result_can_be_formatted(self):
        result = AnalyzerResult(check_name="degenerate_boxes", passed=True, metric_value=0.0)
        check = QualityCheck.model_validate({**result.model_dump(), "action": "fail"})
        self.assertEqual(check.metric_value, result.metric_value)
        other = QualityCheck.model_validate({**result.model_dump(), "action": "fail"})
        check.details["note"] = "example"
        self.assertEqual(other.details, {})

    def test_split_counts_ratios_and_types(self):
        for field, value in (
            ("image_count", -1),
            ("image_count", True),
            ("image_count", "840"),
            ("image_count", 839),
            ("ratio", -0.1),
            ("ratio", 1.1),
            ("ratio", 0.6),
        ):
            with self.subTest(field=field, value=value):
                document = load_example("splits.json")
                document["splits"]["train"][field] = value
                with self.assertRaises(ValidationError):
                    SplitsReport.model_validate(document)

    def test_all_three_named_splits_are_required(self):
        for name in ("train", "validation", "test"):
            with self.subTest(name=name):
                document = load_example("splits.json")
                document["splits"].pop(name)
                with self.assertRaises(ValidationError):
                    SplitsReport.model_validate(document)

    def test_zero_total_and_duplicate_release_are_rejected(self):
        document = load_example("splits.json")
        document["total_images"] = 0
        with self.assertRaises(ValidationError):
            SplitsReport.model_validate(document)
        document = load_example("versions.json")
        document["releases"].append(document["releases"][0].copy())
        with self.assertRaises(ValidationError):
            VersionsReport.model_validate(document)

    def test_report_references_are_relative_and_have_exact_filenames(self):
        for reference in (
            "/tmp/quality.json",
            "../quality.json",
            "https://example.invalid/quality.json",
            "C:\\quality.json",
            "reports//quality.json",
            "other.json",
        ):
            with self.subTest(reference=reference):
                document = load_example("versions.json")
                document["releases"][0]["quality_file"] = reference
                with self.assertRaises(ValidationError):
                    VersionsReport.model_validate(document)
        document = load_example("versions.json")
        document["releases"][0]["quality_file"] = "demo-v1.0.0/quality.json"
        VersionsReport.model_validate(document)

    def test_examples_have_no_secret_markers_or_local_paths(self):
        markers = re.compile(
            r"(?:AKIA|ASIA)[A-Z0-9]{16}|-----BEGIN .*PRIVATE KEY|"
            r"(?i:password|secret_key|access_key|session_token)|"
            r"/Users/|/home/|/tmp/|[A-Za-z]:\\|file://"
        )
        for filename in MODELS:
            with self.subTest(filename=filename):
                self.assertIsNone(markers.search((EXAMPLES / filename).read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()


def test_p230_example_preserves_v1_and_documents_all_six_criteria():
    report = QualityReport.model_validate(load_example("quality.json"))
    expected = {
        "min_images_per_class": (400.0, 300, ">=", True, "fail"),
        "max_imbalance_ratio": (1.0, 20, "<=", True, "warn"),
        "max_small_object_ratio": (0.45, 0.4, "<=", False, "warn"),
        "degenerate_boxes": (0.0, 0, "<=", True, "fail"),
        "duplicate_similarity_threshold": (1.0, 0, "==", False, "warn"),
        "spatial_bias": (0.2, 0.15, ">=", True, "warn"),
    }
    assert {check.check_name for check in report.checks} == expected.keys()
    for check in report.checks:
        value, threshold, operator, passed, action = expected[check.check_name]
        assert check.metric_value == value
        assert check.details["criterion"]["threshold"] == threshold
        assert check.details["criterion"]["operator"] == operator
        assert check.details["criterion"]["metric"] == "metric_value"
        assert check.passed is passed
        assert check.action == action
        assert "threshold" not in check.model_dump()
    assert report.schema_version == "1.0"
    assert report.status == "warning"
    assert QualityReport.model_validate_json(report.model_dump_json()) == report


def test_quality_v1_accepts_historical_six_and_new_seven_checks():
    document = load_example("quality.json")
    historical = QualityReport.model_validate(document)
    assert len(historical.checks) == 6
    document["checks"].append(
        {
            "check_name": "cross_split_leakage",
            "passed": True,
            "metric_value": 0.0,
            "action": "fail",
            "details": {
                "criterion": {"metric": "metric_value", "operator": "<=", "threshold": 0},
                "image_pairs": [],
                "total_pairs_evaluated": 0,
            },
        }
    )
    current = QualityReport.model_validate(document)
    assert len(current.checks) == 7
    assert current.schema_version == historical.schema_version == "1.0"
    assert QualityReport.model_validate_json(current.model_dump_json()) == current
