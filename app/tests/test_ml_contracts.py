"""APP-01: contratos de las pantallas de modelos (Training, Experiments, Evaluation,
Models, Inference).

Los ejemplos de `presentation/examples/ml/` son el corpus compartido con el frontend
(`frontend/tests/ml-contracts.test.ts` valida exactamente los mismos archivos con Zod).
Un cambio incompatible en un solo lado rompe los tests del otro.
"""

import json
import unittest
from pathlib import Path

from pydantic import ValidationError

from presentation.ml_contracts import (
    CONTRACTS,
    EvaluationsResponse,
    InferenceRequest,
    InferenceResponse,
    ModelsResponse,
    ReleaseProvenance,
    RunCurvesResponse,
    RunsResponse,
    TrainingJobRequest,
    TrainingJobsResponse,
    TrainingManifest,
    TrainingParams,
    training_blocked_reason,
    training_request_rejection,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "presentation" / "examples" / "ml"
VALID = {
    "training_jobs": "training_jobs.json",
    "training_logs": "training_logs.json",
    "runs": "runs.json",
    "run_curves": "run_curves.json",
    "evaluations": "evaluations.json",
    "models": "models.json",
    "inference_request": "inference_request.json",
    "inference": "inference.json",
    "error": "error.json",
    "training_request": "training_request.json",
    "provenance": "provenance.json",
    "manifest": "manifest.json",
}


def reject_non_json_number(value):
    raise ValueError(f"Not a JSON number: {value}")


def load(filename):
    return json.loads(
        (EXAMPLES / filename).read_text(encoding="utf-8"),
        parse_constant=reject_non_json_number,
    )


class MlContractExamplesTests(unittest.TestCase):
    def test_every_contract_has_a_valid_example(self):
        self.assertEqual(set(CONTRACTS), set(VALID))

    def test_examples_validate_and_round_trip_without_changing_values(self):
        for contract, filename in VALID.items():
            with self.subTest(contract=contract):
                document = load(filename)
                model = CONTRACTS[contract].model_validate(document)
                self.assertEqual(json.loads(model.model_dump_json()), document)

    def test_schema_version_is_required_and_frozen(self):
        for contract, filename in VALID.items():
            for value in (None, "2.0", 1):
                with self.subTest(contract=contract, value=value):
                    document = load(filename)
                    document["schema_version"] = value
                    with self.assertRaises(ValidationError):
                        CONTRACTS[contract].model_validate(document)

    def test_empty_lists_are_valid_empty_states(self):
        for contract, key in (
            ("training_jobs", "jobs"),
            ("runs", "runs"),
            ("evaluations", "evaluations"),
            ("models", "models"),
        ):
            with self.subTest(contract=contract):
                CONTRACTS[contract].model_validate({"schema_version": "1.0", key: []})


class SharedInvalidCasesTests(unittest.TestCase):
    """Cada caso de `invalid_cases.json` debe fallar aquí y en Zod."""

    def test_invalid_cases_are_rejected(self):
        cases = load("invalid_cases.json")
        self.assertGreater(len(cases), 0)
        names = [case["name"] for case in cases]
        self.assertEqual(len(names), len(set(names)), "nombres de caso duplicados")
        for case in cases:
            with self.subTest(case=case["name"]), self.assertRaises(ValidationError):
                CONTRACTS[case["contract"]].model_validate(case["document"])

    def test_invalid_cases_point_at_the_declared_field(self):
        """Pydantic y Zod deben señalar el mismo campo, no solo rechazar."""
        cases = [case for case in load("invalid_cases.json") if "field" in case]
        self.assertGreater(len(cases), 0)
        for case in cases:
            with self.subTest(case=case["name"]):
                with self.assertRaises(ValidationError) as raised:
                    CONTRACTS[case["contract"]].model_validate(case["document"])
                paths = [
                    ".".join(str(part) for part in error["loc"])
                    for error in raised.exception.errors()
                ]
                self.assertTrue(
                    any(
                        path == case["field"] or path.startswith(f"{case['field']}.")
                        for path in paths
                    ),
                    f"{case['field']} no está en {paths}",
                )


class TrainingRequestTests(unittest.TestCase):
    """APP-02: los parámetros del job y la regla del Quality Gate."""

    def test_request_carries_all_required_hyperparameters(self):
        request = TrainingJobRequest.model_validate(load("training_request.json"))
        self.assertEqual(
            set(type(request.params).model_fields),
            {
                "optimizer",
                "batch_size",
                "max_epochs",
                "learning_rate",
                "image_size",
                "hidden_layers",
                "dropout",
                "seed",
                "patience",
                "min_delta",
            },
        )

    def test_accepts_boundary_values(self):
        document = load("training_request.json")
        document["params"].update(
            batch_size=1,
            max_epochs=1,
            patience=1,
            learning_rate=1.0,
            dropout=0.0,
            image_size=32,
            hidden_layers=[],
            seed=0,
            min_delta=0.0,
        )
        TrainingJobRequest.model_validate(document)

    def release_files(self):
        return (
            ReleaseProvenance.model_validate(load("provenance.json")),
            TrainingManifest.model_validate(load("manifest.json")),
        )

    def test_failed_quality_gate_blocks_training(self):
        provenance, manifest = self.release_files()
        self.assertIsNotNone(training_blocked_reason("failed", provenance, manifest))
        self.assertIsNone(training_blocked_reason("warning", provenance, manifest))
        self.assertIsNone(training_blocked_reason("passed", provenance, manifest))

    def test_missing_provenance_or_manifest_blocks_training(self):
        provenance, manifest = self.release_files()
        self.assertIn("provenance.json", training_blocked_reason("passed", None, manifest))
        self.assertIn("manifest.json", training_blocked_reason("passed", provenance, None))
        other = manifest.model_copy(update={"dataset_version": "demo-v2.0.0"})
        self.assertIsNotNone(training_blocked_reason("passed", provenance, other))

    def test_request_is_bound_to_the_release_manifest_hash(self):
        provenance, manifest = self.release_files()
        request = TrainingJobRequest.model_validate(load("training_request.json"))
        self.assertEqual(request.manifest_hash, manifest.manifest_hash)
        self.assertIsNone(training_request_rejection(request, "warning", provenance, manifest))
        for update, message in (
            ({"manifest_hash": "md5:" + "0" * 32}, "manifest_hash"),
            ({"dataset_version": "demo-v2.0.0"}, "release solicitado"),
        ):
            with self.subTest(update=update):
                other = request.model_copy(update=update)
                reason = training_request_rejection(other, "warning", provenance, manifest)
                self.assertIn(message, reason)
        self.assertIsNotNone(training_request_rejection(request, "passed", None, manifest))

    def test_manifest_accepts_integer_rounding_of_70_20_10(self):
        document = load("manifest.json")
        document["total_images"] = 601
        for name, count in (("train", 421), ("validation", 120), ("test", 60)):
            document["splits"][name] = {"image_count": count, "ratio": count / 601}
        TrainingManifest.model_validate(document)


class ClassificationContractTests(unittest.TestCase):
    """El modelo clasifica recortes: métricas multiclase y una clase por inferencia."""

    def test_training_params_include_the_seven_required(self):
        self.assertLessEqual(
            {
                "optimizer",
                "batch_size",
                "max_epochs",
                "learning_rate",
                "image_size",
                "hidden_layers",
                "dropout",
            },
            set(TrainingParams.model_fields),
        )
        job = TrainingJobsResponse.model_validate(load("training_jobs.json")).jobs[0]
        self.assertIsInstance(job.params, TrainingParams)

    def test_evaluation_metrics_recompute_from_confusion_matrix(self):
        evaluation = EvaluationsResponse.model_validate(load("evaluations.json")).evaluations[0]
        matrix = evaluation.confusion_matrix
        total = sum(map(sum, matrix))
        self.assertEqual(total, len(evaluation.predictions))
        trace = sum(matrix[i][i] for i in range(len(matrix)))
        self.assertAlmostEqual(evaluation.metrics.accuracy_top1, trace / total, places=3)
        self.assertAlmostEqual(
            evaluation.metrics.f1_macro,
            sum(entry.f1 for entry in evaluation.per_class) / len(evaluation.per_class),
            places=3,
        )
        self.assertEqual(
            [entry.support for entry in evaluation.per_class], [sum(row) for row in matrix]
        )

    def test_metrics_accept_rounding_but_not_drift(self):
        for accuracy, valid in ((0.8005, True), (0.802, False)):
            with self.subTest(accuracy=accuracy):
                document = load("evaluations.json")
                document["evaluations"][0]["metrics"]["accuracy_top1"] = accuracy
                if valid:
                    EvaluationsResponse.model_validate(document)
                else:
                    with self.assertRaises(ValidationError):
                        EvaluationsResponse.model_validate(document)

    def test_inference_returns_one_class_for_the_requested_crop(self):
        request = InferenceRequest.model_validate(load("inference_request.json"))
        response = InferenceResponse.model_validate(load("inference.json"))
        self.assertEqual(response.crop, request.crop)
        self.assertEqual(
            (response.model_name, response.model_version),
            (request.model_name, request.model_version),
        )
        self.assertAlmostEqual(sum(response.probabilities.values()), 1.0, places=3)
        self.assertEqual(
            response.predicted_class,
            max(response.probabilities, key=response.probabilities.__getitem__),
        )
        self.assertNotIn("predictions", InferenceResponse.model_fields)


class TraceabilityTests(unittest.TestCase):
    """Los IDs reales encadenan dataset → job → run → checkpoint → modelo → inferencia."""

    def test_dataset_version_and_model_version_are_distinct_fields(self):
        model = ModelsResponse.model_validate(load("models.json")).models[0]
        self.assertNotEqual(model.dataset_version, model.model_version)
        self.assertIn("dataset_version", type(model).model_fields)
        self.assertIn("model_version", type(model).model_fields)

    def test_examples_reference_each_other_consistently(self):
        jobs = TrainingJobsResponse.model_validate(load("training_jobs.json")).jobs
        runs = {run.run_id: run for run in RunsResponse.model_validate(load("runs.json")).runs}
        models = {
            (model.model_name, model.model_version): model
            for model in ModelsResponse.model_validate(load("models.json")).models
        }
        evaluations = EvaluationsResponse.model_validate(load("evaluations.json")).evaluations
        inference = InferenceResponse.model_validate(load("inference.json"))

        for job in jobs:
            if job.run_id is not None:
                run = runs[job.run_id]
                self.assertEqual(
                    (job.dataset_version, job.manifest_hash, job.experiment_id),
                    (run.dataset_version, run.manifest_hash, run.experiment_id),
                )
        for evaluation in evaluations:
            self.assertEqual(runs[evaluation.run_id].manifest_hash, evaluation.manifest_hash)
            if evaluation.model_version is not None:
                model = models[(evaluation.model_name, evaluation.model_version)]
                self.assertEqual(model.checkpoint, evaluation.checkpoint)
        served = models[(inference.model_name, inference.model_version)]
        self.assertEqual(
            (served.run_id, served.dataset_version),
            (inference.run_id, inference.dataset_version),
        )

    def test_curves_end_at_the_run_latest_metrics(self):
        """APP-04: MLflow reporta como métrica del run el último valor de su historial."""
        curves = RunCurvesResponse.model_validate(load("run_curves.json"))
        runs = {run.run_id: run for run in RunsResponse.model_validate(load("runs.json")).runs}
        run = runs[curves.run_id]

        for name, points in curves.curves.items():
            if name in run.metrics:
                self.assertEqual(points[-1].value, run.metrics[name])


if __name__ == "__main__":
    unittest.main()
