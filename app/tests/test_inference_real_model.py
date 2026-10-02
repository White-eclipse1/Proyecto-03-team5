"""APP-07 con el modelo real: el paquete 1.0.0 de OPS-06 sobre los recortes de test.

Necesita `dvc pull data/models.dvc` y `dvc pull crops` (no están en git); sin ellos se
omite. Agent Test del issue #24 con datos reales: la inferencia de `ml-api` reproduce,
recorte por recorte, las probabilidades que guardó ML-09 para el mismo checkpoint.
"""

import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from starlette.testclient import TestClient

from presentation.ml_contracts import InferenceResponse
from training.inference import PackageRegistryResolver
from training.queue import TrainingJobQueue
from training.server import create_app

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "reports" / "models" / "registry.json"
PACKAGE = ROOT / "data" / "models" / "dog-cat-resnet18" / "1.0.0" / "checkpoint" / "best.pt"
CROPS = ROOT / "data" / "crops"

pytestmark = pytest.mark.skipif(
    not PACKAGE.is_file() or not CROPS.is_dir(),
    reason="Requiere dvc pull data/models.dvc y dvc pull crops",
)


def test_real_model_reproduces_the_ml09_test_predictions(tmp_path):
    evaluation = json.loads(
        next((ROOT / "reports" / "evaluations" / "test").glob("*.json")).read_text(encoding="utf-8")
    )
    queue = TrainingJobQueue(create_engine(f"sqlite:///{tmp_path / 'jobs.db'}"))
    queue.create_tables()
    api = TestClient(
        create_app(
            queue=queue,
            reports_dir=ROOT / "reports",
            crops_dir=CROPS,
            models=PackageRegistryResolver(REGISTRY, repo_root=ROOT),
        )
    )

    for expected in evaluation["predictions"]:
        response = api.post(
            "/inference",
            json={
                "schema_version": "1.0",
                "model_name": "dog-cat-resnet18",
                "model_version": "1.0.0",
                "crop": {
                    "dataset_version": evaluation["dataset_version"],
                    "image_id": expected["image_id"],
                    "annotation_id": expected["annotation_id"],
                },
            },
        )
        assert response.status_code == 200, response.text
        result = InferenceResponse.model_validate(response.json())
        assert result.checkpoint_sha256 == (
            "84d6c88b4bd84c3e6f0229dd23f7f188ff85622bbb39e5c3988a97598fa90d52"
        )
        assert result.run_id == evaluation["run_id"]
        assert result.predicted_class == expected["predicted_class"]
        assert result.probabilities == pytest.approx(expected["probabilities"], abs=1e-5)
