"""APP-04 ↔ ML-04: lo que registra el loop de entrenamiento es lo que lee Experiments.

`tracking/run_schema.py` fija los nombres que la pantalla Experiments (y su API) lee
de MLflow; `classification/training.py` (ML-04) es quien los escribe. Si alguno de
los dos cambia un nombre, esta prueba falla en vez de que los runs desaparezcan del
portal o se queden sin curvas.
"""

import inspect
import json
from pathlib import Path

from classification import training
from classification.model import ModelConfig
from presentation.ml_contracts import TrainingParams
from tracking.run_schema import (
    CURVE_METRICS,
    PARAM_NAMES,
    TAG_DATASET_VERSION,
    TAG_GIT_COMMIT,
    TAG_MANIFEST_HASH,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "presentation" / "examples" / "ml"


def test_the_training_loop_logs_exactly_the_curves_experiments_draws():
    assert set(training.METRIC_NAMES) == set(CURVE_METRICS)


def test_the_training_loop_tags_the_provenance_experiments_requires():
    """Sin estos tags, `run_to_contract` descarta el run (no es un entrenamiento trazable)."""
    source = inspect.getsource(training.run_training)
    for tag in (TAG_DATASET_VERSION, TAG_MANIFEST_HASH, TAG_GIT_COMMIT):
        assert f'"{tag}":' in source, f"run_training no registra el tag {tag}"


def test_the_training_loop_logs_every_hyperparameter_under_its_contract_name():
    request = json.loads((EXAMPLES / "training_request.json").read_text(encoding="utf-8"))
    params = TrainingParams.model_validate(request["params"])
    config = ModelConfig.from_params(params, pretrained=False)

    logged = training._mlflow_params(params, config, 469, 128, False, "batch", 42)

    assert set(PARAM_NAMES) <= set(logged)
    assert logged["optimizer"] == params.optimizer
    assert logged["batch_size"] == str(params.batch_size)
    assert logged["hidden_layers"] == ",".join(str(width) for width in params.hidden_layers)
