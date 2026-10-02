"""APP-04 / ML-04 — qué registra en MLflow un run de entrenamiento.

La pantalla Experiments (APP-04) y la evaluación leen estos nombres; el loop de
entrenamiento (ML-04) y el worker (OPS-04) deben escribir exactamente estos. Si
alguno cambia, cambia aquí y en un solo lugar.

Un run sin `TAG_DATASET_VERSION` y `TAG_MANIFEST_HASH` válidos no es un
entrenamiento trazable y no aparece en Experiments.
"""

from presentation.ml_contracts import TrainingParams

# Tags (procedencia del run).
TAG_DATASET_VERSION = "dataset_version"
"""Release de P2 con el que se entrenó, p. ej. `v0.1.1`."""

TAG_MANIFEST_HASH = "manifest_hash"
"""`manifest_hash` del manifiesto 70/20/10 (OPS-02), p. ej. `sha256:<64 hex>`."""

TAG_GIT_COMMIT = "mlflow.source.git.commit"
"""SHA completo (40 hex) del código. Es el tag estándar de MLflow; dentro de Docker
no hay `.git`, así que el worker debe ponerlo explícitamente."""

TAG_TRAINING_JOB = "training_job_id"
"""`job_id` de la cola de APP-03 que lanzó el run."""

# Parámetros: los mismos nombres que `TrainingParams` (los 7 de la rúbrica más
# seed/patience/min_delta), registrados con `mlflow.log_params`.
PARAM_NAMES = tuple(TrainingParams.model_fields)

# Métricas por época, con `step=epoch` (empieza en 1). Son las curvas de Experiments.
CURVE_METRICS = ("train_loss", "val_loss", "train_accuracy", "val_accuracy")
