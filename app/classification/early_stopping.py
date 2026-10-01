"""ML-06 — early stopping sobre la pérdida de validation.

Métrica vigilada (predeclarada): `val_loss`, a minimizar. Una época mejora si su
valor es menor que el mejor anterior por más de `min_delta`; un valor igual, una
mejora menor que `min_delta` o un `NaN` no cuentan. Tras `patience` épocas
seguidas sin mejora, `should_stop` pasa a verdadero en esa época
(`stopped_epoch`). `best_epoch` es la época cuyos pesos se restauran al final.
`patience` y `min_delta` vienen de `TrainingParams`.
"""

import math

MONITOR = "val_loss"
MONITOR_MODE = "min"


class EarlyStopping:
    def __init__(self, *, patience: int, min_delta: float) -> None:
        if patience < 1:
            raise ValueError(f"patience debe ser >= 1: {patience}")
        if min_delta < 0:
            raise ValueError(f"min_delta debe ser >= 0: {min_delta}")
        self.patience = patience
        self.min_delta = min_delta
        self.best_value: float | None = None
        self.best_epoch: int | None = None
        self.stopped_epoch: int | None = None
        self.epochs_without_improvement = 0

    @property
    def should_stop(self) -> bool:
        return self.stopped_epoch is not None

    def update(self, epoch: int, value: float) -> bool:
        """Registra la métrica de `epoch`; devuelve si es la nueva mejor época."""
        improved = not math.isnan(value) and (
            self.best_value is None or value < self.best_value - self.min_delta
        )
        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.patience:
                self.stopped_epoch = epoch
        return improved
