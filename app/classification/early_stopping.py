"""ML-06 — early stopping sobre la pérdida de validation.

Métrica vigilada (predeclarada): `val_loss`, a minimizar. Se separan dos
decisiones:

- **Mejor época** (`best_epoch`, los pesos que se restauran): la del menor
  `val_loss` finito visto, aunque haya bajado menos que `min_delta`.
- **Paciencia**: solo una mejora mayor que `min_delta` respecto del menor valor
  anterior la reinicia. Tras `patience` épocas seguidas sin una mejora así,
  `should_stop` pasa a verdadero en esa época (`stopped_epoch`).

Una pérdida no finita (`NaN`, `+inf`, `-inf`) nunca es la mejor época y cuenta
como época sin mejora. `patience` y `min_delta` vienen de `TrainingParams`.
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
        """Registra la métrica de `epoch`; devuelve si es la nueva mejor época (menor valor)."""
        finite = math.isfinite(value)
        previous = self.best_value
        is_best = finite and (previous is None or value < previous)
        resets_patience = finite and (previous is None or value < previous - self.min_delta)
        if is_best:
            self.best_value = value
            self.best_epoch = epoch
        if resets_patience:
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.patience:
                self.stopped_epoch = epoch
        return is_best
