"""ML-06 — curvas de entrenamiento a partir de las métricas reales de cada época.

`training_curves_png` dibuja loss y accuracy de train y validation por época y
marca la mejor época (`best_epoch`, pesos restaurados) y la época de early
stopping. `history_payload` es el mismo contenido en JSON, para auditar o
redibujar sin la imagen.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin GUI: corre en el worker y en CI

import matplotlib.pyplot as plt

METRICS = ("train_loss", "train_accuracy", "val_loss", "val_accuracy")


def history_payload(
    history: Sequence, *, best_epoch: int, stopped_epoch: int | None, monitor: str
) -> dict:
    return {
        "monitor": monitor,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "epochs": {
            "epoch": [entry.epoch for entry in history],
            **{name: [getattr(entry, name) for entry in history] for name in METRICS},
        },
    }


def training_curves_png(
    history: Sequence, path: Path, *, best_epoch: int, stopped_epoch: int | None
) -> Path:
    epochs = [entry.epoch for entry in history]
    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=100)
    for axis, kind in ((loss_axis, "loss"), (accuracy_axis, "accuracy")):
        for split, style in (("train", "-o"), ("val", "-s")):
            values = [getattr(entry, f"{split}_{kind}") for entry in history]
            axis.plot(epochs, values, style, label=f"{split}_{kind}", markersize=4)
        axis.axvline(
            best_epoch, color="tab:green", linestyle="--", label=f"best_epoch={best_epoch}"
        )
        if stopped_epoch is not None:
            axis.axvline(
                stopped_epoch, color="tab:red", linestyle=":", label=f"early stop={stopped_epoch}"
            )
        axis.set_xlabel("época")
        axis.set_ylabel(kind)
        axis.set_xticks(epochs)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    loss_axis.set_title("Loss por época")
    accuracy_axis.set_title("Accuracy por época")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="png")
    plt.close(figure)
    return path
