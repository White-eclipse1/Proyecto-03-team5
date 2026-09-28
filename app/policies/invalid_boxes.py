"""Parámetro de cajas inválidas usando el cargador de políticas compartido."""

from pathlib import Path

from analyzers.invalid_boxes import InvalidBoxConfig
from policies.models import load_quality_policy


def load_invalid_box_config(path: Path | None = None) -> InvalidBoxConfig:
    policy = load_quality_policy(path)
    return InvalidBoxConfig(threshold=policy.degenerate_boxes.threshold)
