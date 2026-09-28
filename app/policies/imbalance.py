"""Adaptador de la política existente a la configuración de P2-19."""

from pathlib import Path

from analyzers.imbalance import ImbalanceConfig
from policies.models import load_quality_policy


def load_imbalance_config(path: Path | None = None) -> ImbalanceConfig:
    policy = load_quality_policy(path)
    return ImbalanceConfig(
        min_images_per_class=policy.min_images_per_class.threshold,
        threshold=policy.max_imbalance_ratio.threshold,
    )
