"""Parámetro de sesgo espacial usando el cargador de políticas compartido."""

from pathlib import Path

from analyzers.spatial_bias import SpatialBiasConfig
from policies.models import load_quality_policy


def load_spatial_bias_config(path: Path | None = None) -> SpatialBiasConfig:
    policy = load_quality_policy(path)
    return SpatialBiasConfig(min_std_dev=policy.min_spatial_dispersion.threshold)
