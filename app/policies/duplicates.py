"""Configuración pHash a partir del lector de políticas existente."""

from pathlib import Path

from analyzers.duplicates import DuplicateConfig
from policies.models import load_quality_policy


def load_duplicate_config(path: Path | None = None) -> DuplicateConfig:
    policy = load_quality_policy(path)
    return DuplicateConfig(threshold=policy.duplicate_similarity_threshold.threshold)
