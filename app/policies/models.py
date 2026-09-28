"""Tier 3 — modelo Pydantic v2 del contrato completo de `quality.yaml`.

Sin esto, un umbral no numérico, una `action` desconocida o una regla con una
clave de más/de menos llegan a la compuerta como un dict crudo y revientan
con un `KeyError` en vez de señalar el campo roto.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class PolicyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class ThresholdRule(PolicyModel):
    threshold: float = Field(ge=0)
    action: Literal["fail", "warn"]


class SmallObjectRule(ThresholdRule):
    width_px: float = Field(gt=0)
    height_px: float = Field(gt=0)


class QualityPolicy(PolicyModel):
    min_images_per_class: ThresholdRule
    max_imbalance_ratio: ThresholdRule
    max_small_object_ratio: SmallObjectRule
    degenerate_boxes: ThresholdRule
    cross_split_leakage: ThresholdRule
    duplicate_similarity_threshold: ThresholdRule
    min_spatial_dispersion: ThresholdRule


def load_quality_policy(path: Path | None = None) -> QualityPolicy:
    """Lee `quality.yaml` completo y lo valida; nunca se usa el dict crudo."""
    policy_path = path if path is not None else Path(__file__).with_name("quality.yaml")
    raw = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    return QualityPolicy.model_validate(raw)
