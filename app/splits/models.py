"""Tier 4 — modelo Pydantic v2 de la configuración del split estratificado.

`val` es el nombre interno (ver `presentation/contracts.py`, que congela
`validation` como nombre público del reporte); aquí solo se valida que las
tres proporciones sean válidas y sumen 1.0, y que la semilla sea reproducible.
"""

from pathlib import Path
from typing import Annotated, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Ratio = Annotated[float, Field(gt=0, lt=1)]


class SplitsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    train: Ratio
    val: Ratio
    test: Ratio
    seed: int = 42

    @model_validator(mode="after")
    def ratios_sum_to_one(self) -> Self:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"train + val + test debe sumar 1.0; sumó {total}")
        return self


def load_splits_config(path: Path | None = None) -> SplitsConfig:
    """Lee `splits.yaml` completo y lo valida; nunca se usa el dict crudo."""
    config_path = path if path is not None else Path(__file__).with_name("splits.yaml")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return SplitsConfig.model_validate(raw)
