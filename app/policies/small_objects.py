"""Lectura de configuración de P2-13, separada del analizador puro."""

from pathlib import Path

from analyzers.small_objects import SmallObjectConfig


def load_small_object_config(path: Path | None = None) -> SmallObjectConfig:
    """Lee únicamente los parámetros del analizador; no ejecuta la compuerta."""
    import yaml

    policy_path = path if path is not None else Path(__file__).with_name("quality.yaml")
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))["max_small_object_ratio"]
    return SmallObjectConfig.model_validate(
        {name: policy[name] for name in ("width_px", "height_px", "threshold")}
    )
