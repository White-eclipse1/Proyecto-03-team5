"""Tier 5 — configuración de infraestructura y política de calidad.

Único módulo autorizado a leer `.env`/variables de entorno (ver la regla de
la compuerta de acoplamiento en `app/tests/test_architecture.py`).
`quality.yaml` entra por el mismo mecanismo de fuentes de `pydantic-settings`,
así que una variable de entorno faltante o un umbral roto en la política se
rechazan aquí, con un error que nombra el campo, antes de que cualquier otro
tier los use.

`dataset_dir` (padre de `annotations/`/`images/`) y `reports_dir` (donde
`presentation/gate.py` escribe `quality.json`) también viven aquí — dentro
de Docker apuntan a los volúmenes montados (ver docker-compose.yml), no a
rutas calculadas desde `__file__`, porque el Dockerfile aplana `app/` a
`/app` y una ruta relativa al código dejaría de tener sentido ahí.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import SecretStr, field_validator
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from policies.models import QualityPolicy

APP_ROOT = Path(__file__).resolve().parent.parent


class YamlSectionSource(PydanticBaseSettingsSource):
    """Fuente de `pydantic-settings` que anida un YAML completo bajo un campo."""

    def __init__(self, settings_cls: type[BaseSettings], *, field_name: str, yaml_path: Path):
        super().__init__(settings_cls)
        self._field_name = field_name
        self._yaml_path = yaml_path

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        if not self._yaml_path.exists():
            return {}
        raw = yaml.safe_load(self._yaml_path.read_text(encoding="utf-8"))
        return {self._field_name: raw}


def _infrastructure_source(source: EnvSettingsSource) -> EnvSettingsSource:
    """Keep original source options, excluding QUALITY before JSON decoding."""
    source.env_vars = {
        key: value for key, value in source.env_vars.items() if key.lower() != "quality"
    }
    return source


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(APP_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str
    minio_endpoint: str
    minio_port: int
    minio_access_key: str
    minio_secret_key: str
    minio_use_ssl: bool = False
    minio_bucket: str

    dataset_dir: Path
    reports_dir: Path
    dataset_version: str = "local-dev"

    # Copilot (P2-52). Opcional: el gate y el resto del pipeline no la necesitan,
    # así que su ausencia solo deshabilita el chat, no impide arrancar.
    anthropic_api_key: SecretStr | None = None
    copilot_model: str = "claude-opus-5"
    # Escucha solo en loopback por defecto; docker-compose lo abre con COPILOT_HOST.
    copilot_host: str = "127.0.0.1"
    copilot_port: int = 8000

    quality: QualityPolicy

    @field_validator("anthropic_api_key", mode="before")
    @classmethod
    def blank_api_key_means_not_configured(cls, value: Any) -> Any:
        """docker-compose expande `${ANTHROPIC_API_KEY:-}` a "" cuando .env no la define."""
        return None if isinstance(value, str) and not value.strip() else value

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            _infrastructure_source(env_settings),
            _infrastructure_source(dotenv_settings),
            YamlSectionSource(
                settings_cls,
                field_name="quality",
                yaml_path=APP_ROOT / "policies" / "quality.yaml",
            ),
            file_secret_settings,
        )
